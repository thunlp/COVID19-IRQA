import os
import sys
import time
import tqdm
import json
import torch

import logging
import argparse
import traceback
import numpy as np
from tqdm import tqdm

sys.path.append("..")
sys.path.append("../MyCode")

import config
import utils
from functions import filer
from utils import pred_arranger, pred_saver

from bertkpe import Idx2Tag, Tag2Idx, BertTokenizer, RobertaTokenizer, Decode_Candidate_Number
from bertkpe import dataloader, generator
from model import KeyphraseSpanExtraction

torch.backends.cudnn.benchmark=True
from torch.utils.data.distributed import DistributedSampler
from tensorboardX import SummaryWriter
logger = logging.getLogger()


tokenizer_class = {"bert-base-cased":BertTokenizer, "roberta-base":RobertaTokenizer}

# -------------------------------------------------------------------------------------------
# Decoder Selector
# -------------------------------------------------------------------------------------------
def select_decoder(name):
    if name in ["bert2tag", "roberta2tag"]:
        return bert2tag_decoder
    
    elif name in ["bert2gram", "roberta2gram"]:
        return bert2gram_decoder
    
    raise RuntimeError('Invalid retriever class: %s' %name)
    
    
# -------------------------------------------------------------------------------------------
# Bert2Tag
def bert2tag_decoder(args, data_loader, dataset, model, test_input_refactor, pred_arranger, mode):
    logging.info(' Bert2Tag : Start Generating Keyphrases for %s  ...'% mode)
    test_time = utils.Timer()

    tot_examples = 0
    tot_predictions = []
    
    for step, batch in enumerate(tqdm(data_loader)):
        inputs, indices, lengths = test_input_refactor(batch, model.args.device)
        try:
            logit_lists = model.test_bert2tag(inputs, lengths)
        except:
            logging.error(str(traceback.format_exc()))
            continue
            
        # decode logits to phrase per batch
        params = {'examples': dataset.examples, 
                  'logit_lists':logit_lists, 
                  'indices':indices, 
                  'max_phrase_words':args.max_phrase_words, 
                  'pooling':args.tag_pooling, 
                  'stem_flag':False}
        
        if args.run_mode in ["train", "test"]:
            params['return_num'] = Decode_Candidate_Number[args.dataset_class]            
            
        batch_predictions = generator.tag2phrase(**params)
        tot_predictions.extend(batch_predictions)
    
    candidate = pred_arranger(tot_predictions)
    return candidate


# -------------------------------------------------------------------------------------------
# Bert2Gram
def bert2gram_decoder(args, data_loader, dataset, model, test_input_refactor, pred_arranger, mode):
    logging.info(' Bert2Gram : Start Generating Keyphrases for %s  ...'% mode)
    test_time = utils.Timer()

    tot_examples = 0
    tot_predictions = []
        
    for step, batch in enumerate(tqdm(data_loader)):
        
        inputs, indices, lengths = test_input_refactor(batch, model.args.device)
        try:
            logit_lists = model.test_bert2gram(inputs, lengths, args.max_phrase_words)
        except:
            logging.error(str(traceback.format_exc()))
            continue
            
        # decode logits to phrase per batch
        params = {'examples': dataset.examples, 
                  'logit_lists':logit_lists, 
                  'indices':indices, 
                  'max_phrase_words':args.max_phrase_words,
                  'return_num':Decode_Candidate_Number[args.dataset_class], 
                  'stem_flag':False}
        
        batch_predictions = generator.gram2phrase(**params)
        tot_predictions.extend(batch_predictions)
    
    candidate = pred_arranger(tot_predictions)
    return candidate


# -------------------------------------------------------------------------------------------
# Main Function
# -------------------------------------------------------------------------------------------
if __name__ == "__main__":
    
    # setting args
    parser = argparse.ArgumentParser('BertKPE', formatter_class=argparse.ArgumentDefaultsHelpFormatter)
    config.add_default_args(parser)

    args = parser.parse_args()
    config.init_args_config(args)
    
    # -------------------------------------------------------------------------------------------
    # Setup distant debugging if needed
    if args.server_ip and args.server_port:
        # Distant debugging - see https://code.visualstudio.com/docs/python/debugging#_attach-to-a-local-script
        import ptvsd
        print("Waiting for debugger attach")
        ptvsd.enable_attach(address=(args.server_ip, args.server_port), redirect_output=True)
        ptvsd.wait_for_attach()
    
    # -------------------------------------------------------------------------------------------
    # Setup CUDA, GPU & distributed training
    args.cuda = not args.no_cuda and torch.cuda.is_available()
    if args.local_rank == -1 or args.no_cuda:
        device = torch.device("cuda" if torch.cuda.is_available() and not args.no_cuda else "cpu")
        args.n_gpu = torch.cuda.device_count()
    else:  # Initializes the distributed backend which will take care of sychronizing nodes/GPUs
        torch.cuda.set_device(args.local_rank)
        device = torch.device("cuda", args.local_rank)
        torch.distributed.init_process_group(backend='nccl')
        args.n_gpu = 1
    args.device = device
    logger.info("Process rank: %s, device: %s, n_gpu: %s, distributed training: %s, 16-bits training: %s",
                args.local_rank, device, args.n_gpu, bool(args.local_rank != -1), args.fp16)

    # -------------------------------------------------------------------------------------------
    # init tokenizer & Converter 
    logger.info("start setting tokenizer, dataset and dataloader (local_rank = {})... ".format(args.local_rank))
    tokenizer = tokenizer_class[args.pretrain_model_type].from_pretrained(args.cache_dir)
    
    # -------------------------------------------------------------------------------------------
    # Select dataloader
    dataset_examples = filer.load_json(os.path.join(args.preprocess_folder, "%s.json"%args.dataset_class))
    build_dataset, batchify_features_for_train, batchify_features_for_test = dataloader.get_class(args.model_class)
    
    args.test_batch_size = args.per_gpu_test_batch_size * max(1, args.n_gpu)

    # -------------------------------------------------------------------------------------------
    # build eval dataloader 
    eval_dataset = build_dataset(**{'args':args, 
                                    'examples':dataset_examples,
                                    'mode':'eval',
                                    'dataset_name': args.dataset_class,
                                    'tokenizer':tokenizer,
                                    'max_token':args.max_src_len,
                                    'max_phrase_words':args.max_phrase_words,
                                    'preprocess_folder':args.preprocess_folder,
                                    'cached_dataset_dir':args.cached_dataset_dir,
                                    'local_rank':args.local_rank
                                   })
    eval_sampler = torch.utils.data.sampler.SequentialSampler(eval_dataset)
    eval_data_loader = torch.utils.data.DataLoader(
        eval_dataset,
        batch_size=args.test_batch_size,
        sampler=eval_sampler,
        num_workers=args.data_workers,
        collate_fn=batchify_features_for_test,
        pin_memory=args.cuda,
    )
    
    # -------------------------------------------------------------------------------------------
    # Preprare Model & Optimizer
    # -------------------------------------------------------------------------------------------
    logger.info(" ************************** Initilize Model ************************** ")
    try:
        model, checkpoint_epoch = KeyphraseSpanExtraction.load_checkpoint(args.eval_checkpoint, args)
        model.set_device()
    except ValueError:
        print("Could't Load Pretrain Model %s" % args.eval_checkpoint)
    
    if args.local_rank == 0:
        torch.distributed.barrier()
    
    if args.n_gpu > 1:
        model.parallelize()
    
    if args.local_rank != -1:
        model.distribute()
        
    # -------------------------------------------------------------------------------------------
    # Method Select
    # -------------------------------------------------------------------------------------------
    candidate_decoder = select_decoder(args.model_class)
    _, test_input_refactor = utils.select_input_refactor(args.model_class)
    # -----------------------------------------------------------------------------------------------------------------------------
    # eval generator
    candidate = candidate_decoder(args, eval_data_loader, eval_dataset, model, test_input_refactor, pred_arranger, 'eval')

    save_filename = os.path.join(args.pred_folder, '%s.keyphrases.json'% args.dataset_class)
    pred_saver(args, candidate, save_filename)
