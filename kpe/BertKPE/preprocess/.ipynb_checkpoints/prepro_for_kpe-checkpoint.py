import os
import re
import sys
import nltk
import json
import time
import torch
import codecs
import string
import pickle
import logging
import argparse
import unicodedata
import numpy as np
from tqdm import tqdm

sys.path.append('..')
sys.path.append('../MyCode')
punctuations = string.punctuation

from bertkpe import UNK_WORD

from functions import filer
from transformers import BertTokenizer, RobertaTokenizer
tokenizer_class = {"bert-base-cased":BertTokenizer, "roberta-base":RobertaTokenizer}

logger = logging.getLogger()


def add_preprocess_opts(parser):
    parser.add_argument('--source_dataset_dir', type=str, required=True,
                        help="The path to the source data (raw json).")
    parser.add_argument('--output_path', type=str, required=True,
                        help="The dir to save preprocess data")
    parser.add_argument("--pretrain_model_path", type=str, required=True,
                       help="Path to pre-trained model .")
    parser.add_argument("--bert_type", type=str, required=True,
                       help="pretrined bert name : 'bert-base-cased', 'roberta-base'")
    
    
def tokenize_text(text):
    
    # 1. remove line breakers
    text = re.sub(r'[\r\n\t]', ' ', text)

    # 2. remove bracket: {} [] <>
    rm_bracket_pattern = u'\\(.*?\\)|\\{.*?}|\\[.*?]|\\<.*?>'
    text = re.sub(rm_bracket_pattern, '', text)

    # 3. pad spaces to the left and right of special punctuations
    text = re.sub(r'[\.\,]', ' \g<0>', text)

    # 4. tokenize
    tokens = nltk.word_tokenize(text)

    # 5. del stop punct
    tokens = [tk for tk in tokens if tk not in punctuations and len(tk) > 0]
    return tokens


def tokenize_and_clean(examples):
    
    data_list = []
    for idx, ex in enumerate(tqdm(examples)):
        
        title = ex['title']
        abstract = ex['abstract']

        abstract_text = " ".join([title, " ".join(ab for ab in abstract)])
        doc_words = tokenize_text(abstract_text)
        
        if len(doc_words) < 50:
            body_text = ' '.join([' '.join(section['text']) for section in ex['body_text']])
            body_words = tokenize_text(body_text)
            doc_words = doc_words + body_words[:256]

        data = {}
        data['url'] = ex['paper_id']
        data['doc_words'] = doc_words
        data_list.append(data)
    return data_list


def tokenize_for_bert(examples, tokenizer):
    logger.info('strat tokenize for bert training ...')
    Features = []
    for (example_index, example) in enumerate(tqdm(examples)):
        
        valid_mask = []
        all_doc_tokens = []
        tok_to_orig_index = []
        orig_to_tok_index = [] 
        for (i, token) in enumerate(example['doc_words']):
            orig_to_tok_index.append(len(all_doc_tokens))
            sub_tokens = tokenizer.tokenize(token)
            if len(sub_tokens) < 1:
                sub_tokens = [UNK_WORD]
            for num, sub_token in enumerate(sub_tokens):
                tok_to_orig_index.append(i)
                all_doc_tokens.append(sub_token)
                if num == 0:
                    valid_mask.append(1)
                else:
                    valid_mask.append(0)
                    
        InputFeatures = {}
        InputFeatures['url'] = example['url']
        InputFeatures['doc_words'] = example['doc_words']
        
        # -------------------------------------------------------
        # new added
        InputFeatures['tokens'] = all_doc_tokens
        InputFeatures['valid_mask'] = valid_mask
        InputFeatures['tok_to_orig_index'] = tok_to_orig_index
        InputFeatures['orig_to_tok_index'] = orig_to_tok_index # frist_token index in all_doc_tokens
        Features.append(InputFeatures)
    
    return Features

    
def main_preocess(opt, filename, tokenizer):
    
    dataset_name = filename.split('.')[0]
    
    file_path = os.path.join(opt.source_dataset_dir, filename)
    examples = filer.load_jsonl(file_path)
    logger.info("Success loader %s dataset : %d. " %(dataset_name, len(examples)))
    
    feed_data = tokenize_and_clean(examples)
    bert_features = tokenize_for_bert(examples=feed_data, tokenizer=tokenizer)
    logger.info("Success tokenize data : %d. " %len(bert_features))
    
    assert len(examples) == len(bert_features)
    
    save_filename = os.path.join(opt.output_path, "%s.json" % dataset_name)
    filer.save_json(bert_features, save_filename)


if __name__ == "__main__":

    t0 = time.time()
    parser = argparse.ArgumentParser(description='preprocess_for_kpe.py',
                                     formatter_class=argparse.ArgumentDefaultsHelpFormatter)
    # add options
    add_preprocess_opts(parser)
    opt = parser.parse_args()

    # option folder
    if not os.path.exists(opt.source_dataset_dir):
        logger.info("don't exist the source dataset dir: %s" % opt.source_dataset_dir)
        
    if not os.path.exists(opt.output_path):
        os.makedirs(opt.output_path)
        logger.info("don't exist this output dir, remake a new to : %s" % opt.output_path)
        
    # set logger
    opt.log_file = os.path.join(opt.output_path, 'CORD-19_KPE_logging.txt')
    filer.set_logger(opt.log_file)
    
    # load bert tokenizer
    opt.cache_dir = os.path.join(opt.pretrain_model_path, opt.bert_type)
    tokenizer = tokenizer_class[opt.bert_type].from_pretrained(opt.cache_dir)
    
    filenames = os.listdir(opt.source_dataset_dir)
    
    for filename in filenames:
        main_preocess(opt, filename, tokenizer)
