import os
import sys
import json
# sys.path.append("..")
from tqdm import tqdm
import torch
import logging
# from . import loader_utils
from .loader_utils import load_dataset, flat_rank_pos, strict_filter_overlap
from ..constant import BOS_WORD, EOS_WORD, Tag2Idx
from torch.utils.data import Dataset

logger = logging.getLogger()


def get_tag_label(doc_length, start_end_pos):
    # flatten, rank, filter overlap for answer positions
    sorted_positions = flat_rank_pos(start_end_pos)
    filter_positions = strict_filter_overlap(sorted_positions)
    
    if len(filter_positions) != len(sorted_positions):
        overlap_flag = True
    else:
        overlap_flag = False
        
    label = [Tag2Idx['O']] * doc_length
    for s, e in filter_positions:
        if s == e:
            label[s] = Tag2Idx['U']

        elif (e-s) == 1:
            label[s] = Tag2Idx['B']
            label[e] = Tag2Idx['E']

        elif (e-s) >=2:
            label[s] = Tag2Idx['B']
            label[e] = Tag2Idx['E']
            for i in range(s+1, e):
                label[i] = Tag2Idx['I']
        else:
            logger.info('ERROR')
            break

    return {'label':label, 'overlap_flag':overlap_flag}


def prepare_data_for_bert2tag(examples, max_token, mode, pretrain_model):
    logger.info('start preparing data for %s2Tag ...' %pretrain_model)
    overlap_num = 0
    new_examples = []
    for idx, ex in enumerate(tqdm(examples)):
        
        if len(ex['tokens']) < max_token:
            max_word = max_token
        else:
            max_word = ex['tok_to_orig_index'][max_token-1] + 1

        new_ex = {}
        new_ex['url'] = ex['url']
        new_ex['tokens'] = ex['tokens'][:max_token]
        new_ex['valid_mask'] = ex['valid_mask'][:max_token]
        new_ex['doc_words'] = ex['doc_words'][:max_word]
        
        assert len(new_ex['tokens']) == len(new_ex['valid_mask'])
        assert sum(new_ex['valid_mask']) == len(new_ex['doc_words'])
        
        if mode == 'train':
            parameter = {'doc_length': len(ex['doc_words']), 
                         'start_end_pos': ex['start_end_pos']}
            # ------------------------------------------------
            label_dict = get_tag_label(**parameter)
            new_ex['label'] = label_dict['label'][:max_word]
            assert sum(new_ex['valid_mask']) == len(new_ex['label'])
            
            if label_dict['overlap_flag']:
                overlap_num += 1
                
        new_examples.append(new_ex)
    logger.info('Delete Overlap Keyphrase : %d (overlap / total = %.2f' 
                %(overlap_num, float(overlap_num / len(examples) * 100)) + '%)')
    return new_examples



def reload_cached_dataset(cached_dataset_dir, dataset_class, name, pretrain_model):
    logger.info("start Reloading %s2tag %s %s cached dataset ..." %(pretrain_model, dataset_class, name))
    filename = os.path.join(cached_dataset_dir, "%s2tag.cached.%s.%s.json" % (pretrain_model, dataset_class, name))
    
    examples = []
    with open(filename, "r", encoding="utf-8") as f:
        for l in tqdm(f):
            examples.append(json.loads(l))
    f.close()
    logger.info("success loaded %s %s data : %d " %(dataset_class, name, len(examples)))
    return examples



def save_cached_dataset(cached_examples, dataset_name, mode, pretrain_model):
    logger.info("start saving %s2tag %s %s cached dataset ..." %(pretrain_model, dataset_name, mode))
    cached_dataset_dir = "./Cached_Datasets"
    
    if not os.path.exists(cached_dataset_dir):
        os.mkdir(cached_dataset_dir)
        
    filename = os.path.join(cached_dataset_dir, "%s2tag.cached.%s.%s.json" % (pretrain_model, dataset_name, mode))
    with open(filename, 'w', encoding='utf-8') as f_pred:
        for idx, ex in enumerate(tqdm(cached_examples)):
            f_pred.write("{}\n".format(json.dumps(ex)))
        f_pred.close()
    logger.info("successfully saved %s2tag %s %s cached dataset to %s" %(pretrain_model, dataset_name, mode, filename))
    
           
    
class build_bert2tag_dataset(Dataset):
    ''' build datasets for train & eval '''
    def __init__(self, args, examples, dataset_name, tokenizer, max_token, max_phrase_words, 
                 mode, preprocess_folder, cached_dataset_dir=None, local_rank=-1):
        # --------------------------------------------------------------------------------------------
        self.sep_token_extra = False
        self.pretrain_model = "bert"
        if "roberta" in args.model_class:
            self.sep_token_extra = True
            self.pretrain_model = "roberta"            
            
        params = {"examples":examples, "max_token":max_token, "mode":mode, "pretrain_model":self.pretrain_model}
        cached_examples = prepare_data_for_bert2tag(**params)

        # del cache
        examples.clear()

        # Rank cost too much time to preprocess `train_dataset`, so we want to save it
        if local_rank in [-1, 0] and mode == "train":
            save_cached_dataset(cached_examples, dataset_name, mode, self.pretrain_model)
            
        self.examples = cached_examples
        self.mode = mode
        self.tokenizer = tokenizer
            
    def __len__(self):
        return len(self.examples)

    def __getitem__(self, index):
        return convert_examples_to_features(index, self.examples[index], self.tokenizer, self.mode, self.sep_token_extra)
    


def convert_examples_to_features(index, ex, tokenizer, mode, sep_token_extra=False):
    ''' convert each batch data to tensor ; add [CLS] [SEP] tokens ;'''
    
    src_tokens = [BOS_WORD] + ex['tokens'] + [EOS_WORD]
    valid_ids = [0] + ex['valid_mask'] + [0]
    
    if sep_token_extra:
        src_tokens = src_tokens + [EOS_WORD]
        valid_ids = valid_ids + [0]
    
    src_tensor = torch.LongTensor(tokenizer.convert_tokens_to_ids(src_tokens))
    valid_mask = torch.LongTensor(valid_ids)    
    orig_doc_len = sum(valid_ids)

    if mode == 'train':
        label_tensor = torch.LongTensor(ex['label'])
        return index, src_tensor, valid_mask, orig_doc_len, label_tensor
    
    else:
        return index, src_tensor, valid_mask, orig_doc_len
    
    
def batchify_Bert2tag_features_for_train(batch):
    ''' train dataloader & eval dataloader .'''
    
    ids = [ex[0] for ex in batch]
    docs = [ex[1] for ex in batch]
    valid_mask = [ex[2] for ex in batch]
    doc_word_lens = [ex[3] for ex in batch]
    label_list = [ex[4] for ex in batch]
    
    bert_output_dim = 768
    max_word_len = max([word_len for word_len in doc_word_lens]) # word-level

    # ---------------------------------------------------------------
    # [1][2]src tokens tensor
    doc_max_length = max([d.size(0) for d in docs])
    input_ids = torch.LongTensor(len(docs), doc_max_length).zero_()
    input_mask = torch.LongTensor(len(docs), doc_max_length).zero_()    
    for i, d in enumerate(docs):
        input_ids[i, :d.size(0)].copy_(d)
        input_mask[i, :d.size(0)].fill_(1)
        
    # ---------------------------------------------------------------
    # valid mask tensor
    valid_max_length = max([v.size(0) for v in valid_mask])
    valid_ids = torch.LongTensor(len(valid_mask), valid_max_length).zero_()
    for i, v in enumerate(valid_mask):
        valid_ids[i, :v.size(0)].copy_(v)

    # ---------------------------------------------------------------
    # label tensor
    labels = torch.LongTensor(len(label_list), max_word_len).zero_()
    active_mask = torch.LongTensor(len(label_list), max_word_len).zero_()
    for i, t in enumerate(label_list):
        labels[i, :t.size(0)].copy_(t)
        active_mask[i, :t.size(0)].fill_(1)
        
    # -------------------------------------------------------------------
    # [6] Empty Tensor : word-level max_len
    valid_output = torch.zeros(len(docs), max_word_len, bert_output_dim) 
    return input_ids, input_mask, valid_ids, active_mask, valid_output, labels, ids


def batchify_Bert2tag_features_for_test(batch):
    ''' test dataloader for Dev & Public_Valid.'''
    
    ids = [ex[0] for ex in batch]
    docs = [ex[1] for ex in batch]
    valid_mask = [ex[2] for ex in batch]
    doc_word_lens = [ex[3] for ex in batch]
    
    bert_output_dim = 768
    max_word_len = max([word_len for word_len in doc_word_lens]) # word-level

    # ---------------------------------------------------------------
    # [1][2]src tokens tensor
    doc_max_length = max([d.size(0) for d in docs])
    input_ids = torch.LongTensor(len(docs), doc_max_length).zero_()
    input_mask = torch.LongTensor(len(docs), doc_max_length).zero_()    
    for i, d in enumerate(docs):
        input_ids[i, :d.size(0)].copy_(d)
        input_mask[i, :d.size(0)].fill_(1)
        
    # ---------------------------------------------------------------
    # [3] valid mask tensor
    valid_max_length = max([v.size(0) for v in valid_mask])
    valid_ids = torch.LongTensor(len(valid_mask), valid_max_length).zero_()
    for i, v in enumerate(valid_mask):
        valid_ids[i, :v.size(0)].copy_(v)
        
    # ---------------------------------------------------------------
    # valid length tensor
    active_mask = torch.LongTensor(len(doc_word_lens), max_word_len).zero_()
    for i, l in enumerate(doc_word_lens):
        active_mask[i, :l].fill_(1)
        
    # -------------------------------------------------------------------
    # [4] Empty Tensor : word-level max_len
    valid_output = torch.zeros(len(docs), max_word_len, bert_output_dim) 
    return input_ids, input_mask, valid_ids, active_mask, valid_output, doc_word_lens, ids