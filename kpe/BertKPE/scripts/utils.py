import os
import time
import json
import torch
import random
import argparse
import logging
import numpy as np

logger = logging.getLogger()

# *********************************************************************************************
# *** Select Data Refactor : batch → inputs ***
# *********************************************************************************************
def select_input_refactor(name):
    if name in ['bert2tag', 'roberta2tag', 'bert2gram', 'roberta2gram']:
        return train_input_refactor, test_input_refactor
    raise RuntimeError('Invalid retriever class: %s' % name)
    
    
# *********************************************************************************************
# Train Refactor
def train_input_refactor(batch, device):
    ex_indices = batch[-1]
    batch = tuple(b.to(device) for b in batch[:-1])
    inputs = {'input_ids':batch[0],
              'attention_mask':batch[1],
              'valid_ids':batch[2],
              'active_mask':batch[3],
              'valid_output':batch[4],
              'labels':batch[5]
             }        
    return inputs, ex_indices

# *********************************************************************************************
# TEST Refactor
def test_input_refactor(batch, device):
    ex_indices, ex_phrase_numbers = batch[-1], batch[-2]
    
    batch = tuple(b.to(device) for b in batch[:-2])
    inputs = {'input_ids':batch[0],
              'attention_mask':batch[1],
              'valid_ids':batch[2],
              'active_mask':batch[3],
              'valid_output':batch[4],
             }
    return inputs, ex_indices, ex_phrase_numbers
    
# *********************************************************************************************
# *** Select Prediction Arranger
# *********************************************************************************************
def pred_arranger(tot_predictions):
    data_dict = {}
    for prediction in tot_predictions:
        item = {}
        item['url'] = prediction[0]
        item['KeyPhrases'] = [keyphrase.split() for keyphrase in prediction[1]]
        if len(prediction) > 2:
            item['Scores'] = prediction[2]
        data_dict[item['url']] = item
    return data_dict


def pred_saver(args, tot_predictions, filename):
    with open(filename, 'w', encoding='utf-8') as f_pred:
        for url, item in tot_predictions.items():
            data = {}
            data['paper_id'] = url
            data['KeyPhrases'] = [' '.join([w.capitalize() for w in kp]) for kp in item['KeyPhrases']] 
            f_pred.write("{}\n".format(json.dumps(data)))
        f_pred.close()
    logger.info('Success save %s prediction file' % filename)
    
    
# *********************************************************************************************
# *** Common Functions ***
# *********************************************************************************************

def set_seed(args):
    random.seed(args.seed)
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)
    if args.n_gpu > 0:
        torch.cuda.manual_seed_all(args.seed)
        
        
def override_args(old_args, new_args):
    ''' cover old args to new args, log which args has been changed.'''
    old_args, new_args = vars(old_args), vars(new_args)
    for k in new_args.keys():
        if k in old_args:
            if old_args[k] != new_args[k]:
                logger.info('Overriding saved %s: %s --> %s' %(k, old_args[k], new_args[k]))
                old_args[k] = new_args[k]
        else:
            old_args[k] = new_args[k]
    return argparse.Namespace(**old_args) 


class AverageMeter(object):
    """Computes and stores the average and current value."""
    def __init__(self):
        self.reset()

    def reset(self):
        self.val = 0
        self.avg = 0
        self.sum = 0
        self.count = 0

    def update(self, val, n=1):
        self.val = val
        self.sum += val * n
        self.count += n
        self.avg = self.sum / self.count
        
        
class Timer(object):
    """Computes elapsed time."""

    def __init__(self):
        self.running = True
        self.total = 0
        self.start = time.time()

    def reset(self):
        self.running = True
        self.total = 0
        self.start = time.time()
        return self

    def resume(self):
        if not self.running:
            self.running = True
            self.start = time.time()
        return self

    def stop(self):
        if self.running:
            self.running = False
            self.total += time.time() - self.start
        return self

    def time(self):
        if self.running:
            return self.total + time.time() - self.start
        return self.total
    
