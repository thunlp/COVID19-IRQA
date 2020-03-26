
import numpy as np
import os
import torch
import torch.nn as nn
from transformers import *


class RankingFeature(object):
    def __init__(self, query_id, doc_id, d_input_ids, d_input_mask, d_segment_ids):
        self.query_id = query_id
        self.doc_id = doc_id
        self.d_input_ids = d_input_ids
        self.d_input_mask = d_input_mask
        self.d_segment_ids = d_segment_ids


def pack_bert_seq(q_tokens, p_tokens, tokenizer, max_seq_length):
    tokens = []
    segment_ids = []
    tokens.append("[CLS]")
    segment_ids.append(0)
    for token in q_tokens:
        tokens.append(token)
        segment_ids.append(0)
    tokens.append("[SEP]")
    segment_ids.append(0)
    for token in p_tokens:
        tokens.append(token)
        segment_ids.append(1)
    tokens.append("[SEP]")
    segment_ids.append(1)

    input_ids = tokenizer.convert_tokens_to_ids(tokens)
    input_mask = [1] * len(input_ids)
    while len(input_ids) < max_seq_length:
        input_ids.append(0)
        input_mask.append(0)
        segment_ids.append(0)

    assert len(input_ids) == max_seq_length
    assert len(input_mask) == max_seq_length
    assert len(segment_ids) == max_seq_length

    return input_ids, input_mask, segment_ids


def convert_to_ranking_feature(query, doc, query_id, doc_id, tokenizer, max_seq_length, max_query_length):
    q_tokens = tokenizer.tokenize(query)
    if len(q_tokens) > max_query_length:
        q_tokens = q_tokens[:max_query_length]

    max_doc_length = max_seq_length - len(q_tokens) - 3
    d_tokens = tokenizer.tokenize(doc)
    if len(d_tokens) > max_doc_length:
        d_tokens = d_tokens[:max_doc_length]

    d_input_ids, d_input_mask, d_segment_ids = pack_bert_seq(q_tokens, d_tokens, tokenizer, max_seq_length)

    feature = RankingFeature(
        query_id = query_id,
        doc_id = doc_id,
        d_input_ids = d_input_ids,
        d_input_mask = d_input_mask,
        d_segment_ids = d_segment_ids
    )

    return feature 


def DataLoader(features, batch_size):
    n_samples = len(features)
    idx = np.arange(n_samples)

    for start_idx in range(0, n_samples, batch_size):
        batch_idx = idx[start_idx:start_idx+batch_size]

        query_id = [features[i].query_id for i in batch_idx]
        doc_id = [features[i].doc_id for i in batch_idx]
        d_input_ids = torch.tensor([features[i].d_input_ids for i in batch_idx], dtype=torch.long)
        d_input_mask = torch.tensor([features[i].d_input_mask for i in batch_idx], dtype=torch.long)
        d_segment_ids = torch.tensor([features[i].d_segment_ids for i in batch_idx], dtype=torch.long)

        batch = (query_id, doc_id, d_input_ids, d_input_mask, d_segment_ids)
        yield batch
    return

class BertForRanking(BertPreTrainedModel):
    def __init__(self, config):
        super().__init__(config)
        self.bert = BertModel(config)
        self.dense = nn.Linear(config.hidden_size, 2)

        self.init_weights()

    def forward(self, inst, tok, mask):
        output = self.bert(inst, token_type_ids=tok, attention_mask=mask)
        final_res = self.dense(output[1])[:, 1]

        return final_res, output[1]
