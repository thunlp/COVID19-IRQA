
import argparse
import collections
import json
import math
import os
import torch

import torch.nn as nn
from transformers import BertConfig, BertTokenizer, BertModel, BertPreTrainedModel


class Example(object):

    def __init__(self,
                 qid,
                 query,
                 passages):
        self.qid = qid
        self.query = query
        self.passages = passages


class InputFeatures(object):
    """A single set of features of data."""

    def __init__(self,
                 qid,
                 tokens,
                 input_ids,
                 input_mask,
                 segment_ids):
        self.qid = qid
        self.tokens = tokens
        self.input_ids = input_ids
        self.input_mask = input_mask
        self.segment_ids = segment_ids


def convert_examples_to_features(examples, tokenizer, max_seq_length,
                                 max_query_length,
                                 num_passage_per_query=10,
                                 cls_token_at_end=False,
                                 cls_token='[CLS]', sep_token='[SEP]', pad_token=0,
                                 sequence_a_segment_id=0, sequence_b_segment_id=1,
                                 cls_token_segment_id=0, pad_token_segment_id=0,
                                 mask_padding_with_zero=True,
                                 sequence_a_is_doc=False):
    features = []
    # for (example_index, example) in enumerate(tqdm(examples)):
    for (example_index, example) in enumerate(examples):
        query_tokens = tokenizer.tokenize(example.query)

        if len(query_tokens) > max_query_length:
            query_tokens = query_tokens[0:max_query_length]

        # The -3 accounts for [CLS], [SEP] and [SEP]
        max_tokens_for_doc = max_seq_length - len(query_tokens) - 3

        tokens_list = []
        input_ids_list = []
        input_mask_list = []
        segment_ids_list = []
        max_score = 0.0
        start_position = None
        end_position = None
        for pid in range(num_passage_per_query):
            passage = example.passages[pid] if pid < len(example.passages) else ''

            tokens = []
            segment_ids = []

            # p_mask: mask with 1 for token than cannot be in the answer (0 for token which can be in an answer)
            # Original TF implem also keep the classification token (set to 0) (not sure why...)
            p_mask = []

            # CLS token at the beginning
            if not cls_token_at_end:
                tokens.append(cls_token)
                segment_ids.append(cls_token_segment_id)
                p_mask.append(0)
                cls_index = 0

            # XLNet: P SEP Q SEP CLS
            # Others: CLS Q SEP P SEP
            if not sequence_a_is_doc:
                # Query
                tokens += query_tokens
                segment_ids += [sequence_a_segment_id] * len(query_tokens)
                p_mask += [1] * len(query_tokens)

                # SEP token
                tokens.append(sep_token)
                segment_ids.append(sequence_a_segment_id)
                p_mask.append(1)

            passage_tokens = tokenizer.tokenize(passage)
            passage_tokens = passage_tokens[:max_tokens_for_doc]

            # Paragraph
            for i in range(len(passage_tokens)):
                tokens.append(passage_tokens[i])
                if not sequence_a_is_doc:
                    segment_ids.append(sequence_b_segment_id)
                else:
                    segment_ids.append(sequence_a_segment_id)
                p_mask.append(0)

            if sequence_a_is_doc:
                # SEP token
                tokens.append(sep_token)
                segment_ids.append(sequence_a_segment_id)
                p_mask.append(1)

                tokens += query_tokens
                segment_ids += [sequence_b_segment_id] * len(query_tokens)
                p_mask += [1] * len(query_tokens)

            # SEP token
            tokens.append(sep_token)
            segment_ids.append(sequence_b_segment_id)
            p_mask.append(1)

            # CLS token at the end
            if cls_token_at_end:
                tokens.append(cls_token)
                segment_ids.append(cls_token_segment_id)
                p_mask.append(0)
                cls_index = len(tokens) - 1  # Index of classification token

            input_ids = tokenizer.convert_tokens_to_ids(tokens)

            # The mask has 1 for real tokens and 0 for padding tokens. Only real
            # tokens are attended to.
            input_mask = [1 if mask_padding_with_zero else 0] * len(input_ids)

            # Zero-pad up to the sequence length.
            while len(input_ids) < max_seq_length:
                input_ids.append(pad_token)
                input_mask.append(0 if mask_padding_with_zero else 1)
                segment_ids.append(pad_token_segment_id)
                p_mask.append(1)

            assert len(input_ids) == max_seq_length
            assert len(input_mask) == max_seq_length
            assert len(segment_ids) == max_seq_length

            tokens_list.append(tokens)
            input_ids_list.append(input_ids)
            input_mask_list.append(input_mask)
            segment_ids_list.append(segment_ids)

        features.append(
            InputFeatures(
                qid=example.qid,
                tokens=tokens_list,
                input_ids=input_ids_list,
                input_mask=input_mask_list,
                segment_ids=segment_ids_list))

    return features


class BertForQuestionAnswering(BertPreTrainedModel):

    def __init__(self, config):
        super(BertForQuestionAnswering, self).__init__(config)
        self.num_labels = config.num_labels

        self.bert = BertModel(config)
        self.qa_outputs = nn.Linear(config.hidden_size, config.num_labels)
        self.selected_outputs = nn.Linear(config.hidden_size, 1)

        self.init_weights()

    def forward(self, input_ids, attention_mask=None, token_type_ids=None, position_ids=None, head_mask=None):

        outputs = self.bert(input_ids,
                            attention_mask=attention_mask,
                            token_type_ids=token_type_ids,
                            position_ids=position_ids, 
                            head_mask=head_mask)

        sequence_output = outputs[0]
        pooled_output = outputs[1]

        logits = self.qa_outputs(sequence_output)
        start_logits, end_logits = logits.split(1, dim=-1)
        start_logits = start_logits.squeeze(-1)
        end_logits = end_logits.squeeze(-1)
        
        selected_logits = self.selected_outputs(pooled_output).squeeze(-1)

        outputs = (start_logits, end_logits, selected_logits,) + outputs[2:]

        return outputs


def _get_best_indexes(logits, n_best_size):
    """Get the n-best logits from a list."""
    index_and_score = sorted(enumerate(logits), key=lambda x: x[1], reverse=True)

    best_indexes = []
    for i in range(len(index_and_score)):
        if i >= n_best_size:
            break
        best_indexes.append(index_and_score[i][0])
    return best_indexes


def _compute_softmax(scores):
    """Compute softmax probability over raw logits."""
    if not scores:
        return []

    max_score = None
    for score in scores:
        if max_score is None or score > max_score:
            max_score = score

    exp_scores = []
    total_sum = 0.0
    for score in scores:
        x = math.exp(score - max_score)
        exp_scores.append(x)
        total_sum += x

    probs = []
    for score in exp_scores:
        probs.append(score / total_sum)
    return probs


def get_prediction(feature, start_logits, end_logits, n_best_size, max_seq_length, max_answer_length):
    _PrelimPrediction = collections.namedtuple(  # pylint: disable=invalid-name
        "PrelimPrediction",
        ["start_index", "end_index", "start_logit", "end_logit"])

    prelim_predictions = []
    start_indexes = _get_best_indexes(start_logits, n_best_size)
    end_indexes = _get_best_indexes(end_logits, n_best_size)
    # if we could have irrelevant answers, get the min score of irrelevant
    for start_index in start_indexes:
        for end_index in end_indexes:
            # We could hypothetically create invalid predictions, e.g., predict
            # that the start of the span is in the question. We throw out all
            # invalid predictions.
            if start_index >= len(feature.tokens) * max_seq_length:
                continue
            if end_index >= len(feature.tokens) * max_seq_length:
                continue
            if start_index // max_seq_length != end_index // max_seq_length:
                continue
            if end_index < start_index:
                continue
            length = end_index - start_index + 1
            if length > max_answer_length:
                continue
            prelim_predictions.append(
                _PrelimPrediction(
                    start_index=start_index,
                    end_index=end_index,
                    start_logit=start_logits[start_index],
                    end_logit=end_logits[end_index]))
    prelim_predictions = sorted(
        prelim_predictions,
        key=lambda x: (x.start_logit + x.end_logit),
        reverse=True)

    _NbestPrediction = collections.namedtuple(  # pylint: disable=invalid-name
        "NbestPrediction", ["text", "passage_index", "start_logit", "end_logit"])

    seen_predictions = {}
    nbest = []
    for pred in prelim_predictions:
        if len(nbest) >= n_best_size:
            break
        if pred.start_index > 0:  # this is a non-null prediction
            passage_index = pred.start_index // max_seq_length
            start_index = pred.start_index % max_seq_length
            end_index = pred.end_index % max_seq_length
            tok_tokens = feature.tokens[passage_index][start_index:(end_index + 1)]
            tok_text = " ".join(tok_tokens)

            # De-tokenize WordPieces that have been split off.
            tok_text = tok_text.replace(" ##", "")
            tok_text = tok_text.replace("##", "")

            # Clean whitespace
            tok_text = tok_text.strip()
            final_text = " ".join(tok_text.split()).replace(" [UNK] ", " ")
            if final_text in seen_predictions:
                continue

            seen_predictions[final_text] = True
        else:
            passage_index = -1
            final_text = ""
            seen_predictions[final_text] = True

        nbest.append(
            _NbestPrediction(
                text=final_text,
                passage_index=passage_index,
                start_logit=pred.start_logit,
                end_logit=pred.end_logit))

    # In very rare edge cases we could have no valid predictions. So we
    # just create a nonce prediction in this case to avoid failure.
    if not nbest:
        nbest.append(
            _NbestPrediction(text="", start_logit=0.0, end_logit=0.0))

    assert len(nbest) >= 1

    total_scores = []
    for entry in nbest:
        total_scores.append(entry.start_logit + entry.end_logit)
    probs = _compute_softmax(total_scores)

    nbest_json = []
    for (i, entry) in enumerate(nbest):
        output = collections.OrderedDict()
        output["text"] = entry.text
        output["doc_id"] = entry.passage_index
        output["probability"] = probs[i]
        nbest_json.append(output)

    return nbest_json



