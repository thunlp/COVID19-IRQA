import argparse
import collections
import faiss
import json
import logging
import numpy as np
import os
import time
import torch
import tensorflow as tf
import tensorflow_hub as hub
import tensorflow_text

from elasticsearch import Elasticsearch
from transformers import BertConfig, BertTokenizer, BertModel, BertPreTrainedModel

from bert_ranking import BertForRanking, RankingFeature, pack_bert_seq
from bert_ranking import DataLoader as RankingDataLoader

from bert_qa import BertForQuestionAnswering, get_prediction 
from bert_qa import convert_examples_to_features as convert_examples_to_qa_features 
from bert_qa import Example as QAExample
from bert_qa import InputFeatures as QAInputFeatures

logger = logging.getLogger(__name__)
logger.setLevel(logging.INFO)


def to_list(tensor):
    return tensor.detach().cpu().tolist()


def convert_to_ranking_feature(q_tokens, doc, query_id, doc_id, tokenizer, max_seq_length, max_query_length, doc_stride):
    if len(q_tokens) > max_query_length:
        q_tokens = q_tokens[:max_query_length]

    max_tokens_for_doc = max_seq_length - len(q_tokens) - 3
    all_doc_tokens = tokenizer.tokenize(doc)

    _DocSpan = collections.namedtuple(  # pylint: disable=invalid-name
        "DocSpan", ["start", "length"])
    doc_spans = []
    start_offset = 0
    while start_offset < len(all_doc_tokens):
        length = len(all_doc_tokens) - start_offset
        if length > max_tokens_for_doc:
            length = max_tokens_for_doc
        doc_spans.append(_DocSpan(start=start_offset, length=length))
        if start_offset + length == len(all_doc_tokens):
            break
        start_offset += min(length, doc_stride)

    features = []
    for (doc_span_index, doc_span) in enumerate(doc_spans):
        d_tokens = all_doc_tokens[doc_span.start: doc_span.start + doc_span.length]
        d_input_ids, d_input_mask, d_segment_ids = pack_bert_seq(q_tokens, d_tokens, tokenizer, max_seq_length)
        feature = RankingFeature(
            query_id = query_id,
            doc_id = doc_id,
            d_input_ids = d_input_ids,
            d_input_mask = d_input_mask,
            d_segment_ids = d_segment_ids
        )
        features.append(feature)

    return features 


class IRQASystem():

    QUESTION_WORD = ["what", "which", "who", "where", "why", "when", "how", "whose", "is", "are", "was", "were", "does", "do", "did", "tell", "will", "would", "can", "could"]

    def __init__(self, args, device):
        self.num_docs = args.num_docs
        self.num_para_qa = args.num_para_qa
        self.num_para_ann = args.num_para_ann
        self.max_seq_length = args.max_seq_length
        self.max_query_length = args.max_query_length
        self.max_answer_length = args.max_answer_length
        self.doc_stride = args.doc_stride
        self.ranking_batch_size = args.ranking_batch_size
        self.qa_batch_size = args.qa_batch_size
        self.n_best_size = args.n_best_size
        self.use_ann = args.use_ann

        self.tokenizer = BertTokenizer.from_pretrained(args.qa_model_path, do_lower_case=args.do_lower_case)


        # qa model
        logger.info("Loading QA module from path %s..." % args.qa_model_path)
        self.qa_model_config = BertConfig.from_pretrained(args.qa_model_path)
        self.qa_model = BertForQuestionAnswering.from_pretrained(args.qa_model_path, config=self.qa_model_config)
        self.qa_model.to(device)
    
        # ranking model
        logger.info("Loading Ranking module from path %s..." % args.qa_model_path)
        self.ranking_model_config = BertConfig.from_pretrained(args.qa_model_path)
        self.ranking_model = BertForRanking.from_pretrained(args.ranking_model_path, config=self.ranking_model_config)
        self.ranking_model.to(device)

        # elasticsearch index
        logger.info("connect to ElasticSearch service from %s:%d" % (args.es_index_host, args.es_index_port))
        self.es = Elasticsearch([{'host': args.es_index_host, 'port': args.es_index_port}], timeout=600)

        if self.use_ann:
            # Limit GPU usage for TF
            gpus = tf.config.experimental.list_physical_devices('GPU')
            if gpus:
                try:
                    for gpu in gpus:
                        tf.config.experimental.set_memory_growth(gpu, True)
                except RuntimeError as e:
                    print(e)

            # universal sentence encoder module
            logger.info("Loading USE module from path %s..." % args.use_model_path)
            self.use_module = hub.load(args.use_model_path)

            # faiss index
            logger.info("Loading faiss index from file %s..." % args.faiss_index_file)
            self.faiss_index = faiss.read_index(args.faiss_index_file)
            self.faiss_index = faiss.index_cpu_to_all_gpus(self.faiss_index)

            # original paragraphs for faiss index
            self.original_paragraphs = []
            with open(args.original_paragraphs_file, 'r') as fin:
                for line in fin:
                    self.original_paragraphs.append(json.loads(line.strip()))

            # original paragraphs for faiss index
            self.original_paragraphs = []
            with open(args.original_paragraphs_file, 'r') as fin:
                for line in fin:
                    self.original_paragraphs.append(json.loads(line.strip()))

    def _is_question(self, query):
        return query.split()[0].lower() in self.QUESTION_WORD

    def process_query(self, query):
        dsl = {"query": {"multi_match": {"query": "", "fields": ["title^4", "abstract^3", "contents^2", "back_matter^1"]}}, 'from': 0, 'size': self.num_docs}
        dsl["query"]["multi_match"]["query"] = query

        ct = time.time()
        res = self.es.search(index='cord19', body=dsl)
        docs = []
        seen_doc_id = set()
        for hit in res['hits']['hits']:
            doc_id = hit['_source']['id']
            seen_doc_id.add(doc_id)
            title = hit['_source']['title']
            keyphrases = ', '.join(hit['_source']['key_phrase'].split('\n\n'))
            docs.append({"id": doc_id, "title": title, "text": hit['_source']['abstract'], "keyphrases": keyphrases})
            for paragraph in hit['_source']['contents'].split("\n\n"):
                docs.append({"id": doc_id, "title": title, "text": paragraph, "keyphrases": keyphrases})
        logger.info("BM25 retrieval time: %f" % (time.time() - ct))
        logger.info("BM25 retrieved para num: %d" % (len(docs)))

        if self.use_ann:
            original_len = len(docs)
            ct = time.time()
            query_embeddings = self.use_module.signatures['question_encoder'](tf.constant([query]))["outputs"].numpy()
            D, I = self.faiss_index.search(query_embeddings, self.num_para_ann)
            D = list(np.squeeze(D))
            I = list(np.squeeze(I))
            for i in range(self.num_para_ann):
                dist, para_idx = D[i], I[i]
                doc_id = self.original_paragraphs[para_idx]["paper_id"]
                title = ""
                text = self.original_paragraphs[para_idx]["text"]
                keyphrases = ""
                if not doc_id in seen_doc_id:
                    docs.append({"id": doc_id, "title": title, "text": text, "keyphrases": keyphrases})
            logger.info("ANN retrieval time: %f" % (time.time() - ct))
            logger.info("ANN retrieved para num: %d" % (len(docs) - original_len))

        return self.process(query, docs)

    def process(self, query, docs):
        # Ranking Model
        ct = time.time()

        ranking_features = []
        query_idx = 0
        q_tokens = self.tokenizer.tokenize(query)
        for (doc_idx, doc) in enumerate(docs):
            ranking_features.extend(
                convert_to_ranking_feature(
                    q_tokens, doc["text"], query_idx, doc_idx,
                    self.tokenizer, self.max_seq_length, self.max_query_length, self.doc_stride))

        logger.info("num of paragraphs: %d" % len(ranking_features))
        logger.info("Convert ranking feature time: %f" % (time.time() - ct))
        ct = time.time()
        ranking_data = RankingDataLoader(ranking_features, self.ranking_batch_size)

        _RankingResult= collections.namedtuple(
            "RankingResult", ["doc_idx", "title", "text", "keyphrases", "score"])
        ranking_results = []
        for s, batch in enumerate(ranking_data):
            self.ranking_model.eval()
            query_idx, doc_idx = batch[:2]
            batch = tuple(t.to(device) for t in batch[2:])
            (d_input_ids, d_input_mask, d_segment_ids) = batch
            with torch.no_grad():
                doc_scores, _ = self.ranking_model(d_input_ids, d_segment_ids, d_input_mask)
            d_scores = to_list(doc_scores)

            for (did, score) in zip(doc_idx, d_scores):
                ranking_results.append(
                    _RankingResult(doc_idx=did, title=docs[did]["title"], text=docs[did]["text"], keyphrases=docs[did]["keyphrases"], score=score))
        logger.info("Ranking time: %f" % (time.time() - ct))

        # rank data by score
        ranking_results = sorted(
            ranking_results,
            key=lambda x: x.score,
            reverse=True)

        # get search result
        search_results = []
        seen_titles = set()
        for result in ranking_results:
            if not result.title in seen_titles:
                seen_titles.add(result.title)
                result_json = {"title": result.title, "text": result.text, "keyphrases": result.keyphrases, "score": result.score}
                search_results.append(result_json)

        # QA Model
        qa_results = None
        if self._is_question(query):
            ct = time.time()

            selected_passages = []
            selected_passages_idx = []
            for i in range(self.num_para_qa):
                selected_passages.append(ranking_results[i].text)
                selected_passages_idx.append(ranking_results[i].doc_idx)
            qa_example = QAExample(query_idx, query, selected_passages)

            qa_features = convert_examples_to_qa_features(
                examples=[qa_example],
                tokenizer=self.tokenizer,
                max_seq_length=self.max_seq_length,
                max_query_length=self.max_query_length)

            feature = qa_features[0]
            self.qa_model.eval()
            input_ids = torch.tensor([feature.input_ids], dtype=torch.long).to(device)
            input_mask = torch.tensor([feature.input_mask], dtype=torch.long).to(device)
            segment_ids = torch.tensor([feature.segment_ids], dtype=torch.long).to(device)
            with torch.no_grad():
                inputs = {'input_ids':       input_ids.view(-1, input_ids.shape[2]),
                          'attention_mask':  input_mask.view(-1, input_mask.shape[2]),
                          'token_type_ids':  segment_ids.view(-1, segment_ids.shape[2])}
                outputs = self.qa_model(**inputs)
                start_logits, end_logits = outputs[0], outputs[1]
                start_logits = start_logits.view(self.qa_batch_size, self.num_para_qa * self.max_seq_length)
                end_logits = end_logits.view(self.qa_batch_size, self.num_para_qa * self.max_seq_length)

            sl = to_list(start_logits[0])
            el = to_list(end_logits[0])
            nbest_pred = get_prediction(feature, sl, el, self.n_best_size, self.max_seq_length, self.max_answer_length)
            for pred in nbest_pred:
                if pred["doc_id"] != -1:
                    doc_idx = selected_passages_idx[pred["doc_id"]]
                    pred["title"] = docs[doc_idx]["title"]
                else:
                    pred["title"] = ""
                pred["doc_id"] = None

            qa_results = nbest_pred
            logger.info("QA time: %f" % (time.time() - ct))

        return search_results, qa_results



if __name__ == '__main__':
    parser = argparse.ArgumentParser()

    parser.add_argument("--qa_model_path", default="models/bert_qa_model", type=str, 
                        help="Path to qa model")
    parser.add_argument("--ranking_model_path", default="models/bert_ranking_model", type=str,
                        help="Path to ranking model")
    parser.add_argument("--es_index_host", default="127.0.0.1", type=str,
                        help="Host for ES index service")
    parser.add_argument("--es_index_port", default=9200, type=int,
                        help="Port for ES index service")
    parser.add_argument("--use_model_path", default="models/USE_QA_3", type=str,
                        help="Path to universal sentence encoder module")

    parser.add_argument("--use_ann", action='store_true',
                        help="Set this flag if you are using ann.")
    parser.add_argument('--faiss_index_file', type=str, default='retrieval/faiss_index/covid_new.index',
                        help='Path to faiss index file')
    parser.add_argument('--original_paragraphs_file', type=str, default='data/original_paragraphs.jsonl',
                        help='Path to original paragraph file, used by faiss index')

    parser.add_argument("--ranking_batch_size", default=8, type=int,
                        help="Batch size per GPU/CPU for ranking model.")
    parser.add_argument("--qa_batch_size", default=1, type=int,
                        help="Batch size per GPU/CPU for qa model.")
    parser.add_argument("--num_docs", default=20, type=int,
                        help="Number of retrieved docs per question.")
    parser.add_argument("--num_para_ann", default=100, type=int,
                        help="Number of retrieved paragraphs per question using ann.")
    parser.add_argument("--num_para_qa", default=10, type=int,
                        help="Number of passages per question.")
    parser.add_argument("--max_seq_length", default=384, type=int,
                        help="The maximum total input sequence length after WordPiece tokenization. Sequences "
                             "longer than this will be truncated, and sequences shorter than this will be padded.")
    parser.add_argument("--max_query_length", default=64, type=int,
                        help="The maximum number of tokens for the question. Questions longer than this will "
                             "be truncated to this length.")
    parser.add_argument("--max_answer_length", default=200, type=int,
                        help="The maximum length of an answer that can be generated. This is needed because the start "
                             "and end predictions are not conditioned on one another.")
    parser.add_argument("--doc_stride", default=256, type=int,
                        help="doc_stride")
    parser.add_argument("--n_best_size", default=50, type=int,
                        help="The total number of n-best predictions to generate in the nbest_predictions.json output file.")
    parser.add_argument("--do_lower_case", default=True,
                        help="Set this flag if you are using an uncased model.")

    args = parser.parse_args()
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    logging.basicConfig(format = '%(asctime)s - %(levelname)s - %(name)s -   %(message)s',
                        datefmt = '%Y-%m-%d %H:%M:%S',
                        level = logging.INFO)

    irqa = IRQASystem(args, device)

    # Test Example
    query = 'What is known about transmission, incubation, and environmental stability?'
    search_results, qa_results= irqa.process_query(query)

    print("Query:")
    print(query)
    print("Search Result:")
    for (idx, ret) in enumerate(search_results):
        if idx > 5:
            break
        print("#%d score: %f" % (idx, ret["score"]))
        print(ret["title"])
        print(ret["keyphrases"])
        print(ret["text"])
    print("QA Result:")
    if qa_results:
        print(qa_results[0]["text"])
    else:
        print("None")



