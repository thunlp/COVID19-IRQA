# IR and QA Pipeline System for COVID-19
<div  align="center"><img src="https://github.com/EdwardZH/COVID19IRQA/blob/master/System/logo/logo.png"/></div>

The repository is organized by [THUNLP](http://nlp.csai.tsinghua.edu.cn/site2/index.php/en) and [Microsoft AI](https://www.microsoft.com/en-us/). It contains an ongoing work of an IR and QA pipeline system towards the novel coronavirus COVID-19 (SARS-CoV-2). This system is trained with MS-MARCO, a large scale reading comprehension dataset, and directly transferred to the medical area. We hope this repository will help us work together against the COVID-19.

## COVID Dataset
The CORD-19 resource is constructed by Semantic Scholar of Allen Institute and will continue to be updated as new research is published in archival services and peer-reviewed publications. The shared task on [Kaggle](https://www.kaggle.com/allen-institute-for-ai/CORD-19-research-challenge) aims to help specialists in virusology, pharmacy and microbiology to find answers to the problem.

## IR and QA Pipeline

### Document Retrieval
The following models are implemented for an effective document retrieval system.
* BM25
* Approximate Nearest Neighbor (ANN)

### Paragraph Retrieval
* BERT (Base version of BERT with 12 layers)
* Distilled BERT (BERT with 3 layers)

### QA System
* BERT (Base version)

## Running Systems 

Downloading and unzipping checkpoints and index files into ``models`` and ``retrieval`` folders, respectively. You can find all resource on [Tsinghua Cloud](https://cloud.tsinghua.edu.cn/d/7ad972bdc56f45d7a06a/). Then install required packages.
```
pip install -r requirements.txt
```

Setting the CUDA device.
```
export CUDA_VISIBLE_DEVICES=DEVICE_ID
```

Running this pipeline system with the basic instruction.
BM25 document retrieval, BERT paragraph retrieval and BERT QA model.
```
python run_pipeline.py
```

Using ANN in document retrieval.
```
python run_pipeline.py --use_ann
```

Using Distilled BERT in paragraph retrieval.
```
python run_pipeline.py --ranking_model_path ./models/bert_ranking_model_distilled
```

## Running Results
Search result is a list of top-k document information and each document contains following fileds
* "title": Document title 
* "keyphrases": Extracted keyphrases
* "text": Document text

QA results is a list of top-k answers and each answer contains following fileds
* "text": Answer text 
* "title": The document tile where the answer is from


## Contribution
The following people share the same contribution for this repository:

[Aowei Lu](https://github.com/LAW991224), [Jiahua Liu](https://github.com/alphaf52), [Kaitao Zhang](https://github.com/zkt12), [Shi Yu](https://github.com/Yu-Shi), [Si Sun](https://github.com/SunSiShining), [Zhenghao Liu](http://nlp.csai.tsinghua.edu.cn/~lzh/)


## Project Organizers
- Chenyan Xiong
  * Microsoft Research AI, Redmond, USA
  * [Homepage](https://www.microsoft.com/en-us/research/people/cxiong/)
- Zhiyuan Liu
  * Tsinghua University
  * [Homepage](http://nlp.csai.tsinghua.edu.cn/~lzy/)