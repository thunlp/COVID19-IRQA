# Keyphrase Extraction for COVID-19 papers

This repository provides two models: **[Bert2Tag](https://github.com/thunlp/Bert2Tag)** & **Bert2Gram**, which aims to extract keyphrases for CORD-19 papers. We released BERT and RoERBTa versions of our models. Our KPE models are pre-trained on the OpenKP (OpenKeyPhrase) dataset that is a large scale, open-domain keyphrase extraction dataset.

```
Bert2Tag : BERT (RoBERTa) + Sequence Tagging
Bert2Gram : BERT (RoBERTa) + CNN
```


## Quickstart
```
python 3.5
Pytorch 1.3.0
Tensorflow (tested on 1.14.0, only for tensorboardX)
```

First download KPE [**data**](https://drive.google.com/open?id=1RM0D84uVXXoovK4sJdGfQjc3MNvxNubj). The `data` folder includes 3 folders:

```
* pretrain_model : BERT & RoBERTa config files
* checkpoints : 4 pre-trained checkpoints used to generate keyphrases
* CORD-KPE-RoBERTa : This data can be obtained by preprocessing the original CORD-19 dataset (pre-processing scipts in `/BertKPE/preprocess`)
```

Then enter `BertKPE/scripts/` directory:

```
source test.sh
```
```
--dataset_class : 4 CORD-19 subset { biorxiv_medrxiv, noncomm_use_subset, comm_use_subset, custom_license }
--model_class : 4 pre-trained models { bert2gram, roberta2gram, bert2tag, roberta2tag }
--pretrain_model_type : 2 version { bert-base-cased, roberta-base }
--eval_checkpoint : 4 checkpoint files (needed reloaded)
```

Generated keyphrases for CORD-19 subset can be found in `BertKPE/scripts/Results`.

## Code Reference

https://github.com/thunlp/Bert2Tag
