# ******************************************************************************************************
# ******************************************************************************************************
# RoBERTa
python prepro_for_kpe.py --source_dataset_dir /home/sunsi/dataset/CORD-19/CORD19-2/CORD-19-2020.3.24 \
--output_path ../../data/CORD-KPE-RoBERTa \
--pretrain_model_path ../../data/pretrain_model \
--bert_type roberta-base \


# # ******************************************************************************************************
# # ******************************************************************************************************
# # BERT
# python prepro_for_kpe.py --source_dataset_dir /home/sunsi/dataset/CORD-19/CORD19-2/CORD-19-2020.3.24 \
# --output_path ../data/CORD-KPE-BERT \
# --pretrain_model_path ../data/pretrain_model \
# --bert_type bert-base-cased \