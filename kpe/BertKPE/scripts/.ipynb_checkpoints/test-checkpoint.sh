## dataset_class : biorxiv_medrxiv | noncomm_use_subset | comm_use_subset | custom_license
## model_class : bert2gram | roberta2gram | bert2tag | roberta2gram
## pretrain_model_type : bert-base-cased | roberta-base

# ******************************************************************************************************
# ******************************************************************************************************
# roberta2gram
CUDA_VISIBLE_DEVICES=0,1 python test.py --run_mode test \
--local_rank -1 \
--dataset_class biorxiv_medrxiv \
--model_class roberta2gram \
--pretrain_model_type roberta-base \
--per_gpu_test_batch_size 128 \
--preprocess_folder ../../data/CORD-KPE-RoBERTa \
--pretrain_model_path ../../data/pretrain_model \
--eval_checkpoint ../../data/checkpoints/roberta2gram.openkp.epoch_3.checkpoint \
--save_path ./Results \


# # ******************************************************************************************************
# # ******************************************************************************************************
# # bert2tag
# CUDA_VISIBLE_DEVICES=0,1 python test.py --run_mode test \
# --local_rank -1 \
# --dataset_class biorxiv_medrxiv \
# --model_class bert2tag \
# --pretrain_model_type bert-base-cased \
# --per_gpu_test_batch_size 128 \
# --preprocess_folder ../../data/CORD-KPE-BERT \
# --pretrain_model_path ../../data/pretrain_model \
# --eval_checkpoint ../../data/checkpoints/bert2tag.openkp.epoch_2.checkpoint \
# --save_path ./Results \

