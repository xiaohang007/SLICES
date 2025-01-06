#!/bin/bash

# -------------------------------------
# 描述: 调用 run.py 脚本解码SLICES并且计算新材料的novelty
# -------------------------------------

python run.py \
    --input_csv ../1_train_generate/inverse_designed_SLICES_eform.csv \
    --structure_json ../0_dataset/cifs_filtered.json \
    --training_file ../0_dataset/train_data_reduce_zero.csv \
    --output_csv results.csv \
    --threads 8 
