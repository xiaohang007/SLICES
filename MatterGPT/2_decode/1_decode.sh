#!/bin/bash

# -------------------------------------
# 描述: 调用 run_local.py 脚本解码SLICES并且计算新材料的novelty
# -------------------------------------


python run.py \
    --input_csv ../1_train_generate/eform_bandgap.csv \
    --output_csv results.csv \
    --threads 16 
