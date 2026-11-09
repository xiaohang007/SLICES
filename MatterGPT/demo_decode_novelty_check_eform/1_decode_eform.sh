#!/bin/bash

# -------------------------------------
# 描述: 解码 SLICES、计算 novelty，并在结束后自动计算 MAPE/Novelty 指标
#      同时把指标写入 metrics_summary.csv，并渲染到 combined_results.png 右侧面板
# -------------------------------------
set -euo pipefail

TOLERANCE=${TOLERANCE:-0.2}
TOTAL_SAMPLES=${TOTAL_SAMPLES:-0}  # 0 表示自动从 input_csv 推断
THREADS=${THREADS:-16}

python run.py \
    --input_csv ../1_train_generate/eform.csv \
    --structure_json_for_novelty_check ../../data/mp20/cifs.json \
    --training_file ../0_dataset/train_data.csv \
    --output_csv results.csv \
    --threads "${THREADS}" \
    --tolerance "${TOLERANCE}" \
    --total_samples "${TOTAL_SAMPLES}" \
    --metrics_output_csv metrics_summary.csv \
    --figure_output combined_results.png
