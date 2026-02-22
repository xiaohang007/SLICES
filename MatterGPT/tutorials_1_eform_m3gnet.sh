#!/bin/bash

# 如果有任何命令失败，立即退出脚本
set -e

# 定义一个打印分隔线的函数
print_separator() {
    echo "=============================================="
}

# 获取脚本所在的目录
SCRIPT_DIR=$( cd $( dirname $0 ) >/dev/null 2>&1 && pwd )

# 定义常用变量
DATA_DIR=$SCRIPT_DIR/0_dataset
MATTERGPT_DIR=$SCRIPT_DIR
RAW_DATA_PATH=$SCRIPT_DIR/raw_mp20_dataset_mini.csv
TRAIN_DIR=$MATTERGPT_DIR/1_train_generate
DECODE_DIR=$MATTERGPT_DIR/2_decode
NOVELTY_CHECK_DIR=$MATTERGPT_DIR/3_novelty
DECODE_DIR_EFORM=$MATTERGPT_DIR/demo_decode_novelty_check_eform_m3gnet
STRUCTURE_JSON_FOR_NOVELTY_CHECK=$DATA_DIR/cifs_filtered.json
TRAINING_FILE=$DATA_DIR/train_data.csv
VAL_DATASET=$DATA_DIR/val_data.csv
THREADS=8

# 定义模型参数  
# 下面的是快速测试用的参数，实际科研中建议使用至少 512 8 8，如果数据量足够可以用 768 12 12
BATCH_SIZE=36
MAX_EPOCHS=20
N_EMBD=256  
N_LAYER=8
N_HEAD=8
LEARNING_RATE=3.3e-4

# 定义生成参数
GEN_BATCH_SIZE=5
GEN_SIZE=50

# 开始教程 2.1：具有目标形成能的材料逆向设计 (x)
print_separator
echo "开始教程 2.1：具有目标形成能的材料逆向设计"

# 构建训练集
print_separator
echo "构建训练集..."
cd $DATA_DIR
python run.py \
    --raw_data_path $RAW_DATA_PATH \
    --cif_column_index 7 \
    --prop_column_index_list 2 3  \
    --structure_json_for_novelty_check $STRUCTURE_JSON_FOR_NOVELTY_CHECK \
    --complete_train_set_name mp20_eform_bandgap_nonmetal.csv \
    --train_output_name $TRAINING_FILE \
    --val_output_name $VAL_DATASET \
    --threads $THREADS

# 训练 MatterGPT 进行单属性材料逆向设计（以形成能为例）
print_separator
echo "训练 MatterGPT 进行形成能逆向设计..."
cd $TRAIN_DIR

# 训练参数
RUN_NAME=eform
SLICES_COLUMN_INDEX=0
PROP_COLUMN_INDEX_LIST="1"  # 形成能列索引

python train.py \
  --run_name $RUN_NAME \
  --batch_size $BATCH_SIZE \
  --max_epochs $MAX_EPOCHS \
  --n_embd $N_EMBD \
  --n_layer $N_LAYER \
  --n_head $N_HEAD \
  --learning_rate $LEARNING_RATE \
  --train_dataset $TRAINING_FILE \
  --val_dataset $VAL_DATASET \
  --slices_column_index $SLICES_COLUMN_INDEX \
  --prop_column_index_list $PROP_COLUMN_INDEX_LIST

# 生成具有指定形成能的 SLICES 字符串
print_separator
echo "生成具有指定形成能的 SLICES 字符串..."
PROP_TARGETS=[[-1.0],[-2.0],[-3.0],[-4.0]]

python generate.py \
    --model_weight ${RUN_NAME}.pt \
    --output_csv ${RUN_NAME}.csv \
    --batch_size $GEN_BATCH_SIZE \
    --gen_size $GEN_SIZE \
    --train_dataset $TRAINING_FILE \
    --prop_targets $PROP_TARGETS 
# 从 SLICES 重构晶体，评估新颖性，预测形成能并可视化
print_separator
echo "从 SLICES 重构晶体并评估新颖性..."
cd $DECODE_DIR_EFORM
python run.py \
    --input_csv $TRAIN_DIR/${RUN_NAME}.csv \
    --structure_json_for_novelty_check $STRUCTURE_JSON_FOR_NOVELTY_CHECK \
    --training_file $TRAINING_FILE \
    --output_csv ${RUN_NAME}_decode.csv \
    --threads $THREADS 
cp combined_results.png ../${RUN_NAME}_m3gnet.png

