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
RAW_DATA_PATH=$DATA_DIR/raw_mp20_dataset.csv
TRAIN_DIR=$MATTERGPT_DIR/1_train_generate
DECODE_DIR=$MATTERGPT_DIR/2_decode
NOVELTY_CHECK_DIR=$MATTERGPT_DIR/3_novelty
DECODE_DIR_EFORM=$MATTERGPT_DIR/demo_decode_novelty_check_eform_m3gnet
STRUCTURE_JSON_FOR_NOVELTY_CHECK=$DATA_DIR/cifs_filtered.json
TRAINING_FILE=$DATA_DIR/train_data_reduce_zero.csv
VAL_DATASET=$DATA_DIR/val_data_reduce_zero.csv
THREADS=8

# 定义模型参数  
# 下面的是快速测试用的参数，实际科研中建议使用至少 512 8 8，如果数据量足够可以用 768 12 12
BATCH_SIZE=36
MAX_EPOCHS=50
N_EMBD=256  
N_LAYER=4
N_HEAD=4
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
    --n_head $N_HEAD \
    --gen_size $GEN_SIZE \
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

# 在 PBE 水平评估重构晶体的形成能分布（需要工作站或 HPC 快速运行 VASP）
print_separator
# 读取${RUN_NAME}_novelty.csv中的晶体结构，需要把poscar的字符串做处理：.replace('\\n','\n')
echo "使用 VASP 评估形成能分布..."
echo "请确保 VASP 已安装并在您的工作站或 HPC 上正确配置。"
echo "如有必要，请修改工作流程中的脚本..."
# cd ../demo_template_DFT
# python 1_run.py

echo "教程 2.1 成功完成！"

# 教程 2.2：具有目标带隙的材料逆向设计
print_separator
echo "开始教程 2.2：具有目标带隙的材料逆向设计"

# 训练 MatterGPT 进行单属性材料逆向设计（以带隙为例）
print_separator
echo "训练 MatterGPT 进行带隙逆向设计..."
cd $TRAIN_DIR

# 训练参数
RUN_NAME=bandgap
SLICES_COLUMN_INDEX=0
PROP_COLUMN_INDEX_LIST="2"  # 带隙列索引

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

# 生成具有指定带隙的 SLICES 字符串
print_separator
echo "生成具有指定带隙的 SLICES 字符串..."
PROP_TARGETS=[[1.0],[2.0],[3.0],[4.0]]

python generate.py \
    --model_weight ${RUN_NAME}.pt \
    --output_csv ${RUN_NAME}.csv \
    --batch_size $GEN_BATCH_SIZE \
    --n_head $N_HEAD \
    --gen_size $GEN_SIZE \
    --prop_targets $PROP_TARGETS 

# 从 SLICES 重构晶体
print_separator
echo "从 SLICES 重构晶体..."
cd $DECODE_DIR
python run.py \
    --input_csv $TRAIN_DIR/${RUN_NAME}.csv \
    --output_csv ${RUN_NAME}_decode.csv \
    --threads $THREADS 

# 评估晶体结构的新颖性
print_separator
echo "评估晶体结构的新颖性..."
cd $NOVELTY_CHECK_DIR
python run.py \
    --input_csv $DECODE_DIR/${RUN_NAME}_decode.csv \
    --structure_json_for_novelty_check $STRUCTURE_JSON_FOR_NOVELTY_CHECK \
    --output_csv ${RUN_NAME}_novelty.csv \
    --threads $THREADS 
cp ${RUN_NAME}_novelty.csv ../${RUN_NAME}_novelty.csv

# 在 PBE 水平评估重构晶体的带隙分布（需要工作站或 HPC 快速运行 VASP）
print_separator
# 读取${RUN_NAME}_novelty.csv中的晶体结构，需要把poscar的字符串做处理：.replace('\\n','\n')
echo "使用 VASP 评估带隙分布..."
echo "请确保 VASP 已安装并在您的工作站或 HPC 上正确配置。"
echo "如有必要，请修改工作流程中的脚本..."
# cd ../demo_template_DFT
# python 1_run.py

echo "教程 2.2 成功完成！"

# 教程 2.3：具有目标带隙和目标形成能的材料逆向设计
print_separator
echo "开始教程 2.3：具有目标带隙和目标形成能的材料逆向设计"

# 训练 MatterGPT
print_separator
echo "训练 MatterGPT 进行带隙和形成能的联合逆向设计..."
cd $TRAIN_DIR

# 训练参数
RUN_NAME=eform_bandgap
SLICES_COLUMN_INDEX=0
PROP_COLUMN_INDEX_LIST="1 2"  # 形成能和带隙列索引

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

# 生成具有指定形成能和带隙的 SLICES 字符串
print_separator
echo "生成具有指定形成能和带隙的 SLICES 字符串..."
PROP_TARGETS=[[-2.0,1.0],[-1.0,4.0]]

python generate.py \
    --model_weight ${RUN_NAME}.pt \
    --batch_size $GEN_BATCH_SIZE \
    --output_csv ${RUN_NAME}.csv \
    --n_head $N_HEAD \
    --gen_size $GEN_SIZE \
    --prop_targets $PROP_TARGETS 

# 从 SLICES 重构晶体
print_separator
echo "从 SLICES 重构晶体..."
cd $DECODE_DIR
python run.py \
    --input_csv $TRAIN_DIR/${RUN_NAME}.csv \
    --output_csv ${RUN_NAME}_decode.csv \
    --threads $THREADS 

# 评估晶体结构的新颖性
print_separator
echo "评估晶体结构的新颖性..."
cd $NOVELTY_CHECK_DIR
python run.py \
    --input_csv $DECODE_DIR/${RUN_NAME}_decode.csv \
    --structure_json_for_novelty_check $STRUCTURE_JSON_FOR_NOVELTY_CHECK \
    --output_csv ${RUN_NAME}_novelty.csv \
    --threads $THREADS 
cp ${RUN_NAME}_novelty.csv ../${RUN_NAME}_novelty.csv
# 在 PBE 水平评估重构晶体的带隙和形成能分布（需要工作站或 HPC 快速运行 VASP）
print_separator
# 读取${RUN_NAME}_novelty.csv中的晶体结构，需要把poscar的字符串做处理：.replace('\\n','\n')
echo "使用 VASP 评估带隙和形成能分布..."
echo "请确保 VASP 已安装并在您的工作站或 HPC 上正确配置。"
echo "如有必要，请修改工作流程中的脚本..."
# cd ../demo_template_DFT
# python 1_run.py

echo "教程 2.3 成功完成！"

