#!/bin/bash
python train.py \
  --run_name bandgap_Aug1 \
  --batch_size 36 \
  --num_props 1 \
  --max_epochs 20 \
  --n_embd 512 \
  --n_layer 8 \
  --n_head 8 \
  --learning_rate 3.3e-4 \
  --train_dataset "../../mp20_nonmetal/train_data_reduce_zero.csv" \
  --test_dataset "../../mp20_nonmetal/test_data_reduce_zero.csv"
