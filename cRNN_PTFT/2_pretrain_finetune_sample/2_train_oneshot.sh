#!/bin/bash

#python transfer_userinpt.py  --task sample_smiles --voc Voc_prior   --nums 100   --save_smi test.abc --tf_model data/Prior_local.ckpt  


python train_oneshot.py  --voc Voc_prior --train_data_dir ../1_downstream_dataset/train_downstream.sli --valid_data_dir ../1_downstream_dataset/test_downstream.sli  --downstream_model oneshot_local.ckpt  --batch_size 350 --epochs 100 --learning_rate 0.002 --scaler 2 



