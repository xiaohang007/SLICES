#!/bin/bash
python data_structs.py ../0_pretrain_dataset/alex_mp_20_EAH01_crystalnn.sli # datav2_essen_symtbr_dtrad_50_300_1_8_C.txt
#export PATH="/home/absws/miniconda3/bin:$PATH"
python train_prior.py --voc Voc_prior --train_data_dir ../0_pretrain_dataset/train_alex_mp_20_EAH01_crystalnn.sli --valid_data_dir ../0_pretrain_dataset/test_alex_mp_20_EAH01_crystalnn.sli --prior_model Prior_local.ckpt --batch_size 350 --epochs 100
