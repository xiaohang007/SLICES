#!/bin/bash
python data_structs.py ../1_augmentation/prior_aug.sli # datav2_essen_symtbr_dtrad_50_300_1_8_C.txt
#export PATH="/home/absws/miniconda3/bin:$PATH"
python train_prior.py

