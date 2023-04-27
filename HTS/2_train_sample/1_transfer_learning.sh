#!/bin/bash

#python transfer_userinpt.py  --task sample_smiles --voc Voc_prior   --nums 100   --save_smi test.abc --tf_model data/Prior_local.ckpt  


python transfer_userinpt.py  --task train_model --voc Voc_prior --smi ../1_augmentation/transfer_aug.sli --save_process_smi process.csv --prior_model Prior_local.ckpt --tf_model transfer_test.ckpt  




