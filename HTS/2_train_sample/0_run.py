# -*- coding: utf-8 -*-
# Hang Xiao 2023.04
# xiaohang07@live.cn
from invcryrep.utils import temporaryWorkingDirectory
import os
os.system("python data_structs.py ../1_augmentation/prior_aug.sli")
os.system("python train_prior.py --batch_size 1200 --epochs 3")
os.system("python transfer_userinpt.py  --task train_model --voc Voc_prior \
--smi ../1_augmentation/transfer_aug.sli --save_process_smi process.csv \
--prior_model Prior_local.ckpt --tf_model transfer_test.ckpt --batch_size 1200 --epochs 3")

from invcryrep.utils import temporaryWorkingDirectory,splitRun_sample,show_progress,collect_csv
splitRun_sample(threads=8,sample_size=16000) # generate 16000 SLICES in total with 8 CPU threads 
show_progress()
collect_csv(output="sampled.sli", glob_target="job_*/100x.sli",header="",cleanup=True)