# -*- coding: utf-8 -*-
# Hang Xiao 2023.04
# xiaohang07@live.cn
import os, sys

os.system("rm 100x.sli; touch 100x.sli")
for i in range(2):    
    os.system("python ../transfer_userinpt.py  --task sample_smiles --voc ../Voc_prior   --nums 5000   --save_smi temp --tf_model ../transfer_test.ckpt")
    os.system("cat 100x.sli temp > temp2; mv temp2 100x.sli")
os.system("rm temp temp2 tmp.csv")
