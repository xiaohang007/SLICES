# -*- coding: utf-8 -*-
# Hang Xiao 2023.04
# xiaohang07@live.cn
import os, sys
os.environ["CUDA_VISIBLE_DEVICES"]="" # use CPU only to fast enough
os.system("rm 100x.sci; touch 100x.sci")
for i in range(1):    
    os.system("python ../transfer_userinpt.py  --task sample_smiles --voc ../Voc_prior   --nums 250   --save_smi temp --tf_model ../Prior_local.ckpt")
    os.system("cat 100x.sci temp > temp2; mv temp2 100x.sci")
os.system("rm temp temp2 tmp.csv")
