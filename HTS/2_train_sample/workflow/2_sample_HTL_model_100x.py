# -*- coding: utf-8 -*-
# Hang Xiao 2023.04
# xiaohang07@live.cn
import os, sys
import math
os.environ["CUDA_VISIBLE_DEVICES"]="" # use CPU only to fast enough
import configparser
os.environ["OMP_NUM_THREADS"] = "1"
config = configparser.ConfigParser()
config.read('./settings.ini') #path of your .ini file
sample_size = config.getfloat("Settings","sample_size") 
os.system("rm 100x.sli; touch 100x.sli")
for i in range(int(math.ceil(sample_size/1000))):    
    os.system("python ../transfer_userinpt.py  --task sample_smiles --voc ../Voc_prior   --nums 1000   --save_smi temp --tf_model ../transfer_test.ckpt")
    os.system("cat 100x.sli temp > temp2; mv temp2 100x.sli")
os.system("rm temp temp2 tmp.csv")
