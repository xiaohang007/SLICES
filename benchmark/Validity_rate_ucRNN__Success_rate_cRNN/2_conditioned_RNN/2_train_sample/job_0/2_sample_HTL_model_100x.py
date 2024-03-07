# -*- coding: utf-8 -*-
# Hang Xiao 2023.04
# xiaohang07@live.cn
import os, sys, math
import configparser
os.environ["CUDA_VISIBLE_DEVICES"]="" # use CPU only to fast enough
os.system("rm 100x.sli; touch 100x.sli")
os.environ["OMP_NUM_THREADS"] = "1"
config = configparser.ConfigParser()
config.read('./settings.ini') #path of your .ini file
sample_size = config.getint("Settings","sample_size") 
config.read('./settings2.ini') #path of your .ini file
eform = config.getfloat("Settings","target") 
for i in range(int(math.ceil(sample_size/100))):    
    os.system("python ../transfer_userinpt.py  --task sample_smiles --voc ../Voc_prior   --nums 100   --save_smi temp --tf_model ../Prior_local.ckpt --target "+str(eform))
    os.system("cat 100x.sli temp > temp2; mv temp2 100x.sli")
os.system("rm temp temp2 tmp.csv")
