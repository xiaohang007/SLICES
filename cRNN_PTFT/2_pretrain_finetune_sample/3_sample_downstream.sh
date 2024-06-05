#!/bin/bash

python ./generate_downstream.py   --voc ./Voc_prior --gen_size 1000 --batch_size 100  --save_slices downstream.sli --model ./downstream_local.ckpt  --scaler 2


