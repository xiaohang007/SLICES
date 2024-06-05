#!/bin/bash

python ./generate_prior.py   --voc ./Voc_prior   --gen_size 1000 --batch_size 100   --save_slices prior.sli --model ./Prior_local.ckpt


