#!/bin/bash
#SBATCH -p cp1
#SBATCH -n 36
#source /public1/soft/modules/module.sh 
#module purge
#module load anaconda/3-Python-3.8.3-phonopy-phono3py 
#source activate py
#export PATH=~/.conda/envs/py/bin:$PATH
#which python
#source ./env.sh
#which gpaw
module add vasp
#which vasp
#export PYTHONUNBUFFERED=1

python 1.py
