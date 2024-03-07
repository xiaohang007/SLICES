#/bin/bash
#source /root/miniconda3/etc/profile.d/conda.sh
ulimit -s unlimited
#conda activate test
export GPAW_SETUP_PATH=/fs1/home/xiaohang/soft/Anaconda3/envs/py38/bin/gpaw
#conda activate base

python 0_EnthalpyOfFormation0.py
cd EFormation0
yhrun -n 36 vasp_std
cp CONTCAR ../0.vasp
cd ..
python 0_EnthalpyOfFormation0.5.py
cd EFormation0.5
yhrun -n 36 vasp_std
cp CONTCAR ../1.vasp
cd ..
python 0_EnthalpyOfFormation1.py
cd EFormation
yhrun -n 36 vasp_std
cp CONTCAR ../2.vasp
cd ..
python 0_EnthalpyOfFormation2.py
#cd EFormation



