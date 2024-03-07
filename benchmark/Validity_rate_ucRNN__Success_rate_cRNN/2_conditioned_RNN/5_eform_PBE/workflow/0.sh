#/bin/bash
#source /root/miniconda3/etc/profile.d/conda.sh
ulimit -s unlimited
#conda activate test
export GPAW_SETUP_PATH=/fs1/home/xiaohang/soft/Anaconda3/envs/py38/bin/gpaw
#conda activate base

/fs1/software/intel/2020.2/compilers_and_libraries_2020.2.254/linux/mpi/intel64/bin/mpiexec -n 36 --allow-run-as-root gpaw python 0_gpaw.py
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



