#/bin/bash
#source /root/miniconda3/etc/profile.d/conda.sh
ulimit -s unlimited
conda activate test
export GPAW_SETUP_PATH=/root/miniconda3/envs/test/share/gpaw
/root/miniconda3/envs/test/bin/mpiexec -n 8 --allow-run-as-root  gpaw python 0_gpaw.py
conda activate base
python 0_EnthalpyOfFormation0.py
cd EFormation0
mpirun -np 8 /opt/vasp.5.4.4/bin/vasp_std
cd ..
python 0_EnthalpyOfFormation0.5.py
cd EFormation0.5
mpirun -np 8 /opt/vasp.5.4.4/bin/vasp_std
cd ..
python 0_EnthalpyOfFormation1.py
cd EFormation
mpirun -np 8 /opt/vasp.5.4.4/bin/vasp_std
cd ..
python 0_EnergyAboveHull.py


