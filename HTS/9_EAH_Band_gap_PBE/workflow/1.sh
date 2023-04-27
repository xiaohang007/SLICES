#/bin/bash
#source /root/miniconda3/etc/profile.d/conda.sh
ulimit -s unlimited

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


