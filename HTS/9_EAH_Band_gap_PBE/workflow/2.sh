#/bin/bash

ulimit -s unlimited

python 0_scf.py
cd Scf
mpirun -np 8 /opt/vasp.5.4.4/bin/vasp_std
cd ..
python 0_band.py
cp plotBand.py ./Band
cd Band
mpirun -np 8 /opt/vasp.5.4.4/bin/vasp_std
#python plotBand.py

