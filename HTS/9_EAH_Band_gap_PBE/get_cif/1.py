# -*- coding: utf-8 -*-
# Hang Xiao 2023.04
# xiaohang07@live.cn
from jarvis.tasks.vasp.vasp import (
    JobFactory,
    VaspJob,
    GenericIncars,
    write_jobfact,
)
from pymatgen.core.structure import Structure
from jarvis.io.vasp.inputs import Potcar, Incar, Poscar
from jarvis.db.jsonutils import dumpjson
from jarvis.db.figshare import data
from jarvis.core.atoms import Atoms
from jarvis.tasks.queue_jobs import Queue
import os,csv,glob
vasp_cmd = "mpirun -np 8 /opt/vasp.6.3.2/bin/vasp_std_3d"
copy_files = [os.getcwd()+"/plotBand.py"]
submit_cmd = ["qsub", "submit_job"]

# For slurm
# submit_cmd = ["sbatch", "submit_job"]

steps = [
    "SCF",
    "BANDSTRUCT",
    "BANDSTRUCT_MBJ",
]

try:
    for i in glob.glob("*PBE"):
        os.system("rm -r "+i)

except:
    print("no job folders to delete")
os.system("rm -r ./structures ")
os.mkdir("structures")
count=1

with open("results_9_EAH_Band_gap_PBE_filtered.csv", 'r') as f:
    reader = csv.reader(f)
    next(reader)
    for row in reader:
        with open("./structures/"+str(row[0])+".vasp","w") as fn:
            fn.write('\n'.join(row[2].split('\\n')))

os.chdir("./structures/")
cifs =glob.glob("*.vasp") 

for i in cifs:
    structure = Structure.from_file(i)
    structure.to(fmt='cif',filename=i.split(".")[0]+'.cif')
    os.system("rm "+i)




