import shutil
from custodian.vasp.jobs import VaspJob
from pymatgen.io.vasp.sets import MPRelaxSet, MPStaticSet, MPNonSCFSet
from pymatgen.core.structure import Structure
from jarvis.io.vasp.inputs import Poscar
from jarvis.core.kpoints import Kpoints3D as JarvisKpoints
from pymatgen.io.vasp.inputs import Kpoints
from custodian.custodian import Custodian
from custodian.vasp.handlers import VaspErrorHandler, UnconvergedErrorHandler, NonConvergingErrorHandler, \
 PotimErrorHandler, DriftErrorHandler,AliasingErrorHandler,PositiveEnergyErrorHandler,MeshSymmetryErrorHandler
import tempfile
from pymatgen.io.vasp.outputs import Vasprun as Vasprun1
from pymatgen.electronic_structure.plotter import BSPlotter
from jarvis.io.vasp.outputs import Vasprun as Vasprun2
import os,csv,glob
import subprocess
import matplotlib.pyplot as plt
import json
from pymatgen.apps.borg.hive import VaspToComputedEntryDrone
from pymatgen.apps.borg.queen import BorgQueen
from pymatgen.analysis.phase_diagram import PhaseDiagram, PDPlotter
from pymatgen.entries.compatibility import MaterialsProjectCompatibility
from pymatgen.core.composition import Composition
from pymatgen.core.periodic_table import Element
def remove_none_from_lines(file_path):
    with open(file_path, 'r') as file:
        lines = file.readlines()
    # remove "None" in KPOINTS
    cleaned_lines = [line.replace("None", "").strip() + '\n' for line in lines]
    with open(file_path, 'w') as file:
        file.writelines(cleaned_lines)
def structure_optimization(vasp_cmd, structure_file="POSCAR", directory="./relax", **vasp_job_kwargs):
    """
    Returns a list of two jobs for structure optimization.
    
    Args:
        vasp_cmd (list): Command to run vasp as a list of args.
        structure_file (str): Path to the input structure file.
        directory (str): Directory to run the calculations.
        **vasp_job_kwargs: Additional kwargs to pass to VaspJob.

    Returns:
        List of two VaspJob objects for structure optimization.
    """
    structure = Structure.from_file(structure_file)
    if os.path.exists(directory):
        os.system("rm -r " + directory)
    os.makedirs(directory)

    # Job 1: Initial relaxation
    relax1_set = MPRelaxSet(structure, 
                            user_incar_settings={'EDIFF': 0.0001,'EDIFFG': -0.05,'LREAL':".FALSE.",  \
'NSW': 200,'ALGO': 'Normal','PREC': 'Normal','ISMEAR': -5,'ISIF': 3,'LORBIT':'.FALSE.','LCHARG':'.FALSE.',"LWAVE": ".FALSE.","LDAU":".FALSE.","ISPIN":1,"ISYM":0},
                            user_kpoints_settings={'reciprocal_density': 32},
                            force_gamma=True)
    relax1_set.write_input(directory)

    # Job 2: Second relaxation
    relax2_set = MPRelaxSet(structure,force_gamma=True)
    
    settings_relax2 = [
        {"dict": "INCAR", "action": {"_set": relax2_set.incar.as_dict()}},
        {"dict": "KPOINTS", "action": {"_set": relax2_set.kpoints.as_dict()}},
        {"file": "CONTCAR", "action": {"_file_copy": {"dest": "POSCAR"}}},
    ]

    jobs = [
        VaspJob(vasp_cmd, final=False, suffix=".relax1",
                auto_npar=False, auto_gamma=False, copy_magmom=True, **vasp_job_kwargs),
        VaspJob(vasp_cmd, final=True, suffix=".relax2", settings_override=settings_relax2,
                auto_npar=False, auto_gamma=False, **vasp_job_kwargs)
    ]

    return jobs
    
def main():
    # Define error handlers
    vasp_cmd = ["/opt/vasp.6.3.2/bin/vasp_std"] #["mpirun", "-np", "4", "/opt/vasp.6.3.2/bin/vasp_std"]
    handlers = [
        VaspErrorHandler(),
        UnconvergedErrorHandler(),
        NonConvergingErrorHandler(),
        PotimErrorHandler(),
        DriftErrorHandler(),
        AliasingErrorHandler(),
        PositiveEnergyErrorHandler(),
        MeshSymmetryErrorHandler(),
    ]
    # Run Custodian
    with open('./chemPotMP.json') as handle:
        chemPot = json.loads(handle.read())
    os.system("rm result.csv")
    with open("temp.csv", 'r') as f:
        reader = csv.reader(f)
        for row in reader:
            with open("./temp.vasp","w") as fn:
                fn.write('\n'.join(row[3].split('\\n'))) # eform [3] other [2]
            with open("./result.csv","a") as fn: 
                opt_jobs = structure_optimization(vasp_cmd, structure_file="temp.vasp",directory="./relax")
                os.chdir("./relax")
                try:
                    c_opt = Custodian(handlers, opt_jobs, max_errors=5)
                    c_opt.run()
                    drone = VaspToComputedEntryDrone()
                    queen = BorgQueen(drone, "./", 1)
                    entries = queen.get_data()
                    print(entries)
                    #修正计算结果并计算生成焓
                    compat = MaterialsProjectCompatibility()
                    compat.process_entries(entries)
                    entry = entries[0]
                    enthalpyForm=entry.energy
                    temp=entry.composition.get_el_amt_dict()
                    for i in range(len(temp)):
                        enthalpyForm=enthalpyForm-list(temp.values())[i]*chemPot[list(temp.keys())[i]]
                    enthalpyForm=enthalpyForm/entry.composition.num_atoms
                    with open("CONTCAR.relax2", "r") as fn2:
                        poscar=fn2.read()
                        poscar=poscar.replace('\n','\\n')
                    fn.write(row[0]+','+row[1]+','+str(enthalpyForm)+','+poscar+','+row[4]+'\n') 
                except Exception as e:
                    print(e)
                os.chdir("..")
if __name__ == "__main__":
    main()
