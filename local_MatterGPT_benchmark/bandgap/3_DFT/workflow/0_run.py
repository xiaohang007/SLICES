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
from plotter_mod import BSPlotter
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
                auto_npar=True, auto_gamma=False, copy_magmom=True, **vasp_job_kwargs),
        VaspJob(vasp_cmd, final=True, suffix=".relax2", settings_override=settings_relax2,
                auto_npar=True, auto_gamma=False, **vasp_job_kwargs)
    ]

    return jobs

def static_and_bands_calculations(vasp_cmd, optimized_structure_file="CONTCAR", directory="./bands", **vasp_job_kwargs):
    """
    Returns a list of two jobs for static and band structure calculations.
    
    Args:
        vasp_cmd (list): Command to run vasp as a list of args.
        optimized_structure_file (str): Path to the optimized structure file.
        directory (str): Directory to run the calculations.
        **vasp_job_kwargs: Additional kwargs to pass to VaspJob.

    Returns:
        List of two VaspJob objects for static and band structure calculations.
    """
    structure = Structure.from_file(optimized_structure_file)
    if os.path.exists(directory):
        os.system("rm -r " + directory)
    os.makedirs(directory)

    static_set = MPStaticSet(structure, user_incar_settings={'ISMEAR':0, "NELM": 500, "LCHARG": ".TRUE.", "LWAVE": ".FALSE.","LDAU": ".FALSE."},)
    static_set.write_input(directory)

    # Job 4: Band structure calculation
    mat = Poscar.from_file(optimized_structure_file)
    jarvis_kpath = JarvisKpoints().kpath(mat.atoms, line_density=20)

    with tempfile.NamedTemporaryFile(delete=False) as temp_file:
        temp_file_path = temp_file.name
        jarvis_kpath.write_file(temp_file_path)
        remove_none_from_lines(temp_file_path)

    kpts = Kpoints.from_file(temp_file_path)
    os.remove(temp_file_path)  # 删除临时文件

    band_set = MPNonSCFSet(structure,
                           user_incar_settings={'ISMEAR':0,"NELM": 500,"LORBIT": 10,"LCHARG": ".FALSE.","LWAVE": ".FALSE.", "ISTART": 1,"ICHARG": 11,"LDAU": ".FALSE."},
                           force_gamma=True)
    #        ,{"dict": "KPOINTS", "action": {"_set": band_set.kpoints.as_dict()}},         {"dict": "KPOINTS", "action": {"_set": jarvis_kpath.to_dict()}}
    settings_bands = [
        {"dict": "INCAR", "action": {"_set": band_set.incar.as_dict()}},
        {"dict": "KPOINTS", "action": {"_set": kpts.as_dict()}}
    ]
    jobs = [
        VaspJob(vasp_cmd, final=False, suffix=".static",
                auto_npar=True, auto_gamma=False, copy_magmom=True, **vasp_job_kwargs),
        VaspJob(vasp_cmd, final=True, suffix=".bands", settings_override=settings_bands,
                auto_npar=True, auto_gamma=False, **vasp_job_kwargs)
    ]

    return jobs
    
def main():
    # Define error handlers
    vasp_cmd = ["mpirun", "-np", "16", "/opt/vasp.6.3.2/bin/vasp_std"]
    handlers1 = [
        VaspErrorHandler(),
        UnconvergedErrorHandler(),
        NonConvergingErrorHandler(),
        PotimErrorHandler(),
        DriftErrorHandler(),
        AliasingErrorHandler(),
        PositiveEnergyErrorHandler(),
        MeshSymmetryErrorHandler(),
    ]
    handlers2 = [
        VaspErrorHandler(),
        UnconvergedErrorHandler(),
        AliasingErrorHandler(),
        MeshSymmetryErrorHandler(),
    ]
    # Run Custodian
    with open('./chemPotMP.json') as handle:
        chemPot = json.loads(handle.read())
    os.system("rm result.csv")
    with open("temp.csv", 'r') as f:
        reader = csv.reader(f)
        original_dir = os.getcwd()
        for row in reader:
            with open("./temp.vasp","w") as fn:
                fn.write('\n'.join(row[2].split('\\n')))
            with open("./result.csv","a") as fn: 
                try:
                    opt_jobs = structure_optimization(vasp_cmd, structure_file="temp.vasp",directory="./relax")
                    os.chdir("./relax")
                    c_opt = Custodian(handlers1, opt_jobs, max_errors=10)
                    c_opt.run()
                    os.chdir(original_dir)
                # Second part: Static and band structure calculations
                    calc_jobs = static_and_bands_calculations(vasp_cmd, optimized_structure_file="./relax/CONTCAR.relax2",directory="./bands")
                    os.chdir("./bands")
                    c_calc = Custodian(handlers2, calc_jobs, max_errors=10)
                    c_calc.run()
                    vrun = Vasprun2('./vasprun.xml.bands')
                    dir_gap = vrun.get_dir_gap
                    indir_gap, cbm, vbm = vrun.get_indir_gap 
                    #读取计算结果
                    drone = VaspToComputedEntryDrone()
                    queen = BorgQueen(drone, "./", 1)
                    entries = queen.get_data()
                    #修正计算结果并计算生成焓
                    compat = MaterialsProjectCompatibility()
                    compat.process_entries(entries)
                    entry = entries[0]
                    enthalpyForm=entry.energy
                    temp=entry.composition.get_el_amt_dict()
                    for i in range(len(temp)):
                        enthalpyForm=enthalpyForm-list(temp.values())[i]*chemPot[list(temp.keys())[i]]
                    enthalpyForm=enthalpyForm/entry.composition.num_atoms
                    vaspout = Vasprun1("./vasprun.xml.bands")
                    remove_none_from_lines("KPOINTS.bands")
                    bandstr = vaspout.get_band_structure(kpoints_filename="KPOINTS.bands",line_mode=True)
                    plotter = BSPlotter(bandstr)
                    ax = plotter.get_plot(ylim=[-2, 6])
                    fig = ax.figure
                    ax.tick_params(axis='x', labelsize=20)
                    ax.tick_params(axis='y', labelsize=20)
                    os.system("cp CONTCAR.bands ../../candidates/"+row[0]+"_dirGap-"+str(round(dir_gap,2))+"-indirGap-"+str(round(indir_gap,2))+"-eform-"+str(enthalpyForm)+".vasp")
                    with open("CONTCAR.bands", "r") as fn2:
                        poscar=fn2.read()
                        poscar=poscar.replace('\n','\\n')
                    fn.write(row[0]+','+row[1]+','+poscar+','+str(round(dir_gap,4))+','+str(round(indir_gap,4))+','+str(enthalpyForm)+','+row[3]+'\n') 
                    fig.savefig("../../candidates/"+row[0]+"_dirGap-"+str(round(dir_gap,2))+"-indirGap-"+str(round(indir_gap,2))+"-eform-"+str(enthalpyForm)+".png")
                except Exception as e:
                    print(f"Error processing row: {e}")
                finally:
                    os.chdir(original_dir)
                    os.system("rm -rf ./relax ./bands")
if __name__ == "__main__":
    main()
