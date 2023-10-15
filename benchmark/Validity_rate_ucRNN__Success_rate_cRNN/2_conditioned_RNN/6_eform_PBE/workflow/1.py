# -*- coding: utf-8 -*-
# Hang Xiao 2023.04
# xiaohang07@live.cn
import os,csv,glob
from jarvis.io.vasp.outputs import Outcar, Vasprun
from pymatgen.core.structure import Structure

os.system("rm result.csv")

exclude_elements=['Tc','Po','At','Ce','Pr','Nd','Pm','Sm','Eu','Gd',
'Tb','Dy','Ho','Er','Tm','Yb','Lu','Fr' ,'Ra','Ac','Th','Pa','U','Np',
'Pu','Am','Cm','Bk','Cf','Es','Fm','Md','No','Lr','Rf','Db','Sg','Bh','Hs','Mt','Ds','Rg','Cn','Nh','Fl','Mc','Lv','Ts','Og']


with open("temp.csv", 'r') as f:
    reader = csv.reader(f)
    #next(reader)
    for row in reader:
        with open("./temp.vasp","w") as fn:
            fn.write('\n'.join(row[2].split('\\n')))
        
        with open("./result.csv","a") as fn: 
            try:
                flag=0
                s=Structure.from_file("temp.vasp")
                s.to(fmt='cif',filename='temp.cif')
                symbols=[e.symbol for e in s.composition.elements]
                for i in symbols:
                    if i  in exclude_elements:
                        flag=1
                        continue 
                if not flag:
                    os.system("sh 0.sh")
                else:
                    os.system("sh 1.sh")
                os.system("cp ./EFormation/CONTCAR temp2.vasp")
                with open("temp2.vasp") as fn2:
                    lines = fn2.readlines()
                    lines[0] ='temp\n'
                with open("temp2.vasp", "w") as fn2:
                    fn2.writelines(lines)
                with open("temp2.vasp", "r") as fn2:
                    poscar=fn2.read()
                    poscar=poscar.replace('\n','\\n')           
                fn.write(row[0]+','+row[1]+','+poscar+','+','.join(row[3:]))  
                
                # read the formation energy per atom
                with open("results.txt") as r:
                    lines = r.readlines()[0]
                fn.write(','+lines)
            except Exception as e:
                print(e)
            finally:
                fn.write('\n')
                #os.system("rm -r results.txt ./Band ./Scf ./EFormation ./EFormation0 ./EFormation0.5")                
        





