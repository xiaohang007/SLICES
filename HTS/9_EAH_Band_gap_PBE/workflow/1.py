# -*- coding: utf-8 -*-
# Hang Xiao 2023.04
# xiaohang007@gmail.com
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
                with open("ehull.txt", "r") as fn2:
                    ehull=fn2.read()            
                fn.write(row[0]+','+row[1]+','+poscar+','+','.join(row[3:])+','+ehull)  
                if float(ehull.split(',')[-2]) <=  0.05:     
                    os.system("sh 2.sh")
                    try:
                        vrun = Vasprun('./Band/vasprun.xml')
                        dir_gap = vrun.get_dir_gap
                        indir_gap, cbm, vbm = vrun.get_indir_gap 
                        fn.write(','+str(round(dir_gap,4))+','+str(round(indir_gap,4)))
                        if  (dir_gap-indir_gap) < 0.05  and indir_gap >=0.1:
                            print("!!!!!!!!!band success")
                            os.chdir("./Band")
                            try:
                                os.system("python plotBand.py;cp band.png ../candidates/"+row[0]+"_dirGap-"+str(round(dir_gap,4))+"-indirGap-"+str(round(indir_gap,4))+".png")
                            except:
                                pass
                            os.chdir('../')
                    except:
                        print("!!!!!!!!!band failed, try again with ISYM=0")
                        os.system("sh 3.sh")
                        vrun = Vasprun('./Band/vasprun.xml')
                        dir_gap = vrun.get_dir_gap
                        indir_gap, cbm, vbm = vrun.get_indir_gap 
                        fn.write(','+str(round(dir_gap,4))+','+str(round(indir_gap,4)))
                        if  (dir_gap-indir_gap) < 0.05  and indir_gap >=0.1 :
                            os.chdir("./Band")
                            try:
                                os.system("python plotBand.py;cp band.png ../candidates/"+row[0]+"_dirGap-"+str(round(dir_gap,4))+"-indirGap-"+str(round(indir_gap,4))+".png")
                            except:
                                pass
                            os.chdir('../')                        
            except Exception as e:
                print(e)
            finally:
                fn.write('\n')
                os.system("rm -r ehull.txt ./Band ./Scf ./EFormation ./EFormation0 ./EFormation0.5")
        





