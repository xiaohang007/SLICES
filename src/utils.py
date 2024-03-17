# -*- coding: UTF-8 -*-
# Hang Xiao 2024.03
# xiaohang07@live.cn
import os,sys,glob,json,io
import re,time
import numpy as np
import math,json
import tempfile
from tqdm import tqdm
from pymatgen.core.structure import Structure
import warnings
warnings.filterwarnings("ignore")
import contextlib
from itertools import zip_longest
from mp_api.client.mprester import MPRester
import configparser
from contextlib import redirect_stdout

@contextlib.contextmanager
#temporarily change to a different working directory
def temporaryWorkingDirectory(path):
    _oldCWD = os.getcwd()
    os.chdir(os.path.abspath(path))

    try:
        yield
    finally:
        os.chdir(_oldCWD)

def split_list(a, n):
    k, m = divmod(len(a), n)
    return (a[i*k+min(i, m):(i+1)*k+min(i+1, m)] for i in range(n))

def splitRun(filename,threads,skip_header=False):
    os.system('rm -rf job_* structures_ori_opt ./result.csv')
    with open(filename, 'r') as f:
        cifs=json.load(f)
    if skip_header:
        cifs_split=list(split_list(cifs[1:],threads))
    else:
        cifs_split=list(split_list(cifs,threads))
    for i in range(len(cifs_split)):
        os.mkdir('job_'+str(i))
        os.system('cp -r ./workflow/. job_'+str(i))
        with open('temp.json', 'w') as f:
            json.dump(cifs_split[i], f)
        os.system('mv temp.json job_'+str(i))
        os.chdir('job_'+str(i))
        if len(sys.argv)==2:
            if sys.argv[1]=="test":
                os.system('qsub 0_test.pbs')
        else:
            os.system('qsub 0_run.pbs > /dev/null 2>&1')
        os.chdir('..')
    print("Computational tasks have been submitted.")

def splitRun_csv(filename,threads,skip_header=False):
    os.system('rm -rf job_* structures_ori_opt ./result.csv')
    with open(filename, 'r') as f:
        if skip_header:
            cifs=f.readlines()[1:]
        else:
            cifs=f.readlines()
    cifs_split=list(split_list(cifs,threads))
    for i in range(len(cifs_split)):
        os.mkdir('job_'+str(i))
        os.system('cp -r ./workflow/. job_'+str(i))
        with open('temp.csv', 'w') as f:
            f.writelines(cifs_split[i])
        os.system('mv temp.csv job_'+str(i))
        os.chdir('job_'+str(i))
        if len(sys.argv)==2:
            if sys.argv[1]=="test":
                os.system('qsub 0_test.pbs')
        else:
            os.system('qsub 0_run.pbs > /dev/null 2>&1')
        os.chdir('..')
    print("Computational tasks have been submitted.")

def splitRun_sample(threads=8,sample_size=8000):
    config = configparser.ConfigParser()
    os.system('rm -rf job_* structures_ori_opt ./result.csv')
    config["Settings"] = {'sample_size':int(sample_size/threads) }
    with open('./workflow/settings.ini', 'w') as configfile:
        config.write(configfile)
    for i in range(threads):
        os.mkdir('job_'+str(i))
        os.system('cp -r ./workflow/. job_'+str(i))
        os.chdir('job_'+str(i))
        if len(sys.argv)==2:
            if sys.argv[1]=="test":
                os.system('qsub 0_test.pbs')
        else:
            os.system('qsub 0_run.pbs > /dev/null 2>&1')
        os.chdir('..')
    print("Sampling tasks have been submitted.")

def show_progress():
    try:
        countTask = 0
        temp_dir = tempfile.TemporaryDirectory(dir="/tmp")
        with tqdm(total=100, position=0, leave=True,bar_format='{desc:<5.5}{percentage:3.0f}%|{bar:15}{r_bar}') as pbar:
            while True:
                countTask0 = countTask
                os.system('qstat > ' + temp_dir.name + '/temp.log')  # Execute qstat and save output
                log = open(temp_dir.name + '/temp.log').readlines()[2:]  # Read qstat output
                countTask = 0
                for i in log:
                    if i.split()[4] == 'R' or i.split()[4] == 'Q':  # Count running or queued tasks
                        countTask += 1
                # Reset totalTask if new tasks are detected
                if countTask0 < countTask:
                    totalTask = countTask
                    pbar.update(0)
                # Update progress bar if the number of tasks decreases
                if countTask0 > countTask:
                    pbar.update((totalTask - countTask) / totalTask * 100)
                # If all tasks are completed, update the progress bar to 100% before breaking
                if (totalTask - countTask) == totalTask:
                    pbar.n = pbar.total  # Update the progress bar to 100%
                    #pbar.refresh()  # Refresh the display
                    pbar.close()  # Properly close the progress bar
                    temp_dir.cleanup()  # Clean up the temporary directory
                    break
                time.sleep(1)
    except KeyboardInterrupt:
        # press stop button will cancel all jobs
        os.system('scancel --user=root')
        print("All jobs have been canceled")
    finally:
        temp_dir.cleanup()


def collect_json(output,glob_target,cleanup=True):
    data=[]               
    for f in glob.glob(glob_target, recursive=True):
        with open(f,"r") as infile:
            temp=json.load(infile)  # put each cifs into the final list
            for i in temp:
                data.append(i)
    with open(output,'w') as outfile:
        json.dump(data, outfile)     
    if cleanup:
        for i in glob.glob("job_*"):
            os.system("rm -r "+i)
    print("Results have been collected into: "+output)

def collect_csv(output,glob_target,header="",index=False,cleanup=True):
    result_sli=""
    if index:
        index=0
        for i in glob.glob(glob_target):
            with open(i,'r') as result:
                lines=result.readlines()
                for j in range(len(lines)):
                    result_sli+=str(index)+','+lines[j]
                    index+=1
    else:
        for f in glob.glob(glob_target, recursive=True):
            with open(f,"r") as infile:
                result_sli += infile.read()
    with open(output,'w') as result:
        if header!="":
            result.write(header)
        result.write(result_sli)  
    if cleanup:
        for i in glob.glob("job_*"):
            os.system("rm -r "+i)
    print("Results have been collected into: "+output)

def collect_csv_filter(output,glob_target,header,condition,cleanup=True):
    result_csv=''
    result_filtered_csv=''
    for i in glob.glob(glob_target):
        with open(i,'r') as result:
            for j in result.readlines():
                result_csv+=j
                if condition(j):
                    result_filtered_csv+=j
    with open(output,'w') as result:
        if header!="":
            result.write(header)
        result.write(result_csv) 
    with open(output.split('.')[0]+"_filtered."+output.split('.')[1],'w') as result:
        if header!="":
            result.write(header)
        result.write(result_filtered_csv) 
    if cleanup:
        for i in glob.glob("job_*"):
            os.system("rm -r "+i)
    print("Results have been collected into: "+output)

def exclude_elements_json(input_json,exclude_elements):
    print("excluding materials containing elements not supported")
    flitered_json = []
    for i in tqdm(input_json,position=0, leave=True,bar_format='{desc:<5.5}{percentage:3.0f}%|{bar:15}{r_bar}'):
        ori = Structure.from_str(i['cif'],fmt="cif")
        species=[str(j) for j in ori.species]
        flag=0
        for j in species:
            if j in exclude_elements:
                flag+=1
                break
        if not flag and i["material_id"] != None:
            flitered_json.append(i)
    print(str(round((len(input_json)-len(flitered_json))/len(input_json)*100,1))+"% materials excluded")
    return flitered_json

def search_materials(apikeyPath,**search_params):
    def grouper(iterable, n, fillvalue=None):
        args = [iter(iterable)] * n
        return zip_longest(fillvalue=fillvalue, *args)
    config = configparser.ConfigParser()
    config.read(apikeyPath) #path of your .ini file
    apikey = config.get("Settings","API_KEY")
    with MPRester(api_key=apikey) as mpr:
        docs = mpr.summary.search(**search_params)
        oxide_mp_ids = [e.material_id for e in docs]
        data = []
        mpid_groups = [g for g in grouper(oxide_mp_ids, 1000)]
        for group in tqdm(mpid_groups,position=0, leave=True,bar_format='{desc:<5.5}{percentage:3.0f}%|{bar:15}{r_bar}'):
            # The last group may have fewer than 1000 actual ids,
            # so filter the `None`s out.
            temp=[]
            for i in group:
                if i != None:
                    temp.append(i)
            docs = mpr.summary.search(material_ids=temp, fields=["material_id", "structure"])
            data.extend(docs)
        dict_json = [{"material_id":str(e.material_id),"cif":e.structure.to(fmt="cif")} for e in data]
        return dict_json