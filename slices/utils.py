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
import configparser
from contextlib import redirect_stdout
import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
import math

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


def determine_bin_count(data_size, target_values):
    # 使用Sturges规则作为起点
    sturges_bins = math.ceil(math.log2(data_size)) + 1
    
    # 使用Freedman-Diaconis规则
    q75, q25 = np.percentile(target_values, [75, 25])
    iqr = q75 - q25
    bin_width = 2 * iqr * (len(target_values) ** (-1/3))
    fd_bins = math.ceil((max(target_values) - min(target_values)) / bin_width)
    
    # 使用Scott规则
    scott_bins = math.ceil((max(target_values) - min(target_values)) / (3.5 * np.std(target_values) * (len(target_values) ** (-1/3))))
    
    # 取这些方法的平均值，并确保bin数量在合理范围内
    avg_bins = int(np.mean([sturges_bins, fd_bins, scott_bins]))
    return max(min(avg_bins, data_size // 20), 5)  # 至少5个bin，最多数据点数的1/20

def adaptive_dynamic_binning(data, target_column, test_size=0.2, random_state=42):
    # 将目标列转换为数值型，设置errors='coerce'将非数值转换为NaN
    data[target_column] = pd.to_numeric(data[target_column], errors='coerce')
    
    # 删除目标列中的NaN值
    data_cleaned = data.dropna(subset=[target_column])
    
    print(f"\n原始数据行数: {len(data)}")
    print(f"清理后数据行数: {len(data_cleaned)}")
    
    # 自动确定bin数量
    target_values = data_cleaned[target_column].values
    n_bins = determine_bin_count(len(data_cleaned), target_values)
    print(f"自动确定的bin数量: {n_bins}")
    
    # 使用分位数法创建bins
    percentiles = [i * 100 / n_bins for i in range(n_bins + 1)]
    bins = list(data_cleaned[target_column].quantile([p/100 for p in percentiles]))
    
    # 确保bin边界是唯一的
    bins = sorted(set(bins))
    
    # 为bins添加标签
    labels = [f'Bin{i+1}' for i in range(len(bins) - 1)]
    
    # 将目标值分成区间
    data_cleaned['bin'] = pd.cut(data_cleaned[target_column], bins=bins, labels=labels, include_lowest=True)
    
    train_data = pd.DataFrame(columns=data_cleaned.columns)
    test_data = pd.DataFrame(columns=data_cleaned.columns)
    
    # 对每个区间进行分层抽样
    for bin_label in data_cleaned['bin'].unique():
        bin_data = data_cleaned[data_cleaned['bin'] == bin_label]
        if len(bin_data) > 1:
            bin_train, bin_test = train_test_split(bin_data, test_size=test_size, random_state=random_state)
        else:
            bin_train, bin_test = bin_data, pd.DataFrame()
        
        train_data = pd.concat([train_data, bin_train])
        test_data = pd.concat([test_data, bin_test])
    
    # 删除临时的'bin'列
    train_data = train_data.drop('bin', axis=1)
    test_data = test_data.drop('bin', axis=1)
    
    # 打乱数据
    train_data = train_data.sample(frac=1, random_state=random_state).reset_index(drop=True)
    test_data = test_data.sample(frac=1, random_state=random_state).reset_index(drop=True)
    
    return train_data, test_data, bins