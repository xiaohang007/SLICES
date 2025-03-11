# -*- coding: utf-8 -*-
# Hang Xiao 2023.04
# xiaohang07@live.cn
import os,sys,json,gc,math
import subprocess
def run_script(timeout_sec):
    env = os.environ.copy()
    env["OMP_NUM_THREADS"] = "1"
    env["MKL_NUM_THREADS"] = "1"
    env["OPENBLAS_NUM_THREADS"] = "1"
    try:
        result = subprocess.run(
            ["timeout", f"{timeout_sec}s", "python", "-B", "script.py"],
            capture_output=True,
            check=True,
            env=env
        )
        return result.stdout
    except subprocess.TimeoutExpired:
        print("Script timed out")
    except subprocess.CalledProcessError as e:
        print(f"Script failed with error: {e.stderr}")


def split_list(a, n):
    k, m = divmod(len(a), n)
    return (a[i*k+min(i, m):(i+1)*k+min(i+1, m)] for i in range(n))

os.system("rm result2.csv")  # to deal with slurm's twice execution bug
  # This loads the default pre-trained model

with open('temp.csv', 'r') as f:
    slices_list=f.readlines()

batch_size=10
slices_split=list(split_list(slices_list,math.ceil(len(slices_list)/batch_size)))
for i in range(len(slices_split)):
    with open('temp_splited.csv', 'w') as f:
        f.writelines(slices_split[i])
    run_script(batch_size * 120)
os.system("mv result2.csv result.csv")
