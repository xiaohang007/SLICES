# -*- coding: utf-8 -*-
# Hang Xiao 2023.04
# xiaohang07@live.cn
import os,sys,json,gc,math
import subprocess




def split_list(a, n):
    k, m = divmod(len(a), n)
    return (a[i*k+min(i, m):(i+1)*k+min(i+1, m)] for i in range(n))

os.system("rm result2.csv")  # to deal with slurm's twice execution bug
  # This loads the default pre-trained model

with open('temp.csv', 'r') as f:
    slices_list = f.readlines()

batch_size=10  # SLICES per script.py process; each process loads the force fields once (~4 s)
header = None
if slices_list:
    first = slices_list[0].strip()
    if "SLICES" in first or "symSLICES" in first:
        header = slices_list[0]
        slices_list = slices_list[1:]

if not slices_list:
    open("result2.csv", "a").close()
else:
    slices_split=list(split_list(slices_list,math.ceil(len(slices_list)/batch_size)))
    for i in range(len(slices_split)):
        with open('temp_splited.csv', 'w') as f:
            if header:
                f.write(header)
            f.writelines(slices_split[i])
        subprocess.run(["timeout", f"{batch_size * 120}s", sys.executable, "-B", "script.py"])
os.system("mv result2.csv result.csv")
