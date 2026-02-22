# -*- coding: utf-8 -*-
import os
import math
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
            env=env,
        )
        return result.stdout
    except subprocess.TimeoutExpired:
        print("Script timed out")
    except subprocess.CalledProcessError as e:
        print(f"Script failed with error: {e.stderr}")


def split_list(a, n):
    k, m = divmod(len(a), n)
    return (a[i * k + min(i, m):(i + 1) * k + min(i + 1, m)] for i in range(n))


os.system("rm -f result2.csv")

with open('temp.csv', 'r') as f:
    lines = f.readlines()

if not lines:
    print("temp.csv is empty, nothing to do.")
else:
    batch_size = 10
    lines_split = list(split_list(lines, math.ceil(len(lines) / batch_size)))
    for chunk in lines_split:
        with open('temp_splited.csv', 'w') as f:
            f.writelines(chunk)
        run_script(batch_size * 120)

if os.path.exists("result2.csv"):
    os.system("mv result2.csv result.csv")
