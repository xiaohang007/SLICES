#!/bin/bash
#SBATCH --nodes=1                     # 请求1个节点
#SBATCH --ntasks-per-node=1           # 每个节点上的任务数为1
#SBATCH --job-name=Nb3I8L1_vasp       # 作业名称（避免使用路径）
#SBATCH --output=job.out              # 标准输出文件
#SBATCH --error=job.err               # 标准错误文件
#SBATCH --time=48000:00:00            # 最长运行时间 (格式: [days-]hours:minutes:seconds)

echo "Begin time: $(date)"

export OMP_NUM_THREADS=1
export PYTHONDONTWRITEBYTECODE=1
# 执行Python脚本
# SLICES_PYTHON: the interpreter that submitted the job (set by slices.utils), so the job
# uses the same environment even if it is not on PATH
"${SLICES_PYTHON:-python}" 0_run.py

echo "End time: $(date)"
