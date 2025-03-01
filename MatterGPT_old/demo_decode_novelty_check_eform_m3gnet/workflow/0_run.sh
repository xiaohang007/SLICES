#!/bin/bash
#SBATCH --nodes=1                     # 请求1个节点
#SBATCH --ntasks-per-node=1           # 每个节点上的任务数为1
#SBATCH --job-name=Nb3I8L1_vasp       # 作业名称（避免使用路径）
#SBATCH --output=job.out              # 标准输出文件
#SBATCH --error=job.err               # 标准错误文件
#SBATCH --time=48000:00:00            # 最长运行时间 (格式: [days-]hours:minutes:seconds)

echo "Begin time: $(date)"

# 设置Conda环境路径并激活环境
export PATH=/opt/conda/env/chgnet/bin:$PATH
source activate chgnet
export OMP_NUM_THREADS=1
export TF_CPP_MIN_LOG_LEVEL=3
export PYTHONDONTWRITEBYTECODE=1
# 执行Python脚本
python 0_run.py

echo "End time: $(date)"
