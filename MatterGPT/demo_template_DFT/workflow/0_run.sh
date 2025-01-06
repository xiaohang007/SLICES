#!/bin/bash
#SBATCH --nodes=1
#SBATCH -n 8
#SBATCH --job-name=./structures/gcmc_Oremoved_opt.vasp
#SBATCH --output=job.out
#SBATCH --error=job.err



# Slurm doesn't use ulimit directly, but we can set some resource limits
# The unlimited stack size is typically already set in Slurm
# Memory limits are handled by the --mem option above (0 means use all available memory)
# Locked memory limit can be set with --mem-lock, but it's often not necessary
export MKL_DEBUG_CPU_TYPE=5
export MKL_CBWR=AVX2
export I_MPI_PIN_DOMAIN=numa
python 0_run.py
