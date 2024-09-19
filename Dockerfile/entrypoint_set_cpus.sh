#!/bin/bash
source /opt/miniconda/bin/activate
conda activate umat
export LD_LIBRARY_PATH=/usr/local/nvidia/lib:/usr/local/nvidia/lib64:$LD_LIBRARY_PATH
export PATH=/opt/conda/bin:/opt/conda/envs/umat/bin:$PATH
cp /crystal/slurm.conf /etc/slurm-llnl/
service munge restart
service slurmctld restart
service slurmd restart
sinfo
/bin/bash
