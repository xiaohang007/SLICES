#!/bin/bash
export LD_LIBRARY_PATH=/usr/local/nvidia/lib:/usr/local/nvidia/lib64:$LD_LIBRARY_PATH
export PATH=/opt/conda/bin:/opt/conda/envs/chgnet/bin:$PATH
cp /crystal/slurm.conf /etc/slurm-llnl/
service munge restart
service slurmctld restart
service slurmd restart
sinfo

jupyter notebook --allow-root --ip 0.0.0.0 --port=8888 --NotebookApp.iopub_data_rate_limit=1.0e10
