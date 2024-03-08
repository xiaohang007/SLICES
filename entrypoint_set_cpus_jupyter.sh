#!/bin/bash

cp /crystal/slurm.conf /etc/slurm-llnl/
service munge restart
service slurmctld restart
service slurmd restart
sinfo

jupyter notebook --allow-root --ip 0.0.0.0 --port=8888 --NotebookApp.iopub_data_rate_limit=1.0e10
