#!/bin/bash

cp /crystal/slurm.conf /etc/slurm-llnl/
service munge restart
service slurmctld restart
service slurmd restart
sinfo

/bin/bash
