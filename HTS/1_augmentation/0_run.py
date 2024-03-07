# -*- coding: utf-8 -*-
# Hang Xiao 2023.04
# xiaohang07@live.cn
from invcryrep.utils import temporaryWorkingDirectory,splitRun,show_progress,collect_json,collect_csv
with temporaryWorkingDirectory("./prior"):
    splitRun(filename='../../0_get_json_mp_api/prior_model_dataset_filtered.json',threads=16,skip_header=False)
    show_progress()
    collect_csv(output="../prior_aug.sli", glob_target="./**/result.sli",header="",cleanup=True)
with temporaryWorkingDirectory("./transfer"):
    splitRun(filename='../../0_get_json_mp_api/transfer_model_dataset_filtered.json',threads=16,skip_header=False)
    show_progress()
    collect_csv(output="../transfer_aug.sli", glob_target="./**/result.sli",header="",cleanup=True)