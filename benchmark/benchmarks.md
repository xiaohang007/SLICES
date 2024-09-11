## Reproduction of benchmarks
Reproduction of benchmarks and inverse design case study using a docker image [as an example]. One can run these calculaitons without the docker environment but one need to edit the *.pbs files to make sure the job management system on your PC/HPC work.
### General setup
Download this repo and unzipped it.

Put Materials Project's new API key in "APIKEY.ini". 

Edit "CPUs" in "slurm.conf" to set up the number of CPU threads available for the docker container.

```bash
docker pull xiaohang07/slices:v9   # Download SLICES_docker with pre-installed SLICES and other relevant packages. 
# Make entrypoint_set_cpus.sh executable 
sudo chmod +x entrypoint_set_cpus.sh
# Repalce "[]" with the absolute path of this repo's unzipped folder to setup share folder for the docker container.
docker run  -it --privileged=true -h workq --gpus all --shm-size=0.1gb  -v /[]:/crystal -w /crystal xiaohang07/slices:v9 /crystal/entrypoint_set_cpus.sh
```

### Reconstruction benchmark for MP-20
Convert MP-20 dataset to json (cdvae/data/mp_20 at main · txie-93/cdvae. GitHub. https://github.com/txie-93/cdvae (accessed 2023-03-12))

```bash
cd /crystal/benchmark/Match_rate_MP-20/get_json/0_get_mp20_json
python 0_mp20.py
```

Rule out unsupported elements
```bash
cd /crystal/benchmark/Match_rate_MP-20/get_json/1_element_filter
python 1_splitRun.py
#After running python 1_splitRun.py, the computation is only submitted to the queue, 
# not completed. To monitor the progress of the computation, use the qstat command. 
#If all tasks are marked with a status of "C", it indicates that the computation has finished.
python 2_collect.py
#After the computation are finished, running python 2_collect.py to get results.
```

Convert to primitive cell
```bash
cd /crystal/benchmark/Match_rate_MP-20/get_json/2_primitive_cell_conversion
python 1_splitRun.py
#After running python 1_splitRun.py, the computation is only submitted to the queue, 
# not completed. To monitor the progress of the computation, use the qstat command. 
#If all tasks are marked with a status of "C", it indicates that the computation has finished.
python 2_collect.py
#After the computation are finished, running python 2_collect.py to get results.
```

Rule out crystals with low-dimensional units (e.g. molecular crystals or layered crystals)
```bash
cd /crystal/benchmark/Match_rate_MP-20/get_json/3_3d_filter
python 1_splitRun.py
#After running python 1_splitRun.py, the computation is only submitted to the queue, 
# not completed. To monitor the progress of the computation, use the qstat command. 
#If all tasks are marked with a status of "C", it indicates that the computation has finished.
python 2_collect.py
#After the computation are finished, running python 2_collect.py to get results.
```
Calculate reconstruction rate of IAP-refined structures, ZL*-optimized structures, rescaled structures under strict and coarse setting. 
```bash
cd /crystal/benchmark/Match_rate_MP-20/matchcheck3
python 1_ini.py
#After running python 1_ini.py, the computation is only submitted to the queue, 
# not completed. To monitor the progress of the computation, use the qstat command. 
#If all tasks are marked with a status of "C", it indicates that the computation has finished.
python 2_collect_grid_new.py
#After the computation are finished, running python 2_collect_grid_new.py to get "results_collection_matchcheck3.csv"
```
Calculate reconstruction rate of IAP-refined structures, ZL*-optimized structures, IAP-refined rescaled structures, rescaled structures under strict and coarse setting. 
```bash
cd /crystal/benchmark/Match_rate_MP-20/matchcheck4
python 1_ini.py
#After running python 1_ini.py, the computation is only submitted to the queue, 
# not completed. To monitor the progress of the computation, use the qstat command. 
#If all tasks are marked with a status of "C", it indicates that the computation has finished.
python 2_collect_grid_new.py
#After the computation are finished, running python 2_collect_grid_new.py to get "results_collection_matchcheck4.csv"
```
**Reproduction of Table 1:** the table below illustrates the correspondence between the data in "results_collection_matchcheck4.csv" and the match rates of SLI2Cry for the filtered MP-20 dataset (40,330 crystals) presented in Table 1.
<font size="2">
| Setting         | Rescaled Structure | 𝑍𝐿∗-Optimized Structure | IAP-Refined Structure | IAP-Refined Rescaled Structure |
|-----------------|-----------------|-----------------------|---------------------|---------------------------|
| Strict  | std_match_sum      | opt_match_sum         | opt2_match_sum      | std2_match_sum            |
| Loose   | std_match2_sum     | opt_match2_sum        | opt2_match2_sum     | std2_match2_sum           |
</font>

**Reproduction of Table 2:** the match rate of SLI2Cry for the MP-20 dataset (45,229 crystals) = opt2_match2_sum\*40330/45229. 
### Reconstruction benchmark for MP-21-40
Download entries to build the filtered MP-21-40 dataset
```bash
cd /crystal/benchmark/Match_rate_MP-21-40/0_get_json_mp_api
python 0_mp21-40_dataset.py
!!! If “mp_api.client.core.client.MPRestError: REST query returned with error status code” occurs. The solution is:
pip install -U mp-api
```
Rule out crystals with low-dimensional units (e.g. molecular crystals or layered crystals) in general dataset
```bash
cd /crystal/benchmark/Match_rate_MP-21-40/0_get_json_mp_api/1_filter_prior_3d
python 1_splitRun.py
#After running python 1_splitRun.py, the computation is only submitted to the queue, 
# not completed. To monitor the progress of the computation, use the qstat command. 
#If all tasks are marked with a status of "C", it indicates that the computation has finished.
python 2_collect.py
#After the computation are finished, running python 2_collect.py to get results.
```
Calculate reconstruction rate of IAP-refined structures, ZL*-optimized structures, rescaled structures under strict and coarse setting.
```bash
cd /crystal/benchmark/Match_rate_MP-21-40/matchcheck3
python 1_ini.py
#After running python 1_ini.py, the computation is only submitted to the queue, 
# not completed. To monitor the progress of the computation, use the qstat command. 
#If all tasks are marked with a status of "C", it indicates that the computation has finished.
python 2_collect_grid_new.py
#After the computation are finished, running python 2_collect_grid_new.py to get results.
```
**Reproduction of Table S1:** the table below illustrates the correspondence between the data in "results_collection_matchcheck3.csv" and the match rates of SLI2Cry for the filtered MP-21-40 dataset (23,560 crystals) presented in Table S1.
<font size="2">
| Setting         | Filtered MP-21-40 |
|-----------------|-----------------|
| Strict  | opt2_match_sum      | 
| Loose   | opt2_match2_sum     | 
</font>

#### Reconstruction benchmark for QMOF-21-40
Extract MOFs with 21-40 atoms per unit cells in QMOF database to build the QMOF-21-40 dataset ( Figshare: https://figshare.com/articles/dataset/QMOF_Database/13147324 Version 14)
```bash
cd /crystal/benchmark/Match_rate_QMOF-21-40/get_json/0_get_mof_json
python get_json.py
```

Rule out unsupported elements
```bash
cd /crystal/benchmark/Match_rate_QMOF-21-40/get_json/1_element_filter
python 1_splitRun.py
#After running python 1_splitRun.py, the computation is only submitted to the queue, 
# not completed. To monitor the progress of the computation, use the qstat command. 
#If all tasks are marked with a status of "C", it indicates that the computation has finished.
python 2_collect.py
#After the computation are finished, running python 2_collect.py to get results.
```

Convert to primitive cell
```bash
cd /crystal/benchmark/Match_rate_QMOF-21-40/get_json/2_primitive_cell_conversion
python 1_splitRun.py
#After running python 1_splitRun.py, the computation is only submitted to the queue, 
# not completed. To monitor the progress of the computation, use the qstat command. 
#If all tasks are marked with a status of "C", it indicates that the computation has finished.
python 2_collect.py
#After the computation are finished, running python 2_collect.py to get results.
```

Rule out crystals with low-dimensional units (e.g. molecular crystals or layered crystals)
```bash
cd /crystal/benchmark/Match_rate_QMOF-21-40/get_json/3_3d_filter
python 1_splitRun.py
#After running python 1_splitRun.py, the computation is only submitted to the queue, 
# not completed. To monitor the progress of the computation, use the qstat command. 
#If all tasks are marked with a status of "C", it indicates that the computation has finished.
python 2_collect.py
#After the computation are finished, running python 2_collect.py to get results.
```
Calculate reconstruction rate of IAP-refined structures, ZL*-optimized structures, rescaled structures under strict and coarse setting. 
```bash
cd /crystal/benchmark/Match_rate_QMOF-21-40/matchcheck3
python 1_ini.py
#After running python 1_ini.py, the computation is only submitted to the queue, 
# not completed. To monitor the progress of the computation, use the qstat command. 
#If all tasks are marked with a status of "C", it indicates that the computation has finished.
python 2_collect_grid_new.py
#After the computation are finished, running python 2_collect_grid_new.py to get results.
```
**Reproduction of Table S1:** the table below illustrates the correspondence between the data in "results_collection_matchcheck3.csv" and the match rates of SLI2Cry for the filtered QMOF-21-40 dataset (339 MOFs) presented in Table S1.
<font size="2">
| Setting         | Filtered QMOF-21-40  |
|-----------------|-----------------|
| Strict  | opt2_match_sum      | 
| Loose   | opt2_match2_sum     | 
</font>

### Material generation benchmark
Convert MP-20 dataset to json (cdvae/data/mp_20 at main · txie-93/cdvae. GitHub. https://github.com/txie-93/cdvae (accessed 2023-03-12))

```bash
cd /crystal/benchmark/Validity_rate_ucRNN__Success_rate_cRNN/0_get_json/0_get_mp20_json
python 0_mp20.py
```

Rule out unsupported elements
```bash
cd /crystal/benchmark/Validity_rate_ucRNN__Success_rate_cRNN/0_get_json/1_element_filter
python 1_splitRun.py
#After running python 1_splitRun.py, the computation is only submitted to the queue, 
# not completed. To monitor the progress of the computation, use the qstat command. 
#If all tasks are marked with a status of "C", it indicates that the computation has finished.
python 2_collect.py
#After the computation are finished, running python 2_collect.py to get results.
```

Convert to primitive cell
```bash
cd /crystal/benchmark/Validity_rate_ucRNN__Success_rate_cRNN/0_get_json/2_primitive_cell_conversion
python 1_splitRun.py
#After running python 1_splitRun.py, the computation is only submitted to the queue, 
# not completed. To monitor the progress of the computation, use the qstat command. 
#If all tasks are marked with a status of "C", it indicates that the computation has finished.
python 2_collect.py
#After the computation are finished, running python 2_collect.py to get results.
```

Rule out crystals with low-dimensional units (e.g. molecular crystals or layered crystals)
```bash
cd /crystal/benchmark/Validity_rate_ucRNN__Success_rate_cRNN/0_get_json/3_3d_filter
python 1_splitRun.py
#After running python 1_splitRun.py, the computation is only submitted to the queue, 
# not completed. To monitor the progress of the computation, use the qstat command. 
#If all tasks are marked with a status of "C", it indicates that the computation has finished.
python 2_collect.py
#After the computation are finished, running python 2_collect.py to get results.
```

Convert crystal structures in datasets to SLICES strings and conduct data augmentation
```bash
cd /crystal/benchmark/Validity_rate_ucRNN__Success_rate_cRNN/1_unconditioned_RNN/1_augmentation
python 1_splitRun.py
#After running python 1_splitRun.py, the computation is only submitted to the queue, 
# not completed. To monitor the progress of the computation, use the qstat command. 
#If all tasks are marked with a status of "C", it indicates that the computation has finished.
python 2_collect.py
#After the computation are finished, running python 2_collect.py to get results.
```
Train unconditional RNN; sample 10000 SLICES strings
```bash
cd /crystal/benchmark/Validity_rate_ucRNN__Success_rate_cRNN/1_unconditioned_RNN/2_train_sample
sh 0_train_prior_model.sh
```
Modify ./workflow/2_sample_HTL_model_100x.py to define the number of SLICES to be sampled 
```bash
sh 1_sample_in_parallel.sh
#After running sh 1_sample_in_parallel.sh, the computation is only submitted to the queue, 
# not completed. To monitor the progress of the computation, use the qstat command. 
#If all tasks are marked with a status of "C", it indicates that the computation has finished.
python 2_collect_clean_glob_details.py
#After the computation are finished, running python 2_collect_clean_glob_details.py to get results.
```
Removing duplicate edges in SLICES strings to fix the syntax error
```bash
cd /crystal/benchmark/Validity_rate_ucRNN__Success_rate_cRNN/1_unconditioned_RNN/3_fix_syntax_check
python 1_splitRun.py
#After running python 1_splitRun.py, the computation is only submitted to the queue, 
# not completed. To monitor the progress of the computation, use the qstat command. 
#If all tasks are marked with a status of "C", it indicates that the computation has finished.
python 2_collect_clean_glob_details.py
#After the computation are finished, running python 2_collect_clean_glob_details.py to get results.
```

Reconstruct crystal structures from SLICES strings and calculate the number of reconstructed crystals (num_reconstructed)
```bash
cd /crystal/benchmark/Validity_rate_ucRNN__Success_rate_cRNN/1_unconditioned_RNN/4_inverse
python 1_splitRun.py
#After running python 1_splitRun.py, the computation is only submitted to the queue, 
# not completed. To monitor the progress of the computation, use the qstat command. 
#If all tasks are marked with a status of "C", it indicates that the computation has finished.
python 2_collect_clean_glob_details.py
#After the computation are finished, running python 2_collect_clean_glob_details.py to get results.
!!! In order to address the potential memory leaks associated with M3GNet, we implemented a strategy of 
restarting the Python script at regular intervals, with a batch size of 30.
python count.py #calculate the number of reconstructed crystals (num_reconstructed)
```

Evaluate the compositional validity of the reconstructed crystals and calculate the number of compositionally valid reconstructed crystals (num_comp_valid)
```bash
cd /crystal/benchmark/Validity_rate_ucRNN__Success_rate_cRNN/1_unconditioned_RNN/5_check_comp_valid
python 1_splitRun.py
#After running python 1_splitRun.py, the computation is only submitted to the queue, 
# not completed. To monitor the progress of the computation, use the qstat command. 
#If all tasks are marked with a status of "C", it indicates that the computation has finished.
python 2_collect_clean_glob_details.py
#After the computation are finished, running python 2_collect_clean_glob_details.py to get results.
python count.py # calculate the number of compositionally valid reconstructed crystals (num_comp_valid)
```

Evaluate the structural validity of the reconstructed crystals and calculate the number of structurally valid reconstructed crystals (num_struc_valid)
```bash
cd /crystal/benchmark/Validity_rate_ucRNN__Success_rate_cRNN/1_unconditioned_RNN/6_check_struc_validity
python 1_splitRun.py
#After running python 1_splitRun.py, the computation is only submitted to the queue, 
# not completed. To monitor the progress of the computation, use the qstat command. 
#If all tasks are marked with a status of "C", it indicates that the computation has finished.
python 2_collect_clean_glob_details.py
#After the computation are finished, running python 2_collect_clean_glob_details.py to get results.
python count.py # calculate the number of compositionally valid reconstructed crystals (num_struc_valid)
```
**Reproduction of Table 3:** 
Structural validity (%) = num_struc_valid/num_reconstructed\*100
Compositional validity (%) = num_comp_valid/num_reconstructed\*100


### Property optimization benchmark
(1) Convert crystal structures in datasets to SLICES strings and conduct data augmentation
```bash
cd /crystal/benchmark/Validity_rate_ucRNN__Success_rate_cRNN/2_conditioned_RNN/1_augmentation
python 1_splitRun.py # wait for jobs to finish (using qstat to check)
python 2_collect.py
```
(2) Train conditional RNN
```bash
cd /crystal/benchmark/Validity_rate_ucRNN__Success_rate_cRNN/2_conditioned_RNN/2_train_sample
sh 0_train_prior_model.sh
```
(3) Sample 1000 SLICES strings with $E_{form}$ target = -4.5 eV/atom
Modify ./settings.ini to define the $E_{form}$ target and the number of SLICES to be sampled 
```bash
sh 1_sample_in_parallel.sh # wait for jobs to finish (using qstat to check)
python 2_collect_clean_glob_details.py
```
(4) Removing duplicate edges in SLICES strings to fix the syntax error
```bash
cd /crystal/benchmark/Validity_rate_ucRNN__Success_rate_cRNN/2_conditioned_RNN/3_fix_syntax_check
python 1_splitRun.py # wait for jobs to finish (using qstat to check)
python 2_collect_clean_glob_details.py
```

(5) Reconstruct crystal structures from SLICES strings and calculate the number of reconstructed crystals (num_reconstructed)
```bash
cd /crystal/benchmark/Validity_rate_ucRNN__Success_rate_cRNN/2_conditioned_RNN/4_inverse
python 1_splitRun.py # wait for jobs to finish (using qstat to check)
python 2_collect_clean_glob_details.py
!!! In order to address the potential memory leaks associated with M3GNet, we implemented a strategy of 
restarting the Python script at regular intervals, with a batch size of 30.
python count.py #calculate the number of reconstructed crystals (num_reconstructed)
```

(6) Evaluate the formation energy distribution of the reconstructed crystals with the M3GNet model
```bash
cd /crystal/benchmark/Validity_rate_ucRNN__Success_rate_cRNN/2_conditioned_RNN/5_eform_m3gnet
python 1_splitRun.py # wait for jobs to finish (using qstat to check)
python 2_collect_clean_glob_details.py
python 3_normal_distri_plot.py # plot the formation energy distribution (M3GNet) of the reconstructed crystals 
```

(7) Evaluate the formation energy distribution of the reconstructed crystals at PBE level (took less than 1 day to finish with 36*26 cores HPC; need to tweak the ./workflow/0_EnthalpyOfFormation\*.py to deal with some tricky cases of VASP cell optimization)
```bash
cd /crystal/benchmark/Validity_rate_ucRNN__Success_rate_cRNN/2_conditioned_RNN/6_eform_PBE
python 1_splitRun.py # wait for jobs to finish (using qstat to check)
python 2_collect_clean_glob_details.py
python 3_normal_distri_plot.py # plot the formation energy distribution (PBE) of the reconstructed crystals 
```

(8) **Reproduction of Table 3:**  Calculate SR5, SR10, SR15 in Table S1 using formation energies (at PBE level) of crystals generated with a target of -4.5 eV/atom
```bash
cd /crystal/benchmark/Validity_rate_ucRNN__Success_rate_cRNN/2_conditioned_RNN/7_calculate_FigureS2c
python calculate_SR5-10-15_TableS1.py # SR5, SR10, SR15 are printed in the terminal
```

(9) **Reproduction of Fig. S2c:** Repeat step (3-6) with $E_{form}$ target = -3.0, -4.0, -5.0, -6.0 eV/atom. Extract formation energy distributions from "results_5_eform_m3gnet.csv" in step (6) and transfer the data to "/crystal/benchmark/Validity_rate_ucRNN__Success_rate_cRNN/2_conditioned_RNN/7_calculate_FigureS2c/energy_formation_m3gnet_lists.csv". Then:
```bash
cd /crystal/benchmark/Validity_rate_ucRNN__Success_rate_cRNN/2_conditioned_RNN/7_calculate_FigureS2c
python plot_FigureS1c.py # get Fig. S2c as test3.svg
```
The formation energy distributions with $E_{form}$ target = -3.0, -4.0, -5.0, -6.0 eV/atom can be accessed from "/crystal/benchmark/Validity_rate_ucRNN__Success_rate_cRNN/2_conditioned_RNN/Other_targets/3_eform_\*".