Scripts to reproduce results of benchmark and inverse design case study. 
Developed by Hang Xiao 2023.04 xiaohang007@gmail.com https://github.com/xiaohang007
All results are available in the link mentioned in the paper: "SLICES: An Invertible, Symmetry-invariant, and String-based Crystallographic Representation".

How to use:

1. General_setup:
Put Materials Project's new API key in APIKEY.ini. 
Edit "CPUs" in slurm.conf to set up the number of CPU threads available for the docker container.
Put this folder's absolute path in "[]" in rundocker.sh to setup share folder for the docker container.
# run the docker container with pre-installed SLICES and other relevant packages, allowing easy installation and reproduction of the reconstruction benchmark and inverse design case study 
sh rundocker.sh

2. Benchmark:
# convert MP-20 dataset to json (cdvae/data/mp_20 at main · txie-93/cdvae. GitHub. https://github.com/txie-93/cdvae (accessed 2023-03-12))
cd /crystal/benchmark/0_get_mp20_json
python 0_mp20.py

# rule out unsupported elements
cd /crystal/benchmark/1_element_filter
python 1_splitRun.py
# wait for jobs to finish (using qstat to check)
python 2_collect.py

# convert to primitive cell
cd /crystal/benchmark/2_primitive_cell_conversion
python 1_splitRun.py
# wait for jobs to finish (using qstat to check)
python 2_collect.py

# rule out crystals with low-dimensional units (e.g. molecular crystals or layered crystals)
cd /crystal/benchmark/3_3d_filter
python 1_splitRun.py
# wait for jobs to finish (using qstat to check)
python 2_collect.py

# calculate reconstruction rate of IAP-refined structures, L-optimized structures, rescaled structures under strict or coarse setting. 
cd /crystal/benchmark/matchcheck3
python 1_ini.py
# wait for jobs to finish (using qstat to check)
python 2_collect_grid_new.py

# calculate reconstruction rate of IAP-refined structures, L-optimized structures, IAP-refined rescaled structures, rescaled structures under strict or coarse setting. 
cd /crystal/benchmark/matchcheck4
python 1_ini.py
# wait for jobs to finish (using qstat to check)
python 2_collect_grid_new.py

3. Inverse design of direct narrow-gap semiconductors for optical applications
# download entries to build general and transfer datasets
cd /crystal/HTS/0_get_json_mp_api
python 0_prior_model_dataset.py
python 1_transfer_learning_dataset.py

# rule out crystals with low-dimensional units (e.g. molecular crystals or layered crystals) in general dataset
cd /crystal/HTS/0_get_json_mp_api/2_filter_prior_3d
python 1_splitRun.py
# wait for jobs to finish (using qstat to check)
python 2_collect.py

# rule out crystals with low-dimensional units (e.g. molecular crystals or layered crystals) in transfer dataset
cd /crystal/HTS/0_get_json_mp_api/2_filter_transfer_3d
python 1_splitRun.py
# wait for jobs to finish (using qstat to check)
python 2_collect.py

# convert crystal structures in datasets to SLICES strings and conduct data augmentation
cd /crystal/HTS/1_augmentation/prior
python 1_splitRun.py
# wait for jobs to finish (using qstat to check)
python 2_collect.py
cd /crystal/HTS/1_augmentation/transfer
python 1_splitRun.py
# wait for jobs to finish (using qstat to check)
python 2_collect.py

# train general and specialized RNN; sample SLICES
cd /crystal/HTS/2_train_sample
sh 0_train_prior_model.sh
sh 1_transfer_learning.sh
# modify /crystal/HTS/2_train_sample/workflow/2_sample_HTL_model_100x.py to define the number of SLICES to be sampled 
sh 2_sample_in_parallel.sh 
# wait for jobs to finish (using qstat to check)
python 3_collect_clean_glob_details.py

# reconstruct crystal structures from SLICES strings
cd /crystal/HTS/3_inverse
python 1_splitRun.py
# wait for jobs to finish (using qstat to check)
python 2_collect_clean_glob_details.py

# filter out crystals with compositions that exist in the Materials Project database
cd /crystal/HTS/4_composition_filter
python 1_splitRun.py
# wait for jobs to finish (using qstat to check)
python 2_collect_clean_glob_details.py

# find high-symmetry structures in candidates with duplicate compositions
cd /crystal/HTS/5_symmetry_filter_refine
python 1_splitRun.py
# wait for jobs to finish (using qstat to check)
python 2_collect_clean_glob_details.py

# rule out crystals displaying minimum structural dissimilarity value < 0.75 (a dissimilarity threshold used in the Materials Project) with respect to the structures in the training dataset
cd /crystal/HTS/6_structure_dissimilarity_filter
cd ./0_save_structure_fingerprint
cp /crystal/HTS/0_get_json_mp_api/prior_model_dataset_filtered.json ./
python 1_splitRun.py
# wait for jobs to finish (using qstat to check)
python 2_collect_clean_glob_details.py
cd ../
python 1_splitRun.py
# wait for jobs to finish (using qstat to check)
python 2_collect_clean_glob_details.py

# rule out candidates with IAP-predicted energy above hull >= 50 meV/atom
cd /crystal/HTS/7_EAH_prescreen
# download relevant entries for high-throughput energy above hull calculation
sh 0_setup.sh
python 1_splitRun.py
# wait for jobs to finish (using qstat to check)
python 2_collect_clean_glob_details.py
python 3_filter.py

# rule out candidates with ALIGNN predicted band gap E_g < 0.1 eV (less likely to be a semiconductor) 
cd /crystal/HTS/8_band_gap_prescreen
python 1_splitRun.py
# wait for jobs to finish (using qstat to check)
python 2_collect_clean_glob_details.py

# !!! Note that VASP should be installed and POTCAR should be set up for pymatgen using "pmg config -p <EXTRACTED_VASP_POTCAR> <MY_PSP>" before performing this task. Because VASP is a commercial software, it is not installed in the docker image provided.
# perform geometry relaxation and band structure calculation at PBE level using VASP
cd /crystal/HTS/9_EAH_Band_gap_PBE
cp /crystal/HTS/7_EAH_prescreen/competitive_compositions.json.gz ./
python 1_splitRun.py
# wait for jobs to finish (using qstat to check)
python 2_collect_clean_glob_details.py
python 3_filter.py
# check results_7_EAH_prescreenfiltered_0.05eV.csv for details of promising candidates; check ./candidates for band structures

###
How to install invcryrep(SLICES) python package:
cd invcryrep
python setup.py install