# -*- coding: utf-8 -*-
# Hang Xiao 2023.04
# xiaohang07@live.cn

import os
import json
import argparse
import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from slices.utils import *
# Removed: from slices.utils import adaptive_dynamic_binning

def main(args):
    # Load the raw dataset

    if not os.path.exists(args.raw_data_path):
        raise FileNotFoundError(f"The raw data file '{args.raw_data_path}' does not exist.")
    
    data = pd.read_csv(args.raw_data_path)
    print(f"Loaded raw data from {args.raw_data_path} with {len(data)} entries.")

    # Validate column indices
    num_columns = data.shape[1]
    
    if args.cif_column_index >= num_columns or args.cif_column_index < 0:
        raise IndexError(f"cif_column_index {args.cif_column_index} is out of bounds for data with {num_columns} columns.")
    
    for idx in args.prop_column_index_list:
        if idx >= num_columns or idx < 0:
            raise IndexError(f"Property column index {idx} is out of bounds for data with {num_columns} columns.")
    
    if args.mat_id_column_index is not None:
        if args.mat_id_column_index >= num_columns or args.mat_id_column_index < 0:
            raise IndexError(f"mat_id_column_index {args.mat_id_column_index} is out of bounds for data with {num_columns} columns.")
    
    # Extract columns based on provided indices
    cifs = list(data.iloc[:, args.cif_column_index])
    #print("cifs",cifs)
    if args.mat_id_column_index is not None:
        mat_ids = list(data.iloc[:, args.mat_id_column_index])
    # Else: Do not extract mat_ids

    # Extract target properties based on target_indices
    target_columns = [data.columns[i] for i in args.prop_column_index_list]
    targets = [list(data[col]) for col in target_columns]

    # Prepare the output JSON structure
    output = []
    num_entries = len(data)
    for i in range(num_entries):
        entry = {}
        if args.mat_id_column_index is not None:
            entry["material_id"] = mat_ids[i]
        # Else: Do not include "material_id" in the JSON

        entry["cif"] = cifs[i]
        for j, col in enumerate(target_columns):
            entry[col] = targets[j][i]
        output.append(entry)

    # Write the temporary JSON file
    temp_json_path = 'temp_cifs.json'
    with open(temp_json_path, 'w') as f:
        json.dump(output, f)
    print(f"Constructed temporary JSON file '{temp_json_path}' with {len(output)} entries.")

    # Run splitRun with specified threads
    splitRun(filename=temp_json_path, threads=args.threads, skip_header=False)
    show_progress()
    print("Completed splitRun_local and show_progress_local.")

    # Collect JSON outputs
    collect_json(output=args.structure_json_for_novelty_check, 
                glob_target="./**/output.json", cleanup=False)
    print(f"Collected JSON outputs into '{args.structure_json_for_novelty_check}'.")

    # Collect CSV outputs
    collect_csv(output=args.complete_train_set_name, 
               glob_target="./**/result.csv", cleanup=True, 
               header="SLICES," + ",".join(target_columns) + "\n")
    print(f"Collected CSV outputs into '{args.complete_train_set_name}'.")

    # Remove the temporary JSON file
    os.remove(temp_json_path)
    print(f"Removed temporary JSON file '{temp_json_path}'.")

    # Read the collected CSV data
    data_collected = pd.read_csv(args.complete_train_set_name)
    print(f"Loaded collected data from '{args.complete_train_set_name}' with {len(data_collected)} entries.")
    
    if data_collected.empty:
        raise ValueError("The collected training data is empty. Please check the data collection steps.")

    target_column = data_collected.columns[-1]  # Assuming last column is the target
    print(f"Identified target column as '{target_column}'.")

    # Perform standard train-validation split
    train_data, val_data = train_test_split(
        data_collected,
        test_size=args.validation_size,
        random_state=args.random_seed,
        shuffle=True
    )
    print(f"Performed train-validation split with validation size {args.validation_size}.")

    # Derive train and validation file names
    train_file = args.complete_train_set_name.replace('.csv', '_train.csv')
    val_file = args.complete_train_set_name.replace('.csv', '_val.csv')

    # Save the split data
    train_data.to_csv(train_file, index=False)
    val_data.to_csv(val_file, index=False)
    print(f"Saved training data to '{train_file}' and validation data to '{val_file}'.")

    # Optionally, rename to desired output names
    if args.train_output_name:
        os.rename(train_file, args.train_output_name)
        print(f"Renamed training file to '{args.train_output_name}'.")
    if args.val_output_name:
        os.rename(val_file, args.val_output_name)
        print(f"Renamed validation file to '{args.val_output_name}'.")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Process and split MP20 dataset with configurable parameters.")

    parser.add_argument(
        "--raw_data_path",
        type=str,
        default="./",
        help="Path to the raw data directory containing 'raw_mp20_dataset.csv'."
    )

    parser.add_argument(
        "--cif_column_index",
        type=int,
        default=0,
        help="Index of the 'cif' column in the raw CSV."
    )

    parser.add_argument(
        "--prop_column_index_list",
        type=int,
        nargs='+',
        default=[2, 3],
        help="Indices of the target property columns (e.g., formation_energy_per_atom, band_gap)."
    )

    parser.add_argument(
        "--mat_id_column_index",
        type=int,
        default=None,
        help="Index of the 'material_id' column in the raw CSV. If not provided, 'material_id' will be excluded from the JSON output."
    )

    parser.add_argument(
        "--structure_json_for_novelty_check",
        type=str,
        default="cifs_filtered.json",
        help="Output JSON file for structure novelty check."
    )

    parser.add_argument(
        "--complete_train_set_name",
        type=str,
        default="complete_dataset.csv",
        help="Name of the complete training set CSV file."
    )

    parser.add_argument(
        "--threads",
        type=int,
        default=8,
        help="Number of threads to use for processing."
    )

    parser.add_argument(
        "--train_output_name",
        type=str,
        default="train_data.csv",
        help="Optional name for the training split CSV file."
    )

    parser.add_argument(
        "--val_output_name",
        type=str,
        default="val_data.csv",
        help="Optional name for the validation split CSV file."
    )

    # New arguments for train-validation split
    parser.add_argument(
        "--validation_size",
        type=float,
        default=0.2,
        help="Proportion of the dataset to include in the validation split (e.g., 0.2 for 20%)."
    )

    parser.add_argument(
        "--random_seed",
        type=int,
        default=42,
        help="Random seed for reproducibility of the train-validation split."
    )

    args = parser.parse_args()
    main(args)
