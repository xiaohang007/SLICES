import shutil
import math
import os
import csv
import pickle
import json
import argparse
from collections import defaultdict
from pymatgen.core.structure import Structure
from slices.utils import *

def split_database(pkl_file="structure_database.pkl", shard_size=1000, output_dir="split_db"):
    """
    Split the structure_database.pkl into shards and save them to the specified directory.

    Parameters:
    - pkl_file (str): Path to the original pickle file.
    - shard_size (int): Number of structures per shard.
    - output_dir (str): Directory where shard files will be saved.
    """
    # If output_dir exists, delete its contents; otherwise, create the directory
    if os.path.exists(output_dir):
        print(f"Clearing existing directory: {output_dir}")
        shutil.rmtree(output_dir)
    os.makedirs(output_dir)
    print(f"Created new directory: {output_dir}")

    # Load the original structure database
    print(f"Loading structure data from {pkl_file}...")
    with open(pkl_file, 'rb') as f:
        structure_database = pickle.load(f)

    total_structures = len(structure_database)
    num_shards = math.ceil(total_structures / shard_size)
    print(f"Total structures: {total_structures}. Splitting into {num_shards} shards with up to {shard_size} structures each.")

    for shard_index in range(num_shards):
        start_idx = shard_index * shard_size
        end_idx = min(start_idx + shard_size, total_structures)
        shard_data = structure_database[start_idx:end_idx]
        shard_filename = os.path.join(output_dir, f"structure_database_shard_{shard_index + 1}.pkl")
        with open(shard_filename, 'wb') as shard_file:
            pickle.dump(shard_data, shard_file)
        print(f"Saved shard {shard_index + 1}/{num_shards} to {shard_filename}")
    
    print("All shards have been successfully saved.")

def load_and_save_structure_database(structure_json_path):
    """
    Load CIF data from a JSON file and save it as a pickle file.

    Parameters:
    - structure_json_path (str): Path to the JSON file containing CIF data.
    """
    with open(structure_json_path, 'r', encoding='utf-8') as f:
        cifs = json.load(f)

    structure_database = []
    for i, cif_entry in enumerate(cifs):
        cif_string = cif_entry.get("cif", "")
        if not cif_string:
            print(f"Missing CIF string in entry {i}. Skipping.")
            continue
        try:
            stru = Structure.from_str(cif_string, "cif")
            structure_database.append([stru, '1.0'])
        except Exception as e:
            print(f"Error parsing CIF in entry {i}: {e}")

    with open('structure_database.pkl', 'wb') as f:
        pickle.dump(structure_database, f)
    print(f"Saved {len(structure_database)} structures to structure_database.pkl")

def build_database_by_comp(pkl_file="structure_database.pkl"):
    """
    Load structures from a pickle file and build a composition index.

    Parameters:
    - pkl_file (str): Path to the pickle file containing structures.

    Returns:
    - defaultdict: A dictionary with reduced composition strings as keys and lists of structures as values.
    """
    with open(pkl_file, 'rb') as f:
        structure_database = pickle.load(f)

    database_by_comp = defaultdict(list)
    for entry in structure_database:
        struct = entry[0]
        reduced_comp = struct.composition.reduced_composition
        comp_str = str(reduced_comp)
        database_by_comp[comp_str].append(struct)

    return database_by_comp

def prepare(args):
    """
    Prepare the structure database and split it into shards.

    Parameters:
    - args: Parsed command-line arguments.
    """
    # 1) Generate structure_database.pkl
    #    (Uncomment if you need to regenerate the pickle file)
    load_and_save_structure_database(args.structure_json_for_novelty_check)
    
    # 2) Split the database into shards
    print("Starting to split structure_database.pkl into shards...")
    split_database(pkl_file="structure_database.pkl", shard_size=10000, output_dir="split_db")

    # 3) Build composition index
    print("Loading structure_database.pkl and building composition index...")
    database_by_comp = build_database_by_comp("structure_database.pkl")
    print(f"database_by_comp size = {len(database_by_comp)}")

    # 4) Pre-filter input CSV based on composition
    print(f"Processing input CSV: {args.input_csv}")
    with open(args.input_csv, 'r', encoding='utf-8') as fin, \
         open("temp_novel.csv", 'w', encoding='utf-8', newline='') as fout_novel, \
         open("temp_non_novel.csv", 'w', encoding='utf-8', newline='') as fout_non_novel:
        
        reader = csv.reader(fin)
        writer_novel = csv.writer(fout_novel)
        writer_non_novel = csv.writer(fout_non_novel)

        header = next(reader, None)  # Read header
        if header:
            writer_non_novel.writerow(header)

        for row_idx, row in enumerate(reader, start=1):
            try:
                # Assuming the last column contains the POSCAR string
                poscar_input = row[-1].replace('\\n','\n')
                query_struc = Structure.from_str(poscar_input, fmt="poscar")
                reduced_comp_query = query_struc.composition.reduced_composition
                comp_query = str(reduced_comp_query)

                if comp_query in database_by_comp:
                    # Potentially non-novel
                    writer_non_novel.writerow(row)
                else:
                    # Novel structure
                    writer_novel.writerow(row + ["1"])

                if row_idx % 1000 == 0:
                    print(f"Processed {row_idx} rows...")
            except Exception as e:
                print(f"Error parsing POSCAR in input.csv row {row_idx}: {e}")

    print("Pre-filtering completed. Generated temp_novel.csv and temp_non_novel.csv.")

def main_work(args):
    """
    Execute the main processing workflow.

    Parameters:
    - args: Parsed command-line arguments.
    """
    # 5) Process non-novel entries with multiple threads
    print("Processing non-novel entries with splitRun_csv...")
    splitRun_csv(filename='temp_non_novel.csv', threads=args.threads, skip_header=True)
    show_progress()
    csv_file_to_run = 'temp_non_novel.csv'

    # 4) 查看输入 CSV 首行，以获取动态 header
    try:
        with open(csv_file_to_run, 'r', encoding='utf-8') as f:
            reader = csv.reader(f)
            header_in = next(reader)  # 读取 CSV 的第一行做 header
    except FileNotFoundError:
        print(f"Error: The input CSV file '{csv_file_to_run}' does not exist.")
        exit(1)
    except Exception as e:
        print(f"Error reading CSV file '{csv_file_to_run}': {e}")
        exit(1)
    
    # 假设最后一列是 "SLICES"，前面若干列是各种属性名称
    # 结果里要再添加 "poscar" 和 "novelty" 两列
    dynamic_header = header_in + ["novelty"]
    # 变成字符串
    result_header_line = ",".join(dynamic_header) + "\n"

    # 6) Collect results from job shards
    print(f"Collecting results into {args.output_csv} and suspect_rows.csv...")
    collect_csv(output=args.output_csv,
               glob_target="./job_*/result.csv", cleanup=False,
               header=result_header_line)
    collect_csv(output="suspect_rows.csv",
               glob_target="./job_*/suspect_rows.csv", cleanup=True,
               header=result_header_line)

    # 6.1) 去重 suspect_rows.csv
    print("Removing duplicates from suspect_rows.csv...")
    unique_rows = set()
    temp_dedup_file = "suspect_rows_dedup.csv"
    
    with open("suspect_rows.csv", 'r', encoding='utf-8') as fin, \
         open(temp_dedup_file, 'w', encoding='utf-8', newline='') as fout:
        reader = csv.reader(fin)
        writer = csv.writer(fout)
        header = next(reader, None)
        if header:
            writer.writerow(header)
        for row in reader:
            row_tuple = tuple(row)
            if row_tuple not in unique_rows:
                unique_rows.add(row_tuple)
                writer.writerow(row)
    
    # 用去重后的文件替换原始文件
    os.replace(temp_dedup_file, "suspect_rows.csv")
    print("Duplicates removed from suspect_rows.csv.")

    # 7) Append novel entries to the final results
    temp_novel_path = "temp_novel.csv"
    results_path = args.output_csv

    if os.path.exists(temp_novel_path):
        print(f"Appending contents of {temp_novel_path} to {results_path}")
        with open(temp_novel_path, "r", encoding="utf-8") as fin:
            reader = csv.reader(fin)
            with open(results_path, "a", encoding="utf-8", newline='') as fout:
                writer = csv.writer(fout)
                for row in reader:
                    writer.writerow(row)
        print(f"Successfully appended {temp_novel_path} to {results_path}")
    else:
        print(f"{temp_novel_path} does not exist; skipping append operation.")

def parse_args():
    """
    Parse command-line arguments.

    Returns:
    - argparse.Namespace: Parsed arguments.
    """
    parser = argparse.ArgumentParser(description="Process structure database and CSV files.")
    parser.add_argument('--input_csv', type=str, required=True,
                        help='Path to the input CSV file (e.g., ../2_inverse/results.csv).')
    parser.add_argument('--structure_json_for_novelty_check', type=str, required=True,
                        help='Path to the JSON file containing CIF data for novelty checks (e.g., cifs_filtered.json).')
    parser.add_argument('--threads', type=int, default=8,
                        help='Number of threads to use for processing (default: 8).')
    parser.add_argument('--output_csv', type=str, default='results.csv',
                        help='Path to the output CSV file (default: results.csv).')
    return parser.parse_args()

def main():
    """
    Main function to execute the script.
    """
    args = parse_args()
    prepare(args)
    main_work(args)
    os.system("rm suspect_rows.csv temp_non_novel.csv temp_novel.csv")

if __name__ == "__main__":
    main()
