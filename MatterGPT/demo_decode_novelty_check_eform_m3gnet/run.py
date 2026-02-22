
import os
import glob
import argparse  
from slices.utils import *
import pandas as pd
import matplotlib.pyplot as plt
from scipy.stats import norm
import numpy as np
import matplotlib.ticker as ticker
import pickle
import json
from pymatgen.core.structure import Structure
import sqlite3 
from pymatgen.symmetry.analyzer import SpacegroupAnalyzer 

from tqdm import tqdm  # 在文件顶部 import 处添加这一行（如果还没有）

def load_and_save_structure_database(structure_json_path):
    """
    Loads CIF data from a JSON file, converts them to pymatgen Structures,
    analyzes them, and saves them into an indexed SQLite database.

    Parameters:
    - structure_json_path (str): Path to the JSON file containing CIFs.
    """
    db_path = 'structure_database.db'

    # 估算时间提示（可根据实际数据集调整数字）
    print("开始构建结构数据库（structure_database.db）...")
    print("提示：对于 Materials Project 全集（约 150,000 条结构），"
          "构建过程通常需要 40-50 分钟（取决于 CPU 性能和 I/O 速度），初次构建后新颖性检查无需二次重复构建。"
          "过程中会显示实时进度条，请耐心等待。")

    # 连接数据库并创建表
    try:
        conn = sqlite3.connect(db_path)
        c = conn.cursor()
        c.execute('''
        CREATE TABLE IF NOT EXISTS structures (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            composition TEXT,
            spacegroup INTEGER,
            primitive_cif TEXT,
            band_gap REAL
        )
        ''')
        # 如果是全新创建，提示一下
        if c.execute("SELECT name FROM sqlite_master WHERE type='table' AND name='structures'").fetchone():
            print(f"表 'structures' 已存在，将追加或覆盖数据。")
        else:
            print(f"已创建新表 'structures'。")

    except sqlite3.Error as e:
        print(f"数据库连接或创建失败: {e}")
        exit(1)

    # 加载 JSON 文件
    try:
        with open(structure_json_path, 'r') as f:
            cifs = json.load(f)
    except FileNotFoundError:
        print(f"错误：结构 JSON 文件 '{structure_json_path}' 不存在。")
        exit(1)
    except json.JSONDecodeError as e:
        print(f"错误：无法解析 JSON 文件 '{structure_json_path}': {e}")
        exit(1)

    total_count = len(cifs)
    print(f"JSON 文件加载完成，共 {total_count:,} 条结构记录。")

    if total_count == 0:
        print("警告：JSON 文件中没有结构数据，跳过数据库构建。")
        conn.close()
        return

    processed_count = 0
    failed_count = 0

    # 使用 tqdm 显示进度条
    with tqdm(total=total_count, desc="构建数据库", unit="结构",
              bar_format="{l_bar}{bar}| {n_fmt}/{total_fmt} [{elapsed}<{remaining}, {rate_fmt}]") as pbar:

        for i, cif_entry in enumerate(cifs):
            cif_string = cif_entry.get("cif")
            if not cif_string:
                pbar.update(1)
                continue

            try:
                stru = Structure.from_str(cif_string, "cif")
                band_gap = cif_entry.get("band_gap", None)

                finder = SpacegroupAnalyzer(stru)
                primitive_stru = finder.get_primitive_standard_structure()
                spacegroup = finder.get_space_group_number()

                composition = primitive_stru.composition.reduced_formula
                primitive_cif = primitive_stru.to(fmt="cif")

                c.execute(
                    "INSERT INTO structures (composition, spacegroup, primitive_cif, band_gap) "
                    "VALUES (?, ?, ?, ?)",
                    (composition, spacegroup, primitive_cif, band_gap)
                )
                processed_count += 1

            except Exception as e:
                failed_count += 1
                # 可选：打印严重错误，但不中断进度条
                if failed_count <= 10:  # 只显示前10个错误，避免刷屏
                    print(f"\n警告：第 {i} 条结构处理失败（已跳过）：{e}")

            pbar.update(1)

            # 每插入 1000 条提交一次，加快写入速度并减少内存占用
            if (i + 1) % 1000 == 0:
                conn.commit()

    # 最终提交剩余数据
    conn.commit()

    # 创建索引（放在插入完成后，避免边插边建索引拖慢速度）
    print("\n插入完成，正在创建索引以加速后续新颖性查询...")
    c.execute("CREATE INDEX IF NOT EXISTS idx_composition ON structures (composition)")
    c.execute("CREATE INDEX IF NOT EXISTS idx_spacegroup ON structures (spacegroup)")
    conn.commit()
    conn.close()

    print(f"\n结构数据库构建完成！")
    print(f"   → 成功处理：{processed_count:,} 条")
    if failed_count > 0:
        print(f"   → 处理失败：{failed_count:,} 条（已自动跳过）")
    print(f"   → 数据库保存至：{db_path}")
    print(f"   → 已创建 composition 索引，后续新颖性检查将显著加速（>100x）")

def process_data(input_csv, output_csv, structure_json_path, threads):
    """
    Processes the input CSV file by splitting it into jobs, running them locally,
    and collecting the results.
    Parameters:
    - input_csv (str): Path to the input CSV file to be processed. [cite: 13]
    - structure_json_path (str): Path to the JSON file containing CIFs. [cite: 14]
    - threads (int): Number of threads to use for processing. [cite: 15]
    """

    print("Cleaning up old job directories...")
    os.system("rm -rf job_*")
    print(f"Loading and saving structure database from '{structure_json_path}'...")

    db_path = 'structure_database.db' 
    if os.path.exists(db_path):
        print(f"Existed database '{db_path}'.")
    else:
        load_and_save_structure_database(structure_json_path)
    print("Splitting the input CSV into job files...")
    splitRun_csv(filename=input_csv, threads=threads, skip_header=True)


    print("Showing progress of local jobs...")
    show_progress()


    print("Collecting results into 'results.csv'...")
    collect_csv(
        output=output_csv,
        glob_target="./job_*/result.csv",  
        cleanup=True,
        header="eform_target,SLICES,eform_chgnet,spacegroup_number,poscar,novelty\n"
    ) 
    print("Results collected into 'results.csv'.")

def prepare_data(results_file, training_file):
    """
    Prepares the data by reading results and training files, organizing the data
    into dictionaries, and sorting the keys. [cite: 17]

    Parameters:
    - results_file (str): Path to the results CSV file. [cite: 17]
    - training_file (str): Path to the training CSV file. [cite: 18]

    Returns:
    - data_dict (dict): Dictionary containing all and novel values. [cite: 19]
    - sorted_keys (list): Sorted list of headers. [cite: 19]
    - trainingset_values (list): List of training dataset values. [cite: 19]
    """
    results_1 = pd.read_csv(results_file) 
    trainingset = pd.read_csv(training_file, header=0) 

    header_values = results_1.iloc[:, 0].tolist()
    data_values = results_1.iloc[:, 2].tolist()

    if trainingset.shape[1] > 1:
        trainingset_values = trainingset.iloc[:, 1].tolist() 
    else:
        trainingset_values = []
        print("Warning: Training file does not have formation energy column. Skipping training data in plot.") 

    novelty_values = results_1.iloc[:, 5].tolist()

    data_dict = {}
    for header, value, novelty in zip(header_values, data_values, novelty_values):
        if header not in data_dict:
            data_dict[header] = {'all': [], 'novel': []}
        data_dict[header]['all'].append(value)
        if novelty == 1:
            data_dict[header]['novel'].append(value)

    sorted_keys = sorted(data_dict.keys(), reverse=True) 

    return data_dict, sorted_keys, trainingset_values


def create_dataframe(data_dict, sorted_keys, trainingset_values, data_type='all'):
    """
    Creates a pandas DataFrame from the data dictionary.
    Parameters:
    - data_dict (dict): Dictionary containing data. [cite: 23]
    - sorted_keys (list): Sorted list of headers. [cite: 23]
    - trainingset_values (list): List of training dataset values. [cite: 24]
    - data_type (str): Type of data to include ('all' or 'novel'). [cite: 24]
    Returns:
    - df (pd.DataFrame): Constructed DataFrame. [cite: 25]
    """
    df = pd.DataFrame({k: pd.Series(data_dict[k][data_type], index=range(len(data_dict[k][data_type]))) for k in sorted_keys})
    df = pd.concat([df, pd.Series(trainingset_values, name='training_dataset')], axis=1)
    return df


def plot_combined_histograms(all_data, novel_data, output_file):
    num_cols = len(all_data.columns)
    fig, axs = plt.subplots(num_cols, 2, figsize=(12, 3*num_cols), sharex=True)
    bins = np.linspace(-6, 0, 50)
    # Ensure axs is always 2D
    if num_cols == 1:
        axs = axs.reshape(1, -1)
    for i, col_name in enumerate(all_data.columns):
        for j, (data, title) in enumerate(zip([all_data, novel_data], ['All Materials', 'Novel Materials'])): 
            color = 'violet' if col_name == 'training_dataset' else 'lightblue'
            # Convert to numeric, coercing errors to NaN
            numeric_data = pd.to_numeric(data[col_name], errors='coerce').dropna()

            # Only plot if we have valid numeric data
            if len(numeric_data) > 0:
                axs[i, j].hist(numeric_data, bins=bins, density=True, color=color, edgecolor='black', alpha=0.7)
                mu, std = norm.fit(numeric_data)
                mean_val = numeric_data.mean()
                axs[i, j].axvline(mean_val, color='red', linestyle='--', linewidth=1)
                axs[i, j].text(mean_val, axs[i, j].get_ylim()[1]*0.9, f"{mean_val:.2f}", color='red', fontsize=6, ha='left') 

            axs[i, j].text(0.05, 0.95, f"{col_name}\n{title}", transform=axs[i, j].transAxes, fontsize=8, va='top')
    for ax in axs.flat:
        ax.set_xlim(-6, 0)
        ax.tick_params(axis='both', which='major', labelsize=6)
        ax.xaxis.set_major_locator(ticker.MultipleLocator(1))
    fig.text(0.5, 0.02, 'Formation Energy (eV/atom)', ha='center', va='center', fontsize=10)
    fig.text(0.02, 0.5, 'Density', ha='center', va='center', rotation='vertical', fontsize=10)
    plt.subplots_adjust(hspace=0, wspace=0.1)
    plt.tight_layout(rect=[0.03, 0.03, 1, 0.97])
    plt.savefig(output_file, dpi=300, bbox_inches='tight')
    return fig 


def main():
    """
    Main function to parse command-line arguments and initiate data processing. 
    """
    # Set up command-line argument parsing
    parser = argparse.ArgumentParser(
        description="Process and analyze structure data.",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter
    )
    parser.add_argument(
        "--input_csv",
        type=str,
        help="Path to the input CSV file to be processed."
    )
    parser.add_argument(
        "--structure_json_for_novelty_check",
        type=str,
        help="Path to the JSON file containing CIFs (structure database)." 
    )
    parser.add_argument(
        "--training_file",
        type=str,
        help="Path to the training CSV file (e.g., 'train_data_reduce_zero.csv')."
    )
    parser.add_argument(
        "--output_csv",
        type=str,
        default="results.csv",
        help="Path for the output CSV file to be generated." 
    )
    parser.add_argument(
        "--threads",
        type=int,
        default=8,
        help="Number of threads to use for processing."
    )
    parser.add_argument(
        "--cleanup",
        action='store_true',
        help="If set, cleanup intermediate files after processing."
    )

    args = parser.parse_args()

    
    # Check if input files exist
    if not os.path.isfile(args.input_csv):
        print(f"Error: The input CSV file '{args.input_csv}' does not exist.") 
        exit(1)

    if not os.path.isfile(args.structure_json_for_novelty_check):
        print(f"Error: The structure JSON file '{args.structure_json_for_novelty_check}' does not exist.") 
        exit(1)

    if not os.path.isfile(args.training_file):
        print(f"Error: The training file '{args.training_file}' does not exist.") 
        exit(1)

    # Process data
    process_data(args.input_csv, args.output_csv, args.structure_json_for_novelty_check, args.threads) 

    # Prepare data
    data_dict, sorted_keys, trainingset_values = prepare_data(args.output_csv, args.training_file)

    # Process all materials
    all_materials_df = create_dataframe(data_dict, sorted_keys, trainingset_values, 'all')

    # Process novel materials
    novel_materials_df = create_dataframe(data_dict, sorted_keys, trainingset_values, 'novel')
    fig = plot_combined_histograms(all_materials_df, novel_materials_df, "combined_results.png")

    # Optional cleanup
    if args.cleanup:
        os.system("rm energy_formation_chgnet_lists.csv energy_formation_chgnet_lists_novel.csv")
        print("Cleaned up intermediate CSV files.")


if __name__ == "__main__":
    main() 
