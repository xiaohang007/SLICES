import os
import csv
import pickle
import json
import argparse  # Import argparse for command-line argument parsing
from slices.utils import splitRun_csv, show_progress, collect_csv
import pandas as pd
import matplotlib.pyplot as plt
from scipy.stats import norm
import numpy as np
import matplotlib.ticker as ticker
from pymatgen.core.structure import Structure

def process_data(input_csv, output_csv, threads):
    """
    Processes the input CSV file, splits it into jobs, runs them locally,
    and collects the results into an output CSV file.

    Parameters:
    - input_csv (str): Path to the input CSV file to be processed.
    - output_csv (str): Path for the output CSV file to be generated.
    - structure_json_path (str): Path to the JSON file containing CIFs.
    """
    # 1) 清理旧目录
    print("Cleaning up old job directories...")
    os.system("rm -rf job_*")

    # 3) 需要处理的 CSV 文件(使用命令行输入)
    csv_file_to_run = input_csv

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
    dynamic_header = header_in + ["space_group_num","poscar"]
    # 变成字符串
    result_header_line = ",".join(dynamic_header) + "\n"

    # 5) 分割 CSV 文件并执行任务
    print("Splitting the input CSV into job files...")
    splitRun_csv(filename=csv_file_to_run, threads=threads, skip_header=True)
    show_progress()

    # 7) 收集所有 job_* 子文件夹中的 result.csv 文件到一个总的 results.csv
    print(f"Collecting results into '{output_csv}'...")
    collect_csv(
        output=output_csv,
        glob_target="./job_*/result.csv",  # 所有分块跑出的部分结果
        cleanup=True,
        header=result_header_line  # 动态生成的表头
    )
    print(f"Results collected into '{output_csv}'.")

def main():
    """
    Main function to parse command-line arguments and initiate data processing.
    """
    # 设置命令行参数解析
    parser = argparse.ArgumentParser(
        description="Process CSV files with structure data.",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter
    )
    parser.add_argument(
        "--input_csv",
        type=str,
        help="Path to the input CSV file to be processed."
    )
    parser.add_argument(
        "--output_csv",
        type=str,
        help="Path for the output CSV file to be generated."
    )
    parser.add_argument(
        "--threads",
        type=int,
        default=8,
        help="Number of threads to use for processing."
    )

    args = parser.parse_args()
    
    # 检查输入文件是否存在
    if not os.path.isfile(args.input_csv):
        print(f"Error: The input file '{args.input_csv}' does not exist.")
        exit(1)
    
    
    # 调用 process_data 函数并传递参数
    process_data(args.input_csv, args.output_csv, args.threads)

if __name__ == "__main__":
    main()  
