import os
import csv
import json
import argparse
import sqlite3
import glob
from pymatgen.core.structure import Structure
from pymatgen.symmetry.analyzer import SpacegroupAnalyzer
from slices.utils import splitRun_csv, show_progress, collect_csv
try:
    from tqdm import tqdm
except ImportError:
    tqdm = None


def build_structure_database(structure_json_path, db_path):
    """
    Build a SQLite structure database from a JSON file containing CIFs.

    The database schema matches the demo workflow and supports novelty checks
    by reduced formula composition and primitive CIF.
    """
    print("开始构建结构数据库（structure_database.db）...")
    print(
        "提示：对于 Materials Project 全集（约 150,000 条结构），"
        "构建过程通常需要 40-50 分钟（取决于 CPU 性能和 I/O 速度），初次构建后新颖性检查无需二次重复构建。"
        "过程中会显示实时进度条，请耐心等待。"
    )

    try:
        conn = sqlite3.connect(db_path)
        cursor = conn.cursor()
        cursor.execute(
            """
            CREATE TABLE IF NOT EXISTS structures (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                composition TEXT,
                spacegroup INTEGER,
                primitive_cif TEXT
            )
            """
        )
        if cursor.execute(
            "SELECT name FROM sqlite_master WHERE type='table' AND name='structures'"
        ).fetchone():
            print("表 'structures' 已存在，将追加或覆盖数据。")
        else:
            print("已创建新表 'structures'。")
    except sqlite3.Error as e:
        print(f"数据库连接或创建失败: {e}")
        exit(1)

    try:
        with open(structure_json_path, 'r', encoding='utf-8') as f:
            cifs = json.load(f)
    except FileNotFoundError:
        print(f"Error: Structure JSON file '{structure_json_path}' does not exist.")
        exit(1)
    except json.JSONDecodeError as e:
        print(f"Error: Failed to parse JSON file '{structure_json_path}': {e}")
        exit(1)

    total = len(cifs)
    print(f"JSON 文件加载完成，共 {total:,} 条结构记录。")
    if total == 0:
        print("警告：JSON 文件中没有结构数据，跳过数据库构建。")
        conn.close()
        return

    processed = 0
    failed = 0

    if tqdm is None:
        print("Error: tqdm not installed. Please install it to show progress, e.g., `pip install tqdm`.")
        exit(1)

    with tqdm(
        total=total,
        desc="构建数据库",
        unit="结构",
        bar_format="{l_bar}{bar}| {n_fmt}/{total_fmt} [{elapsed}<{remaining}, {rate_fmt}]",
    ) as pbar:
        for idx, cif_entry in enumerate(cifs, start=1):
            cif_string = cif_entry.get("cif")
            if not cif_string:
                pbar.update(1)
                continue
            try:
                stru = Structure.from_str(cif_string, "cif")
                finder = SpacegroupAnalyzer(stru)
                primitive_stru = finder.get_primitive_standard_structure()
                spacegroup = finder.get_space_group_number()

                composition = primitive_stru.composition.reduced_formula
                primitive_cif = primitive_stru.to(fmt="cif")

                cursor.execute(
                    "INSERT INTO structures (composition, spacegroup, primitive_cif) VALUES (?, ?, ?)",
                    (composition, spacegroup, primitive_cif),
                )
                processed += 1
            except Exception as e:
                failed += 1
                if failed <= 10:
                    print(f"\n警告：第 {idx} 条结构处理失败（已跳过）：{e}")

            pbar.update(1)

            if idx % 1000 == 0:
                conn.commit()

    conn.commit()
    print("\n插入完成，正在创建索引以加速后续新颖性查询...")
    cursor.execute("CREATE INDEX IF NOT EXISTS idx_composition ON structures (composition)")
    conn.commit()
    conn.close()

    print("\n结构数据库构建完成！")
    print(f"   → 成功处理：{processed:,} 条")
    if failed:
        print(f"   → 处理失败：{failed:,} 条（已自动跳过）")
    print(f"   → 数据库保存至：{db_path}")
    print("   → 已创建 composition 索引，后续新颖性检查将显著加速（>100x）")


def ensure_structure_database(structure_json_path, db_path):
    if os.path.exists(db_path):
        print(f"Found existing database '{db_path}'.")
        return
    build_structure_database(structure_json_path, db_path)


def _read_csv_header(csv_path):
    try:
        with open(csv_path, 'r', encoding='utf-8') as f:
            reader = csv.reader(f)
            return next(reader)
    except FileNotFoundError:
        print(f"Error: The input CSV file '{csv_path}' does not exist.")
        exit(1)
    except Exception as e:
        print(f"Error reading CSV file '{csv_path}': {e}")
        exit(1)


def process_data(input_csv, output_csv, threads):
    print("Cleaning up old job directories...")
    os.system("rm -rf job_*")

    print("Splitting the input CSV into job files...")
    splitRun_csv(filename=input_csv, threads=threads, skip_header=True)
    show_progress()

    header_in = _read_csv_header(input_csv)
    dynamic_header = header_in + ["novelty"]
    result_header_line = ",".join(dynamic_header) + "\n"

    print(f"Collecting results into '{output_csv}'...")
    collect_csv(
        output=output_csv,
        glob_target="./job_*/result.csv",
        cleanup=False,
        header=result_header_line,
    )

    suspect_files = glob.glob("./job_*/suspect_rows.csv")
    if suspect_files:
        print("Collecting suspect rows into 'suspect_rows.csv'...")
        collect_csv(
            output="suspect_rows.csv",
            glob_target="./job_*/suspect_rows.csv",
            cleanup=True,
            header=result_header_line,
        )
        _dedup_suspect_rows("suspect_rows.csv")
    else:
        os.system("rm -rf job_*")
        print("No suspect rows found.")

    print(f"Results collected into '{output_csv}'.")


def _dedup_suspect_rows(path):
    if not os.path.exists(path):
        return
    print("Removing duplicates from suspect_rows.csv...")
    unique_rows = set()
    temp_dedup_file = "suspect_rows_dedup.csv"

    with open(path, 'r', encoding='utf-8') as fin, \
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

    os.replace(temp_dedup_file, path)
    print("Duplicates removed from suspect_rows.csv.")


def parse_args():
    parser = argparse.ArgumentParser(description="Process structure database and CSV files.")
    parser.add_argument(
        '--input_csv',
        type=str,
        required=True,
        help='Path to the input CSV file (e.g., ../2_inverse/results.csv).'
    )
    parser.add_argument(
        '--structure_json_for_novelty_check',
        type=str,
        required=True,
        help='Path to the JSON file containing CIF data for novelty checks (e.g., cifs_filtered.json).'
    )
    parser.add_argument(
        '--threads',
        type=int,
        default=8,
        help='Number of threads to use for processing (default: 8).'
    )
    parser.add_argument(
        '--output_csv',
        type=str,
        default='results.csv',
        help='Path to the output CSV file (default: results.csv).'
    )
    return parser.parse_args()


def main():
    args = parse_args()

    if not os.path.isfile(args.input_csv):
        print(f"Error: The input CSV file '{args.input_csv}' does not exist.")
        exit(1)

    if not os.path.isfile(args.structure_json_for_novelty_check):
        print(f"Error: The structure JSON file '{args.structure_json_for_novelty_check}' does not exist.")
        exit(1)

    db_path = "structure_database.db"
    ensure_structure_database(args.structure_json_for_novelty_check, db_path)

    process_data(args.input_csv, args.output_csv, args.threads)


if __name__ == "__main__":
    main()
