import os

# Limit per-process native threads to avoid CPU oversubscription when DB build uses multiprocessing.
# You can override via MATTERGPT_NUMERIC_THREADS=<n>.
_numeric_threads = os.environ.get("MATTERGPT_NUMERIC_THREADS", "1")
os.environ["OMP_NUM_THREADS"] = _numeric_threads
os.environ["OPENBLAS_NUM_THREADS"] = _numeric_threads
os.environ["MKL_NUM_THREADS"] = _numeric_threads
os.environ["NUMEXPR_NUM_THREADS"] = _numeric_threads
os.environ["VECLIB_MAXIMUM_THREADS"] = _numeric_threads
os.environ["BLIS_NUM_THREADS"] = _numeric_threads

import argparse
import json
import sqlite3
import hashlib
import time
import multiprocessing as mp

from slices.utils import *
import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
import matplotlib.ticker as ticker
from pymatgen.core.structure import Structure
from pymatgen.symmetry.analyzer import SpacegroupAnalyzer
from tqdm import tqdm


def _compute_file_sha256(file_path, chunk_size=1024 * 1024):
    hasher = hashlib.sha256()
    with open(file_path, "rb") as f:
        while True:
            chunk = f.read(chunk_size)
            if not chunk:
                break
            hasher.update(chunk)
    return hasher.hexdigest()


def _build_source_fingerprint(structure_json_path):
    return {
        "source_json_sha256": _compute_file_sha256(structure_json_path),
    }


def _check_structure_database_freshness(db_path, structure_json_path):
    """
    Return (needs_rebuild, reason, source_fingerprint).
    """
    source_fingerprint = _build_source_fingerprint(structure_json_path)
    if not os.path.exists(db_path):
        return True, f"Database '{db_path}' not found.", source_fingerprint

    try:
        conn = sqlite3.connect(db_path)
        c = conn.cursor()

        c.execute("SELECT name FROM sqlite_master WHERE type='table' AND name='structures'")
        if not c.fetchone():
            conn.close()
            return True, "Missing table 'structures'.", source_fingerprint

        c.execute("SELECT COUNT(*) FROM structures")
        row_count = int(c.fetchone()[0])
        if row_count == 0:
            conn.close()
            return True, "Table 'structures' is empty.", source_fingerprint

        c.execute("SELECT name FROM sqlite_master WHERE type='table' AND name='db_meta'")
        if not c.fetchone():
            conn.close()
            return True, "Missing table 'db_meta' (old DB format).", source_fingerprint

        c.execute("SELECT key, value FROM db_meta")
        meta = {k: v for k, v in c.fetchall()}
        conn.close()
    except Exception as e:
        return True, f"Failed to inspect DB: {e}", source_fingerprint

    required_keys = ["source_json_sha256"]
    missing_keys = [k for k in required_keys if k not in meta]
    if missing_keys:
        return True, f"Missing metadata keys: {missing_keys}", source_fingerprint

    mismatches = []
    for key in required_keys:
        if str(meta.get(key, "")) != str(source_fingerprint.get(key, "")):
            mismatches.append(key)
    if mismatches:
        return True, "JSON content hash changed.", source_fingerprint

    return False, "Database is up-to-date with novelty JSON.", source_fingerprint


def _prepare_db_record(cif_entry):
    cif_string = cif_entry.get("cif")
    if not cif_string:
        return None, "empty_cif"
    try:
        stru = Structure.from_str(cif_string, "cif")
        band_gap = cif_entry.get("band_gap", None)
        finder = SpacegroupAnalyzer(stru)
        primitive_stru = finder.get_primitive_standard_structure()
        spacegroup = finder.get_space_group_number()
        composition = primitive_stru.composition.reduced_formula
        primitive_cif = primitive_stru.to(fmt="cif")
        return (composition, spacegroup, primitive_cif, band_gap), None
    except Exception as e:
        return None, str(e)


def load_and_save_structure_database(structure_json_path, source_fingerprint=None, db_build_workers=1):
    """
    Loads CIF data from a JSON file, converts them to pymatgen Structures,
    analyzes them, and saves them into an indexed SQLite database.
    """
    db_path = "structure_database.db"

    if os.path.exists(db_path):
        try:
            os.remove(db_path)
            print(f"已删除旧数据库 '{db_path}'，准备全量重建。")
        except OSError as e:
            print(f"无法删除旧数据库 '{db_path}': {e}")
            exit(1)

    print("开始构建结构数据库（structure_database.db）...")
    print(
        "提示：对于 Materials Project 全集（约 150,000 条结构），"
        "构建过程通常需要 40-50 分钟（取决于 CPU 性能和 I/O 速度），初次构建后新颖性检查无需二次重复构建。"
        "过程中会显示实时进度条，请耐心等待。"
    )

    try:
        conn = sqlite3.connect(db_path)
        c = conn.cursor()
        c.execute(
            """
        CREATE TABLE IF NOT EXISTS structures (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            composition TEXT,
            spacegroup INTEGER,
            primitive_cif TEXT,
            band_gap REAL
        )
        """
        )
        c.execute(
            """
        CREATE TABLE IF NOT EXISTS db_meta (
            key TEXT PRIMARY KEY,
            value TEXT
        )
        """
        )
        print("已创建新表 'structures' 和 'db_meta'。")
    except sqlite3.Error as e:
        print(f"数据库连接或创建失败: {e}")
        exit(1)

    try:
        with open(structure_json_path, "r") as f:
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
    insert_sql = "INSERT INTO structures (composition, spacegroup, primitive_cif, band_gap) VALUES (?, ?, ?, ?)"
    pending_rows = []
    batch_commit_size = 1000
    db_build_workers = max(int(db_build_workers), 1)
    parallel_failed = False
    if db_build_workers > 1:
        print(f"使用并行构建：{db_build_workers} 个进程")
        try:
            with tqdm(
                total=total_count,
                desc="构建数据库",
                unit="结构",
                bar_format="{l_bar}{bar}| {n_fmt}/{total_fmt} [{elapsed}<{remaining}, {rate_fmt}]",
            ) as pbar:
                with mp.Pool(processes=db_build_workers) as pool:
                    for idx, (row, err_msg) in enumerate(
                        pool.imap_unordered(_prepare_db_record, cifs, chunksize=32), start=1
                    ):
                        if row is not None:
                            pending_rows.append(row)
                            processed_count += 1
                            if len(pending_rows) >= batch_commit_size:
                                c.executemany(insert_sql, pending_rows)
                                conn.commit()
                                pending_rows.clear()
                        elif err_msg != "empty_cif":
                            failed_count += 1
                            if failed_count <= 10 and err_msg:
                                print(f"\n警告：第 {idx} 条结构处理失败（已跳过）：{err_msg}")
                        pbar.update(1)
        except Exception as e:
            parallel_failed = True
            print(f"并行构建不可用（{type(e).__name__}: {e}），自动回退单进程。")
            c.execute("DELETE FROM structures")
            conn.commit()
            processed_count = 0
            failed_count = 0
            pending_rows = []

    if db_build_workers <= 1 or parallel_failed:
        with tqdm(
            total=total_count,
            desc="构建数据库",
            unit="结构",
            bar_format="{l_bar}{bar}| {n_fmt}/{total_fmt} [{elapsed}<{remaining}, {rate_fmt}]",
        ) as pbar:
            for i, cif_entry in enumerate(cifs):
                row, err_msg = _prepare_db_record(cif_entry)
                if row is not None:
                    pending_rows.append(row)
                    processed_count += 1
                    if len(pending_rows) >= batch_commit_size:
                        c.executemany(insert_sql, pending_rows)
                        conn.commit()
                        pending_rows.clear()
                elif err_msg != "empty_cif":
                    failed_count += 1
                    if failed_count <= 10 and err_msg:
                        print(f"\n警告：第 {i} 条结构处理失败（已跳过）：{err_msg}")
                pbar.update(1)

    if pending_rows:
        c.executemany(insert_sql, pending_rows)
        conn.commit()

    if source_fingerprint is None:
        source_fingerprint = _build_source_fingerprint(structure_json_path)
    source_fingerprint = dict(source_fingerprint)
    source_fingerprint["db_built_at_epoch"] = str(int(time.time()))
    c.execute("DELETE FROM db_meta")
    for key, value in source_fingerprint.items():
        c.execute("INSERT OR REPLACE INTO db_meta (key, value) VALUES (?, ?)", (str(key), str(value)))

    conn.commit()
    print("\n插入完成，正在创建索引以加速后续新颖性查询...")
    c.execute("CREATE INDEX IF NOT EXISTS idx_composition ON structures (composition)")
    c.execute("CREATE INDEX IF NOT EXISTS idx_spacegroup ON structures (spacegroup)")
    conn.commit()
    conn.close()

    print("\n结构数据库构建完成！")
    print(f"   → 成功处理：{processed_count:,} 条")
    if failed_count > 0:
        print(f"   → 处理失败：{failed_count:,} 条（已自动跳过）")
    print(f"   → 数据库保存至：{db_path}")
    print("   → 已创建 composition 索引，后续新颖性检查将显著加速（>100x）")


def _infer_prop_columns_from_input(input_csv):
    prop_cols = []
    try:
        with open(input_csv, "r") as f:
            first_line = f.readline().strip()
        if first_line:
            headers = [h.strip() for h in first_line.split(",")]
            slices_col = "SLICES" if "SLICES" in headers else ("symSLICES" if "symSLICES" in headers else None)
            if slices_col:
                prop_cols = [h for h in headers if h not in {slices_col, "crystal_system"}]
    except Exception:
        prop_cols = []
    if not prop_cols:
        prop_cols = ["prop_0"]
    return prop_cols


def _infer_total_samples_from_input(input_csv):
    with open(input_csv, "r") as f:
        lines = [line for line in f.readlines() if line.strip()]
    if not lines:
        return 0
    first_line = lines[0].strip()
    if "SLICES" in first_line or "symSLICES" in first_line:
        return max(len(lines) - 1, 0)
    return len(lines)


def process_data(input_csv, output_csv, structure_json_path, threads):
    """
    Process the input CSV by splitting jobs, running local/SLURM jobs,
    and collecting all result.csv files.
    """
    print("Cleaning up old job directories...")
    os.system("rm -rf job_*")
    print(f"Loading and saving structure database from '{structure_json_path}'...")

    db_path = "structure_database.db"
    needs_rebuild, reason, fingerprint = _check_structure_database_freshness(db_path, structure_json_path)
    if needs_rebuild:
        print(f"Need rebuild database: {reason}")
        db_build_workers = min(max(1, int(threads)), max(1, os.cpu_count() or 1))
        load_and_save_structure_database(
            structure_json_path,
            source_fingerprint=fingerprint,
            db_build_workers=db_build_workers,
        )
    else:
        print(f"Reusing existing database '{db_path}'. {reason}")

    print("Splitting the input CSV into job files...")
    # Keep header so workflow/script.py can map property columns by name.
    splitRun_csv(filename=input_csv, threads=threads, skip_header=False)

    print("Showing progress of local jobs...")
    show_progress()

    prop_cols = _infer_prop_columns_from_input(input_csv)
    header = ",".join(prop_cols + ["SLICES", "eform_pred", "spacegroup", "poscar", "novelty"]) + "\n"
    print(f"Collecting results into '{output_csv}'...")
    collect_csv(
        output=output_csv,
        glob_target="./job_*/result.csv",
        cleanup=True,
        header=header,
    )
    print(f"Results collected into '{output_csv}'.")


def _detect_result_columns(df):
    slices_col = "SLICES" if "SLICES" in df.columns else ("symSLICES" if "symSLICES" in df.columns else None)
    # "eform_chgnet" is the column name in results written by older versions
    pred_col = next((c for c in ("eform_pred", "eform_chgnet") if c in df.columns), df.columns[2] if len(df.columns) > 2 else None)
    novelty_col = "novelty" if "novelty" in df.columns else (df.columns[-1] if len(df.columns) > 0 else None)

    exclude_cols = {slices_col, pred_col, novelty_col, "spacegroup", "spacegroup_number", "poscar", "crystal_system", None}
    target_candidates = [c for c in df.columns if c not in exclude_cols]
    target_col = "prop_0" if "prop_0" in df.columns else ("eform_target" if "eform_target" in df.columns else None)
    if target_col is None:
        target_col = target_candidates[0] if target_candidates else df.columns[0]
    return target_col, pred_col, novelty_col


def prepare_data(results_file, training_file):
    """
    Read results/training and organize hist data for all materials vs novel materials.
    """
    results_df = pd.read_csv(results_file)
    trainingset = pd.read_csv(training_file, header=0)

    target_col, pred_col, novelty_col = _detect_result_columns(results_df)
    if pred_col is None or novelty_col is None:
        raise ValueError(f"Cannot detect prediction/novelty columns in '{results_file}'.")
    header_values = results_df[target_col].tolist()
    data_values = pd.to_numeric(results_df[pred_col], errors="coerce").tolist()

    if trainingset.shape[1] > 1:
        trainingset_values = trainingset.iloc[:, 1].tolist()
    else:
        trainingset_values = []
        print("Warning: Training file does not have formation energy column. Skipping training data in plot.")

    novelty_values = pd.to_numeric(results_df[novelty_col], errors="coerce").fillna(0).astype(int).tolist()

    data_dict = {}
    for header, value, novelty in zip(header_values, data_values, novelty_values):
        if pd.isna(value):
            continue
        if header not in data_dict:
            data_dict[header] = {"all": [], "novel": []}
        data_dict[header]["all"].append(value)
        if novelty == 1:
            data_dict[header]["novel"].append(value)

    sorted_keys = sorted(data_dict.keys(), reverse=True)
    return data_dict, sorted_keys, trainingset_values


def create_dataframe(data_dict, sorted_keys, trainingset_values, data_type="all"):
    """
    Creates a pandas DataFrame from the grouped data dictionary.
    """
    df = pd.DataFrame(
        {
            k: pd.Series(data_dict[k][data_type], index=range(len(data_dict[k][data_type])))
            for k in sorted_keys
        }
    )
    df = pd.concat([df, pd.Series(trainingset_values, name="training_dataset")], axis=1)
    return df


def calculate_decode_metrics(results_file, tolerance=0.2, total_samples=0):
    """
    Calculate MAPE / within-tolerance / novelty metrics grouped by target value.
    """
    df = pd.read_csv(results_file)
    target_col, pred_col, novelty_col = _detect_result_columns(df)
    if pred_col is None or novelty_col is None:
        return pd.DataFrame(), {
            "decoded_count": 0,
            "novel_count": 0,
            "novelty_percent": 0.0,
            "mape": np.nan,
            "within_tolerance_ratio": np.nan,
            "target_groups": 0,
            "total_samples": total_samples,
            "target_column": target_col,
            "prediction_column": pred_col,
        }

    work_df = df[[target_col, pred_col, novelty_col]].copy()
    work_df[target_col] = pd.to_numeric(work_df[target_col], errors="coerce")
    work_df[pred_col] = pd.to_numeric(work_df[pred_col], errors="coerce")
    work_df[novelty_col] = pd.to_numeric(work_df[novelty_col], errors="coerce").fillna(0).astype(int)
    work_df = work_df.dropna(subset=[target_col, pred_col])

    if work_df.empty:
        return pd.DataFrame(), {
            "decoded_count": 0,
            "novel_count": 0,
            "novelty_percent": 0.0,
            "mape": np.nan,
            "within_tolerance_ratio": np.nan,
            "target_groups": 0,
            "total_samples": total_samples,
            "target_column": target_col,
            "prediction_column": pred_col,
        }

    result_rows = []
    grouped = work_df.groupby(target_col, dropna=False)
    for target_value, group in grouped:
        preds = group[pred_col].to_numpy(dtype=float)
        novelty = group[novelty_col].to_numpy(dtype=int)
        decode_count = len(preds)
        novel_count = int((novelty == 1).sum())

        if decode_count > 0 and abs(target_value) > 1e-12:
            mape = float(np.mean(np.abs((preds - target_value) / target_value)) * 100.0)
        else:
            mape = np.nan
        if decode_count > 0:
            proportion = float(np.mean(np.abs(preds - target_value) <= tolerance))
            novelty_percent = float(novel_count / decode_count * 100.0)
        else:
            proportion = np.nan
            novelty_percent = 0.0

        result_rows.append(
            {
                "target": target_value,
                "decode_count": decode_count,
                "novel_count": novel_count,
                "novelty_percent": novelty_percent,
                "mape": mape,
                "within_tolerance_ratio": proportion,
            }
        )

    metrics_df = pd.DataFrame(result_rows).sort_values(by="target", ascending=False).reset_index(drop=True)

    decoded_count = int(len(work_df))
    novel_count = int((work_df[novelty_col] == 1).sum())
    novelty_percent = float(novel_count / decoded_count * 100.0) if decoded_count > 0 else 0.0

    nonzero_target = work_df[np.abs(work_df[target_col]) > 1e-12]
    if nonzero_target.empty:
        overall_mape = np.nan
    else:
        overall_mape = float(
            np.mean(np.abs((nonzero_target[pred_col] - nonzero_target[target_col]) / nonzero_target[target_col])) * 100.0
        )
    overall_within_tol = float(np.mean(np.abs(work_df[pred_col] - work_df[target_col]) <= tolerance))

    overall_metrics = {
        "decoded_count": decoded_count,
        "novel_count": novel_count,
        "novelty_percent": novelty_percent,
        "mape": overall_mape,
        "within_tolerance_ratio": overall_within_tol,
        "target_groups": int(len(metrics_df)),
        "total_samples": total_samples,
        "target_column": target_col,
        "prediction_column": pred_col,
    }
    return metrics_df, overall_metrics


def format_metrics_text(metrics_df, overall_metrics, tolerance):
    """
    Convert metrics to a compact monospace text block for embedding in PNG.
    """
    def _fmt_pct(value):
        if value is None or pd.isna(value):
            return "N/A"
        return f"{value:.1f}%"

    def _fmt_ratio(value):
        if value is None or pd.isna(value):
            return "N/A"
        return f"{value * 100.0:.1f}%"

    lines = [
        "Decode Metrics Summary",
        "----------------------",
        f"Decoded: {overall_metrics.get('decoded_count', 0)}",
        f"Novel: {overall_metrics.get('novel_count', 0)} ({_fmt_pct(overall_metrics.get('novelty_percent'))})",
        f"Overall MAPE: {_fmt_pct(overall_metrics.get('mape'))}",
        f"|error| <= {tolerance}: {_fmt_ratio(overall_metrics.get('within_tolerance_ratio'))}",
        f"Target groups: {overall_metrics.get('target_groups', 0)}",
        f"Input samples: {overall_metrics.get('total_samples', 0)}",
        "",
        "By target",
        "target      N   novel  nov%   MAPE   tol%",
    ]

    max_rows = 24
    shown = metrics_df.head(max_rows)
    for _, row in shown.iterrows():
        target_str = f"{row['target']:.3g}" if pd.notna(row["target"]) else "N/A"
        line = (
            f"{target_str:>7} "
            f"{int(row['decode_count']):>6} "
            f"{int(row['novel_count']):>7} "
            f"{_fmt_pct(row['novelty_percent']):>6} "
            f"{_fmt_pct(row['mape']):>7} "
            f"{_fmt_ratio(row['within_tolerance_ratio']):>6}"
        )
        lines.append(line)
    if len(metrics_df) > max_rows:
        lines.append(f"... ({len(metrics_df) - max_rows} more targets)")
    return "\n".join(lines)


def plot_combined_histograms(all_data, novel_data, output_file, metrics_text=""):
    """
    Plot histogram panels and append a right-side text panel with decode metrics.
    """
    num_cols = len(all_data.columns)
    fig, axs = plt.subplots(num_cols, 2, figsize=(14, 3 * max(num_cols, 1)), sharex=True)
    bins = np.linspace(-6, 0, 50)

    if num_cols == 1:
        axs = axs.reshape(1, -1)

    for i, col_name in enumerate(all_data.columns):
        for j, (data, title) in enumerate(zip([all_data, novel_data], ["All Materials", "Novel Materials"])):
            color = "violet" if col_name == "training_dataset" else "lightblue"
            numeric_data = pd.to_numeric(data[col_name], errors="coerce").dropna()

            if len(numeric_data) > 0:
                axs[i, j].hist(
                    numeric_data,
                    bins=bins,
                    density=True,
                    color=color,
                    edgecolor="black",
                    alpha=0.7,
                )
                mean_val = numeric_data.mean()
                axs[i, j].axvline(mean_val, color="red", linestyle="--", linewidth=1)
                axs[i, j].text(
                    mean_val,
                    axs[i, j].get_ylim()[1] * 0.9,
                    f"{mean_val:.2f}",
                    color="red",
                    fontsize=6,
                    ha="left",
                )

            axs[i, j].text(
                0.05,
                0.95,
                f"{col_name}\n{title}",
                transform=axs[i, j].transAxes,
                fontsize=8,
                va="top",
            )

    for ax in axs.flat:
        ax.set_xlim(-6, 0)
        ax.tick_params(axis="both", which="major", labelsize=6)
        ax.xaxis.set_major_locator(ticker.MultipleLocator(1))

    fig.text(0.35, 0.02, "Formation Energy (eV/atom)", ha="center", va="center", fontsize=10)
    fig.text(0.02, 0.5, "Density", ha="center", va="center", rotation="vertical", fontsize=10)
    plt.subplots_adjust(hspace=0, wspace=0.12, right=0.72)

    if metrics_text:
        metrics_ax = fig.add_axes([0.735, 0.08, 0.255, 0.84])
        metrics_ax.axis("off")
        metrics_ax.text(
            0.0,
            1.0,
            metrics_text,
            va="top",
            ha="left",
            fontsize=8,
            family="monospace",
            bbox=dict(facecolor="#f7f7f7", edgecolor="#cccccc", boxstyle="round,pad=0.5"),
        )

    plt.savefig(output_file, dpi=300, bbox_inches="tight")
    return fig


def main():
    parser = argparse.ArgumentParser(
        description="Process and analyze structure data.",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    parser.add_argument("--input_csv", type=str, help="Path to the input CSV file to be processed.")
    parser.add_argument(
        "--structure_json_for_novelty_check",
        type=str,
        help="Path to the JSON file containing CIFs (structure database).",
    )
    parser.add_argument(
        "--training_file",
        type=str,
        help="Path to the training CSV file (e.g., 'train_data.csv').",
    )
    parser.add_argument(
        "--output_csv",
        type=str,
        default="results.csv",
        help="Path for the output CSV file to be generated.",
    )
    parser.add_argument("--threads", type=int, default=8, help="Number of threads to use for processing.")
    parser.add_argument("--tolerance", type=float, default=0.2, help="Tolerance threshold for metric calculation.")
    parser.add_argument(
        "--total_samples",
        type=int,
        default=0,
        help="Total generated sample count for report display. <=0 means infer from input_csv.",
    )
    parser.add_argument(
        "--metrics_output_csv",
        type=str,
        default="metrics_summary.csv",
        help="Output path for per-target metrics CSV.",
    )
    parser.add_argument(
        "--figure_output",
        type=str,
        default="combined_results.png",
        help="Output figure path.",
    )

    args = parser.parse_args()

    if not os.path.isfile(args.input_csv):
        print(f"Error: The input CSV file '{args.input_csv}' does not exist.")
        exit(1)
    if not os.path.isfile(args.structure_json_for_novelty_check):
        print(f"Error: The structure JSON file '{args.structure_json_for_novelty_check}' does not exist.")
        exit(1)
    if not os.path.isfile(args.training_file):
        print(f"Error: The training file '{args.training_file}' does not exist.")
        exit(1)

    total_samples = args.total_samples if args.total_samples > 0 else _infer_total_samples_from_input(args.input_csv)

    process_data(args.input_csv, args.output_csv, args.structure_json_for_novelty_check, args.threads)

    data_dict, sorted_keys, trainingset_values = prepare_data(args.output_csv, args.training_file)
    all_materials_df = create_dataframe(data_dict, sorted_keys, trainingset_values, "all")
    novel_materials_df = create_dataframe(data_dict, sorted_keys, trainingset_values, "novel")

    metrics_df, overall_metrics = calculate_decode_metrics(
        results_file=args.output_csv,
        tolerance=args.tolerance,
        total_samples=total_samples,
    )
    metrics_df.to_csv(args.metrics_output_csv, index=False)
    print(f"Saved per-target metrics to '{args.metrics_output_csv}'.")

    metrics_text = format_metrics_text(metrics_df, overall_metrics, args.tolerance)
    plot_combined_histograms(all_materials_df, novel_materials_df, args.figure_output, metrics_text=metrics_text)
    print(f"Saved figure to '{args.figure_output}'.")
    print(metrics_text)



if __name__ == "__main__":
    main()
