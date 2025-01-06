# 0_run.py

import os
import sys
import csv
import pickle
import gc

def main():
    # 0) 若已存在 result2.csv，删掉，避免干扰
    if os.path.exists("result2.csv"):
        os.remove("result2.csv")
        
    # 1) 读取需要进一步匹配的条目（temp_non_novel.csv）
    input_csv = "temp.csv"
    if not os.path.exists(input_csv):
        print(f"No {input_csv} found, nothing to do.")
        return

    with open(input_csv, 'r', encoding='utf-8') as f:
        lines = f.readlines()

    if not lines:
        print(f"{input_csv} is empty, nothing to do.")
        return

    # 2) 把这个大列表 lines 作为“待匹配数据”放到 unmatched_lines 变量里
    unmatched_lines = lines

    # 3) 查看数据库分块目录 split_db 下有多少个 pkl 文件
    #    这些文件是由 1_run.py 或者别的脚本提前切好的
    db_dir = "../split_db"  # 也可能是当前目录 "./split_db"，看你的目录结构
    if not os.path.exists(db_dir):
        print(f"Error: {db_dir} does not exist! You must split the DB beforehand.")
        return

    # 按名称排序，比如 structure_database_0.pkl, structure_database_1.pkl ...
    db_files = sorted(
        [f for f in os.listdir(db_dir) if f.startswith("structure_database_") and f.endswith(".pkl")],
        key=lambda x: int(x.split("_")[-1].split(".")[0])  # 根据最后的数字排序
    )
    if not db_files:
        print(f"No database chunks found in {db_dir}")
        return

    # 4) 对于每一个数据库分块，调用 script.py 去做匹配
    #    脚本的逻辑：1) 载入该分块；2) 对 unmatched_lines 做匹配；3) 把匹配到的写 result2.csv；4) 把未匹配的行写 unmatched_tmp.csv
    for i, dbf in enumerate(db_files):
        print(f"==> Matching with DB chunk: {dbf}, current unmatched lines = {len(unmatched_lines)}")

        # 如果没有待匹配的行了，就可以提前结束
        if not unmatched_lines:
            break

        # 4.1) 先将 unmatched_lines 写到一个临时文件 temp_splited.csv，让 script.py 去读
        with open("temp_splited.csv", "w", encoding="utf-8") as fout:
            fout.writelines(unmatched_lines)

        # 4.2) 调用 script.py，同时通过环境变量或者命令行参数传递哪个数据库分块
        #      这里演示环境变量 DB_CHUNK_FILE=../split_db/structure_database_i.pkl
        db_chunk_path = os.path.join(db_dir, dbf)
        os.environ["DB_CHUNK_FILE"] = db_chunk_path

        # 让 script.py 内部自己去读 DB_CHUNK_FILE 环境变量
        # script.py 处理完之后，会把未匹配到的行写到 unmatched_tmp.csv
        # 已匹配到的行会直接附加写到 result2.csv
        cmd = "python -B script.py"
        os.system(cmd)

        # 4.3) 读取 script.py 生成的 unmatched_tmp.csv，更新 unmatched_lines
        #      script.py 用 unmatched_tmp.csv 来存储“没匹配到”的条目
        if os.path.exists("unmatched_tmp.csv"):
            with open("unmatched_tmp.csv", "r", encoding="utf-8") as f:
                unmatched_lines = f.readlines()
            os.remove("unmatched_tmp.csv")
        else:
            # 如果没有 unmatched_tmp.csv，说明全部匹配了？
            unmatched_lines = []

        # 强制一次 gc
        gc.collect()

    # 5) 当所有数据库分块都处理完，如果此时 unmatched_lines 还不为空，
    #    则它们就是“真正的新结构”(novel=1)
    if unmatched_lines:
        print(f"{len(unmatched_lines)} lines remain unmatched; write them as novel=1 to result2.csv")
        with open("result2.csv", "a", encoding="utf-8", newline='') as fout:
            writer = csv.writer(fout)
            for line in unmatched_lines:
                row = line.strip().split(",")
                # row + [novel=1]
                # 这里要注意 input.csv 的列数是不是固定，比如 [bandgap_target, SLICES, POSCAR]
                writer.writerow(row + ["1"])

    # 6) 最后将 result2.csv 改名为 result.csv 以示完成
    if os.path.exists("result2.csv"):
        os.rename("result2.csv", "result.csv")

if __name__ == "__main__":
    main()

