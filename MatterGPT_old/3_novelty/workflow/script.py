# -*- coding: utf-8 -*-
# Yan Chen 2023.10
# yanchen@xjtu.edu.com

import os
import csv
import pickle
import gc
from collections import defaultdict

from pymatgen.core.structure import Structure
from pymatgen.analysis.structure_matcher import StructureMatcher, ElementComparator

os.environ["OMP_NUM_THREADS"] = "1"

# 设置公差
ltol = 0.2
stol = 0.3
angle_tol = 5

def main():
    # 0) 从环境变量里读出本次要加载的数据库分块路径
    db_chunk_path = os.environ.get("DB_CHUNK_FILE", "")
    if not db_chunk_path or not os.path.exists(db_chunk_path):
        print("Error: DB_CHUNK_FILE not set or file not found:", db_chunk_path)
        return

    # 1) 读取本次需要匹配的行(temp_splited.csv)
    if not os.path.exists("temp_splited.csv"):
        print("No temp_splited.csv found.")
        return

    with open("temp_splited.csv", 'r', encoding='utf-8') as fin:
        lines = fin.readlines()
    if not lines:
        print("temp_splited.csv is empty, nothing to do.")
        return

    # 2) 尝试解析并构造 poscar 结构；若出错，则写入 suspect_rows.csv
    query_data = []
    suspect_indices = set()  # 用于记录怀疑的行的索引
    for idx, line in enumerate(lines):
        row = line.strip().split(",")
        try:
            # 假设 poscar 在最后一列
            stru = Structure.from_str(row[-1].replace('\\n','\n'), fmt='poscar')
            query_data.append((line, stru))
        except Exception as e:
            # 解析poscar出错，记录到 suspect_rows.csv，并直接标记为“无效”或暂不匹配
            with open("suspect_rows.csv", "a", encoding='utf-8', newline='') as fsus:
                writer_suspect = csv.writer(fsus)
                writer_suspect.writerow(
                    row + ["POSCAR parse error", repr(e)]
                )
            # 标记为suspect
            suspect_indices.add(idx)
            continue

    if not query_data:
        print("All lines invalid or empty after parsing.")
        return

    # 3) 加载本分块的数据库
    print(f"Loading DB chunk {db_chunk_path}")
    with open(db_chunk_path, 'rb') as fdb:
        sub_database = pickle.load(fdb)
    print(f"Loaded {len(sub_database)} structures in this chunk.")

    # 3.1) 按组分分类
    database_by_comp = defaultdict(list)
    for entry in sub_database:
        struct = entry[0]
        comp_str = str(struct.composition)
        database_by_comp[comp_str].append(struct)

    # 4) 创建结构匹配器
    sm = StructureMatcher(
        ltol=ltol,
        stol=stol,
        angle_tol=angle_tol,
        primitive_cell=True,  # 不进行原胞转换
        scale=True,            # 根据需要保留或移除此选项
        attempt_supercell=False,
        comparator=ElementComparator()
    )

    # 5) 匹配过程
    matched_indices = set()  # 哪些条目已成功匹配
    for idx, (line_str, q_struct) in enumerate(query_data):
        # 直接使用原始结构进行匹配，无需对称性处理
        comp_q = str(q_struct.composition)
        if comp_q not in database_by_comp:
            # 该组分在本分块里没有候选，直接跳过
            continue

        candidate_list = database_by_comp[comp_q]
        is_matched = False

        # 逐个 candidate 尝试 fit
        for candidate in candidate_list:
            try:
                if sm.fit(candidate, q_struct):
                    # 一旦匹配成功，写 result2.csv (novel=0)
                    row0 = line_str.strip().split(",")
                    with open("result2.csv", "a", encoding='utf-8', newline='') as fout:
                        writer = csv.writer(fout)
                        writer.writerow(row0 + ["0"])
                    matched_indices.add(idx)
                    is_matched = True
                    break
            except Exception as e_fit:
                # 如果 sm.fit 抛异常，不中断其它 candidate 的匹配
                # 而是记录到 suspect_rows.csv，以备后续复查
                row0 = line_str.strip().split(",")
                with open("suspect_rows.csv", "a", encoding='utf-8', newline='') as fsus:
                    writer_suspect = csv.writer(fsus)
                    # 可以把 candidate 的某些信息如 composition 也记录一下
                    writer_suspect.writerow(
                        row0 + [
                            f"CandidateComp={candidate.composition}",
                            f"sm.fit error: {repr(e_fit)}"
                        ]
                    )
                # 标记为suspect
                suspect_indices.add(idx)
                # 继续尝试下一个 candidate
                continue

        # 释放资源
        del q_struct

    # 6) 把未匹配且未被标记为suspect的行写到 unmatched_tmp.csv，给下一个分块(或最终判为 novel=1)
    unmatched_count = 0
    with open("unmatched_tmp.csv", "w", encoding='utf-8') as fout:
        for idx, (line_str, _) in enumerate(query_data):
            if idx not in matched_indices and idx not in suspect_indices:
                fout.write(line_str)
                unmatched_count += 1

    print(f"Matched {len(matched_indices)} lines in this chunk, still {unmatched_count} unmatched left.")

    # 7) 清理内存
    del sub_database
    del database_by_comp
    gc.collect()

if __name__ == "__main__":
    main()

