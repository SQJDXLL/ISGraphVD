# coding = utf-8

import os
import sys
import json
# from tqdm import tqdm
# from multiprocessing import Pool
from tqdm import tqdm
from tqdm.contrib.concurrent import process_map
from painter import get_graph
from functools import partial
import random
import time
from argparse import ArgumentParser

parser = ArgumentParser("process pseudo for construct graphs.")
parser.add_argument("--project", type=str, default="curl")
parser.add_argument("--cve_id", type=str, default="CVE-2021-22901")
parser.add_argument("--RL", type=bool, default=False)
args = parser.parse_args()

def replace_last_occurrence(s, old, new):  
    # 找到最后一个old在s中的位置  
    li = s.rsplit(old, 1)  
    # 如果找到了，就替换它  
    if len(li) == 2:  
        return new.join(li)  
    # 如果没有找到，就返回原字符串  
    return s  

def get_decompiled_files(decompiled_dir, outputDirs):
    ret_box = []
    for file in os.listdir(decompiled_dir):
        filePath = os.path.join(decompiled_dir, file)
        file_detail = {}
        # file_detail['filename'] = file.replace(".c", "")
        file_detail['filename'] = replace_last_occurrence(file, '.c', '')  
        file_detail['path'] = filePath
        ret_box.append((file_detail, outputDirs))
    return ret_box


def gen_and_save_graph(task: tuple):
    decompiled_f,  output_dir = task
    stored_dir = '{}/{}'.format(output_dir, decompiled_f['filename'])
    return get_graph(decompiled_f['path'], stored_dir)


if __name__ == '__main__':
    time_start = time.time()
    project, cve_id, rl = args.project, args.cve_id, args.RL
    if rl:
        pseudo_path = os.path.join("../../realWorld/data/", project, cve_id, "pseudo")
        outputDir = os.path.join("../../realWorld/data/", project, cve_id, "graph")
    else:
        pseudo_path = os.path.join("../../data/", project, cve_id, "pseudo")
        outputDir = os.path.join("../../data/", project, cve_id, "graph")
    tasks = get_decompiled_files(pseudo_path, outputDir)

    res = process_map(gen_and_save_graph, tasks, max_workers=10, chunksize=4)

    time_end = time.time()
    time_cost = time_end = time_start
    print("time_cost_pdg", time_cost)