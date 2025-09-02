# coding = utf-8

import os
import sys
import json
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
parser.add_argument("--RL", action="store_true", default=False)
parser.add_argument("--mission_id", type=str, default="hchdvbjsjci-bhjdsbcj")
args = parser.parse_args()

def replace_last_occurrence(s, old, new):  
    li = s.rsplit(old, 1)  
    if len(li) == 2:  
        return new.join(li)  
    return s  

def get_decompiled_files(decompiled_dir, outputDirs):
    ret_box = []
    for file in os.listdir(decompiled_dir):
        filePath = os.path.join(decompiled_dir, file)
        file_detail = {}
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
    if args.mission_id:
        project, cve_id, rl, mission_id  = args.project, args.cve_id, args.RL, args.mission_id
    else:
        project, cve_id, rl  = args.project, args.cve_id, args.RL
    if rl:
        pseudo_path = os.path.join("../../data_detect/data/", mission_id, project, cve_id, "pseudo")
        outputDir = os.path.join("../../data_detect/data/", mission_id, project, cve_id, "graph")
    else:
        pseudo_path = os.path.join("../../data/", project, cve_id, "pseudo")
        outputDir = os.path.join("../../data/", project, cve_id, "graph")
        
    tasks = get_decompiled_files(pseudo_path, outputDir)

    res = process_map(gen_and_save_graph, tasks, max_workers=10, chunksize=4)

    time_end = time.time()
    time_cost = time_end = time_start
