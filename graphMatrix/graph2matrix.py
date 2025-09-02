#!/usr/bin/env python3
# -*- coding:utf-8 -*-

import glob
import itertools
import json
import os
import os.path as osp
import pprint
from datetime import datetime
import numpy as np
import pygraphviz as pgv
from scipy import sparse
from loguru import logger
from utils import find_files, multi_run, normalize_edges, normalize_nodes_dot
import config as cfg
from argparse import ArgumentParser

parser = ArgumentParser("change graph to node and adj matrix.")
parser.add_argument("--project", type=str, default="curl")
parser.add_argument("--cve_id", type=str, default="CVE-2021-22901")
args = parser.parse_args()

log_name = "{}_{}.log".format(__file__[:-3], datetime.now().strftime("%Y%m%d"))
log_path = osp.join(cfg.LOG_DIR, log_name)
logger.add(log_path, format="{time} | {message}")

def filter_dot_file(path):
    if not osp.isfile(path):
        return False
    fname = osp.basename(path)
    # print("fname",fname)
    if fname in cfg.GRAPH_DOT_FILES.keys():
        return True
    return False

def get_tag(fname):
    '''
        Extract labels (fix,vul) from file names
    '''
    if len(fname.split("_")) >= 5:
        tag = fname.split("_")[4]
        if tag in ['vul', 'fix']:
            return tag
    else:
        return "unknown"

def get_prefix(path, dataset, outdir):
    if dataset[-1] != os.sep:
        dataset += os.sep
    if outdir[-1] != os.sep:
        outdir += os.sep

    prefix = osp.dirname(path.replace(dataset, outdir))
    return prefix

def split2task(files, dataset, outdir):
    func_dots = {}
    for path in files:
        dirname = osp.dirname(path)
        if dirname not in func_dots:
            func_dots[dirname] = []
        func_dots[dirname].append(path)
    tasks = []
    for key, value in func_dots.items():
        dots = [osp.basename(x) for x in value]
        graph_dots = cfg.GRAPH_DOT_FILES.keys()
        if set(dots) ^ set(graph_dots):
            logger.error("[!] Graphs of {} must be {}, but it is {}", key, graph_dots, dots)
            continue
        tag = get_tag(osp.basename(key))
        prefix = get_prefix(key, dataset, outdir)
        tasks.append(
            (key, prefix, tag)
        )
    return tasks

def json2matrix_dot(path, graph_types=cfg.GRAPH_TYPES):
    select_dot = list(filter(lambda x: x[1] in graph_types, cfg.GRAPH_DOT_FILES.items()))
    nodes = {}
    adj_edges = dict([(x, {}) for x in graph_types])
    for dot_name, graph_type in select_dot:
        dot_path = osp.join(path, dot_name)
        G = pgv.AGraph(dot_path)
        
        label_key = cfg.DOT_NODE_ATTR
        G.node_attr ['label']
        if label_key not in list(G.node_attr):
            logger.error("[!] Cannot find {} in graph {}", label_key, G)
            return False
        for node in G.nodes():
            if cfg.GRAPH_MODE == "disjoint":
                node_id = "{}_{}".format(graph_type, node.name)
            elif cfg.GRAPH_MODE == 'single':
                node_id = node.name

            label_raw = node.attr[label_key]
            if label_raw:
                if label_raw[0] == "(":
                    label_raw = label_raw[1:]
                if label_raw[-1] == ")":
                    label_raw = label_raw[:-1]
                label = label_raw.split(",")[0]
                nodes[node_id] = label
      
        for in_edge, out_edge in G.edges():
            if cfg.GRAPH_MODE == "disjoint":
                in_edge_id = "{}_{}".format(graph_type, in_edge.name)
                out_edge_id = "{}_{}".format(graph_type, out_edge.name)
            elif cfg.GRAPH_MODE == 'single':
                in_edge_id = in_edge.name
                out_edge_id = out_edge.name
            if in_edge_id not in adj_edges[graph_type]:
                adj_edges[graph_type][in_edge_id] = []
            if out_edge_id not in adj_edges[graph_type][in_edge_id]:
                adj_edges[graph_type][in_edge_id].append(out_edge_id)

    node_matrix, node_dic = normalize_nodes_dot(nodes)
    adj_matrix = normalize_edges(adj_edges, graph_types, node_dic)

    return node_matrix, adj_matrix

def run_graph2matrix_dot(task, iindex=1):
    path, prefix, tag = task
    name = osp.basename(path)
    tags = ["vul", "fix"]
    types = ["node", "adj"]

    assert tag in ["vul", "fix"]
    if not osp.isdir(prefix):
        if osp.exists(prefix):
            logger.error("[!] prefix ({}) already exists!", prefix)
            return False
        os.makedirs(prefix, exist_ok=True)
        for _tag, _type in itertools.product(tags, types):
            os.makedirs(osp.join(prefix, _tag, _type), exist_ok=True)

    node_matrix, adj_matrix = json2matrix_dot(path)
    
    if not node_matrix.any():
        logger.error("[!] Failed to generate node/adjacency matrix for graph ({})")
        return False
    node_matrix_path = osp.join(prefix, tag, "node", name + "_node")
    adj_matrix_path = osp.join(prefix, tag, "adj", name + "_adj")
    if os.path.exists(node_matrix_path):
        print("Removing existing directory:", node_matrix_path)
        os.rmdir(node_matrix_path)
    os.makedirs(node_matrix_path)
    if os.path.exists(adj_matrix_path):
        print("Removing existing directory:", adj_matrix_path)
        os.rmdir(adj_matrix_path)
    os.makedirs(adj_matrix_path)

    logger.info(
        '[+] Generate node/adjacency matrix successfully (Graph: "{}", Node Matrix: "{}.npy", Adjacency Matrix: "{}.npy")',
        path, node_matrix_path, adj_matrix_path
    )
    adj_matrix = adj_matrix.astype("int8")
    np.savez_compressed(node_matrix_path, node_matrix)
    np.savez_compressed(adj_matrix_path, adj_matrix)

    if os.path.exists(node_matrix_path):
        print("Removing existing directory:", node_matrix_path)
        os.rmdir(node_matrix_path)
    
    if os.path.exists(adj_matrix_path):
        print("Removing existing directory:", adj_matrix_path)
        os.rmdir(adj_matrix_path)
    
    return True

def graph2matrix_dot(dataset, outdir):
    dot_files = find_files(dataset, filter_func=filter_dot_file)
    tasks = split2task(dot_files, dataset, outdir)
    if not osp.isdir(outdir):
        if osp.exists(outdir):
            logger.error("[!] outdir ({}) already exists!", outdir)
            return False
        os.makedirs(outdir, exist_ok=True)

    for task in tasks:
        run_graph2matrix_dot(task, 1)


def main():
    project, cve_id = args.project, args.cve_id
    dataset = os.path.join("../data", project, cve_id, "graph")
    outdir = os.path.join("../data/", project, cve_id, "matrix_" + cfg.GRAPH_MODE)
     
    dataset = dataset.replace("graph", "dividegraph")
    graph2matrix_dot(dataset, outdir)

if __name__ == "__main__":
    main()
