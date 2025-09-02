#!/usr/bin/env python3
# -*- coding:utf-8 -*-

import glob
import itertools
import json
import os
import os.path as osp
import shutil
import pprint
from datetime import datetime
import numpy as np
import pygraphviz as pgv
from scipy import sparse
from loguru import logger
from utils import find_files, multi_run, normalize_nodes_dot, normalize_edges_gat, normalize_edges_gmn_highlight, normalize_edges_gmn
import config as cfg
from argparse import ArgumentParser

parser = ArgumentParser("change graph to node and adj matrix.")
parser.add_argument("--project", type=str, default="curl")
parser.add_argument("--cve_id", type=str, default="CVE-2021-22901")
parser.add_argument("--hl", action="store_true", default=False)
parser.add_argument("--RL", action="store_true", default=False)
parser.add_argument("--mission_id", type=str, default="hchdvbjsjci-bhjdsbcj")
args = parser.parse_args()

log_name = "{}_{}.log".format(__file__[:-3], datetime.now().strftime("%Y%m%d"))
log_path = osp.join(cfg.LOG_DIR, log_name)
logger.add(log_path, format="{time} | {message}")

def filter_dot_file(path):
    if not osp.isfile(path):
        return False
    fname = osp.basename(path)
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

def json2matrix_dot(path, hl, graph_types=cfg.GRAPH_TYPES):
    select_dot = list(filter(lambda x: x[1] in graph_types, cfg.GRAPH_DOT_FILES.items()))
    nodes = {}
    if hl:
        adj_edges = {"True":{},"False":{}}
        adj_edges_in_true = dict([(x, {}) for x in graph_types])
        adj_edges_in_false = dict([(x, {}) for x in graph_types])

        adj_edges["True"] = adj_edges_in_true
        adj_edges["False"] = adj_edges_in_false
    else:
        adj_edges = dict([(x, {}) for x in graph_types])
    
    for dot_name, graph_type in select_dot:
        dot_path = osp.join(path, dot_name)
        G = pgv.AGraph(dot_path)
        edges = G.edges()
        if edges:
        
            label_key = cfg.DOT_NODE_ATTR
            G.node_attr['label']
            if label_key not in list(G.node_attr):
                logger.error("[!] Cannot find {} in graph {}", label_key, G)
                return False
            for node in G.nodes():
                if cfg.GRAPH_MODE == "disjoint":
                    node_id = "{}_{}".format(graph_type, node.name)
                elif cfg.GRAPH_MODE == 'single' or cfg.GRAPH_MODE == 'new_disjoint':
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
                    if hl:
                        in_edge_id = "{}_{}".format(graph_type, in_edge.name)
                        out_edge_id = "{}_{}".format(graph_type, out_edge.name)
                        edge_color = G.get_edge(in_edge, out_edge).attr['color']
                        
                        if edge_color == "green": 
                            if in_edge_id not in adj_edges["False"][graph_type]:
                                adj_edges["False"][graph_type][in_edge_id] = []

                            if out_edge_id not in adj_edges["False"][graph_type][in_edge_id]:
                                adj_edges["False"][graph_type][in_edge_id].append(out_edge_id)

                        elif edge_color == "red":

                            if in_edge_id not in adj_edges["True"][graph_type]:
                                adj_edges["True"][graph_type][in_edge_id] = []

                            if out_edge_id not in adj_edges["True"][graph_type][in_edge_id]:
                                adj_edges["True"][graph_type][in_edge_id].append(out_edge_id)

                    else:
                        in_edge_id = "{}_{}".format(graph_type, in_edge.name)
                        out_edge_id = "{}_{}".format(graph_type, out_edge.name)
                        if in_edge_id not in adj_edges[graph_type]:
                            adj_edges[graph_type][in_edge_id] = []
                        if out_edge_id not in adj_edges[graph_type][in_edge_id]:
                            adj_edges[graph_type][in_edge_id].append(out_edge_id)
                    
                else:
                    if hl:
                        in_edge_id = in_edge.name
                        out_edge_id = out_edge.name
                        edge_color = G.get_edge(in_edge, out_edge).attr['color']

                        if edge_color == "green": 
                            if in_edge_id not in adj_edges["False"][graph_type]:
                                adj_edges["False"][graph_type][in_edge_id] = []

                            if out_edge_id not in adj_edges["False"][graph_type][in_edge_id]:
                                adj_edges["False"][graph_type][in_edge_id].append(out_edge_id)

                        elif edge_color == "red":

                            if in_edge_id not in adj_edges["True"][graph_type]:
                                adj_edges["True"][graph_type][in_edge_id] = []

                            if out_edge_id not in adj_edges["True"][graph_type][in_edge_id]:
                                adj_edges["True"][graph_type][in_edge_id].append(out_edge_id)

                    else:
                        in_edge_id = in_edge.name
                        out_edge_id = out_edge.name
                        if in_edge_id not in adj_edges[graph_type]:
                            adj_edges[graph_type][in_edge_id] = []
                        if out_edge_id not in adj_edges[graph_type][in_edge_id]:
                            adj_edges[graph_type][in_edge_id].append(out_edge_id)
    node_matrix, node_dic = normalize_nodes_dot(nodes)
    
    if cfg.MODEL == 'GAT':
        logger.debug("[*] GAT mode: normalize_edges = normalize_edges_gat")
        normalize_edges = normalize_edges_gat
    elif cfg.MODEL == "GMN":
        if hl:
            logger.debug("[*] GMN mode: normalize_edges = normalize_edges_gmn_highlight")
            normalize_edges = normalize_edges_gmn_highlight
        else:
            logger.debug("[*] GMN mode: normalize_edges = normalize_edges_gmn")
            normalize_edges = normalize_edges_gmn
    adj_matrix = normalize_edges(adj_edges, graph_types, node_dic)

    return node_matrix, adj_matrix

def run_graph2matrix_dot(task, hl, rl, iindex=1):

    path, prefix, tag = task
    name = osp.basename(path)
    tags = ["vul", "fix"]
    types = ["node", "adj"]

    if not rl:
        assert tag in ["vul", "fix"]
    if not osp.isdir(prefix):
        if osp.exists(prefix):
            logger.error("[!] prefix ({}) already exists!", prefix)
            return False
        os.makedirs(prefix, exist_ok=True)
        if rl:
            for _type in itertools.product(types):
                os.makedirs(osp.join(prefix, _type), exist_ok=True)
        else:
            for _tag, _type in itertools.product(tags, types):
                os.makedirs(osp.join(prefix, _tag, _type), exist_ok=True)

    node_matrix, adj_matrix = json2matrix_dot(path, hl)
    
    if not node_matrix.any():
        logger.error("[!] Failed to generate node/adjacency matrix for graph ({})")
        return False
    if rl:
        node_matrix_path = osp.join(prefix, "node", name + "_node")
        adj_matrix_path = osp.join(prefix, "adj", name + "_adj")
    else:
        node_matrix_path = osp.join(prefix, tag, "node", name + "_node")
        adj_matrix_path = osp.join(prefix, tag, "adj", name + "_adj")
    if os.path.exists(node_matrix_path):
        os.rmdir(node_matrix_path)
    os.makedirs(node_matrix_path)
    if os.path.exists(adj_matrix_path):
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
        os.rmdir(node_matrix_path)
    
    if os.path.exists(adj_matrix_path):
        os.rmdir(adj_matrix_path)
    
    return True

def graph2matrix_dot(dataset, outdir, hl, rl):
    dot_files = find_files(dataset, filter_func=filter_dot_file)
    tasks = split2task(dot_files, dataset, outdir)
    if not osp.isdir(outdir):
        if osp.exists(outdir):
            logger.error("[!] outdir ({}) already exists!", outdir)
            return False
        os.makedirs(outdir, exist_ok=True)

    for task in tasks:
        run_graph2matrix_dot(task, hl, rl, 1)

def divide_by_datatype(dataset ,hl):

    all_edge_type = ["AST", "LastUse", "ComputedFrom", "CFG", "CDG", "DDG"]
    list_divide_file = ["AST", "LastUse", "ComputedFrom", "CFG", "CDG", "DDG"]
    
    delete_type = [x for x in all_edge_type if x not in list_divide_file]

    file_list = os.listdir(dataset)

    for index, filename in enumerate(file_list):
        if filename == ".DS_Store":
            continue
        source_file_path = os.path.join(dataset, filename)
        for indexd, divide_file in enumerate(list_divide_file):
            destination_directory = os.path.join(source_file_path, divide_file +".dot")
            if hl:
                destination_directory = destination_directory.replace("graph", "dividegraph_hl")
            else:
                destination_directory = destination_directory.replace("graph", "dividegraph")
            source_file = source_file_path + "/ast_deform.dot"

            with open(source_file, 'r') as input_f:
                lines = input_f.readlines()
                filtered_lines_public = [line for line in lines if not any(keyword in line for keyword in list_divide_file)]
                filtered_lines_public = [line for line in filtered_lines_public if not any(keyword in line for keyword in delete_type)]
                filtered_edges = [line for line in lines if divide_file in line]
            os.makedirs(os.path.dirname(destination_directory), exist_ok=True)
            with open(destination_directory, 'w') as output_f:
                lines = filtered_lines_public[:-1] + filtered_edges + [filtered_lines_public[-1]]
                output_f.writelines(lines)
    

def main():
    if args.mission_id:
        project, cve_id, hl, rl, mission_id  = args.project, args.cve_id, args.hl, args.RL, args.mission_id
    else:
        project, cve_id, hl, rl  = args.project, args.cve_id, args.hl, args.RL

    if hl:
        if rl:
            dataset = os.path.join("../data_detect/data", mission_id, project, cve_id, "graph_hl")
            hl = "hl"
            outdir = os.path.join("../data_detect/data/", mission_id, project, cve_id, "matrix_" + cfg.GRAPH_MODE + "_" + hl)
            divide_by_datatype(dataset, hl) 
            dataset = dataset.replace("graph", "dividegraph_hl")
        else:
            dataset = os.path.join("../data", project, cve_id, "graph_hl")
            hl = "hl"
            outdir = os.path.join("../data/", project, cve_id, "matrix_" + cfg.GRAPH_MODE + "_" + hl)
            divide_by_datatype(dataset, hl) 
            dataset = dataset.replace("graph", "dividegraph_hl")
    else:
        if rl:
            dataset = os.path.join("../data_detect/data", mission_id, project, cve_id, "graph")
            outdir = os.path.join("../data_detect/data/", mission_id, project, cve_id, "matrix_" + cfg.GRAPH_MODE)
            divide_by_datatype(dataset, hl) 
            dataset = dataset.replace("graph", "dividegraph")
        else:
            dataset = os.path.join("../data", project, cve_id, "graph")
            outdir = os.path.join("../data/", project, cve_id, "matrix_" + cfg.GRAPH_MODE)
            divide_by_datatype(dataset, hl) 
            dataset = dataset.replace("graph", "dividegraph")

    graph2matrix_dot(dataset, outdir, hl, rl)

if __name__ == "__main__":
    main()
