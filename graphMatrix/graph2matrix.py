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
# from graphConstruct.preprocess.changeFormat import change_deform_for_matrix, divide_by_datatype
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

# 将图形数据从.dot文件解析为节点矩阵和邻接矩阵
def json2matrix_dot(path, graph_types=cfg.GRAPH_TYPES):
    select_dot = list(filter(lambda x: x[1] in graph_types, cfg.GRAPH_DOT_FILES.items()))
    # select_dot [('AST.dot', 'AST'), ('LastUse.dot', 'LastUse'), ('ComputedFrom.dot', 'ComputedFrom')]
    nodes = {}
    adj_edges = dict([(x, {}) for x in graph_types])
    # adj_edges {'AST': {}, 'LastUse': {}, 'ComputedFrom': {}}
    for dot_name, graph_type in select_dot:
        dot_path = osp.join(path, dot_name)
        # print("dot_path", dot_path)
        G = pgv.AGraph(dot_path)
        # try:
        #     G = pgv.AGraph(dot_path)
        #     print("G*************",G)
        # except Exception as e:
        #     print("Error loading graph with dot_path:", dot_path)
        #     print("Error message:", str(e))
        
        label_key = cfg.DOT_NODE_ATTR
        G.node_attr ['label']
        if label_key not in list(G.node_attr):
            logger.error("[!] Cannot find {} in graph {}", label_key, G)
            return False
        for node in G.nodes():
            if cfg.GRAPH_MODE == "disjoint":
                # disjoint mode
                # graph_type:AST node.name:"8"
                node_id = "{}_{}".format(graph_type, node.name)
            elif cfg.GRAPH_MODE == 'single':
                # single mode
                node_id = node.name

            label_raw = node.attr[label_key]
            if label_raw:
            # print("label_raw",label_raw)
                if label_raw[0] == "(":
                    label_raw = label_raw[1:]
                if label_raw[-1] == ")":
                    label_raw = label_raw[:-1]
                label = label_raw.split(",")[0]
                nodes[node_id] = label
        # print("node_each_graph", len(nodes))
        
            # single nodes{'7':'METHOD,...}
            # disjoint nodes {'AST_7': 'METHOD', 'AST_8': 'PARAM',...'ComputedFrom_175': 'IDENTIFIER', 'ComputedFrom_176': 'UNKNOWN', 'ComputedFrom_177': '<operator>.assignment', 'ComputedFrom_178': 'IDENTIFIER', 'ComputedFrom_179': 'LITERAL', 'ComputedFrom_180': 'RETURN'}
        for in_edge, out_edge in G.edges():
            if cfg.GRAPH_MODE == "disjoint":
                # disjoint mode
                in_edge_id = "{}_{}".format(graph_type, in_edge.name)
                out_edge_id = "{}_{}".format(graph_type, out_edge.name)
            elif cfg.GRAPH_MODE == 'single':
                # single mode
                in_edge_id = in_edge.name
                out_edge_id = out_edge.name
            # 开始组织边的集合
            if in_edge_id not in adj_edges[graph_type]:
                adj_edges[graph_type][in_edge_id] = []
            if out_edge_id not in adj_edges[graph_type][in_edge_id]:
                adj_edges[graph_type][in_edge_id].append(out_edge_id)
        # adj_edges single {'AST':{'7':['8', '10'], '8':['11','23'}...}
        # adj_edges disjoint {'AST': {'AST_7': ['AST_8', 'AST_9', 'AST_10', 'AST_182'], 'AST_10': ['AST_11', 'AST_12', 'AST_13', 'AST_14', 'AST_15', 'AST_17', 'AST_18', 'AST_19', 'AST_20', 'AST_21', 'AST_22', 'AST_23', 'AST_24', 'AST_25', 'AST_33', 'AST_44', 'AST_52', 'AST_180'], 'AST_15': ['AST_16'], 'AST_25': ['AST_26', 'AST_27'], 'AST_27': ['AST_28'], 'AST_28': ['AST_29', 'AST_30'], 'AST_30': ['AST_31', 'AST_32'], 'AST_33': ['AST_34', 'AST_42'], 'AST_34': ['AST_35', 'AST_41'], 'AST_35': ['AST_36'], 'AST_36': ['AST_37', 'AST_38'], 'AST_38': ['AST_39', 'AST_40'], 'AST_42': ['AST_43'], 'AST_44': ['AST_45', 'AST_46'], 'AST_46': ['AST_47'], 'AST_47': ['AST_48', 'AST_49'], 'AST_49': ['AST_50', 'AST_51'], 'AST_52': ['AST_53', 'AST_54', 'AST_69'], 'AST_54': ['AST_55', 'AST_58'], 'AST_55': ['AST_56', 'AST_57'], 'AST_58': ['AST_59', 'AST_67'], 'AST_59': ['AST_60', 'AST_61'], 'AST_61': ['AST_62'], 'AST_62': ['AST_63', 'AST_64'], 'AST_64': ['AST_65', 'AST_66'], 'AST_67': ['AST_68'], 'AST_69': ['AST_70'], 'AST_70': ['AST_71', 'AST_75', 'AST_76', 'AST_116'], 'AST_71': ['AST_72', 'AST_73'], 'AST_73': ['AST_74'], 'AST_76': ['AST_77', 'AST_78'], 'AST_78': ['AST_79', 'AST_80', 'AST_86', 'AST_92', 'AST_98', 'AST_108', 'AST_114', 'AST_115'], 'AST_80': ['AST_81'], 'AST_81': ['AST_82', 'AST_83'], 'AST_83': ['AST_84', 'AST_85'], 'AST_86': ['AST_87'], 'AST_87': ['AST_88', 'AST_89'], 'AST_89': ['AST_90', 'AST_91'], 'AST_92': ['AST_93'], 'AST_93': ['AST_94', 'AST_95'], 'AST_95': ['AST_96', 'AST_97'], 'AST_98': ['AST_99', 'AST_107'], 'AST_99': ['AST_100', 'AST_106'], 'AST_100': ['AST_101'], 'AST_101': ['AST_102', 'AST_103'], 'AST_103': ['AST_104', 'AST_105'], 'AST_108': ['AST_109'], 'AST_109': ['AST_110', 'AST_111'], 'AST_111': ['AST_112', 'AST_113'], 'AST_116': ['AST_117', 'AST_120'], 'AST_117': ['AST_118', 'AST_119'], 'AST_120': ['AST_121'], 'AST_121': ['AST_122', 'AST_130', 'AST_142'], 'AST_122': ['AST_123', 'AST_129'], 'AST_123': ['AST_124'], 'AST_124': ['AST_125', 'AST_126'], 'AST_126': ['AST_127', 'AST_128'], 'AST_130': ['AST_131', 'AST_139'], 'AST_131': ['AST_132', 'AST_138'], 'AST_132': ['AST_133'], 'AST_133': ['AST_134', 'AST_135'], 'AST_135': ['AST_136', 'AST_137'], 'AST_139': ['AST_140', 'AST_141'], 'AST_142': ['AST_143'], 'AST_143': ['AST_144', 'AST_145', 'AST_177'], 'AST_145': ['AST_146', 'AST_176'], 'AST_146': ['AST_147', 'AST_151', 'AST_160', 'AST_170', 'AST_174'], 'AST_147': ['AST_148', 'AST_149'], 'AST_149': ['AST_150'], 'AST_151': ['AST_152', 'AST_153'], 'AST_153': ['AST_154'], 'AST_154': ['AST_155', 'AST_156'], 'AST_156': ['AST_157', 'AST_159'], 'AST_157': ['AST_158'], 'AST_160': ['AST_161', 'AST_164'], 'AST_161': ['AST_162', 'AST_163'], 'AST_164': ['AST_165'], 'AST_165': ['AST_166', 'AST_167'], 'AST_167': ['AST_168', 'AST_169'], 'AST_170': ['AST_171', 'AST_172'], 'AST_172': ['AST_173'], 'AST_174': ['AST_175'], 'AST_177': ['AST_178', 'AST_179'], 'AST_180': ['AST_181']}, 'LastUse': {'LastUse_39': ['LastUse_26'], 'LastUse_50': ['LastUse_39'], 'LastUse_53': ['LastUse_45'], 'LastUse_60': ['LastUse_53'], 'LastUse_65': ['LastUse_50'], 'LastUse_84': ['LastUse_50'], 'LastUse_104': ['LastUse_31'], 'LastUse_118': ['LastUse_77'], 'LastUse_127': ['LastUse_84'], 'LastUse_129': ['LastUse_118'], 'LastUse_138': ['LastUse_118'], 'LastUse_150': ['LastUse_16'], 'LastUse_158': ['LastUse_150'], 'LastUse_162': ['LastUse_152'], 'LastUse_168': ['LastUse_162'], 'LastUse_175': ['LastUse_158', 'LastUse_175'], 'LastUse_181': ['LastUse_56', 'LastUse_140', 'LastUse_178']}, 'ComputedFrom': {'ComputedFrom_31': ['ComputedFrom_26'], 'ComputedFrom_50': ['ComputedFrom_45'], 'ComputedFrom_84': ['ComputedFrom_77', 'ComputedFrom_90', 'ComputedFrom_96', 'ComputedFrom_104', 'ComputedFrom_112'], 'ComputedFrom_90': ['ComputedFrom_77'], 'ComputedFrom_96': ['ComputedFrom_77'], 'ComputedFrom_104': ['ComputedFrom_77'], 'ComputedFrom_112': ['ComputedFrom_77'], 'ComputedFrom_138': ['ComputedFrom_136'], 'ComputedFrom_150': ['ComputedFrom_148'], 'ComputedFrom_158': ['ComputedFrom_152'], 'ComputedFrom_168': ['ComputedFrom_166']}}
    # print("node_all", len(nodes))
    # 转matrix
    node_matrix, node_dic = normalize_nodes_dot(nodes)
    adj_matrix = normalize_edges(adj_edges, graph_types, node_dic)

    return node_matrix, adj_matrix

# 将解析后的节点矩阵和邻接矩阵保存到指定目录中
def run_graph2matrix_dot(task, iindex=1):
    path, prefix, tag = task
    name = osp.basename(path)
    tags = ["vul", "fix"]
    types = ["node", "adj"]
    # print("tag", tag)
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
        # break
    # multi_run(tasks, run_graph2matrix_dot)

def main():
    project, cve_id = args.project, args.cve_id
    dataset = os.path.join("../data", project, cve_id, "graph")
    outdir = os.path.join("../data/", project, cve_id, "matrix_" + cfg.GRAPH_MODE)
    
    # change_deform_for_matrix(dataset)
    # divide_by_datatype(dataset)   
    dataset = dataset.replace("graph", "dividegraph")
    graph2matrix_dot(dataset, outdir)

if __name__ == "__main__":
    main()
