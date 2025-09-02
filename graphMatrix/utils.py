# -*- coding: utf-8 -*-

import glob
import json
import os.path
import re
from concurrent.futures import ProcessPoolExecutor

import numpy as np
from loguru import logger
import config as cfg

NORMAL_FLAG = 1
HIGHLIGHT_FLAG = 2

id_pattern = re.compile(r"^([a-zA-Z0-9.]+)\[label=([A-Z_]+); id=(\d+)\]$")

def load_json(path):
    try:
        with open(path, 'r') as f:
            data = f.read()
        return json.loads(data)
    except Exception as e:
        print("[!] failed to load jsons in {}".format(path))
        return None

def parse_id(_id):
    if not id_pattern.match(_id):
        return None, None, None
    return id_pattern.findall(_id)[0]

def normalize_label(label):
    label_matrix = list(map(lambda  x: int(x == label), cfg.NODE_LABELS))
    if not any(label_matrix):
        raise BaseException('[!] Found new node label: {}'.format(label))
        exit(1)
    return np.array(label_matrix, dtype=int)

def normalize_label_dot(label):
    if label in cfg.NODE_LABELS_DOT:
        key = label
    elif label in cfg.OPERATOR_LABEL:
        key = "OPERATOR"
    elif label in cfg.C_BUILT_IN_FUNCS:
        key = "CALL_IN"
    else:
        key = "CALL_OUT"
    idx = cfg.MAP_DOT_LABEL[key]

    label_matrix = list(map(lambda x: int(x == idx), range(len(cfg.MAP_DOT_LABEL))))

    if not any(label_matrix):
        raise BaseException('[!] Found new node label: {}'.format(label))
        exit(1)
    return np.array(label_matrix, dtype=int)

def normalize_edges_gat(adj_edges, graph_types, node_dic):
    """[summary]
    GAT input

    Args:
        adj_edges ([type]): [description]
        graph_types ([type]): [description]
        node_dic ([type]): [description]

    Returns:
        adj_matrix (dict): adj_matrix[3, num_edge, num_connected_edge]
    """
    adj_matrix = [dict([(i, []) for i in range(len(node_dic.keys()))]) for _ in range(len(graph_types))]
    for graph_type, i in zip(graph_types, range(len(graph_types))):
        for _in, _outs in adj_edges[graph_type].items():
            _in_n = node_dic[_in]
            _outs_n = list(map(lambda x: node_dic[x], _outs))
            adj_matrix[i][_in_n] = _outs_n
    return adj_matrix

def normalize_edges_gmn(adj_edges, graph_types, node_dic):

    n_graph = len(graph_types)
    n_node = len(node_dic.keys())
    adj_matrix = np.zeros((n_graph, n_node, n_node), dtype=int)
    for graph_type, x in zip(graph_types, range(n_graph)):
        for _in, _outs in adj_edges[graph_type].items():
            y = node_dic[_in]
            for z in map(lambda x: node_dic[x], _outs):
                adj_matrix[x][y][z] = 1
    return adj_matrix


def normalize_edges_gmn_highlight(adj_edges, graph_types, node_dic):

    n_graph = len(graph_types)
    n_node = len(node_dic.keys())
    adj_matrix = np.zeros((n_graph, n_node, n_node), dtype=int)

    for isHighlight in adj_edges.keys():
        if isHighlight == "True":
            for graph_type, x in zip(graph_types, range(n_graph)):
                for _in, _outs in adj_edges[isHighlight][graph_type].items():
                    y = node_dic[_in]
                    for z in map(lambda x: node_dic[x], _outs):

                        adj_matrix[x][y][z] = NORMAL_FLAG 
        else:
            for graph_type, x in zip(graph_types, range(n_graph)):
                for _in, _outs in adj_edges[isHighlight][graph_type].items():
                    y = node_dic[_in]
                    for z in map(lambda x: node_dic[x], _outs):
                        adj_matrix[x][y][z] = HIGHLIGHT_FLAG 

    return adj_matrix


def normalize_nodes(nodes):
    node_list = sorted(nodes.keys())
    node_dic = dict([(x, i) for x, i in zip(node_list, range(len(node_list)))])
    node_matrix = np.zeros((len(node_list), len(cfg.NODE_LABELS)), dtype=int)
    for key, i in node_dic.items():
        node_matrix[i] = normalize_label(nodes[key])
    return node_matrix, node_dic


def normalize_nodes_dot(nodes):
    node_list = sorted(nodes.keys())
    node_dic = dict([(x, i) for x, i in zip(node_list, range(len(node_list)))])

    node_matrix = np.zeros((len(node_list), len(cfg.MAP_DOT_LABEL.keys())), dtype=int)

    for key, i in node_dic.items():
        node_matrix[i] = normalize_label_dot(nodes[key])
    return node_matrix, node_dic

def multi_run(inputs, func, max_workers=10):
    logger.info(
        "[+] Run {fu} with {mw} Processes for {num} tasks",
        fu=func.__name__, mw=max_workers, num=len(inputs)
    )
    outputs = []
    with ProcessPoolExecutor(max_workers=max_workers) as pool:
        for res, index in zip(pool.map(func, inputs, [i+1 for i in range(len(inputs))]), [i+1 for i in range(len(inputs))]):
            logger.info("[+] {}/{}: Task {} return {}", index, len(inputs), index, res)

def find_files(dir_path, filter_func=None):
    files = []
    _path = dir_path
    while True:
        _path = os.path.join(_path, "*")
        _files = glob.glob(_path)
        if not _files:
            break
        files.extend(_files)
    logger.debug("[*] Found {} files in {}", len(files), dir_path)
    if filter_func:
        files = filter(filter_func, files)
    files = sorted(files)
    return files

def parse_install_name(name):
    res = re.findall(r"^([a-zA-Z0-9]+)_([a-zA-Z0-9]+)-(\d+)_([a-zA-Z0-9]+)+_(.*)_((?:fix|vul))", name)
    if not len(res) == 1:
        return None
    return res[0]
