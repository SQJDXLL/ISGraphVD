import re
import os
import time
import numpy as np
import chardet
from graph_tool.all import *

from html import unescape


from argparse import ArgumentParser
from tqdm import tqdm
from tqdm.contrib.concurrent import process_map

from utils import remove_comments

# --dot_path /data_hdd/lx20/anonymous_workspace/diff-gmn/newDiff/graph/dnsmasq/CVE-2015-8899
# --statements_path /data_hdd/lx20/anonymous_workspace/diff-gmn/diffBypseudo/result/dnsmasq/CVE-2015-8899 
# --output_dir /data_hdd/lx20/anonymous_workspace/diff-gmn/diffBypseudo/result_dot/dnsmasq/CVE-2015-8899
parser = ArgumentParser("according statesment diff dot graph")
parser.add_argument("--project", type=str, default="curl")
parser.add_argument("--cve_id", type=str, default="CVE-2021-22901")
parser.add_argument("--RL", action="store_true", default=False)
# parser.add_argument("--dot_path", type=str, required=True, help="dot file path")
# parser.add_argument("--statements_path", type=str, required=True, help="statements that will not be removed")
# parser.add_argument("--output_dir", type=str, required=True)
args = parser.parse_args()

        
def process_dot_file(file_path):
    with open(file_path, 'r', encoding='utf-8') as file:
        dot_content = file.read()
    # print("before", dot_content)
    dot_content = unescape(dot_content)
    # print("end", dot_content)
    
    with open(file_path, 'w', encoding='utf-8') as file:
        file.write(dot_content)
        
def process_files_in_folder(folder_path):
    for root, dirs, files in os.walk(folder_path):
        for file_name in files:
            if file_name.endswith('.dot'):
                file_path = os.path.join(root, file_name)
                process_dot_file(file_path)

def get_root_vertex(graph):
    """get root vertex from graph
    return:
        if root exist return vertex index otherwise -1"""
    result = np.argwhere(graph.get_in_degrees(graph.get_vertices()) == 0)
    return result[0, 0] if result.size > 0 else -1


def get_remove_vertices_strict(
    graph,
    root_idx: int,
    statements: list[str],
    label_process: callable = lambda x: re.split(",(?! )", re.sub("\(|\)", "", x))[1],
):
    """get vertices that need to be removed
    params:
        graph: graph_tool.Graph
        root_idx: root vertex index
        statements: statements that will not be removed
        label_process: process label function, split by comma without space
    return:
        list of vertices that need to be removed"""
    statements = statements.copy()
    remove_vertices = set()
    rollback_vertices = []
    current_trajectory_pre_match = np.array([0])

    root_vertex = graph.vertex(root_idx)
    for idx, stmt in enumerate(statements):
        if label_process(graph.vp["label"][root_vertex]) == stmt:
            break
        elif idx == len(statements) - 1:
            remove_vertices.add(root_vertex)

    for e in dfs_iterator(graph, graph.vertex(root_idx)):
        v = e.target()

        # * main logic
        for pre_idx, pre_match in enumerate(current_trajectory_pre_match):
            if pre_match == -1:
                continue
            for stmt_idx, stmt in enumerate(statements[pre_match:]):
                if not label_process(graph.vp["label"][v]) == stmt:
                    remove_vertices.add(v)
                    if current_trajectory_pre_match[0]:
                        current_trajectory_pre_match[pre_idx] = -1
                else:
                    if current_trajectory_pre_match[0]:
                        current_trajectory_pre_match[0] = stmt_idx + pre_match
                    else:
                        np.append(current_trajectory_pre_match, np.array([stmt_idx + pre_match]))
                    rollback_vertices.append(v)

        # * if reach a leaf node
        if v.out_degree() == 0:
            current_trajectory_pre_match = 0
            if np.any(current_trajectory_pre_match != -1):
                remove_vertices.update(rollback_vertices)
            current_trajectory_pre_match = np.array([0])

    return remove_vertices


def get_remove_vertices(
    graph,
    root_idx: int,
    statements: list[str],
    label_process: callable = lambda x: re.split(",(?! )", x[1:-1])[1],
):
    """get vertices that need to be removed
    params:
        graph: graph_tool.Graph
        root_idx: root vertex index
        statements: statements that will not be removed
        label_process: process label function, split by comma without space
    return:
        list of vertices that need to be removed"""
    statements = statements.copy()
    remove_vertices = set()
    retain_vertices = set()

    root_vertex = graph.vertex(root_idx)
    for idx, stmt in enumerate(statements):
        if label_process(graph.vp["label"][root_vertex]) == stmt.strip():
            retain_vertices.add(root_vertex)
            break
        elif idx == len(statements) - 1:
            remove_vertices.add(root_vertex)

    for e in dfs_iterator(graph, graph.vertex(root_idx)):
        v = e.target()

        if e in retain_vertices:
            continue

        for stmt in statements:
            if not label_process(graph.vp["label"][v]).replace(" ","") == stmt.strip().strip(";"):
                remove_vertices.add(v)
            else:
                retain_vertices.add(v)
                for e2 in dfs_iterator(graph, v):
                    retain_vertices.add(e2.target())

    return remove_vertices - retain_vertices


def statements2graph(
    graph: Graph, statements: list, output_path: str, filter_edge: str = "AST: ", strict_mode: bool = False
):
    origin_numbers = graph.num_vertices()
    root_idx = get_root_vertex(graph)
    u = GraphView(graph, efilt=lambda e: graph.ep["label"][e] == filter_edge)
    if strict_mode:
        remove_vertices = get_remove_vertices_strict(u, root_idx, statements)
    else:
        remove_vertices = get_remove_vertices(u, root_idx, statements)

    graph.remove_vertex(remove_vertices, fast=True)
    os.makedirs(output_path, exist_ok=True)
    graph.save(os.path.join(output_path, "ast_deform.dot"), fmt="dot")

    return len(remove_vertices), origin_numbers


def runner(data: tuple):
    statement, graph, store_path = data
    # print(store_path)
    return statements2graph(graph, statement, store_path, strict_mode=False)


if __name__ == "__main__":
    # change_deform_for_matrix(args.dot_path)
    time_start = time.time()
    proj, cve, rl = args.project, args.cve_id, args.RL
    if rl:
        dot_path= os.path.join("../../realWorld/data/", proj, cve, "graph")
    else:
        dot_path = os.path.join("../../data/", proj, cve, "graph")
    dot_file_set = set(os.listdir(dot_path))
    # dot_file_set = set(os.listdir(args.dot_path))
    if rl:
        statements_path = os.path.join("../../realWorld/data/", proj, cve, "pseudo_hl")
    else:
        statements_path = os.path.join("../../data/", proj, cve, "pseudo_hl")
    statements_file_set = set(map(lambda x:x.replace(".c", ""), os.listdir(statements_path)))
    # statements_file_set = set(map(lambda x:x.replace(".c", ""), os.listdir(args.statements_path)))
    target_files = list(dot_file_set & statements_file_set)
    num_files = len(target_files)
    print(f"target files: {num_files}")
    dataset = []
    for file in tqdm(target_files):
        if file == ".DS_Store":
            continue

        dot_file = file
        stmt_file = file + ".c"

        with open(os.path.join(statements_path, stmt_file), "r") as f:
        # with open(os.path.join(args.statements_path, stmt_file), "r") as f:
            statements = list(map(lambda x: remove_comments(x.replace(" ", "")), f.readlines()))
        # path_graph = os.path.join(args.dot_path, dot_file, "ast_deform.dot")
            path_graph = os.path.join(dot_path, dot_file, "ast_deform.dot")
        # path_graph_without = os.path.join(args.dot_path, dot_file)
        # process_files_in_folder(path_graph_without)
        try:
            graph = load_graph(path_graph)
        except Exception as e:
            print(file)
            raise e

        # store_path = os.path.join(args.output_dir, dot_file)
        # store_path = os.path.join(args.output_dir, dot_file)
        if rl:
            output_dir = os.path.join("../../realWorld/data/", proj, cve, "diffDot_hl")
        else:
            output_dir = os.path.join("../../data/", proj, cve, "diffDot_hl")
        store_path = os.path.join(output_dir, dot_file)
        dataset.append((statements, graph, store_path))
        
    res = process_map(runner, dataset, max_workers=1, chunksize=1)

    time_end = time.time()
    time_cose = time_end - time_start
    print("diff2Dot_timeCost", time_cose)
    res = np.array(res)
    print(res[..., 0].sum() / res[..., 1].sum())
