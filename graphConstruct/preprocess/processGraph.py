import os
import re
from html import unescape
from changeFormat import change_deform_for_matrix
import pydot
# import pygraphviz as pgv
from tqdm import tqdm
from tqdm.contrib.concurrent import process_map
from argparse import ArgumentParser


parser = ArgumentParser("Handling HTML encoding issues in graphs and modifying content that dot cannot handle.")
parser.add_argument("--project", type=str, default="curl")
parser.add_argument("--cve_id", type=str, default="CVE-2021-22901")
parser.add_argument("--RL", type=bool, default=False)
args = parser.parse_args()

def process_dot_file(file_path):
    with open(file_path, 'r', encoding='utf-8') as file:
        dot_content = file.read()
    dot_content = unescape(dot_content)
    with open(file_path, 'w', encoding='utf-8') as file:
        file.write(dot_content)
        
def process_files_in_folder(folder_path):
    """
        use after generating graphs
        for change coding from html to normal to fix the unsuitable encoding
    """
    for root, dirs, files in os.walk(folder_path):
        for file_name in files:
            # print("file_name", file_name)
            if file_name.endswith('.dot'):
                file_path = os.path.join(root, file_name)
                process_dot_file(file_path)
                process_Escape_characters(file_path)

    
def process_Escape_characters(file_path):
    with open(file_path, "r") as f:
        lines = f.readlines()

    modified_lines = []
    for line in lines:
        modified_line = re.sub(r'(DDG: ).*$', r'\1"]', line.rstrip())
        modified_lines.append(modified_line)
    # modified_string = re.sub(r'(DDG: ).*$', r'\1"]', original_string, flags=re.MULTILINE)
    # dot_content_cleaned = dot_content.replace("\\t", "").replace("\\r", "").replace("\\n", "")

    # 将清理后的内容写回原始的 DOT 文件
    with open(file_path, "w") as f:
        f.write("\n".join(modified_lines))


def merge_dot_files(main_file, file_to_merge1, file_to_merge2, label1, label2, label3):
    # Load main dot file
    main_graph = pydot.graph_from_dot_file(main_file)[0]
    graph_to_merge1 = pydot.graph_from_dot_file(file_to_merge1)[0]
    graph_to_merge2 = pydot.graph_from_dot_file(file_to_merge2)[0]

    # Add edges from the first dot file with label1
    for edge in graph_to_merge1.get_edges():
        edge.set_label(label1)
        main_graph.add_edge(edge)
        
    # Add edges from the second dot file with label2
    for edge in graph_to_merge2.get_edges():
        if "CDG" in edge.get_label():
            # edge.del_attr('label')
            edge.set('label', label2)
            main_graph.add_edge(edge)
        if 'DDG' in edge.get_label():
            # edge.del_attr('label')
            source = edge.get_source()
            target = edge.get_destination()
            main_graph.del_edge(edge)
            new_edge = pydot.Edge(source, target, label=label3)
            main_graph.add_edge(new_edge)
        
        
    # Write the merged dot file
    main_graph.write(main_file)


def runner(task: tuple):
    main_file, file_to_merge1, file_to_merge2, label1, label2, label3 = task
    return merge_dot_files(main_file, file_to_merge1, file_to_merge2, label1, label2, label3)

if __name__ == "__main__":
    project, cve_id, rl = args.project, args.cve_id, args.RL
    if rl:
        graph_path = os.path.join("../../realWorld/data/", project, cve_id, "graph")
    else:
        graph_path = os.path.join("../../data/", project, cve_id, "graph")
    
    process_files_in_folder(graph_path)
    change_deform_for_matrix(graph_path)

    tasks = []
    for file in os.listdir(graph_path):
        main_file = os.path.join(graph_path, file, "ast_deform.dot")
        file_to_merge1 = os.path.join(graph_path, file, "CFG.dot")
        file_to_merge2 = os.path.join(graph_path, file, "pdg.dot")
        label1 = "CFG"
        label2 = "CDG"
        label3 = "DDG"
        tasks.append((main_file, file_to_merge1, file_to_merge2, label1, label2, label3))
    
    res = process_map(runner, tasks, max_workers=10, chunksize=4)

    # tasks = []
    # for file in os.listdir(graph_path):
    #     main_file = os.path.join(graph_path, file, "ast_deform.dot")
    #     file_to_merge1 = os.path.join(graph_path, file, "CFG.dot")
    #     file_to_merge2 = os.path.join(graph_path, file, "pdg.dot")
    #     label1 = "CFG"
    #     label2 = "CDG"
    #     label3 = "DDG"
    #     merge_dot_files(main_file, file_to_merge1, file_to_merge2, label1, label2, label3)
    




