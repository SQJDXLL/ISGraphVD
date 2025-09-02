import os
from html import unescape
from changeFormat import change_deform_for_matrix
from argparse import ArgumentParser

parser = ArgumentParser("Handling HTML encoding issues in graphs and modifying content that dot cannot handle.")
parser.add_argument("--project", type=str, default="curl")
parser.add_argument("--cve_id", type=str, default="CVE-2021-22901")
args = parser.parse_args()


def remove_ast_edges_from_folder(folder_path):

    for root, dirs, files in os.walk(folder_path):
        for file_name in files:
            if file_name.endswith('.dot'):
                file_path = os.path.join(root, file_name)
                remove_ast_edges(file_path)


def remove_ast_edges(dot_file_path):

    with open(dot_file_path, 'r') as f:
        lines = f.readlines()

    new_lines = []

    for line in lines:

        if '->' in line:
            if 'AST:' in line or 'CDG:' in line:
                continue
        new_lines.append(line)

    with open(dot_file_path, 'w') as f:
        f.writelines(new_lines)




if __name__ == "__main__":
    project, cve_id = args.project, args.cve_id
    graph_path = os.path.join("../../data/", project, cve_id, "graph")
    
    remove_ast_edges_from_folder(graph_path)
