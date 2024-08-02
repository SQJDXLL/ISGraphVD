import os
from html import unescape
from changeFormat import change_deform_for_matrix
from argparse import ArgumentParser

parser = ArgumentParser("Handling HTML encoding issues in graphs and modifying content that dot cannot handle.")
parser.add_argument("--project", type=str, default="curl")
parser.add_argument("--cve_id", type=str, default="CVE-2021-22901")
args = parser.parse_args()


def remove_ast_edges_from_folder(folder_path):
    # 遍历文件夹下的所有文件
    for root, dirs, files in os.walk(folder_path):
        for file_name in files:
            if file_name.endswith('.dot'):
                file_path = os.path.join(root, file_name)
                remove_ast_edges(file_path)


def remove_ast_edges(dot_file_path):
    # 读取 dot 文件内容
    with open(dot_file_path, 'r') as f:
        lines = f.readlines()

    # 存储不包含 label 为 "AST" 的边的新内容
    new_lines = []

    # 遍历 dot 文件的每一行
    for line in lines:
        # 检查是否为边定义行
        if '->' in line:
            # 检查是否包含 label
            if 'AST:' in line or 'CDG:' in line:
                # 如果包含 label 为 "AST" 的边，则跳过该行
                continue
        # 将不包含 label 为 "AST" 的边的行添加到新内容中
        new_lines.append(line)

    # 将更新后的内容写入 dot 文件中
    with open(dot_file_path, 'w') as f:
        f.writelines(new_lines)




if __name__ == "__main__":
    project, cve_id = args.project, args.cve_id
    graph_path = os.path.join("../../data/", project, cve_id, "graph")
    
    remove_ast_edges_from_folder(graph_path)
