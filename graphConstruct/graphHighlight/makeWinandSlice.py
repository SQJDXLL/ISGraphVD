import difflib
import os
import sys

sys.path.append(os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))
import re
import shutil
from utils import *
from argparse import ArgumentParser
import time

parser = ArgumentParser("get diff statement")
parser.add_argument("--project", type=str, required=True, help="project")
parser.add_argument("--cve_id", type=str, required=True, help="CVE")
parser.add_argument("--RL", action="store_true", default=False)
parser.add_argument("--mission_id", type=str, default="hchdvbjsjci-bhjdsbcj")
args = parser.parse_args()


def recreate_folder(folder_path):
    if os.path.exists(folder_path):
        shutil.rmtree(folder_path)
    os.makedirs(folder_path)


def compare_files(list_fix, list_vul):

    differ = difflib.Differ()
    diff = list(differ.compare(list_fix, list_vul))
    line_number_file1 = 1
    line_number_file2 = 1

    diff_fix = {}
    diff_vul = {}
    for line in diff:
        if line.startswith("- "):
            diff_fix[line_number_file1] = line[2:].strip()
            line_number_file1 += 1
        elif line.startswith("+ "):
            diff_vul[line_number_file1] = line[2:].strip()
            line_number_file2 += 1
        else:
            line_number_file1 += 1
            line_number_file2 += 1
    return diff_fix, diff_vul


if __name__ == "__main__":
    time_start = time.time()
    proj, cve, rl, mission_id = args.project, args.cve_id, args.RL, args.mission_id
    if rl:
        path_pseudoCode = os.path.join("../../data_detect/data/", mission_id, proj, cve, "pseudo")
        path_graphCode = os.path.join("../../data_detect/data/", mission_id, proj, cve, "graph")
        path_return = os.path.join("../../data_detect/data/", mission_id, proj, cve, "pseudo_hl")
    else:
        path_pseudoCode = os.path.join("../../data/", proj, cve, "pseudo")
        path_graphCode = os.path.join("../../data/", proj, cve, "graph")
        path_return = os.path.join("../../data/", proj, cve, "pseudo_hl")

    recreate_folder(path_return)
    if not rl:
        list_s = []
        list_l = []
        for file_fix in os.listdir(path_pseudoCode):
            if "fix" in file_fix:
                file_vul = file_fix.replace("fix", "vul")
                if os.path.exists(os.path.join(path_pseudoCode, file_vul)):
                    fix_path = os.path.join(path_pseudoCode, file_fix)
                    vul_path = os.path.join(path_pseudoCode, file_vul)
                    with open(fix_path, "r") as f1, open(vul_path, "r") as f2:
                        list_fix = remove_comments(f1.read()).split("\n")
                        list_vul = remove_comments(f2.read()).split("\n")
                    list_fix = [item for item in list_fix if item.strip() != ""]
                    list_vul = [item for item in list_vul if item.strip() != ""]
                    list_fix = list_fix[1:-1]
                    list_vul = list_vul[1:-1]

                    differ = list(difflib.unified_diff(list_fix, list_vul, lineterm="", fromfile=fix_path, tofile=vul_path))

                    list_line = []
                    for line_num in range(len(differ)):
                        if differ[line_num].startswith("@@"):
                            match = re.match(r"@@ -(\d+),(\d+) \+(\d+),(\d+) @@", differ[line_num])
                            if match:
                                start_line_src, num_lines_src, start_line_dest, num_lines_dest = map(int, match.groups())
                                list_line.append((start_line_src, num_lines_src, start_line_dest, num_lines_dest))

                    if not list_line:
                        print(f"Identical files found: {file_fix} and {file_vul}")
                        continue

                    fix_start = list_line[0][0]
                    vul_start = list_line[0][2]
                    fix_end = list_line[-1][0] + list_line[-1][1]
                    vul_end = list_line[-1][2] + list_line[-1][3]

                    if fix_start > vul_start:
                        start = vul_start
                    else:
                        start = fix_start

                    if fix_end > vul_end:
                        end = fix_end
                    else:
                        end = vul_end
                    list_s.append(start)
                    list_l.append(end)

        win_start = min(list_s)
        win_end = max(list_l)

        from data.winConfig import win

        key = proj
        value = {cve: [win_start, win_end]}
        if key not in win:
            win[key] = value
        else:
            win[key].update(value)

        with open("../../data/winConfig.py", "w") as f:
            f.write(f"win = {win}")

    if rl:
        from data.winConfig import win

        win_start = win[proj][cve][0]
        win_end = win[proj][cve][1]
        for file in os.listdir(path_pseudoCode):
            file_path = os.path.join(path_pseudoCode, file)
            with open(file_path, "r") as f:
                list_file = remove_comments(f.read()).split("\n")
            list_file = [item for item in list_file if item.strip() != ""]
            list_file = list_file[1:-1]
            if len(list_file) > win_end:
                file_final = list_file[win_start - 1 : win_end]
            else:
                file_final = list_file[win_start - 1 :]

            max_window_string = "\n".join(file_final)
            file_path = os.path.join(path_return, file)
            with open(file_path, "w") as file:
                file.write(max_window_string)
    else:
        for file in os.listdir(path_pseudoCode):
            file_path = os.path.join(path_pseudoCode, file)
            with open(file_path, "r") as f:
                list_file = remove_comments(f.read()).split("\n")
            list_file = [item for item in list_file if item.strip() != ""]
            list_file = list_file[1:-1]
            if len(list_file) > win_end:
                file_final = list_file[win_start - 1 : win_end]
            else:
                file_final = list_file[win_start - 1 :]

            max_window_string = "\n".join(file_final)
            file_path = os.path.join(path_return, file)
            with open(file_path, "w") as file:
                file.write(max_window_string)
    time_end = time.time()
    cost_time = time_end - time_start
