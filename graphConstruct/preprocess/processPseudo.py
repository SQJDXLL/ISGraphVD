# -*- coding: utf-8 -*-
import os
import re
from argparse import ArgumentParser

parser = ArgumentParser("process pseudo for construct graphs.")
parser.add_argument("--project", type=str, default="curl")
parser.add_argument("--cve_id", type=str, default="CVE-2021-22901")
parser.add_argument("--RL", type=bool, default=False)
args = parser.parse_args()

def changeSuffix(folder_path):
    """
        change file_name to support joern
        example: change "file.c123456" to "file.c"
    """
    for filename in os.listdir(folder_path):
        if filename.endswith(".c"):
            continue  
        file_name, old_extension = os.path.splitext(filename)

        suffix = "c" 
        new_filename = f"{file_name}.{suffix}"
        old_filepath = os.path.join(folder_path, filename)
        new_filepath = os.path.join(folder_path, new_filename)

        os.rename(old_filepath, new_filepath)

def remove_angle_brackets_bytes(data):
    pattern = re.compile(b'<.*?>')
    result = re.sub(pattern, b'', data)
    return result

def deleteKeywords(folder_path):
    """
        Delete keywords that Joern cannot handle and are not related to vulnerability detection
    """
    file_list = os.listdir(folder_path)

    for filename in file_list:
        file_path = os.path.join(folder_path, filename)
        # print("FILENAME", filename)
        
        if os.path.isfile(file_path):
        #     with open(file_path, 'r+') as file:
        #         lines = file.readlines()
        #         file.seek(0)
        #         file.truncate(0)

        #         in_function = False
        #         for line in lines:
        #             # 判断是否为函数定义
        #             if re.match(r'^\s*[\w\s]+\([^)]*\)\s*{', line):
        #                 print("in", line)
        #                 in_function = True
        #             elif re.match(r'^\s*}', line):
        #                 print("out", line)
        #                 in_function = False

        #             # 删除特定字符串
        #             if not in_function:
        #                 line = line.replace("unsigned", '')
        #             file.write(line)
            try:
                with open(file_path, 'rb') as file:
                    lines = file.readlines()
                # *__cdecl
                if lines:
                    # 先把所有行的unsigned处理掉
                    modified_lines = []
                    for line in lines:
                        modified_line = line.decode(errors='ignore')
                        modified_line  =  modified_line .replace("unsigned", "")
                        modified_line = modified_line.replace("signed", "")
                        modified_line = modified_line.replace("*__usercall", "*__fastcall")
                        modified_line = modified_line.replace("**__usercall", "")
                        modified_line = modified_line.replace("__usercall", "__fastcall")
                        modified_line = modified_line.replace("*__cdecl", "")
                        modified_line = modified_line.replace("__cdecl", "")
                        # modified_line  = re.sub(r'\s+', ' ', modified_line .replace("unsigned", "")).strip()
                        # modified_line = re.sub(r'\s+', ' ', modified_line.replace("signed", "")).strip()
                        # modified_line = modified_line.replace("*__usercall", "*__fastcall")
                        # modified_line = re.sub(r'\s+', ' ', modified_line.replace("**__usercall", "")).strip()
                        # modified_line = modified_line.replace("__usercall", "__fastcall")
                        # modified_line = re.sub(r'\s+', ' ', modified_line.replace("*__cdecl", "")).strip()
                        # modified_line = re.sub(r'\s+', ' ', modified_line.replace("__cdecl", "")).strip()
                        # modified_line  = modified_line  + "\n"
                        modified_line = modified_line.encode()
                        modified_line = remove_angle_brackets_bytes(modified_line)
                        modified_lines.append(modified_line)
                            # print("line", line)

                with open(file_path, 'wb') as file:
                    file.writelines(modified_lines)
                    
            except Exception as e:
                print(f"处理文件 {file_path} 时出现错误: {str(e)}")

def main():
    project, cve_id, rl = args.project, args.cve_id, args.RL
    if rl:
        pseudo_path = os.path.join("../../realWorld/data/", project, cve_id, "pseudo")
    else:
        pseudo_path = os.path.join("../../data/", project, cve_id, "pseudo")
    # pseudo_path = os.path.join("../../data/", project, cve_id, "pseudo")
    changeSuffix(pseudo_path)
    # graph_path = os.path.join("../../data/", project, cve_id, "graph")
    deleteKeywords(pseudo_path)


if __name__ == "__main__":
    main()