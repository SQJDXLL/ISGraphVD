# -*- coding: utf-8 -*-
import os
import re
from argparse import ArgumentParser

parser = ArgumentParser("process pseudo for construct graphs.")
parser.add_argument("--project", type=str, default="curl")
parser.add_argument("--cve_id", type=str, default="CVE-2021-22901")
parser.add_argument('--RL', action='store_true', help='Disable all output')
parser.add_argument("--mission_id", type=str, default="hchdvbjsjci-bhjdsbcj")
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
        
        if os.path.isfile(file_path):
            try:
                with open(file_path, 'rb') as file:
                    lines = file.readlines()
                # *__cdecl
                if lines:
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
                        modified_line = modified_line.encode()
                        modified_line = remove_angle_brackets_bytes(modified_line)
                        modified_lines.append(modified_line)

                with open(file_path, 'wb') as file:
                    file.writelines(modified_lines)
                    
            except Exception as e:
                print(f"处理文件 {file_path} 时出现错误: {str(e)}")

def main():
    if args.mission_id:
        project, cve_id, rl, mission_id  = args.project, args.cve_id, args.RL, args.mission_id
    else:
        project, cve_id, rl  = args.project, args.cve_id, args.RL
    if rl:
        pseudo_path = os.path.join("../../data_detect/data/", mission_id, project, cve_id, "pseudo")
    else:
        pseudo_path = os.path.join("../../data/", project, cve_id, "pseudo")
    changeSuffix(pseudo_path)
    deleteKeywords(pseudo_path)


if __name__ == "__main__":
    main()
