import re
import os
import chardet
import shutil

# Change the format to make it consistent

def change_deform_for_matrix(dataset):

    pattern_sub = r'<SUB>.*?</SUB>'
    pattern_brackets = r'<(\(.+)>'
    edge_label = ['AST', 'CFG', 'LastUse', 'ComputedFrom', 'CDG', 'DDG']

    # 遍历文件夹下的所有文件
    for root, dirs, files in os.walk(dataset):
        for filename in files:
            if filename == ".DS_Store":
                continue
            file_path = os.path.join(root, filename)
            # print("FILE_PATH",file_path)

            with open(file_path, 'rb') as file:
                result = chardet.detect(file.read())
                file_encoding = result['encoding']
            
            with open(file_path, 'r', encoding=file_encoding) as file:
                lines = file.readlines()
            
            with open(file_path, 'w') as file:
                filtered_lines = [line for line in lines if all(label not in line for label in edge_label)]
                for line in lines:  
                    if line in filtered_lines:  
                        # 使用正则表达式删除<SUB>16</SUB>及其中间内容
                        modified_line = re.sub(pattern_sub, '', line)
                        modified_line = re.sub(pattern_brackets, lambda match: match.group().replace('"', ''), modified_line)
                        modified_line = re.sub(pattern_brackets, r'"\1"', modified_line)
                        file.write(modified_line)
                    else:
                        file.write(line)
                # list_edge = [item for item in lines if item not in filtered_lines]
                # for line in list_edge:
                #     # 将修改后的行写入文件
                #     file.write(line)


def divide_by_datatype_diff(dataset):
    list_divide_file = ["AST", "LastUse", "ComputedFrom"]
    file_list = []
    for root, dirs, files in os.walk(dataset):
        # 将当前文件夹中的文件加入列表
        for file in files:
            file_path = os.path.join(root, file)
            file_list.append(file_path)
    # file_list = os.listdir(dataset)
    # print("file_list", file_list)
    # # 根据划分规则将文件分配到不同目标文件夹
    for index, filepath in enumerate(file_list):
        if os.path.basename(filepath) == ".DS_Store":
            continue
        for indexd, divide_file in enumerate(list_divide_file):
            destination_directory = os.path.join(os.path.dirname(filepath), divide_file +".dot")
            print(destination_directory)
            with open(filepath, 'r') as input_f:
                lines = input_f.readlines()
                # 公有的
                filtered_lines_public = [line for line in lines if not any(keyword in line for keyword in list_divide_file)]
                # 当前文件特有的边
                filtered_edges = [line for line in lines if divide_file in line]
            with open(destination_directory, 'w') as output_f:
                lines = filtered_lines_public[:-1] + filtered_edges + [filtered_lines_public[-1]]
                output_f.writelines(lines)

        # 删除原来的ast_deform.dot文件
        if os.path.exists(filepath):
            os.remove(filepath)
            print(f"{filepath} has been deleted.")
        else:
            print(f"{filepath} does not exist.")
        

# def divide_by_datatype(dataset):
#     list_divide_file = ["AST", "LastUse", "ComputedFrom"]
#     # file_list = []
#     # for root, dirs, files in os.walk(dataset):
#     #     # 将当前文件夹中的文件加入列表
#     #     for file in files:
#     #         file_path = os.path.join(root, file)
#     #         file_list.append(file_path)
#     file_list = os.listdir(dataset)

#     # 根据划分规则将文件分配到不同目标文件夹
#     for index, filename in enumerate(file_list):
#         # print(filename,file_list)
#         if filename == ".DS_Store":
#             continue
#         source_file_path = os.path.join(dataset, filename)
#         for indexd, divide_file in enumerate(list_divide_file):
#             # print("indexd, divide_file",indexd, divide_file)
#             destination_directory = os.path.join(source_file_path, divide_file +".dot")
            
#             destination_directory = destination_directory.replace("graph", "dividegraph")
#             # print("destination_directory", destination_directory)
#             source_file = source_file_path + "/ast_deform.dot"
#             with open(source_file, 'r') as input_f:
#                 lines = input_f.readlines()
#                 # 公有的
#                 filtered_lines_public = [line for line in lines if not any(keyword in line for keyword in list_divide_file)]
#                 # 当前文件特有的边
#                 filtered_edges = [line for line in lines if divide_file in line]
#             os.makedirs(os.path.dirname(destination_directory), exist_ok=True)
#             with open(destination_directory, 'w') as output_f:
#                 lines = filtered_lines_public[:-1] + filtered_edges + [filtered_lines_public[-1]]
#                 output_f.writelines(lines)

        # # 删除原来的ast_deform.dot文件
        # if os.path.exists(source_file):
        #     os.remove(source_file)
        #     print(f"{source_file} has been deleted.")
        # else:
        #     print(f"{source_file} does not exist.")

def divide_by_datatype_detect(dataset):
    list_divide_file = ["AST", "LastUse", "ComputedFrom"]
    file_list = []
    for root, dirs, files in os.walk(dataset):
        # 将当前文件夹中的文件加入列表
        for file in files:
            file_path = os.path.join(root, file)
            file_list.append(file_path)
    # file_list = os.listdir(dataset)

    # 根据划分规则将文件分配到不同目标文件夹
    for index, filename in enumerate(file_list):
        # print("filename", filename)
        if filename == ".DS_Store":
            continue
        source_file_path = os.path.dirname(filename)
        print("source_file_path", source_file_path)
        # source_file_path = dataset
        for indexd, divide_file in enumerate(list_divide_file):
            print("indexd, divide_file",indexd, divide_file)
            destination_directory = os.path.join(source_file_path, divide_file +".dot")
            source_file = source_file_path + "/ast_deform.dot"
            # source_file = source_file_path 
            with open(source_file, 'r') as input_f:
                lines = input_f.readlines()
                # 公有的
                filtered_lines_public = [line for line in lines if not any(keyword in line for keyword in list_divide_file)]
                # 当前文件特有的边
                filtered_edges = [line for line in lines if divide_file in line]
            with open(destination_directory, 'w') as output_f:
                lines = filtered_lines_public[:-1] + filtered_edges + [filtered_lines_public[-1]]
                output_f.writelines(lines)

        # 删除原来的ast_deform.dot文件
        if os.path.exists(source_file):
            os.remove(source_file)
            print(f"{source_file} has been deleted.")
        else:
            print(f"{source_file} does not exist.")

# dataset = "../../dataset_Bifrost/generated_graphs1/"
# # /home/dslab/zyl/new_SCVulPecker/dataset_Bifrost/generated_graphs1/

# dataset = "../../generated_graphs_CVE-2015-6031"
# change_deform_for_matrix(dataset)
# divide_by_datatype(dataset)
