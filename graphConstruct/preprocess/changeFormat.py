import re
import os
import chardet
import shutil

# Change the format to make it consistent

def change_deform_for_matrix(dataset):

    pattern_sub = r'<SUB>.*?</SUB>'
    pattern_brackets = r'<(\(.+)>'
    edge_label = ['AST', 'CFG', 'LastUse', 'ComputedFrom', 'CDG', 'DDG']

    for root, dirs, files in os.walk(dataset):
        for filename in files:
            if filename == ".DS_Store":
                continue
            file_path = os.path.join(root, filename)

            with open(file_path, 'rb') as file:
                result = chardet.detect(file.read())
                file_encoding = result['encoding']
            
            with open(file_path, 'r', encoding=file_encoding) as file:
                lines = file.readlines()
            
            with open(file_path, 'w') as file:
                filtered_lines = [line for line in lines if all(label not in line for label in edge_label)]
                for line in lines:  
                    if line in filtered_lines:  
                        modified_line = re.sub(pattern_sub, '', line)
                        modified_line = re.sub(pattern_brackets, lambda match: match.group().replace('"', ''), modified_line)
                        modified_line = re.sub(pattern_brackets, r'"\1"', modified_line)
                        file.write(modified_line)
                    else:
                        file.write(line)


def divide_by_datatype_diff(dataset):
    list_divide_file = ["AST", "LastUse", "ComputedFrom"]
    file_list = []
    for root, dirs, files in os.walk(dataset):
        for file in files:
            file_path = os.path.join(root, file)
            file_list.append(file_path)

    for index, filepath in enumerate(file_list):
        if os.path.basename(filepath) == ".DS_Store":
            continue
        for indexd, divide_file in enumerate(list_divide_file):
            destination_directory = os.path.join(os.path.dirname(filepath), divide_file +".dot")
            print(destination_directory)
            with open(filepath, 'r') as input_f:
                lines = input_f.readlines()
                filtered_lines_public = [line for line in lines if not any(keyword in line for keyword in list_divide_file)]
                filtered_edges = [line for line in lines if divide_file in line]
            with open(destination_directory, 'w') as output_f:
                lines = filtered_lines_public[:-1] + filtered_edges + [filtered_lines_public[-1]]
                output_f.writelines(lines)

        if os.path.exists(filepath):
            os.remove(filepath)
            print(f"{filepath} has been deleted.")
        else:
            print(f"{filepath} does not exist.")
        

def divide_by_datatype_detect(dataset):
    list_divide_file = ["AST", "LastUse", "ComputedFrom"]
    file_list = []
    for root, dirs, files in os.walk(dataset):
        for file in files:
            file_path = os.path.join(root, file)
            file_list.append(file_path)

    for index, filename in enumerate(file_list):

        if filename == ".DS_Store":
            continue
        source_file_path = os.path.dirname(filename)
        print("source_file_path", source_file_path)

        for indexd, divide_file in enumerate(list_divide_file):
            print("indexd, divide_file",indexd, divide_file)
            destination_directory = os.path.join(source_file_path, divide_file +".dot")
            source_file = source_file_path + "/ast_deform.dot"

            with open(source_file, 'r') as input_f:
                lines = input_f.readlines()

                filtered_lines_public = [line for line in lines if not any(keyword in line for keyword in list_divide_file)]

                filtered_edges = [line for line in lines if divide_file in line]
            with open(destination_directory, 'w') as output_f:
                lines = filtered_lines_public[:-1] + filtered_edges + [filtered_lines_public[-1]]
                output_f.writelines(lines)


        if os.path.exists(source_file):
            os.remove(source_file)
            print(f"{source_file} has been deleted.")
        else:
            print(f"{source_file} does not exist.")

