import os
import filecmp
import difflib
from argparse import ArgumentParser

parser = ArgumentParser("change graph to node and adj matrix.")
parser.add_argument("--project", type=str, default="curl")
parser.add_argument("--cve_id", type=str, default="CVE-2021-22901")
args = parser.parse_args()

def get_all_files(directory):
    """Recursively get all files in a directory."""
    files = []
    for root, _, filenames in os.walk(directory):
        for filename in filenames:
            files.append(os.path.join(root, filename))
    return files

def get_relative_paths(files, base_directory):
    """Get relative paths of files with respect to base_directory."""
    return [os.path.relpath(file, base_directory) for file in files]

def compare_files(file1, file2):
    """Compare two files and return True if they are identical, otherwise False."""
    with open(file1, 'r') as f1, open(file2, 'r') as f2:
        diff = difflib.unified_diff(
            f1.readlines(), f2.readlines(), fromfile=file1, tofile=file2)
        differences = list(diff)
        return len(differences) == 0

def find_common_files(dir1, dir2):
    """Find files with the same content in both directories."""
    files1 = get_all_files(dir1)
    files2 = get_all_files(dir2)

    relative_files1 = get_relative_paths(files1, dir1)
    relative_files2 = get_relative_paths(files2, dir2)

    common_files = set(relative_files1) & set(relative_files2)
    same_content_files = []

    for file in common_files:
        file1 = os.path.join(dir1, file)
        file2 = os.path.join(dir2, file)
        if compare_files(file1, file2):
            same_content_files.append(file)

    return same_content_files


if __name__ == "__main__":
    project, cve_id = args.project, args.cve_id
    dir1 = os.path.join('../data', project, cve_id, "pseudo")
    dir2 = os.path.join('data', project, cve_id, "pseudo")
    
    common_files = find_common_files(dir1, dir2)
    
    if common_files:
        print("Files with the same content:")
        for file in common_files:
            print(file)
    else:
        print("No files with the same content found.")
