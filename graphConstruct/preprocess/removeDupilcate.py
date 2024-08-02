import os
import hashlib
from argparse import ArgumentParser


parser = ArgumentParser("remove duplicates pseudos.")
parser.add_argument("--project", type=str, default="curl")
parser.add_argument("--cve_id", type=str, default="CVE-2021-22901")
args = parser.parse_args()

def calculate_file_hash(file_path, hash_algo=hashlib.sha256):
    """Calculate the hash of a file."""
    hash_func = hash_algo()
    with open(file_path, 'rb') as f:
        while chunk := f.read(8192):
            hash_func.update(chunk)
    return hash_func.hexdigest()

def remove_duplicate_files(folder_path):
    """Remove duplicate files in a folder."""
    files_seen = {}
    for root, _, files in os.walk(folder_path):
        for file_name in files:
            file_path = os.path.join(root, file_name)
            file_hash = calculate_file_hash(file_path)
            
            if file_hash in files_seen:
                print(f"Duplicate found: {file_path} (removing)")
                os.remove(file_path)
            else:
                files_seen[file_hash] = file_path

if __name__ == "__main__":
    # folder_path = input("Enter the path to the folder: ")
    project, cve_id = args.project, args.cve_id
    pseudo_path = os.path.join("../../data/", project, cve_id, "pseudo")
    if os.path.isdir(pseudo_path):
        remove_duplicate_files(pseudo_path)
        print("Duplicate files removed.")
    else:
        print("The provided path is not a valid directory.")
