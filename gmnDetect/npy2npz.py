import os
import numpy as np

def convert_npy_to_npz(folder_path):
    for root, dirs, files in os.walk(folder_path):
        for file in files:
            if file.endswith('.npy'):
                npy_path = os.path.join(root, file)
                file_name, _ = os.path.splitext(file)
                npz_path = os.path.join(root, file_name + '.npz')   

                data = np.load(npy_path)
                np.savez_compressed(npz_path, **{'arr_0': data})   
                print(f"Converted {npy_path} to {npz_path}")
                os.remove(npy_path)
                print(f"Deleted {npy_path}")
