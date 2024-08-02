import os
import numpy as np

def convert_npy_to_npz(folder_path):
    for root, dirs, files in os.walk(folder_path):
        for file in files:
            if file.endswith('.npy'):
                npy_path = os.path.join(root, file)
                file_name, _ = os.path.splitext(file)
                npz_path = os.path.join(root, file_name + '.npz')  # 修改后缀名

                data = np.load(npy_path)
                np.savez_compressed(npz_path, **{'arr_0': data})  # 使用默认的关键字参数 arr_0

                print(f"Converted {npy_path} to {npz_path}")

                # 删除原始的 .npy 文件
                os.remove(npy_path)
                print(f"Deleted {npy_path}")

if __name__ == "__main__":
    folder_path = "/home/anonymous/anonymous/SSGraph/data/dnsmasq/CVE-2015-8899/matrix_single_divide"
    convert_npy_to_npz(folder_path)
    print("Conversion completed.")
