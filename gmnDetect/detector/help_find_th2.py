# choose th base on acc
# 如果没有直接分割线的情况，调用该脚本以牺牲假阳性率的方式找出得到最高acc的阈值
# 找到find_th中的VUL Min DIFF和FIX Min DIFF
# 给定两个文件，分别包含相似图的距离和不相似图的距离，根据这些距离选出使验证集acc最高的值

import numpy as np
import collections
import os
import sys
sys.path.append(os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))
import torch
from tqdm import tqdm


# from settings import model_path, vul_samples, vul_dict, vul_sample, vul_dict_validate
# from choose_sample import pack_batch_disjoint

# from utils import load_model, pack_batch_disjoint

import utils

smi_box = []
dis_smi_box = []

# GraphData = collections.namedtuple('GraphData', [
#     'from_idx',
#     'to_idx',
#     'node_features',
#     'edge_features',
#     'graph_idx',
#     'n_graphs']
# )

# def load_model(project, cve_id):
#     cve_pkl_path = model_path + project + '/' + cve_id + '-disjoint.pkl'
#     model = torch.load(cve_pkl_path)
#     return model.to(device)

def gen_detect_pair(adj_npy, node_npy, project, cve_id, hl):
    adj = np.load(adj_npy)
    node = np.load(node_npy)
    vul_adj, vul_node = utils.load_vul_sample(project, cve_id, hl)
    return adj, node, vul_adj, vul_node

def gen_detect_pair_large(adj_npy, node_npy, project, cve_id, hl):
    adj = np.load(adj_npy)['arr_0'].astype(int)
    node = np.load(node_npy)['arr_0']
    vul_adj, vul_node = utils.load_vul_sample_large(project, cve_id, hl)
    return adj, node, vul_adj, vul_node


def detect_vul(model, project, cve_id, adj_npy, node_npy, th, device, hl, large):
    if large:
        adj, node, vul_adj, vul_node = gen_detect_pair_large(adj_npy, node_npy, project, cve_id, hl)
    else:
        adj, node, vul_adj, vul_node = gen_detect_pair(adj_npy, node_npy, project, cve_id, hl)
    
    batch = utils.pack_batch_disjoint([node, vul_node], [adj, vul_adj])
    node_features, edge_features, from_idx, to_idx, graph_idx = utils.get_graph(batch)

    eval_pairs = model(node_features.to(device), edge_features.to(device), from_idx.to(device), to_idx.to(device), graph_idx.to(device), 128)

    x, y = utils.reshape_and_split_tensor(eval_pairs, 2)
    similarity = float(utils.compute_similarity(x, y)[0])
    # print("similarity",  similarity)

    if similarity <= th:
        smi_box.append(similarity)
        return True
    else:
        dis_smi_box.append(similarity)
        return False

def help_find(project, cve_id, gpu, large, vul_dict, hl, vul_min, fix_min):
    #确定最终的阈值
    minValue = vul_min
    maxValue = fix_min
    print(minValue, maxValue)

    if hl:
        pathBox = os.path.join("../../data/", project, cve_id, "detect_hl", utils.GRAPH_MODE)
    else:
        pathBox = os.path.join("../../data/", project, cve_id, "detect", utils.GRAPH_MODE)

    
    pathFixDiff = os.path.join(pathBox, "fix_diff.txt")
    pathVulDiff = os.path.join(pathBox, "vul_diff.txt")

    with open(pathFixDiff, 'r') as filefix:
        # 逐行读取文件内容并保存到列表
        datafix = filefix.readlines()

    with open(pathVulDiff, 'r') as filevul:
        # 逐行读取文件内容并保存到列表
        datavul = filevul.readlines()

    dict_th = { }
    for th in tqdm(np.arange(minValue, maxValue, 0.001)):
        print("TH", th)
        true_positive_count = 0
        false_negative_count = 0
        for vul in datavul:
            if float(vul) <= th :
                true_positive_count += 1
            else:
                false_negative_count += 1
        true_negative_count = 0
        false_positive_count = 0
        for fix in datafix:
            if float(fix) <= th:
                false_positive_count += 1
            else:
                true_negative_count += 1

        # Max SMI为与样例相似（即相似度分数大于阈值，被判定为漏洞）的最大相似分数
        # Min DIS-SMI与样本样例不相似（即相似度分数小于阈值）的最小相似度得分
        # print('[Info] Max SMI:', max(smi_box), 'Min DIS-SMI:', min(dis_smi_box))
        print('[Info] FN: {}, FP: {}, TP: {}, TN {}'.format(false_negative_count, false_positive_count, true_positive_count, true_negative_count))
        acc = (true_negative_count + true_positive_count)/(true_negative_count + true_positive_count + false_positive_count + false_negative_count + 1e-10)
        fpr = false_positive_count/(false_positive_count + true_negative_count + 1e-10)
        dict_th[th] = acc
    print("dict_th", dict_th)
    max_acc_th = max(dict_th, key=lambda k: dict_th[k])
    print("final_th_validate", max_acc_th)

    return max_acc_th
