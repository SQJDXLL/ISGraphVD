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
    # device = gpu
    # print(project, cve_id, gpu, large, vul_dict, hl, vul_min, fix_min)
    os.environ["CUDA_DEVICE_ORDER"] = "PCI_BUS_ID"
    os.environ["CUDA_VISIBLE_DEVICES"] = gpu
    use_cuda = torch.cuda.is_available()
    device = torch.device('cuda:0' if use_cuda else 'cpu')

    # 记录所有的vul样本和fix样本的特征矩阵地址(注意的是detect的时候要把test文件夹下的sim文件夹和dissim文件夹都考虑进去)
    # 检测一个文件是否为漏洞，需要与vul_sample对应的求diff之前的图做diff再与vul_sample计算相似度，
    current_directory = os.path.dirname(os.path.abspath(__file__))
    # base_dir = os.path.join(current_directory, "validate_matrix")
    if hl:
        base_dir = os.path.join("../../data/", project, cve_id, "matrix_{}_divide_hl".format(utils.GRAPH_MODE), "validate")
    else:
        base_dir = os.path.join("../../data/", project, cve_id, "matrix_{}_divide".format(utils.GRAPH_MODE), "validate")
    # base_dir = '../../gmn_detect/graph_matrix/output/' + project + '/' + cve_id + '/test/'
    # base_dir = '../eval_box/' + project + '/' + cve_id
    if large:
        adj_find_cmd = 'find {} -name \'*_adj.npz\' -type f'.format(base_dir)
    else:
        adj_find_cmd = 'find {} -name \'*_adj.npy\' -type f'.format(base_dir)
    adjs = utils.run_cmd(adj_find_cmd)
    # print("here")
    dataset = {'vul': [], 'fix': []}
    if large:
        for adj in adjs:
            if '/vul/' in adj: # vul
                vul_adj = adj.strip()
                vul_node = vul_adj.replace('_adj.npz', '_node.npz').replace('/adj/', '/node/')
                item = {'adj': vul_adj, 'node': vul_node}
                dataset['vul'].append(item)
            elif '/fix/' in adj: # fix and other projects
                fix_adj = adj.strip()
                fix_node = fix_adj.replace('_adj.npz', '_node.npz').replace('/adj/', '/node/')
                item = {'adj': fix_adj, 'node': fix_node}
                dataset['fix'].append(item)
        print('[Info] Vul: {}, Normal: {}'.format(len(dataset['vul']), len(dataset['fix'])))
    else:
        for adj in adjs:
            if '/vul/' in adj: # vul
                vul_adj = adj.strip()
                vul_node = vul_adj.replace('_adj.npy', '_node.npy').replace('/adj/', '/node/')
                item = {'adj': vul_adj, 'node': vul_node}
                dataset['vul'].append(item)
            elif '/fix/' in adj: # fix and other projects
                fix_adj = adj.strip()
                fix_node = fix_adj.replace('_adj.npy', '_node.npy').replace('/adj/', '/node/')
                item = {'adj': fix_adj, 'node': fix_node}
                dataset['fix'].append(item)
            print('[Info] Vul: {}, Normal: {}'.format(len(dataset['vul']), len(dataset['fix'])))


    model = utils.load_model(project, cve_id, device, hl)

    #确定最终的阈值
    minValue = vul_min
    maxValue = fix_min
    print(minValue, maxValue)
    dict_th = { }
    for th in tqdm(np.arange(minValue, maxValue, 0.001)):
        true_positive_count = 0
        false_negative_count = 0
        for vul in dataset['vul']:
            # print("vul", vul)
            if not detect_vul(model, project, cve_id, vul['adj'], vul['node'], th, device, hl, large):
                false_negative_count += 1
            else:
                true_positive_count += 1

        true_negative_count = 0
        false_positive_count = 0
        for fix in dataset['fix']:
            # print("fix", fix)
            if detect_vul(model, project, cve_id, fix['adj'], fix['node'], th, device, hl, large):
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
    max_acc_th = max(dict_th, key=lambda k: dict_th[k])
    print("final_th_validate", max_acc_th)

    return max_acc_th
