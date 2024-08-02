import numpy as np
import collections
import os
import sys
sys.path.append(os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))
import torch

import utils
from help_find_th2 import help_find
from argparse import ArgumentParser


parser = ArgumentParser("choosesample")
parser.add_argument("--project", type=str, default="curl")
parser.add_argument("--cve_id", type=str, default="CVE-2021-22901")
parser.add_argument("--hl", action="store_true", default=False)
parser.add_argument("--batch_size", type=int, default=32)
parser.add_argument("--learning_rate", type=float, default= 1e-4)
parser.add_argument("--num_epoch", type=int, default= 10)
parser.add_argument("--gpu", type=str, default="0")
parser.add_argument("--largeOrnot", type=bool, default=False)
args = parser.parse_args()

os.environ["CUDA_DEVICE_ORDER"] = "PCI_BUS_ID"
os.environ["CUDA_VISIBLE_DEVICES"] = args.gpu
use_cuda = torch.cuda.is_available()
device = torch.device('cuda:0' if use_cuda else 'cpu')

vul_diff_box = []
fix_diff_box = []


def gen_detect_pair(adj_npy, node_npy, project, cve_id):
    adj = np.load(adj_npy)
    node = np.load(node_npy)
    vul_adj, vul_node = utils.load_vul_sample(project, cve_id)
    return adj, node, vul_adj, vul_node

def gen_detect_pair_large(adj_npy, node_npy, project, cve_id, hl):
    adj = np.load(adj_npy)['arr_0'].astype(int)
    node = np.load(node_npy)['arr_0']
    vul_adj, vul_node = utils.load_vul_sample_large(project, cve_id, hl)
    return adj, node, vul_adj, vul_node

def label_data(model, project, cve_id, adj_npy, node_npy, label, hl):
    if large:
        adj, node, vul_adj, vul_node = gen_detect_pair_large(adj_npy, node_npy, project, cve_id, hl)
    else:
        adj, node, vul_adj, vul_node = gen_detect_pair(adj_npy, node_npy, project, cve_id)
    #adj, node, vul_adj, vul_node = gen_detect_pair(adj_npy, node_npy, project, cve_id)
    batch = utils.pack_batch_disjoint([node, vul_node], [adj, vul_adj])
    node_features, edge_features, from_idx, to_idx, graph_idx = utils.get_graph(batch)
    eval_pairs = model(node_features.to(device), edge_features.to(device), from_idx.to(device), to_idx.to(device), graph_idx.to(device), 128)
    x, y = utils.reshape_and_split_tensor(eval_pairs, 2)
    similarity = float(utils.compute_similarity(x, y)[0])

    if label == 'vul':
        vul_diff_box.append(similarity)
    else:
        fix_diff_box.append(similarity)


if __name__ == "__main__":
    # project, cve_id, gpu, large, hl = args.project, args.cve_id, args.gpu, args.largeOrnot, args.hl

    project, cve_id, large, bs, lr, epoch, hl = args.project, args.cve_id, args.largeOrnot, args.batch_size, args.learning_rate, args.num_epoch, args.hl

    model = utils.load_model_ckpt(project, cve_id, device, bs, lr, epoch, hl)

    # Main Test
    if hl:
        base_dir = os.path.join("../../data/", project, cve_id, "matrix_{}_divide_hl".format(utils.GRAPH_MODE), "train")
        from data.settings_hl_disjoint import vul_dict
    else:
        base_dir = os.path.join("../../data/", project, cve_id, "matrix_{}_divide".format(utils.GRAPH_MODE), "train")
        from data.settings_disjoint import vul_dict
    if large:
        adj_find_cmd = 'find {} -name \'*_adj.npz\' -type f'.format(base_dir)
    else:
        adj_find_cmd = 'find {} -name \'*_adj.npy\' -type f'.format(base_dir)
        

    adjs = utils.run_cmd(adj_find_cmd)
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


    true_positive_count = 0
    false_negative_count = 0
    for vul in dataset['vul']:
        label_data(model, project, cve_id, vul['adj'], vul['node'], 'vul', hl)

    true_negative_count = 0
    false_positive_count = 0
    for fix in dataset['fix']:
        label_data(model, project, cve_id, fix['adj'], fix['node'], 'fix', hl)

    print('[Info] VUL Max DIFF:', max(vul_diff_box), 'VUL Min DIFF:', min(vul_diff_box))
    print('[Info] FIX Max DIFF:', max(fix_diff_box), 'FIX Min DIFF:', min(fix_diff_box))

    if hl:
        with open("../../data/settings_hl_{}.py".format(utils.GRAPH_MODE), "w") as f:
            f.write(f"vul_dict = {vul_dict}")
        utils.write_lines('../../data/{}/{}/detect_hl/{}/vul_diff.txt'.format(project, cve_id, utils.GRAPH_MODE), vul_diff_box)
        utils.write_lines('../../data/{}/{}/detect_hl/{}/fix_diff.txt'.format(project, cve_id, utils.GRAPH_MODE), fix_diff_box)
    else:

        with open("../../data/settings_{}.py".format(utils.GRAPH_MODE), "w") as f:
            f.write(f"vul_dict = {vul_dict}")

        utils.write_lines('../../data/{}/{}/detect/{}/vul_diff.txt'.format(project, cve_id, utils.GRAPH_MODE), vul_diff_box)
        utils.write_lines('../../data/{}/{}/detect/{}/fix_diff.txt'.format(project, cve_id, utils.GRAPH_MODE), fix_diff_box)

    # if max(vul_diff_box) < min(fix_diff_box):
    #     middle_value = (max(vul_diff_box) + min(fix_diff_box)) / 2
    #     th = middle_value
        
    # else:
    #     print("can not find the best threshold.")
    #     # 使用helpFindTh寻找最佳阈值
    #     th = help_find(project, cve_id, gpu, large, vul_dict, hl, min(vul_diff_box), min(fix_diff_box))
    # print("find_best_th:", th)

    
    if max(vul_diff_box) < min(fix_diff_box):
        th = help_find(project, cve_id, gpu, large, vul_dict, hl, max(vul_diff_box), min(fix_diff_box))  
    else:
        print("can not find the best threshold.")
        # 使用helpFindTh寻找最佳阈值
        th = help_find(project, cve_id, gpu, large, vul_dict, hl, min(vul_diff_box), min(fix_diff_box))
    print("find_best_th:", th)

    # 使用helpFindTh寻找最佳阈值
    # th = help_find(project, cve_id, gpu, large, vul_dict, hl, min(vul_diff_box), min(fix_diff_box))
    # print("find_best_th:", th)
    

    key = project
    value = {cve_id:th}
    if key not in vul_dict:
        print("here")
        vul_dict[key] = value
    else:
        # if cve_id in vul_dict[key]:
        #     print("The vulnerability has already passed the threshold calculation!")
        # # 如果键已经存在，只更新值
        # else:
        vul_dict[key].update(value)

    if hl:
        with open("../../data/settings_hl_{}.py".format(utils.GRAPH_MODE), "w") as f:
            f.write(f"vul_dict = {vul_dict}")
    else:

        with open("../../data/settings_{}.py".format(utils.GRAPH_MODE), "w") as f:
            f.write(f"vul_dict = {vul_dict}")



    

    
