import numpy as np
import collections
import os
import torch
import time

import utils
from argparse import ArgumentParser

parser = ArgumentParser("Detect")
parser.add_argument("--project", type=str, default="curl")
parser.add_argument("--cve_id", type=str, default="CVE-2021-22901")
parser.add_argument("--largeOrnot", type=bool, default=False)
parser.add_argument("--hl", action="store_true", default=False)
parser.add_argument("--gpu", type=str, default="0")
args = parser.parse_args()

os.environ["CUDA_DEVICE_ORDER"] = "PCI_BUS_ID"
os.environ["CUDA_VISIBLE_DEVICES"] = args.gpu
use_cuda = torch.cuda.is_available()
device = torch.device('cuda:0' if use_cuda else 'cpu')

smi_box = []
dis_smi_box = []

def gen_detect_pair(adj_npy, node_npy, project, cve_id):
    adj = np.load(adj_npy)
    node = np.load(node_npy)
    vul_adj, vul_node = utils.load_vul_sample(project, cve_id, hl)
    return adj, node, vul_adj, vul_node

def gen_detect_pair_large(adj_npy, node_npy, project, cve_id):
    adj = np.load(adj_npy)['arr_0'].astype(int)
    node = np.load(node_npy)['arr_0']

    vul_adj, vul_node = utils.load_vul_sample_large(project, cve_id, hl)
    return adj, node, vul_adj, vul_node


def detect_vul(model, project, cve_id, adj_npy, node_npy):
    if large:
        adj, node, vul_adj, vul_node = gen_detect_pair_large(adj_npy, node_npy, project, cve_id)
    else:
        adj, node, vul_adj, vul_node = gen_detect_pair(adj_npy, node_npy, project, cve_id)
    batch = utils.pack_batch_disjoint([node, vul_node], [adj, vul_adj])
    node_features, edge_features, from_idx, to_idx, graph_idx = utils.get_graph(batch)
    eval_pairs = model(node_features.to(device), edge_features.to(device), from_idx.to(device), to_idx.to(device), graph_idx.to(device), 128)
    x, y = utils.reshape_and_split_tensor(eval_pairs, 2)
    similarity = float(utils.compute_similarity(x, y)[0])
    # print("similar", similarity)
    # print(vul_dict[project][cve_id])
    if similarity <= vul_dict[project][cve_id]:
        smi_box.append(similarity)
        # print("sim")
        return True
    else:
        dis_smi_box.append(similarity)
        # print("dissim")
        return False

if __name__ == "__main__":

    project, cve_id, large, hl = args.project, args.cve_id, args.largeOrnot, args.hl
    if hl:
        if utils.GRAPH_MODE == 'single':
            from data.settings_hl_single import vul_dict
            base_dir = os.path.join("../../data/", project, cve_id, "matrix_{}_divide_hl".format(utils.GRAPH_MODE), "test")
        else:
            from data.settings_hl_disjoint import vul_dict
            base_dir = os.path.join("../../data/", project, cve_id, "matrix_{}_divide_hl".format(utils.GRAPH_MODE), "test")
    else:
        if utils.GRAPH_MODE == 'single':
            from data.settings_single import vul_dict
            base_dir = os.path.join("../../data/", project, cve_id, "matrix_{}_divide".format(utils.GRAPH_MODE), "test")
        else:
            from data.settings_disjoint import vul_dict
            base_dir = os.path.join("../../data/", project, cve_id, "matrix_{}_divide".format(utils.GRAPH_MODE), "test")
        
    for proj, cve_dict in vul_dict.items():
        if project != proj:
            continue
        for cve in cve_dict.keys():
            if cve != cve_id:
                continue

            model = utils.load_model(project, cve_id, device, hl)

            # Main Test
            print('[Info]', project, cve_id)
                

            if large:
                adj_find_cmd = 'find {} -name \'*_adj.npz\' -type f'.format(base_dir)
            else:
                adj_find_cmd = 'find {} -name \'*_adj.npy\' -type f'.format(base_dir)
            #adj_find_cmd = 'find {} -name \'*_adj.npy\' -type f'.format(base_dir)
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

            time_start = time.time()

            true_positive_count = 0
            false_negative_count = 0
            for vul in dataset['vul']:
                if not detect_vul(model, project, cve_id, vul['adj'], vul['node']):
                    false_negative_count += 1
                else:
                    true_positive_count += 1

            true_negative_count = 0
            false_positive_count = 0
            for fix in dataset['fix']:
                if detect_vul(model, project, cve_id, fix['adj'], fix['node']):
                    false_positive_count += 1
                else:
                    true_negative_count += 1

            time_end = time.time()
            detect_time_code = time_end - time_start
            print("detect_time_code", detect_time_code)
            
            print('[Info] Max SMI:', max(smi_box), 'Min DIS-SMI:', min(dis_smi_box))
            print('[Info] FN: {}, FP: {}, TP: {}, TN {}'.format(false_negative_count, false_positive_count, true_positive_count, true_negative_count))
# FP 代表把fix误判为vul，那就是该样本与sample的相似度小于阈值，应该降低阈值（去/data/settings上修改）