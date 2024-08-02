# coding = utf-8
import numpy as np
import collections
import os
import sys
sys.path.append(os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))
import torch
import shutil

import utils
from argparse import ArgumentParser

parser = ArgumentParser("Evaluate the method for diff graph.")
parser.add_argument("--project", type=str, default="curl")
parser.add_argument("--cve_id", type=str, default="CVE-2021-22901")
parser.add_argument("--largeOrnot", type=bool, default=False)
parser.add_argument("--hl", action="store_true", default=False)
parser.add_argument("--gpu", type=str, default="0")
args = parser.parse_args()

os.environ["CUDA_DEVICE_ORDER"] = "PCI_BUS_ID"
os.environ["CUDA_VISIBLE_DEVICES"] = args.gpu

# 在一个"漏洞"样本集合中，找到一个与已有样本最不相似的样本，并将该样本添加到样本集合中，
# 以进一步扩充训练数据。具体的逻辑和效果可能需要根据实际的数据和模型情况进行验证和确认。

use_cuda = torch.cuda.is_available()
device = torch.device('cuda:0' if use_cuda else 'cpu')



vul_diff_box = []
fix_diff_box = []


def gen_detect_pair(adj_npy1, node_npy1, adj_npy2, node_npy2):
    adj1 = np.load(adj_npy1)
    node1 = np.load(node_npy1)
    adj2 = np.load(adj_npy2)
    node2 = np.load(node_npy2)
    return adj1, node1, adj2, node2
   
def gen_detect_pair_large(adj_npy1, node_npy1, adj_npy2, node_npy2):
    adj1 = np.load(adj_npy1)['arr_0'].astype(int)
    node1 = np.load(node_npy1)['arr_0']            
    adj2 = np.load(adj_npy2)['arr_0'].astype(int)                 
    node2 = np.load(node_npy2)['arr_0']
    return adj1, node1, adj2, node2
    

def get_dissim(model, adj_npy1, node_npy1, adj_npy2, node_npy2):
   
    if large:
        adj1, node1, adj2, node2 = gen_detect_pair_large(adj_npy1, node_npy1, adj_npy2, node_npy2)
    else:
        adj1, node1, adj2, node2 = gen_detect_pair(adj_npy1, node_npy1, adj_npy2, node_npy2)
    batch = utils.pack_batch_disjoint([node1, node2], [adj1, adj2])
    node_features, edge_features, from_idx, to_idx, graph_idx = utils.get_graph(batch)

    eval_pairs = model(node_features.to(device), edge_features.to(device), from_idx.to(device), to_idx.to(device), graph_idx.to(device), 128)

    x, y = utils.reshape_and_split_tensor(eval_pairs, 2)
    similarity = float(utils.compute_similarity(x, y)[0])

    return similarity


if __name__ == "__main__":

    project, cve_id, large, hl = args.project, args.cve_id, args.largeOrnot, args.hl

    model = utils.load_model(project, cve_id, device, hl)

    # Main Test
    if hl:
        base_dir = os.path.join("../../data/", project, cve_id, "matrix_{}_divide_hl".format(utils.GRAPH_MODE), "train")
    else:
        base_dir = os.path.join("../../data/", project, cve_id, "matrix_{}_divide".format(utils.GRAPH_MODE), "train")
    # base_dir = '~/workspace/output/' + project + '/' + cve_id + '/train/'
    # base_dir = '../output0106/dnsmasq/CVE-2017-14492/train/'
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

    vul_total_dissim = {}
    for vul1 in dataset['vul']:
        try:
            print('[Info] Processing:', vul1['adj'])
            vul_total_dissim[vul1['adj']] = 0
            for vul2 in dataset['vul']:
                if vul1 != vul2:
                    dis_sim = get_dissim(model, vul1['adj'], vul1['node'], vul2['adj'], vul2['node'])
                    vul_total_dissim[vul1['adj']] += dis_sim
            torch.cuda.empty_cache()
        except Exception as e:
            print('[Error] Pass:', vul1['adj'], str(e))
            continue

    smallest_dissim = 999999999999
    smallest_adj = ''
    for adj, total_dissim in vul_total_dissim.items():
        if total_dissim < smallest_dissim:
            smallest_dissim = total_dissim
            smallest_adj = adj
    
    print('[Info] SMALLEST:', smallest_adj, smallest_dissim)

    if hl:
        saved_path = '../../data/{}/{}/detect_hl/{}/vul_samples/'.format(project, cve_id, utils.GRAPH_MODE)
        os.makedirs(saved_path, exist_ok=True)
        shutil.copy(smallest_adj, saved_path + 'adj.npz')
        node_npy = smallest_adj.replace('/adj/', '/node/').replace('_adj.npz', '_node.npz')
        shutil.copy(node_npy, saved_path + 'node.npz')
    else:
        saved_path = '../../data/{}/{}/detect/{}/vul_samples/'.format(project, cve_id, utils.GRAPH_MODE)
        os.makedirs(saved_path, exist_ok=True)
        shutil.copy(smallest_adj, saved_path + 'adj.npz')
        node_npy = smallest_adj.replace('/adj/', '/node/').replace('_adj.npz', '_node.npz')
        shutil.copy(node_npy, saved_path + 'node.npz')
        

    
