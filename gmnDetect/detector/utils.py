# coding = utf-8
import os
import sys
sys.path.append(os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
print("sys.path", sys.path)
import torch
import collections
import numpy as np
from data.settings_hl_disjoint import *
from data.settings_disjoint import *
from graphMatrix.config import GRAPH_MODE
# from gmn.train_disjoint import edge_feature_dim, node_feature_dim
# import gmn.configure as config
# from gmn.utils import build_model


# import gmnDetect.gmn.graphembeddingnetwork
# import gmnDetect.gmn.graphmatchingnetwork

def compute_similarity(x, y):
    """This is the squared Euclidean distance."""
    return torch.sum((x - y) ** 2, dim=-1)

def load_model(project, cve_id, device, hl):
    if hl:
        cve_pkl_path = "../../data/{}/{}/model_hl/{}/".format(project, cve_id, GRAPH_MODE) + cve_id + '.pkl'
    else:
        cve_pkl_path = "../../data/{}/{}/model/{}/".format(project, cve_id, GRAPH_MODE) + cve_id + '.pkl'

    # cve_pkl_path = model_path + project + '/' + cve_id + '.pkl'
    model = torch.load(cve_pkl_path)
    return model.to(device)

# def load_model_ckpt(project, cve_id, device, bs, lr, epoch, hl, edge_feature_dim, node_feature_dim):
#     ckptName = 'b{}_lr{}_{}_log_ckpt'.format(bs, lr, cve_id)
#     if hl:
#         cve_ckpt_path = "../../data/{}/{}/model_hl/{}/saved_ckpt/".format(project, cve_id, GRAPH_MODE) + ckptName
#     else:
#         cve_ckpt_path = "../../data/{}/{}/model/{}/saved_ckpt/".format(project, cve_id, GRAPH_MODE) + ckptName
#     model, optimizer = build_model(config.get_disjoint_config(hl, edge_state_dim=edge_feature_dim), node_feature_dim, edge_feature_dim)
#     model.to(device)
#     checkpoint = torch.load(cve_ckpt_path)
#     model.load_state_dict(checkpoint["model_state_dict"])
#     # model.load_state_dict(cve_ckpt_path)
#     # optimizer.load_state_dict(cve_ckpt_path)
#     optimizer.load_state_dict(checkpoint["optimizer_state_dict"])
#     # model = torch.load(cve_ckpt_path)
#     return model.to(device)

GraphData = collections.namedtuple('GraphData', [
    'from_idx',
    'to_idx',
    'node_features',
    'edge_features',
    'graph_idx',
    'n_graphs']
)

def pack_batch_disjoint(nm_list, ad_list):
    num_node_list = [0]
    num_edge_list = []
    total_num_node = 0
    total_num_edge = 0
    batch_size = len(nm_list)
    for nm, am in zip(nm_list, ad_list):
        num_node_of_this_graph = nm.shape[0]
        num_node_list.append(num_node_of_this_graph)
        total_num_node += num_node_of_this_graph
        num_edge_of_this_graph = am.shape[0]
        num_edge_list.append(num_edge_of_this_graph)
        total_num_edge += num_edge_of_this_graph
    cumsum = np.cumsum(num_node_list)
    indices = np.repeat(np.arange(batch_size), num_edge_list)  # [num_edge_this_batch]
    scattered = cumsum[indices]  # [num_edge_this_batch, ]
    edges = np.concatenate(ad_list, axis=0)
    # edges[..., 0] += scattered
    # edges[..., 1] += scattered
    edges[..., 0] = edges[..., 0] + scattered
    edges[..., 1] = edges[..., 1] + scattered

    # * 引入 edge 特征
    if edges.shape[-1] > 2:
        edge_features = edges[..., 2:]
    else:
        edge_features = np.ones((total_num_edge, 1), dtype=np.float32)

    segment = np.repeat(np.arange(batch_size), np.array(num_node_list[1:]))
    return GraphData(
        from_idx=edges[..., 0],
        to_idx=edges[..., 1],
        node_features=np.concatenate(nm_list, axis=0),
        edge_features=edge_features,
        graph_idx=segment,
        n_graphs=batch_size,
    )

def write_lines(path, lines):
    f = open(path, 'w')
    w_box = []
    for line in lines:
        w_box.append(str(line) + '\n')
    f.writelines(w_box)
    f.close()

def load_vul_sample(project, cve_id, hl):
    if hl:
        vul_sample_path = '../../data/{}/{}/detect_hl/{}/vul_samples/'.format(project, cve_id, GRAPH_MODE)
    else:
        vul_sample_path = '../../data/{}/{}/detect/{}/vul_samples/'.format(project, cve_id, GRAPH_MODE)
    adj_path = os.path.join(vul_sample_path, 'adj.npy')
    nodes_path = os.path.join(vul_sample_path, 'node.npy')
    adj = np.load(adj_path)
    nodes = np.load(nodes_path)
    return adj, nodes

def load_vul_sample_large(project, cve_id, hl):
    if hl:
        vul_sample_path = '../../data/{}/{}/detect_hl/{}/vul_samples/'.format(project, cve_id, GRAPH_MODE)
    else:
        vul_sample_path = '../../data/{}/{}/detect/{}/vul_samples/'.format(project, cve_id, GRAPH_MODE)
    adj_path = os.path.join(vul_sample_path, 'adj.npz')
    nodes_path = os.path.join(vul_sample_path, 'node.npz')
    adj = np.load(adj_path)['arr_0'].astype(int)
    nodes = np.load(nodes_path)['arr_0']

    #adj = np.load(adj_path)
    #nodes = np.load(nodes_path)
    return adj, nodes


def get_graph(batch):
    graph = batch
    node_features = torch.from_numpy(graph.node_features.astype('float32'))
    edge_features = torch.from_numpy(graph.edge_features.astype('float32'))
    from_idx = torch.from_numpy(graph.from_idx).long()
    to_idx = torch.from_numpy(graph.to_idx).long()
    graph_idx = torch.from_numpy(graph.graph_idx).long()
    return node_features, edge_features, from_idx, to_idx, graph_idx

def reshape_and_split_tensor(tensor, n_splits):
    feature_dim = tensor.shape[-1]
    tensor = torch.reshape(tensor, [-1, feature_dim * n_splits])
    tensor_split = []
    for i in range(n_splits):
        tensor_split.append(tensor[:, feature_dim * i: feature_dim * (i + 1)])
    return tensor_split

'''
    Func: run_cmd
    @ Description:
        Run system command and get ret
    @ Parameters:
        cmd: string, command to run
    @ Ret:
        list with lines
'''
def run_cmd(cmd):
    ret_box = []
    pipe = os.popen(cmd)
    lines = pipe.readlines()
    pipe.close()

    for line in lines:
        # print("line", line)
        if line.strip():
            ret_box.append(line)
    return ret_box
