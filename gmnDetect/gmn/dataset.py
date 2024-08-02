import abc
import os
import os.path as osp
import sys
# sys.path.append(os.path.split(sys.path[0])[0])
sys.path.append(os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))
# print(sys.path)
import math
import random
import collections
import copy

import numpy as np
import torch
import networkx as nx
import torch.nn.functional as F

from rich.progress import track

import json
# from diffPcode import *
# from graphMatrix.changeFormat import change_deform_for_matrix

GraphData = collections.namedtuple('GraphData', [
    'from_idx',
    'to_idx',
    'node_features',
    'edge_features',
    'graph_idx',
    'n_graphs'])

"""A general Interface"""


class GraphSimilarityDataset(object):
    """Base class for all the graph similarity learning datasets.
  This class defines some common interfaces a graph similarity dataset can have,
  in particular the functions that creates iterators over pairs and triplets.
  """

    @abc.abstractmethod
    def triplets(self):
        """Create an iterator over triplets.
    Args:
      batch_size: int, number of triplets in a batch.
    Yields:
      graphs: a `GraphData` instance.  The batch of triplets put together.  Each
        triplet has 3 graphs (x, y, z).  Here the first graph is duplicated once
        so the graphs for each triplet are ordered as (x, y, x, z) in the batch.
        The batch contains `batch_size` number of triplets, hence `4*batch_size`
        many graphs.
    """
        pass

    @abc.abstractmethod
    def pairs(self):
        """Create an iterator over pairs.
    Args:
      batch_size: int, number of pairs in a batch.
    Yields:
      graphs: a `GraphData` instance.  The batch of pairs put together.  Each
        pair has 2 graphs (x, y).  The batch contains `batch_size` number of
        pairs, hence `2*batch_size` many graphs.
      labels: [batch_size] int labels for each pair, +1 for similar, -1 for not.
    """
        pass


"""Graph Edit Distance Task"""


# Graph Manipulation Functions
def permute_graph_nodes(g):
    """Permute node ordering of a graph, returns a new graph."""
    n = g.number_of_nodes()
    new_g = nx.Graph()
    new_g.add_nodes_from(range(n))
    perm = np.random.permutation(n)
    edges = g.edges()
    new_edges = []
    for x, y in edges:
        new_edges.append((perm[x], perm[y]))
    new_g.add_edges_from(new_edges)
    return new_g


def substitute_random_edges(g, n):
    """Substitutes n edges from graph g with another n randomly picked edges."""
    g = copy.deepcopy(g)
    n_nodes = g.number_of_nodes()
    edges = list(g.edges())
    # sample n edges without replacement
    e_remove = [
        edges[i] for i in np.random.choice(np.arange(len(edges)), n, replace=False)
    ]
    edge_set = set(edges)
    e_add = set()
    while len(e_add) < n:
        e = np.random.choice(n_nodes, 2, replace=False)
        # make sure e does not exist and is not already chosen to be added
        if (
                (e[0], e[1]) not in edge_set
                and (e[1], e[0]) not in edge_set
                and (e[0], e[1]) not in e_add
                and (e[1], e[0]) not in e_add
        ):
            e_add.add((e[0], e[1]))

    for i, j in e_remove:
        g.remove_edge(i, j)
    for i, j in e_add:
        g.add_edge(i, j)
    return g


class yzdDataset(GraphSimilarityDataset):
    def __init__(self, dataset_dir, num_epoch, batch_size, num_pairs=None, num_triplets=None, compare_path=None, name='default'):
        self.name = name
        self.dataset_dir = dataset_dir
        self.num_epoch = num_epoch
        self.batch_size = batch_size
        self.compare_path = compare_path
        self.vul_dir = os.path.join(self.dataset_dir, 'vul')
        self.fix_dir = os.path.join(self.dataset_dir, 'fix') 

        self.vul_nm_dir_of_this_cve = os.path.join(self.vul_dir, 'node')
        self.vul_nm_filename_of_this_cve = [os.path.join(self.vul_nm_dir_of_this_cve, file) for file in
                                            os.listdir(self.vul_nm_dir_of_this_cve) if not file.startswith('.')]
        tmp_record = [item.split('/')[-1].rsplit('.', 1)[0].rsplit('_', 1)[0] for item in
                      self.vul_nm_filename_of_this_cve]
        self.vul_ad_dir_of_this_cve = os.path.join(self.vul_dir, 'adj')
        self.vul_ad_filename_of_this_cve = [os.path.join(self.vul_ad_dir_of_this_cve, id + '_adj.npz') for id in
                                            tmp_record]
        self.zipped_vul_of_this_cve = list(zip(self.vul_nm_filename_of_this_cve, self.vul_ad_filename_of_this_cve))
        # print('zipped_vul_of_this_cve = {}'.format(self.zipped_vul_of_this_cve))
        self.fix_nm_dir_of_this_cve = os.path.join(self.fix_dir, 'node')
        self.fix_nm_filename_of_this_cve = [os.path.join(self.fix_nm_dir_of_this_cve, file) for file in
                                            os.listdir(self.fix_nm_dir_of_this_cve) if not file.startswith('.')]
        tmp_record = [item.split('/')[-1].rsplit('.', 1)[0].rsplit('_', 1)[0] for item in
                      self.fix_nm_filename_of_this_cve]
        self.fix_ad_dir_of_this_cve = os.path.join(self.fix_dir, 'adj')
        self.fix_ad_filename_of_this_cve = [os.path.join(self.fix_ad_dir_of_this_cve, id + '_adj.npz') for id in
                                            tmp_record]
        self.zipped_fix_of_this_cve = list(zip(self.fix_nm_filename_of_this_cve, self.fix_ad_filename_of_this_cve))

        self.num_vul_samples = len(self.vul_nm_filename_of_this_cve)
        self.num_fix_samples = len(self.fix_nm_filename_of_this_cve)

        #vul和vul组成相似对
        similar_pairs_list = [(*item1, *item2, 1) for item1 in self.zipped_vul_of_this_cve for item2 in
                              self.zipped_vul_of_this_cve if not item1 == item2]  # num_vul*num_vul
        #vul和fix组成不相似对
        dissimilar_pairs_list = [(*item1, *item2, -1) for item1 in self.zipped_vul_of_this_cve for item2 in
                                 self.zipped_fix_of_this_cve]  # num_vul * num_fix
        
        mixed_pairs_list = similar_pairs_list + dissimilar_pairs_list
        self.num_pairs_in_total = len(mixed_pairs_list)

        if num_pairs:
            if num_pairs < self.num_pairs_in_total:
                self.selected_pairs = random.sample(population=mixed_pairs_list, k=num_pairs)
            else:
                self.selected_pairs = mixed_pairs_list  # use all sample
        else:
            self.selected_pairs = mixed_pairs_list  # use all sample

    @property
    # 由于一个epoch将所有的数据都投入训练，所以num_pairs_batch就是一个epoch有几个batchsize
    def num_pairs_batch(self):
        return int(len(self.selected_pairs) / self.batch_size)

    # @property
    # def num_triplets_batch(self):
    #     return int(self.num_triplets / self.batch_size)

    # 直接把所有epoch的所有对生成
    @property
    def generate_all_data(self):
        nm_ad_of_all_epoch = []
        # ad_of_all_epoch = []
        label_of_all_epoch = []
        for epoch_idx in range(self.num_epoch):
            # 在这里打乱顺序，方便后面将tensor统一shape
            random.shuffle(self.selected_pairs)
            for pair in track(self.selected_pairs):
                nm1_path, ad1_path, nm2_path, ad2_path, label = pair
                # 将列表中所有元素转换成tensor,以便后续使用pytorch的dataloader
                nm_ad_of_all_epoch.append([torch.from_numpy(np.load(nm1_path)['arr_0']), torch.from_numpy(np.load(nm2_path)['arr_0']), torch.from_numpy(np.load(ad1_path)['arr_0']), torch.from_numpy(np.load(ad2_path)['arr_0'])])
                tensor_label = torch.tensor(label)
                label_of_all_epoch.append(tensor_label)
        return nm_ad_of_all_epoch, label_of_all_epoch


    @property
    def _pairs_generator(self):

        if self.compare_path:
            print("[+] Load pair list from {}".format(self.compare_path))
            compare_path = osp.join(self.compare_path, "compare.json")
            with open(compare_path) as f:
                selected_pairs = json.load(f)
            print("[+] Load {} pairs from {}".format(len(selected_pairs), compare_path))
        # num_epoch 在config文件里，默认为10
        for epoch_idx in range(self.num_epoch):
            # 每个epoch把数据集中所有的数据都跑了
            print('{}: epoch_{}: the length of seleted_pairs is: {}'.format(
                self.name, epoch_idx, len(self.selected_pairs))
            )
            # i += 1
            # j = 0
            random.shuffle(self.selected_pairs)
            for pair in self.selected_pairs:
                nm1_path, ad1_path, nm2_path, ad2_path, label = pair
                nm1 = np.load(nm1_path)['arr_0']
                ad1 = np.load(ad1_path)['arr_0']
                nm2 = np.load(nm2_path)['arr_0']
                ad2 = np.load(ad2_path)['arr_0']
                # print("nm1, nm2", nm1_path, nm2_path)
                yield nm1, ad1, nm2, ad2, label

    def pairs(self):
        pairs_generator = self._pairs_generator
        while True:
            nm_of_this_batch = []
            ad_of_this_batch = []
            label_of_this_batch = []
            # 使用生成器组成每个batch的数据，函数返回的就是node列表，对应的边列表，以及对应的标签数组
            for sample_idx in range(self.batch_size):
                try:
                    nm1, ad1, nm2, ad2, label = next(pairs_generator)
                except StopIteration:
                    return
                nm_of_this_batch.append(nm1)
                nm_of_this_batch.append(nm2)
                ad_of_this_batch.append(ad1)
                ad_of_this_batch.append(ad2)
                label_of_this_batch.append(label)
            batched_label = np.array(label_of_this_batch, dtype=int)
            yield self._pack_batch(nm_of_this_batch, ad_of_this_batch), batched_label

    @property
    def _triplets_generator(self):
        vul2_fix1 = [(*item1, *item2, *item3) for item1 in self.zipped_vul_of_this_cve for item2 in
                     self.zipped_vul_of_this_cve for item3 in self.zipped_fix_of_this_cve if
                     not item1 == item2]  # vul, vul, fix
        mixed_triplet_list = vul2_fix1
        if self.num_triplets < self.num_triplets_in_total:
            seleted_triplets = random.sample(population=mixed_triplet_list, k=self.num_triplets)
        else:
            seleted_triplets = mixed_triplet_list  # use all
        for epoch_idx in range(self.num_epoch):
            random.shuffle(seleted_triplets)
            for triplet in seleted_triplets:
                nm1_path, ad1_path, nm2_path, ad2_path, nm3_path, ad3_path = triplet
                nm1 = np.load(nm1_path)['arr_0']
                ad1 = np.load(ad1_path)['arr_0']
                nm2 = np.load(nm2_path)['arr_0']
                ad2 = np.load(ad2_path)['arr_0']
                nm3 = np.load(nm3_path)['arr_0']
                ad3 = np.load(ad3_path)['arr_0']
                yield nm1, ad1, nm2, ad2, nm3, ad3

    def triplets(self):
        triplet_generator = self._triplets_generator
        while True:
            nm_of_this_batch = []
            ad_of_this_batch = []
            for sample_idx in range(self.batch_size):
                try:
                    nm1, ad1, nm2, ad2, nm3, ad3 = next(triplet_generator)
                except StopIteration:
                    return
                nm_of_this_batch.append(nm1)
                nm_of_this_batch.append(nm2)
                nm_of_this_batch.append(nm1)
                nm_of_this_batch.append(nm3)
                ad_of_this_batch.append(ad1)
                ad_of_this_batch.append(ad2)
                ad_of_this_batch.append(ad1)
                ad_of_this_batch.append(ad3)
                # batchsize个图对应的节点矩阵和边矩阵
            yield self._pack_batch(nm_of_this_batch, ad_of_this_batch)

    def _pack_batch(self, nm_list, ad_list):
        '''
        :param graphs: a list of (nm_matrix[num_nodes, node_feature_dims], am_matrix[num_edge, 2]) pairs. nm/am_matrixes are all numpy array.
        :return:
        '''
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

        #for ad in ad_list:
            #print(ad.shape)
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
        return GraphData(from_idx=edges[..., 0],
                         to_idx=edges[..., 1],
                         node_features=np.concatenate(nm_list, axis=0),
                         edge_features=edge_features,
                         graph_idx=segment,
                         n_graphs=batch_size
                         )
 
  

  
# 假设您已经有了一系列的tensor数据nm_tensors和ad_tensors在GPU上  
# nm_tensors = [...]  # 节点特征张量列表  
# ad_tensors = [...]  # 邻接矩阵张量列表，其中每个张量的形状为(num_edges, 2)  
# device = 'cuda:0'  # 假设GPU设备为'cuda:0'  
  
# 使用修改后的_pack_batch函数  
# graph_data = _pack_batch(nm_tensors, ad_tensors, device)


#for diffGraph
class zylDataset(GraphSimilarityDataset):
    def __init__(self, dataset_dir, num_epoch, batch_size,  max_nodes_limit, num_pairs=None, num_triplets=None, compare_path=None, name='default'):
        self.name = name
        self.dataset_dir = dataset_dir
        self.num_epoch = num_epoch
        self.batch_size = batch_size
        self.compare_path = compare_path
        self.max_nodes_limit = max_nodes_limit

        # src/gmn_detect/dataset_diff_matrix/miniupnpc/CVE-2015-6031/train/dissim
        # 构建相似与不相似路径共np.load

        #output:(vul_path1,vul_path1,1) (vul_path1,fix_path2,-1)
        # path为node.npz和adj.npz路径组成的列表 vul_path=[path_node.npz, path_adj.npz]

        #先由dataset_dir得到sim和dissim的路径
        self.sim_path = os.path.join(dataset_dir, "sim")
        self.dissim_path = os.path.join(dataset_dir, "dissim")

        self.fix_nm_filename_of_this_cve = [ ]
        self.fix_ad_filename_of_this_cve = [ ]
        self.vul_nm_filename_of_this_cve = [ ]
        self.vul_ad_filename_of_this_cve = [ ]
        self.similar_pairs_list = [ ]
        self.dissimilar_pairs_list = [ ]
        #开始组相似对和不相似对
        for item1 in os.listdir(self.sim_path):
            
            names = item1.split('&VS&')
            # print("names", names)
            if len(names) == 2:
                name1 = names[0].strip().split("\'")[0]
                name2 = names[1].strip().split("\'")[0]
                # print(os.system('pwd'))
                # print(name1,name2,len(name1), len(name2))
            item1 = "\'" + item1 + "\'"

            path_name1 = os.path.join(self.sim_path, item1, name1)
            path_name2 = os.path.join(self.sim_path, item1, name2)

            path_node1  = os.path.join(path_name1, name1 + "_node.npz")
            path_adj1  = os.path.join(path_name1, name1 + "_adj.npz")
            path_node2  = os.path.join(path_name2, name2 + "_node.npz")
            path_adj2  = os.path.join(path_name2, name2 + "_adj.npz")
            self.similar_pairs_list.append((path_node1, path_adj1, path_node2, path_adj2, 1))

        for item1 in os.listdir(self.dissim_path):
            names = item1.split('&VS&')
            if len(names) == 2:
                name1 = names[0].strip().split("\'")[0]
                name2 = names[1].strip().split("\'")[0]
            item1 = "\'" + item1 + "\'"

            path_name1 = os.path.join(self.dissim_path, item1, name1)
            path_name2 = os.path.join(self.dissim_path, item1, name2)

            path_node1  = os.path.join(path_name1, name1 + "_node.npz")
            path_adj1  = os.path.join(path_name1, name1 + "_adj.npz")
            path_node2  = os.path.join(path_name2, name2 + "_node.npz")
            path_adj2  = os.path.join(path_name2, name2 + "_adj.npz")
            self.dissimilar_pairs_list.append((path_node1, path_adj1, path_node2, path_adj2, -1))

        mixed_pairs_list = self.similar_pairs_list + self.dissimilar_pairs_list
        
        self.num_pairs_in_total = len(mixed_pairs_list)

        if num_pairs:
            if num_pairs < self.num_pairs_in_total:
                self.selected_pairs = random.sample(population=mixed_pairs_list, k=num_pairs)
            else:
                self.selected_pairs = mixed_pairs_list  # use all sample
        else:
            self.selected_pairs = mixed_pairs_list  # use all sample

    @property
    # 数据集一共能组成多少对
    def num_pairs_batch(self):
        return int(len(self.selected_pairs) / self.batch_size)

    # @property
    # def num_triplets_batch(self):
    #     return int(self.num_triplets / self.batch_size)

    @property
    def _pairs_generator(self):

        if self.compare_path:
            print("[+] Load pair list from {}".format(self.compare_path))
            compare_path = osp.join(self.compare_path, "compare.json")
            with open(compare_path) as f:
                selected_pairs = json.load(f)
            print("[+] Load {} pairs from {}".format(len(selected_pairs), compare_path))
        # num_epoch 在config文件里，默认为10
        for epoch_idx in range(self.num_epoch):
            # 每个epoch把数据集中所有的数据都跑了
            print('{}: epoch_{}: the length of seleted_pairs is: {}'.format(
                self.name, epoch_idx, len(self.selected_pairs))
            )
            # i += 1
            # j = 0
            random.shuffle(self.selected_pairs)
            for pair in self.selected_pairs:
                # j += 1
                # print('inner loop:', i, j)
                # for item in pair:
                #     print(item)
                nm1_path, ad1_path, nm2_path, ad2_path, label = pair
                nm1_path = nm1_path.replace("'", "").replace("&", "\&").replace("\\", "")
                ad1_path = ad1_path.replace("'", "").replace("&", "\&").replace("\\", "")
                nm2_path = nm2_path.replace("'", "").replace("&", "\&").replace("\\", "")
                ad2_path = ad2_path.replace("'", "").replace("&", "\&").replace("\\", "")
                if os.path.exists(nm1_path) and os.path.exists(ad1_path) and os.path.exists(nm2_path) and os.path.exists(ad2_path) :
                    
                    # print("^^^^^^^",nm2_path)
                    
                    # print("***********nm1_path", nm1_path)

                    # if os.path.exists(nm1_path):
                    #     print("1")
                    # else:
                    #     raise BaseException(nm1_path, os.path.dirnsame(nm1_path))

                    nm1 = np.load(nm1_path)['arr_0']
                    ad1 = np.load(ad1_path)['arr_0']
                    nm2 = np.load(nm2_path)['arr_0']
                    ad2 = np.load(ad2_path)['arr_0']
                else:
                    continue
                yield nm1, ad1, nm2, ad2, label


    def pairs(self):
        pairs_generator = self._pairs_generator
        while True:
            nm_of_this_batch = []
            ad_of_this_batch = []
            label_of_this_batch = []
            # 使用生成器组成每个batch的数据，函数返回的就是node列表，对应的边列表，以及对应的标签数组
            for sample_idx in range(self.batch_size):
                try:
                    nm1, ad1, nm2, ad2, label = next(pairs_generator)
                except StopIteration:
                    return
                nm_of_this_batch.append(nm1)
                nm_of_this_batch.append(nm2)
                ad_of_this_batch.append(ad1)
                ad_of_this_batch.append(ad2)
                label_of_this_batch.append(label)
            batched_label = np.array(label_of_this_batch, dtype=int)
            # print(ad_of_this_batch)
            yield self._pack_batch(nm_of_this_batch, ad_of_this_batch), batched_label


    def _pack_batch(self, nm_list, ad_list):
        '''
        :param graphs: a list of (nm_matrix[num_nodes, node_feature_dims], am_matrix[num_edge, 2]) pairs. nm/am_matrixes are all numpy array.
        :return:
        '''

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
        return GraphData(from_idx=edges[..., 0],
                         to_idx=edges[..., 1],
                         node_features=np.concatenate(nm_list, axis=0),
                         edge_features=edge_features,
                         graph_idx=segment,
                         n_graphs=batch_size
                         )





class yzdDatasetNew(GraphSimilarityDataset):
    def __init__(self, nm_dir_vul, am_dir_vul, nm_dir_fix, am_dir_fix, num_epoch, batch_size, max_num_node_of_one_graph,
                 step_per_epoch, num_edge_type=1):
        self.nm_dir_vul = nm_dir_vul
        self.am_dir_vul = am_dir_vul
        self.nm_dir_fix = nm_dir_fix
        self.am_dir_fix = am_dir_fix
        self.num_epoch = num_epoch
        self.batch_size = batch_size
        self.num_edge_type = num_edge_type
        self.max_num_node_of_one_graph = max_num_node_of_one_graph
        if self.max_num_node_of_one_graph:
            self.inital_nm = np.ones(shape=(max_num_node_of_one_graph + 1, 1))
        self.step_per_epoch = step_per_epoch
        self.label_list = ['vul', 'fix']
        self.label_to_sample_sha_dict = {'vul': [], 'fix': []}
        for filename in os.listdir(self.nm_dir_vul):
            if filename.endswith('.npz'):
                sample_sha = filename[:-9]
                self.label_to_sample_sha_dict['vul'].append(sample_sha)
        for filename in os.listdir(self.nm_dir_fix):
            if filename.endswith('.npz'):
                sample_sha = filename[:-9]
                self.label_to_sample_sha_dict['fix'].append(sample_sha)
    @property
    def num_pairs_batch(self):
        return self.step_per_epoch

    @property
    def num_triplets_batch(self):
        return self.step_per_epoch

    def _get_nm_am_dir(self, label_name):
        if label_name == 'vul':
            return self.nm_dir_vul, self.am_dir_vul
        elif label_name == 'fix':
            return self.nm_dir_fix, self.am_dir_fix
        else:
            print('label error!')
            sys.exit(1)

    def sim_pair_sampling(self):
        sampled_label = random.choice(seq=self.label_list)
        nm_dir, am_dir = self._get_nm_am_dir(sampled_label)
        if self.max_num_node_of_one_graph:
            nm_1 = self.inital_nm
            nm_2 = self.inital_nm
            while nm_1.shape[0] >= self.max_num_node_of_one_graph or nm_2.shape[0] >= self.max_num_node_of_one_graph:
                sample_sha_1, sample_sha_2 = random.choices(population=self.label_to_sample_sha_dict[sampled_label], k=2)
                nm_path_1 = os.path.join(nm_dir, sample_sha_1 + '_node.npz')
                nm_path_2 = os.path.join(nm_dir, sample_sha_2 + '_node.npz')
                am_path_1 = os.path.join(am_dir, sample_sha_1 + '_adj.npz')
                am_path_2 = os.path.join(am_dir, sample_sha_2 + '_adj.npz')
                nm_1 = np.load(nm_path_1)['arr_0']
                nm_2 = np.load(nm_path_2)['arr_0']
                am_1 = np.load(am_path_1)['arr_0'].astype(int)
                am_2 = np.load(am_path_2)['arr_0'].astype(int)
        else:
            sample_sha_1, sample_sha_2 = random.choices(population=self.label_to_sample_sha_dict[sampled_label], k=2)
            nm_path_1 = os.path.join(nm_dir, sample_sha_1 + '_node.npz')
            nm_path_2 = os.path.join(nm_dir, sample_sha_2 + '_node.npz')
            am_path_1 = os.path.join(am_dir, sample_sha_1 + '_adj.npz')
            am_path_2 = os.path.join(am_dir, sample_sha_2 + '_adj.npz')
            nm_1 = np.load(nm_path_1)['arr_0']
            nm_2 = np.load(nm_path_2)['arr_0']
            am_1 = np.load(am_path_1)['arr_0'].astype(int)
            am_2 = np.load(am_path_2)['arr_0'].astype(int)
        return nm_1, am_1, nm_2, am_2, 1

    def diff_pair_sampling(self):
        sampled_label_1, sampled_label_2 = random.choices(population=self.label_list, k=2)
        nm_dir_1, am_dir_1 = self._get_nm_am_dir(sampled_label_1)
        nm_dir_2, am_dir_2 = self._get_nm_am_dir(sampled_label_2)
        if self.max_num_node_of_one_graph:
            nm_1 = self.inital_nm
            nm_2 = self.inital_nm
            while nm_1.shape[0] >= self.max_num_node_of_one_graph:
                sample_sha_1 = random.choice(seq=self.label_to_sample_sha_dict[sampled_label_1])
                nm_path_1 = os.path.join(nm_dir_1, sample_sha_1 + '_node.npz')
                am_path_1 = os.path.join(am_dir_1, sample_sha_1 + '_adj.npz')
                nm_1 = np.load(nm_path_1)['arr_0']
                am_1 = np.load(am_path_1)['arr_0'].astype(int)
            while nm_2.shape[0] >= self.max_num_node_of_one_graph:
                sample_sha_2 = random.choice(seq=self.label_to_sample_sha_dict[sampled_label_2])
                nm_path_2 = os.path.join(nm_dir_2, sample_sha_2 + '_node.npz')
                am_path_2 = os.path.join(am_dir_2, sample_sha_2 + '_adj.npz')
                nm_2 = np.load(nm_path_2)['arr_0']
                am_2 = np.load(am_path_2)['arr_0'].astype(int)
        else:
            sample_sha_1 = random.choice(seq=self.label_to_sample_sha_dict[sampled_label_1])
            nm_path_1 = os.path.join(nm_dir_1, sample_sha_1 + '_node.npz')
            am_path_1 = os.path.join(am_dir_1, sample_sha_1 + '_adj.npz')
            nm_1 = np.load(nm_path_1)['arr_0']
            am_1 = np.load(am_path_1)['arr_0'].astype(int)
            sample_sha_2 = random.choice(seq=self.label_to_sample_sha_dict[sampled_label_2])
            nm_path_2 = os.path.join(nm_dir_2, sample_sha_2 + '_node.npz')
            am_path_2 = os.path.join(am_dir_2, sample_sha_2 + '_adj.npz')
            nm_2 = np.load(nm_path_2)['arr_0']
            am_2 = np.load(am_path_2)['arr_0'].astype(int)
        return nm_1, am_1, nm_2, am_2, -1

    def _pair_generator(self):
        while True:
            random_num = random.uniform(0, 1)
            if random_num < 0.5:
                yield self.sim_pair_sampling()
            else:
                yield self.diff_pair_sampling()

    def _triplet_generator(self):
        while True:
            sampled_label_1, sampled_label_2 = random.choices(population=self.label_list, k=2)
            nm_dir_1, am_dir_1 = self._get_nm_am_dir(sampled_label_1)
            nm_dir_2, am_dir_2 = self._get_nm_am_dir(sampled_label_2)
            if self.max_num_node_of_one_graph:
                nm_1 = self.inital_nm
                nm_2 = self.inital_nm
                nm_3 = self.inital_nm
                while nm_1.shape[0] >= self.max_num_node_of_one_graph or nm_2.shape[0] >= self.max_num_node_of_one_graph:
                    sample_sha_1, sample_sha_2 = random.choices(population=self.label_to_sample_sha_dict[sampled_label_1],
                                                                k=2)
                    nm_path_1 = os.path.join(nm_dir_1, sample_sha_1 + '_node.npz')
                    am_path_1 = os.path.join(am_dir_1, sample_sha_1 + '_adj.npz')
                    nm_path_2 = os.path.join(nm_dir_1, sample_sha_2 + '_node.npz')
                    am_path_2 = os.path.join(am_dir_1, sample_sha_2 + '_adj.npz')
    
                    nm_1 = np.load(nm_path_1)['arr_0']
                    am_1 = np.load(am_path_1)['arr_0'].astype(int)
                    nm_2 = np.load(nm_path_2)['arr_0']
                    am_2 = np.load(am_path_2)['arr_0'].astype(int)
    
                while nm_3.shape[0] >= self.max_num_node_of_one_graph:
                    sample_sha_3 = random.choice(seq=self.label_to_sample_sha_dict[sampled_label_2])
                    nm_path_3 = os.path.join(nm_dir_2, sample_sha_3 + '_node.npz')
                    am_path_3 = os.path.join(am_dir_2, sample_sha_3 + '_adj.npz')
                    nm_3 = np.load(nm_path_3)['arr_0']
                    am_3 = np.load(am_path_3)['arr_0'].astype(int)
            else:
                sample_sha_1, sample_sha_2 = random.choices(population=self.label_to_sample_sha_dict[sampled_label_1],
                                                            k=2)
                nm_path_1 = os.path.join(nm_dir_1, sample_sha_1 + '_node.npz')
                am_path_1 = os.path.join(am_dir_1, sample_sha_1 + '_adj.npz')
                nm_path_2 = os.path.join(nm_dir_1, sample_sha_2 + '_node.npz')
                am_path_2 = os.path.join(am_dir_1, sample_sha_2 + '_adj.npz')

                nm_1 = np.load(nm_path_1)['arr_0']
                am_1 = np.load(am_path_1)['arr_0'].astype(int)
                nm_2 = np.load(nm_path_2)['arr_0']
                am_2 = np.load(am_path_2)['arr_0'].astype(int)
                sample_sha_3 = random.choice(seq=self.label_to_sample_sha_dict[sampled_label_2])
                nm_path_3 = os.path.join(nm_dir_2, sample_sha_3 + '_node.npz')
                am_path_3 = os.path.join(am_dir_2, sample_sha_3 + '_adj.npz')
                nm_3 = np.load(nm_path_3)['arr_0']
                am_3 = np.load(am_path_3)['arr_0'].astype(int)
            yield nm_1, am_1, nm_2, am_2, nm_3, am_3

    def pairs(self):
        pair_generator = self._pair_generator()
        num_batch_in_total = self.num_epoch * self.step_per_epoch
        for batch_idx in range(num_batch_in_total):
            nm_of_one_batch = []
            am_of_one_batch = []
            label_of_one_batch = []
            for sample_idx in range(self.batch_size):
                nm_1, am_1, nm_2, am_2, label = next(pair_generator)
                nm_of_one_batch.append(nm_1)
                nm_of_one_batch.append(nm_2)
                am_of_one_batch.append(am_1)
                am_of_one_batch.append(am_2)
                label_of_one_batch.append(label)
            batched_label = np.array(label_of_one_batch, dtype=int)
            yield self._pack_batch(nm_of_one_batch, am_of_one_batch), batched_label

    def triplets(self):
        triplet_generator = self._triplet_generator()
        num_batch_in_total = self.num_epoch * self.step_per_epoch
        for batch_idx in range(num_batch_in_total):
            nm_of_one_batch = []
            am_of_one_batch = []
            for sample_idx in range(self.batch_size):
                nm_1, am_1, nm_2, am_2, nm_3, am_3 = next(triplet_generator)
                nm_of_one_batch.append(nm_1)
                nm_of_one_batch.append(nm_2)
                nm_of_one_batch.append(nm_1)
                nm_of_one_batch.append(nm_3)
                am_of_one_batch.append(am_1)
                am_of_one_batch.append(am_2)
                am_of_one_batch.append(am_1)
                am_of_one_batch.append(am_3)
            yield self._pack_batch(nm_of_one_batch, am_of_one_batch)

    def _pack_batch(self, nm_list, ad_list):
        '''
        :param graphs: a list of (nm_matrix[num_nodes, node_feature_dims], am_matrix[num_edge, 2]) pairs. nm/am_matrixes are all numpy array.
        :return: 组成一个大的节点和边的特征矩阵
        '''
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
        node_features = np.concatenate(nm_list, axis=0)
        # node_features [总节点数, node_feature_dims]

        cumsum = np.cumsum(num_node_list)
        # print("cumsum", cumsum.shape, cumsum)

        # 每个图的边由边的个数个整数占位[0，0，0，0，1，1，1，1，1，2，2，2，...,batchsize-1,batchsize-1,...]
        indices = np.repeat(np.arange(batch_size), num_edge_list)  # [num_edge_this_batch]
        # print("indices", indices.shape, indices)
        # 为每个边取到其在的图的节点的个数[22,22,22,22,12+22,12+22,12+22,12+22,12+22,13+22+12,13+22+12,13+22+12,...,]
        scattered = cumsum[indices]  # [num_edge_this_batch, ]
        # print("scattered", scattered.shape, scattered)
        
        edges = np.concatenate(ad_list, axis=0)
        # print("edges", edges.shape, edges)
        # print("before edges[..., 0]", edges[..., 0].shape, edges[..., 0])
        edges[..., 0] += scattered
        # print("after edges[..., 0]", edges[..., 0].shape, edges[..., 0])
        
        # print("before edges[..., 1]", edges[..., 1].shape, edges[..., 1])
        edges[..., 1] += scattered
        # print("after edges[..., 1]", edges[..., 1].shape, edges[..., 1])

        if self.num_edge_type == 1:
            edge_features = np.ones((total_num_edge, 1), dtype=np.float32)
        else:
            edge_features = np.zeros(shape=(total_num_edge, self.num_edge_type), dtype=np.float32)
            edge_features[np.arange(total_num_edge), edges[:, 2]] = 1

        return GraphData(from_idx=edges[..., 0],
                         to_idx=edges[..., 1],
                         node_features=node_features,
                         edge_features=edge_features,
                         graph_idx=np.repeat(np.arange(batch_size), np.array(num_node_list[1:])),
                         n_graphs=batch_size
                         )


from torch.utils.data import Dataset
class CustomDataset(Dataset):
    def __init__(self, data_same_shape, labels):
        self.data_same_shape = data_same_shape
        self.labels = labels

    def __len__(self):
        return len(self.data_same_shape)

    def __getitem__(self, index):
        # 在这个地方将数据都转化成tensor
        graph = self.data_same_shape[index]
        label = self.labels[index]
        return graph, label