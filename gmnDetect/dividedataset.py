 
import os
import sys
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
print(sys.path)
import copy
import random
import numpy as np
import networkx as nx
from graphMatrix.config import GRAPH_MODE
from argparse import ArgumentParser

parser = ArgumentParser("Divide the dataset into training, validation, and testing sets.")
parser.add_argument("--project", type=str, default="curl")
parser.add_argument("--cve_id", type=str, default="CVE-2021-22901")
parser.add_argument("--hl", action="store_true", default=False)
parser.add_argument("--RL", action="store_true", default=False)
parser.add_argument("--mission_id", type=str, default="hchdvbjsjci-bhjdsbcj")
args = parser.parse_args()

def mkdirs_if_not_exists(path):
    if not os.path.exists(path):
        os.makedirs(path)

 
def prepare_dirs(proj):
     
    train_vul_adj_path = '{}/train/vul/adj'.format(proj)
    train_vul_node_path = '{}/train/vul/node'.format(proj)
    train_fix_adj_path = '{}/train/fix/adj'.format(proj)
    train_fix_node_path = '{}/train/fix/node'.format(proj)

    validate_vul_adj_path = '{}/validate/vul/adj'.format(proj)
    validate_vul_node_path = '{}/validate/vul/node'.format(proj)
    validate_fix_adj_path = '{}/validate/fix/adj'.format(proj)
    validate_fix_node_path = '{}/validate/fix/node'.format(proj)

    test_vul_adj_path = '{}/test/vul/adj'.format(proj)
    test_vul_node_path = '{}/test/vul/node'.format(proj)
    test_fix_adj_path = '{}/test/fix/adj'.format(proj)
    test_fix_node_path = '{}/test/fix/node'.format(proj)

    mkdirs_if_not_exists(train_vul_adj_path)
    mkdirs_if_not_exists(train_vul_node_path)
    mkdirs_if_not_exists(train_fix_adj_path)
    mkdirs_if_not_exists(train_fix_node_path)

    mkdirs_if_not_exists(validate_vul_adj_path)
    mkdirs_if_not_exists(validate_vul_node_path)
    mkdirs_if_not_exists(validate_fix_adj_path)
    mkdirs_if_not_exists(validate_fix_node_path)

    mkdirs_if_not_exists(test_vul_adj_path)
    mkdirs_if_not_exists(test_vul_node_path)
    mkdirs_if_not_exists(test_fix_adj_path)
    mkdirs_if_not_exists(test_fix_node_path)

def prepare_dirs_rl(proj):
    adj_path = '{}/adj'.format(proj)
    node_path = '{}/node'.format(proj)
    mkdirs_if_not_exists(adj_path)
    mkdirs_if_not_exists(node_path)


 
def merge_adj_metrix(path):
    metrix = np.load(path, allow_pickle=True)
    new_metrix = {}
    for type_edges in metrix:
        for k, v_box in type_edges.items():
            if k not in new_metrix:
                new_metrix[k] = []
            for v in v_box:
                if v not in new_metrix[k]:
                    new_metrix[k].append(v)
                     
                     
    ret = nx.adjacency_matrix(nx.from_dict_of_lists(new_metrix, nx.DiGraph)).toarray()
     
    return np.transpose(np.nonzero(ret))

def merge_adj_metrix_new_single(path):
    metrix = np.load(path, allow_pickle=True)['arr_0']
    new_metrix = {}
    
    for index, type_edges in enumerate(metrix):
        new_metrix[index] = type_edges
    combined_adj_matrix = np.zeros_like(new_metrix[0])
    for adj_matrix in new_metrix:   
        combined_adj_matrix = np.logical_or(combined_adj_matrix, new_metrix[adj_matrix])
    combined_adj_matrix = combined_adj_matrix.astype(int)
    ret = combined_adj_matrix
    return np.transpose(np.nonzero(ret))

def merge_adj_metrix_new_single_highlight(path):
    matrix = np.load(path, allow_pickle=True)['arr_0']  
    result = adj_matrix2table(matrix.transpose(1, 2, 0), GRAPH_MODE)
    return result

def merge_adj_metrix_new_disjoint(path):
    metrix = np.load(path, allow_pickle=True)['arr_0']  
    new_metrix = {}
    new_matrix = []
    label_embeddings = np.eye(metrix.shape[0])  

    for index, type_edges in enumerate(metrix):
        new_metrix[index] = type_edges
        adj_metric = np.transpose(np.nonzero(new_metrix[index]))  
        label_metric = label_embeddings[index][None, :].repeat(adj_metric.shape[0], axis=0)  
        adj_metric = np.concatenate((adj_metric, label_metric), axis=-1)
        new_matrix.append(adj_metric)

    result_matrix = np.concatenate([new_matrix[0], new_matrix[1], new_matrix[2]], axis=0)
    return result_matrix

def adj_matrix2table(adj_matrix, mode):

    matrix_shape = adj_matrix.shape
    indices = np.indices(matrix_shape[:2]).reshape(2, -1).T
    flatten_adj_matrix = adj_matrix.reshape(-1, matrix_shape[-1])
    zero_indices = np.all(flatten_adj_matrix == 0, axis=-1)  
    highlight_indices = np.any(flatten_adj_matrix >= 1, axis=-1)
    flatten_adj_matrix[flatten_adj_matrix >= 2] = 1
    hightlight_feature = np.float32(highlight_indices)[..., None]
    if mode == "disjoint" or mode == "new_disjoint":
        return np.concatenate((indices, flatten_adj_matrix, hightlight_feature), axis=1)[~zero_indices]
    else:
        return np.concatenate((indices, hightlight_feature), axis=1)[~zero_indices]

def adj_matrix2table_2(adj_matrix):

    matrix_shape = adj_matrix.shape
    indices = np.indices(matrix_shape[:2]).reshape(2, -1).T
    flatten_adj_matrix = adj_matrix.reshape(-1, matrix_shape[-1])
    zero_indices = np.all(flatten_adj_matrix == 0, axis=-1)  
    highlight_indices = np.any(flatten_adj_matrix >= 2, axis=-1)
    flatten_adj_matrix[flatten_adj_matrix >= 2] = 1
    hightlight_feature = np.float32(highlight_indices)[..., None]
    edge_array = np.concatenate((flatten_adj_matrix, hightlight_feature), axis=1)

    def binary_to_decimal(binary_array):
         
        decimal_array = np.sum(binary_array * np.array([4, 2, 1]), axis=1, keepdims=True)
        return decimal_array

    def decimal_to_binary(decimal_array, output_shape):
         
        binary_array = np.unpackbits(decimal_array.astype(np.uint8), axis=1)
        binary_array = binary_array[:, -output_shape[1]:]
        return binary_array.reshape(output_shape)

    binary_data = edge_array[:, :3]
    flag = edge_array[:, 3]

    def update_array(flag_array, array_3d):
         
        indices_to_update = np.where(flag_array == 1)[0]
        for index in indices_to_update:
            if array_3d[index, 0] == 1:  
                array_3d[index, 2] = 1
            elif array_3d[index, 1] == 1:
                array_3d[index, 0] = 1
            elif array_3d[index, 2] == 1:
                array_3d[index, 1] = 1
             
        return array_3d

    updated_array_3d = update_array(flag, binary_data)
    return np.concatenate((indices, updated_array_3d), axis=1)[~zero_indices]
 
def merge_adj_metrix_new_disjoint_highlight(path):
    matrix = np.load(path, allow_pickle=True)['arr_0']  
    result = adj_matrix2table(matrix.transpose(1, 2, 0), GRAPH_MODE)

    return result
    
 
def proc_source(proj, cve, rl):
   
    output_dict = {}
    if rl:
        dir_name = '{}/'.format(proj)
        f_raw_nodes = os.listdir(dir_name + 'node/')
    else:  
        fix_dir_name = '{}/fix/'.format(proj)
        vul_dir_name = '{}/vul/'.format(proj)
        fix_f_raw_nodes = os.listdir(fix_dir_name + 'node/')
        vul_f_raw_nodes = os.listdir(vul_dir_name + 'node/')
    if rl:
        output_dict[cve] = []
        for item in f_raw_nodes:
            path = '{}/node/{}'.format(proj, item)
            adj_path = path.replace('node.npz', 'adj.npz').replace('/node/', '/adj/')
             
            if hl:
                if GRAPH_MODE == "disjoint" or GRAPH_MODE == "new_disjoint":
                    item_dict = {'node_f': item, 'adj_f': item.replace('node.npz', 'adj.npz'),'node': np.load(path)['arr_0'], 'adj': merge_adj_metrix_new_disjoint_highlight(adj_path)}
                else:
                    item_dict = {'node_f': item, 'adj_f': item.replace('node.npz', 'adj.npz'),'node': np.load(path)['arr_0'], 'adj': merge_adj_metrix_new_single_highlight(adj_path)}
            else:
                if GRAPH_MODE == "disjoint" or GRAPH_MODE == "new_disjoint":
                    item_dict = {'node_f': item, 'adj_f': item.replace('node.npz', 'adj.npz'),'node': np.load(path)['arr_0'], 'adj': merge_adj_metrix_new_disjoint(adj_path)}
                else:
                    item_dict = {'node_f': item, 'adj_f': item.replace('node.npz', 'adj.npz'),'node': np.load(path)['arr_0'], 'adj': merge_adj_metrix_new_single(adj_path)}
             
             
            output_dict[cve].append(item_dict)
    else:
        output_dict[cve] = {'fix': [], 'vul': []}
        for item in fix_f_raw_nodes:
            path = '{}/fix/node/{}'.format(proj, item)
            adj_path = path.replace('node.npz', 'adj.npz').replace('/node/', '/adj/')
             
            if hl:
                if GRAPH_MODE == "disjoint" or GRAPH_MODE == "new_disjoint":
                    item_dict = {'node_f': item, 'adj_f': item.replace('node.npz', 'adj.npz'),'node': np.load(path)['arr_0'], 'adj': merge_adj_metrix_new_disjoint_highlight(adj_path)}
                else:
                    item_dict = {'node_f': item, 'adj_f': item.replace('node.npz', 'adj.npz'),'node': np.load(path)['arr_0'], 'adj': merge_adj_metrix_new_single_highlight(adj_path)}
            else:
                if GRAPH_MODE == "disjoint" or GRAPH_MODE == "new_disjoint":
                    item_dict = {'node_f': item, 'adj_f': item.replace('node.npz', 'adj.npz'),'node': np.load(path)['arr_0'], 'adj': merge_adj_metrix_new_disjoint(adj_path)}
                else:
                    item_dict = {'node_f': item, 'adj_f': item.replace('node.npz', 'adj.npz'),'node': np.load(path)['arr_0'], 'adj': merge_adj_metrix_new_single(adj_path)}
             
             
            output_dict[cve]['fix'].append(item_dict)

        for item in vul_f_raw_nodes:
            path = '{}/vul/node/{}'.format(proj, item)
            adj_path = path.replace('node.npz', 'adj.npz').replace('/node/', '/adj/')
             
             
             
            if hl:
                if GRAPH_MODE == "disjoint":
                    item_dict = {'node_f': item, 'adj_f': item.replace('node.npz', 'adj.npz'),'node': np.load(path)['arr_0'], 'adj': merge_adj_metrix_new_disjoint_highlight(adj_path)}
                else:
                    item_dict = {'node_f': item, 'adj_f': item.replace('node.npz', 'adj.npz'),'node': np.load(path)['arr_0'], 'adj': merge_adj_metrix_new_single_highlight(adj_path)}
            else:
                if GRAPH_MODE == "disjoint":
                    item_dict = {'node_f': item, 'adj_f': item.replace('node.npz', 'adj.npz'),'node': np.load(path)['arr_0'], 'adj': merge_adj_metrix_new_disjoint(adj_path)}
                else:
                    item_dict = {'node_f': item, 'adj_f': item.replace('node.npz', 'adj.npz'),'node': np.load(path)['arr_0'], 'adj': merge_adj_metrix_new_single(adj_path)}
            output_dict[cve]['vul'].append(item_dict)
    return output_dict

 
def save_output(output, proj, cve_used=None):
    for cve, cve_data in output.items():

        if cve_used:
            if cve != cve_used:
                continue
        prepare_dirs(proj)
        fix_count = len(cve_data['fix'])
        vul_count = len(cve_data['vul'])

        train_fix_dataset = random.sample(cve_data['fix'], int(fix_count * 0.6))
        train_vul_dataset = random.sample(cve_data['vul'], int(vul_count * 0.6))

        test_and_vali_fix_dataset = []
        test_and_vali_vul_dataset = []

        for item in cve_data['fix']:
            if item not in train_fix_dataset:
                test_and_vali_fix_dataset.append(item)
        for item in cve_data['vul']:
            if item not in train_vul_dataset:
                test_and_vali_vul_dataset.append(item)

        test_fix_dataset = []
        test_vul_dataset = []
        vali_fix_dataset = []
        vali_vul_dataset = []

        test_fix_dataset = random.sample(test_and_vali_fix_dataset, int(len(test_and_vali_fix_dataset) * 0.5))
        test_vul_dataset = random.sample(test_and_vali_vul_dataset, int(len(test_and_vali_vul_dataset) * 0.5))

        for item in test_and_vali_fix_dataset:
            if item not in test_fix_dataset:
                vali_fix_dataset.append(item)

        for item in test_and_vali_vul_dataset:
            if item not in test_vul_dataset:
                vali_vul_dataset.append(item)

         

        for item in train_fix_dataset:
            np.savez_compressed('{}/train/fix/node/{}'.format(proj, item['node_f']), item['node'])
            np.savez_compressed('{}/train/fix/adj/{}'.format(proj, item['adj_f']), item['adj'])

        for item in train_vul_dataset:
            np.savez_compressed('{}/train/vul/node/{}'.format(proj, item['node_f']), item['node'])
            np.savez_compressed('{}/train/vul/adj/{}'.format(proj, item['adj_f']), item['adj'])

        for item in test_fix_dataset:
            np.savez_compressed('{}/test/fix/node/{}'.format(proj, item['node_f']), item['node'])
            np.savez_compressed('{}/test/fix/adj/{}'.format(proj, item['adj_f']), item['adj'])
        for item in test_vul_dataset:
            np.savez_compressed('{}/test/vul/node/{}'.format(proj, item['node_f']), item['node'])
            np.savez_compressed('{}/test/vul/adj/{}'.format(proj, item['adj_f']), item['adj'])

        for item in vali_fix_dataset:
            np.savez_compressed('{}/validate/fix/node/{}'.format(proj, item['node_f']), item['node'])
            np.savez_compressed('{}/validate/fix/adj/{}'.format(proj, item['adj_f']), item['adj'])
        for item in vali_vul_dataset:
            np.savez_compressed('{}/validate/vul/node/{}'.format(proj, item['node_f']), item['node'])
            np.savez_compressed('{}/validate/vul/adj/{}'.format(proj, item['adj_f']), item['adj'])

def save_output_rl(output, proj, cve_used=None):
    for cve, cve_data in output.items():
        if cve_used:
            if cve != cve_used:
                continue
        prepare_dirs_rl(proj)
        for item in cve_data:
            np.savez_compressed('{}/node/{}'.format(proj, item['node_f']), item['node'])
            np.savez_compressed('{}/adj/{}'.format(proj, item['adj_f']), item['adj'])



if __name__ == "__main__":
     
    if args.mission_id:
        project, cve_id, hl, rl, mission_id  = args.project, args.cve_id, args.hl, args.RL, args.mission_id
    else:
        project, cve_id, hl, rl  = args.project, args.cve_id, args.hl, args.RL

    if hl:
        if rl:
            matrix_path = os.path.join("../data_detect/data/", mission_id, project, cve_id, "matrix_{}_hl".format(GRAPH_MODE))
            output_path = os.path.join("../data_detect/data/", mission_id, project, cve_id, "matrix_{}_divide_hl".format(GRAPH_MODE))
        else:
            matrix_path = os.path.join("../data/", project, cve_id, "matrix_{}_hl".format(GRAPH_MODE))
            output_path = os.path.join("../data/", project, cve_id, "matrix_{}_divide_hl".format(GRAPH_MODE))
    else:
        if rl:
            matrix_path = os.path.join("../data_detect/data/", mission_id, project, cve_id, "matrix_{}".format(GRAPH_MODE))
            output_path = os.path.join("../data_detect/data/", mission_id, project, cve_id, "matrix_{}_divide".format(GRAPH_MODE))
        else:
            matrix_path = os.path.join("../data/", project, cve_id, "matrix_{}".format(GRAPH_MODE))
            output_path = os.path.join("../data/", project, cve_id, "matrix_{}_divide".format(GRAPH_MODE))

    
    if len(sys.argv) == 2:
        output_dict = proc_source(matrix_path)
        save_output(output_dict, matrix_path)
    elif len(sys.argv) >= 3:
         
         
        output_dict = proc_source(matrix_path, cve_id, rl)
        if rl:
            save_output_rl(output_dict, output_path, cve_id)
        else:
            save_output(output_dict, output_path, cve_id)
