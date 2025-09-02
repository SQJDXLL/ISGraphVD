import collections
from dataset import *

import time

 
from graphembeddingnetwork import GraphEmbeddingNet, GraphEncoder, GraphAggregator
from graphmatchingnetwork import GraphMatchingNet
import copy
import torch
import random

GraphData = collections.namedtuple(
    "GraphData", ["from_idx", "to_idx", "node_features", "edge_features", "graph_idx", "n_graphs"]
)
 
 

def reshape_and_split_tensor(tensor, n_splits):
    """Reshape and split a 2D tensor along the last dimension.

    Args:
      tensor: a [num_examples, feature_dim] tensor.  num_examples must be a
        multiple of `n_splits`.
      n_splits: int, number of splits to split the tensor into.

    Returns:
      splits: a list of `n_splits` tensors.  The first split is [tensor[0],
        tensor[n_splits], tensor[n_splits * 2], ...], the second split is
        [tensor[1], tensor[n_splits + 1], tensor[n_splits * 2 + 1], ...], etc..
    """
    num_examples, feature_dim = tensor.shape
     

     
    tensor = tensor.view(-1, n_splits, feature_dim)

     
    tensor_split = torch.chunk(tensor, n_splits, dim=1)

     
    return [chunk.squeeze(1) for chunk in tensor_split]


def build_model(config, node_feature_dim, edge_feature_dim):
     
    """Create model for training and evaluation.

    Args:
      config: a dictionary of configs, like the one created by the
        `get_default_config` function.
      node_feature_dim: int, dimensionality of node features.
      edge_feature_dim: int, dimensionality of edge features.

    Returns:
      tensors: a (potentially nested) name => tensor dict.
      placeholders: a (potentially nested) name => tensor dict.
      AE_model: a GraphEmbeddingNet or GraphMatchingNet instance.

    Raises:
      ValueError: if the specified model or training settings are not supported.
    """
    config["encoder"]["node_feature_dim"] = node_feature_dim
    config["encoder"]["edge_feature_dim"] = edge_feature_dim


    encoder = GraphEncoder(**config["encoder"])
    print("~~~~~~~~~~~~~~~~~encoder done~")

    aggregator = GraphAggregator(**config["aggregator"])
    print("~~~~~~~~~~~~~~~~~aggregator done~")
    if config["model_type"] == "embedding":

        model = GraphEmbeddingNet(encoder, aggregator, **config["graph_embedding_net"])
    elif config["model_type"] == "matching":
        print("matching model type entered~")
        model = GraphMatchingNet(encoder, aggregator, **config["graph_matching_net"])
    else:
        raise ValueError("Unknown model type: %s" % config["model_type"])

    optimizer = torch.optim.Adam((model.parameters()), lr=config["training"]["learning_rate"], weight_decay=1e-5)

    return model, optimizer


def build_datasets(config):
    """Build the training and evaluation datasets."""
    config = copy.deepcopy(config)

    if config["data"]["problem"] == "malicious_detection":
        if config["if_sampling"]:
            training_set = yzdDatasetNew(
                nm_dir_vul=config["data"]["dataset_params"]["training_dataset_dir_vul_nm"],
                am_dir_vul=config["data"]["dataset_params"]["training_dataset_dir_vul_am"],
                nm_dir_fix=config["data"]["dataset_params"]["training_dataset_dir_fix_nm"],
                am_dir_fix=config["data"]["dataset_params"]["training_dataset_dir_fix_am"],
                num_epoch=config["training"]["num_epoch"],
                batch_size=config["training"]["batch_size"],
                step_per_epoch=config["training"]["step_per_train_epoch"],
                max_num_node_of_one_graph=config["data"]["dataset_params"]["max_num_node_of_one_graph"],
            )
            validation_set = yzdDatasetNew(
                nm_dir_vul=config["data"]["dataset_params"]["validation_dataset_dir_vul_nm"],
                am_dir_vul=config["data"]["dataset_params"]["validation_dataset_dir_vul_am"],
                nm_dir_fix=config["data"]["dataset_params"]["validation_dataset_dir_fix_nm"],
                am_dir_fix=config["data"]["dataset_params"]["validation_dataset_dir_fix_am"],
                num_epoch=config["training"]["num_epoch"],
                batch_size=config["training"]["batch_size"],
                step_per_epoch=config["training"]["step_per_vali_epoch"],
                max_num_node_of_one_graph=config["data"]["dataset_params"]["max_num_node_of_one_graph"],
            )
        else:
            training_set = yzdDataset(
                dataset_dir=config["data"]["dataset_params"]["train_dataset_dir"],
                num_epoch=config["training"]["num_epoch"],
                batch_size=config["training"]["batch_size"],
                num_pairs=None,
                num_triplets=None,
                name="train",
            )
            validation_set = yzdDataset(
                dataset_dir=config["data"]["dataset_params"]["vali_dataset_dir"],
                num_epoch=config["training"]["num_epoch"],
                batch_size=config["training"]["batch_size"],
                num_pairs=None,
                num_triplets=None,
                name="vali",
            )
        return training_set, validation_set
    elif config["data"]["problem"] == "malicious_detection_test":
        validation_set = yzdDataset(
            config["data"]["dataset_params"]["eval_dataset_dir"],
            num_epoch=1,
            batch_size=32,
            num_pairs=config["training"]["num_validation_pairs"],
            num_triplets=config["training"]["num_validation_triplets"],
            name="validation",
        )
        return validation_set
    elif config["data"]["problem"] == "malicious_detection_compare":
        compare_set = yzdDataset(
            config["data"]["dataset_params"]["eval_dataset_dir"],
            num_epoch=1,
            num_pairs=config["training"]["num_validation_pairs"],
            num_triplets=config["training"]["num_validation_triplets"],
            compare_path=config["data"]["dataset_params"]["compare_path"],
            name="comparation",
        )
        return compare_set
    else:
        raise ValueError("Unknown problem type: %s" % config["data"]["problem"])


def build_datasets_DiffGraph(config):
    """Build the training and evaluation datasets."""
    config = copy.deepcopy(config)

    if config["data"]["problem"] == "malicious_detection":
        if config["if_sampling"]:
            training_set = yzdDatasetNew(
                nm_dir_vul=config["data"]["dataset_params"]["training_dataset_dir_vul_nm"],
                am_dir_vul=config["data"]["dataset_params"]["training_dataset_dir_vul_am"],
                nm_dir_fix=config["data"]["dataset_params"]["training_dataset_dir_fix_nm"],
                am_dir_fix=config["data"]["dataset_params"]["training_dataset_dir_fix_am"],
                num_epoch=config["training"]["num_epoch"],
                batch_size=config["training"]["batch_size"],
                step_per_epoch=config["training"]["step_per_train_epoch"],
                max_num_node_of_one_graph=config["data"]["dataset_params"]["max_num_node_of_one_graph"],
            )
            validation_set = yzdDatasetNew(
                nm_dir_vul=config["data"]["dataset_params"]["validation_dataset_dir_vul_nm"],
                am_dir_vul=config["data"]["dataset_params"]["validation_dataset_dir_vul_am"],
                nm_dir_fix=config["data"]["dataset_params"]["validation_dataset_dir_fix_nm"],
                am_dir_fix=config["data"]["dataset_params"]["validation_dataset_dir_fix_am"],
                num_epoch=config["training"]["num_epoch"],
                batch_size=config["training"]["batch_size"],
                step_per_epoch=config["training"]["step_per_vali_epoch"],
                max_num_node_of_one_graph=config["data"]["dataset_params"]["max_num_node_of_one_graph"],
            )
        else:
            training_set = zylDataset(
                dataset_dir=config["data"]["dataset_params"]["train_dataset_dir"],
                num_epoch=config["training"]["num_epoch"],
                batch_size=config["training"]["batch_size"],
                max_nodes_limit=50000,
                num_pairs=None,
                num_triplets=None,
                name="train",
            )
            validation_set = zylDataset(
                dataset_dir=config["data"]["dataset_params"]["vali_dataset_dir"],
                num_epoch=config["training"]["num_epoch"],
                batch_size=config["training"]["batch_size"],
                max_nodes_limit=50000,
                num_pairs=None,
                num_triplets=None,
                name="vali",
            )
        return training_set, validation_set
    elif config["data"]["problem"] == "malicious_detection_test":
        validation_set = zylDataset(
            config["data"]["dataset_params"]["eval_dataset_dir"],
            num_epoch=1,
            batch_size=32,
            max_nodes_limit=500,
            num_pairs=config["training"]["num_validation_pairs"],
            num_triplets=config["training"]["num_validation_triplets"],
            name="validation",
        )
        return validation_set
    elif config["data"]["problem"] == "malicious_detection_compare":
        compare_set = zylDataset(
            config["data"]["dataset_params"]["eval_dataset_dir"],
            num_epoch=1,
            num_pairs=config["training"]["num_validation_pairs"],
            num_triplets=config["training"]["num_validation_triplets"],
            compare_path=config["data"]["dataset_params"]["compare_path"],
            name="comparation",
        )
        return compare_set
    else:
        raise ValueError("Unknown problem type: %s" % config["data"]["problem"])


def get_each_graph(sample):
    graph = sample
    node_features = torch.from_numpy(graph.node_features.astype("float32"))
    edge_features = torch.from_numpy(graph.edge_features.astype("float32"))
    from_idx = torch.from_numpy(graph.from_idx).long()
    to_idx = torch.from_numpy(graph.to_idx).long()
    graph_idx = torch.from_numpy(graph.graph_idx).long()
    return node_features, edge_features, from_idx, to_idx, graph_idx


def pack_batch_torch(nm_list, ad_list):  
    '''  
    Packs a batch of graphs represented by node and adjacency matrices into a single GraphData object.  
    
    :param nm_list: A list of node matrix tensors (num_nodes, node_feature_dims) on the given device.  
    :param ad_list: A list of adjacency matrix tensors (num_edges, 2) on the given device.  
    :param device: The device where the tensors are stored (e.g., 'cuda:0' or 'cpu').  
    :return: A GraphData object containing the packed batch.  
    '''    
    
    num_node_list = [0] + [nm.size(0) for nm in nm_list]   
    num_edge_list = [am.size(0) for am in ad_list]   

    total_num_node = sum(num_node_list[1:])  
    total_num_edge = sum(num_edge_list)  

    cumsum = torch.cumsum(torch.tensor(num_node_list), dim=0)  

    indices = torch.repeat_interleave(torch.arange(len(nm_list)), torch.tensor(num_edge_list))  
    scattered = cumsum[indices]  

    edges = torch.cat(ad_list, dim=0)  
    edges[..., 0] += scattered  
    edges[..., 1] += scattered  

    if edges.size(-1) > 2:  
        edge_features = edges[..., 2:]
    else:  
        edge_features = torch.ones((total_num_edge, 1))  

    segment = torch.repeat_interleave(torch.arange(len(nm_list)), torch.tensor(num_node_list[1:]))
    
     
    node_features = torch.cat(nm_list, dim=0)
    
     
    return GraphData(from_idx=edges[..., 0],  
                    to_idx=edges[..., 1],  
                    node_features=node_features,  
                    edge_features=edge_features,  
                    graph_idx=segment,  
                    n_graphs=len(nm_list))  


def get_graph_torch(batch):  
    if len(batch) != 2:  
         
        graph = batch  
  
        node_features = graph.node_features.float()   
        edge_features = graph.edge_features.float()   
        from_idx = graph.from_idx.long()   
        to_idx = graph.to_idx.long()   
        graph_idx = graph.graph_idx.long()   

        return node_features, edge_features, from_idx, to_idx, graph_idx
    else:  
         
        graph, labels = batch  
        node_features = graph.node_features.float()   
        edge_features = graph.edge_features.float()   
        from_idx = graph.from_idx.long()   
        to_idx = graph.to_idx.long()   
        graph_idx = graph.graph_idx.long()   
  
    return node_features, edge_features, from_idx, to_idx, graph_idx


def get_graph(batch):
    if len(batch) != 2:
         
        graph = batch
        node_features = torch.from_numpy(graph.node_features.astype("float32"))
        edge_features = torch.from_numpy(graph.edge_features.astype("float32"))
        from_idx = torch.from_numpy(graph.from_idx).long()
        to_idx = torch.from_numpy(graph.to_idx).long()
        graph_idx = torch.from_numpy(graph.graph_idx).long()
        return node_features, edge_features, from_idx, to_idx, graph_idx
    else:
        graph, labels = batch
        node_features = torch.from_numpy(graph.node_features.astype("float32"))
        edge_features = torch.from_numpy(graph.edge_features.astype("float32"))
        from_idx = torch.from_numpy(graph.from_idx).long()
        to_idx = torch.from_numpy(graph.to_idx).long()
        graph_idx = torch.from_numpy(graph.graph_idx).long()
        labels = torch.from_numpy(labels).long()

    return node_features, edge_features, from_idx, to_idx, graph_idx, labels
