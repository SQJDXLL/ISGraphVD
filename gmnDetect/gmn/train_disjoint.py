import os
import time
import sys
# print(sys.path)
from evaluation import compute_similarity, auc
from loss import pairwise_loss, triplet_loss
from utils import *
from configure import *
import collections
import pickle
from graphMatrix.config import GRAPH_TYPES, GRAPH_MODE
from argparse import ArgumentParser

parser = ArgumentParser("train disjoint.")
parser.add_argument("--project", type=str, default="curl")
parser.add_argument("--cve_id", type=str, default="CVE-2021-22901")
parser.add_argument("--gpu", type=str, default="0")
parser.add_argument("--hl", action="store_true", default=False)
parser.add_argument("--batch_size", type=int, default=32)
parser.add_argument("--learning_rate", type=float, default= 1e-4)
parser.add_argument("--num_epoch", type=int, default= 10)
args = parser.parse_args()

os.environ["CUDA_DEVICE_ORDER"] = "PCI_BUS_ID"
os.environ["CUDA_VISIBLE_DEVICES"] = args.gpu

import numpy as np
import torch.nn as nn

# Set GPU

use_cuda = torch.cuda.is_available()
print("****************", use_cuda, args.gpu)
device = torch.device("cuda:0" if use_cuda else "cpu")


hl = args.hl
batchsize = args.batch_size
lr = args.learning_rate
project = args.project
cve = args.cve_id
epoch = args.num_epoch

if GRAPH_MODE == 'single':
    if hl:
        # edge_feature_dim = 2
        edge_feature_dim = 1
    else:
        edge_feature_dim = 1
else:
    if hl:
        edge_feature_dim = len(GRAPH_TYPES) + 1
    else:
        edge_feature_dim = len(GRAPH_TYPES)



if GRAPH_MODE == "disjoint":

    config = get_disjoint_config(hl, edge_feature_dim, batchsize, lr, project, cve, epoch, GRAPH_MODE)
    for k, v in config.items():
        print("%s= %s" % (k, v))
else:
    config = get_single_config(hl, edge_feature_dim, batchsize, lr, project, cve, epoch, GRAPH_MODE)
    for k, v in config.items():
        print("%s= %s" % (k, v))

# Set random seeds
seed = config["seed"]
random.seed(seed)
np.random.seed(seed + 1)
torch.manual_seed(seed + 2)
torch.backends.cudnn.deterministic = False
torch.backends.cudnn.benchmark = True

training_set, validation_set = build_datasets(config)

# num_pairs_batch_in_training_set 训练集中一个epoch的batchsize的数量
num_pairs_batch_in_training_set = training_set.num_pairs_batch  
num_pairs_batch_in_validation_set = validation_set.num_pairs_batch 

if config["training"]["mode"] == "pair":
    training_data_iter = training_set.pairs()
    # print("training_data_iter",training_data_iter)
    first_batch_graphs, _ = next(training_data_iter)
validation_pairs_iter = validation_set.pairs()

node_feature_dim = first_batch_graphs.node_features.shape[-1]

model, optimizer = build_model(config, node_feature_dim, edge_feature_dim)
model.to(device)

# ###############################################
if os.path.isfile(config["ckpt_save_path"]):
    checkpoint = torch.load(config["ckpt_save_path"])
    model.load_state_dict(checkpoint["model_state_dict"])
    optimizer.load_state_dict(checkpoint["optimizer_state_dict"])
    print("model reloaded from ckpt~")
else:
    print("learning from scratch~")


training_n_graphs_in_batch = config["training"]["batch_size"]
if config["training"]["mode"] == "pair":
    training_n_graphs_in_batch *= 2
    num_samples_batch_in_training_set = num_pairs_batch_in_training_set
else:
    raise ValueError("Unknown training mode: %s" % config["training"]["mode"])



epoch_idx = 0
info_str = ""
list_time = [ ]
t_start = time.time()

# ! change
avg_loss_50 = 0
# # torch.autograd.set_detect_anomaly(True)

for train_batch_idx, train_batch in enumerate(training_data_iter):

    batch_idx_of_epoch = train_batch_idx - epoch_idx * num_samples_batch_in_training_set
    #print("train_batch", train_batch)
    model.train(mode=True)
    # batch = next(training_data_iter)
    if config["training"]["mode"] == "pair":
        node_features, edge_features, from_idx, to_idx, graph_idx, labels = get_graph(train_batch)
        labels = labels.to(device)
    else:
        node_features, edge_features, from_idx, to_idx, graph_idx = get_graph(train_batch)

    graph_vectors = model(
        node_features.to(device),
        edge_features.to(device),
        from_idx.to(device),
        to_idx.to(device),
        graph_idx.to(device),
        training_n_graphs_in_batch,
    )
    # print("graph_vectors", graph_vectors.shape, graph_vectors)

    if config["training"]["mode"] == "pair":
        # print(f"========================\ngraph: {graph_vectors}\n======================")
        x, y = reshape_and_split_tensor(graph_vectors, 2)


        loss = pairwise_loss(x, y, labels, loss_type=config["training"]["loss"], margin=config["training"]["margin"])
        is_pos = (labels == torch.ones(labels.shape).long().to(device)).float()
        is_neg = 1 - is_pos
        n_pos = torch.sum(is_pos)
        n_neg = torch.sum(is_neg)
        sim = compute_similarity(config, x, y)

        sim_pos = torch.sum(sim * is_pos) / (n_pos + 1e-8)
        sim_neg = torch.sum(sim * is_neg) / (n_neg + 1e-8)

        pair_auc_train = auc(sim, labels)
    else:
        x_1, y, x_2, z = reshape_and_split_tensor(graph_vectors, 4)
        loss = triplet_loss(x_1, y, x_2, z, loss_type=config["training"]["loss"], margin=config["training"]["margin"])
        sim1_train = compute_similarity(config, x_1, y)
        sim2_train = compute_similarity(config, x_2, z)

        sim_pos = torch.mean(compute_similarity(config, x_1, y))
        sim_neg = torch.mean(compute_similarity(config, x_2, z))
        triplet_acc_train = torch.mean((sim1_train > sim2_train).float())

    graph_vec_scale = torch.mean(graph_vectors**2)
    if config["training"]["graph_vec_regularizer_weight"] > 0:
        loss += config["training"]["graph_vec_regularizer_weight"] * 0.5 * graph_vec_scale

    optimizer.zero_grad()
    # loss = torch.sum(loss)
    loss.backward(torch.ones_like(loss))  #
    # loss.backward()  #
    nn.utils.clip_grad_value_(model.parameters(), config["training"]["clip_value"])
    optimizer.step()

    sim_diff = sim_pos - sim_neg  
    batch_loss = torch.sum(loss)

    avg_loss_50 += batch_loss
    if (train_batch_idx + 1) % 50 == 0:

        print("----------------------------------------------------------------------------")
        print(f"Batch {train_batch_idx-50} - {train_batch_idx} avg loss: {avg_loss_50 / 50}")
        print("----------------------------------------------------------------------------")
        avg_loss_50 = 0

    # evaluation
    if config["training"]["mode"] == "pair":
        new_info = "batch{}_epoch_{}: batch_loss = {}; sim_pos = {}; sim_neg = {}; sim_diff = {}; pair_auc(train) = {}\n".format(
            batch_idx_of_epoch,
            epoch_idx,
            batch_loss.cpu().detach().numpy().item(),
            sim_pos,
            sim_neg,
            sim_diff,
            pair_auc_train,
        )
        # print(new_info)
        info_str += new_info
    else:
        new_info = "batch{}_epoch_{}: batch_loss = {}; sim_pos = {}; sim_neg = {}; sim_diff = {}; triplet_acc(train) = {}\n".format(
            batch_idx_of_epoch,
            epoch_idx,
            batch_loss.cpu().detach().numpy().item(),
            sim_pos,
            sim_neg,
            sim_diff,
            triplet_acc_train,
        )
        # print(new_info)
        info_str += new_info

    if (train_batch_idx + 1) % 50 == 0:  # dump to log every 50 batch
        log_path = config["training_log_path"]
        log_dir = os.path.dirname(log_path)
        if not os.path.exists(log_dir):
            os.makedirs(log_dir)
    
        with open(log_path, "a") as info_logger:
            info_logger.write(info_str)
            info_str = ""

    if (train_batch_idx + 1) % num_samples_batch_in_training_set == 0:
        t_end = time.time()
        t_diff = t_end - t_start
        list_time.append(t_diff)
        new_info = "start: {}, end: {}; time consumption of epoch {}: {}".format(t_start, t_end, epoch_idx, t_diff)
        print(new_info)
        info_str += new_info
        model.eval()
        with torch.no_grad():
            accumulated_pair_auc = []
            # for vali_pair_batch_idx in range(config['training']['step_per_vali_epoch']):
            for vali_pair_batch_idx in range(num_pairs_batch_in_validation_set):
                batch = next(validation_pairs_iter)
                node_features, edge_features, from_idx, to_idx, graph_idx, labels = get_graph(batch)
                labels = labels.to(device)
                eval_pairs = model(
                    node_features.to(device),
                    edge_features.to(device),
                    from_idx.to(device),
                    to_idx.to(device),
                    graph_idx.to(device),
                    config["evaluation"]["batch_size"] * 2,
                )

                x, y = reshape_and_split_tensor(eval_pairs, 2)
                similarity = compute_similarity(config, x, y)
                pair_auc = auc(similarity, labels)
                accumulated_pair_auc.append(pair_auc)
                new_info = "batch_{}_of_validation_epoch_{}(pair): pair_auc = {}\n".format(
                    vali_pair_batch_idx, epoch_idx, pair_auc
                )
                # print(new_info)
                info_str += new_info
                if (vali_pair_batch_idx + 1) % 50 == 0:
                    with open(config["training_log_path"], "a") as info_logger:
                        info_logger.write(info_str)
                        info_str = ""

            info_str += "validation_epoch_{}: mean_accumulated_pair_auc = {}\n".format(
                epoch_idx, np.mean(accumulated_pair_auc)
            )
            log_path = config["training_log_path"]
            log_dir = os.path.dirname(log_path)
            os.makedirs(log_dir, exist_ok=True)
            with open(config["training_log_path"], "a") as info_logger:
                info_logger.write(info_str)
                info_str = ""
            if not os.path.exists(os.path.dirname(config["ckpt_save_path"])):
                os.makedirs(os.path.dirname(config["ckpt_save_path"]))
            torch.save(
                {"model_state_dict": model.state_dict(), "optimizer_state_dict": optimizer.state_dict()},
                config["ckpt_save_path"],
            )
            print("model saved~")
        model.train()
        epoch_idx += 1

total_time = sum(list_time)
print("train_total_time", total_time)

if hl:
    model_path = os.path.join("../../data/", args.project, args.cve_id, "model_hl", GRAPH_MODE) + '/'
else:
    model_path = os.path.join("../../data/", args.project, args.cve_id, "model", GRAPH_MODE) + '/'

os.makedirs(model_path, exist_ok=True)
torch.save(model, model_path + args.cve_id + ".pkl")
