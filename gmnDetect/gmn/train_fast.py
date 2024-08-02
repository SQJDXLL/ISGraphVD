import os
# from torch.utils.data import Dataset
# from torch.utils.data import DataLoader

from torch_geometric.data import Data
from torch_geometric.loader import DataLoader
from evaluation import compute_similarity, auc
from loss import pairwise_loss, triplet_loss
from utils import *
from configure import *
import collections
import time
import sys
import pickle
from graphMatrix.config import GRAPH_TYPES, GRAPH_MODE
from argparse import ArgumentParser

parser = ArgumentParser("train disjoint.")
parser.add_argument("--project", type=str, default="curl")
parser.add_argument("--cve_id", type=str, default="CVE-2021-22901")
parser.add_argument("--gpu", type=str, default="0")
parser.add_argument("--hl", action="store_true", default=False)
parser.add_argument("--batch_size", type=int, default=32)
parser.add_argument("--learning_rate", type=float, default=1e-4)
parser.add_argument("--num_epoch", type=int, default=10)
args = parser.parse_args()

os.environ["CUDA_DEVICE_ORDER"] = "PCI_BUS_ID"
os.environ["CUDA_VISIBLE_DEVICES"] = args.gpu

import numpy as np
import torch.nn as nn

# Set GPU

use_cuda = torch.cuda.is_available()
print("****************", use_cuda, args.gpu)
device = torch.device("cuda:0" if use_cuda else "cpu")

# # Print configure
# config = get_yzd_config()
# for (k, v) in config.items():
#     print("%s= %s" % (k, v))

# anonymous

# ! 因为模型的build过程参数的维度大小是由config决定的
# ! 直接在model初始化过程里面传入 edge_featue_dim 可能会有问题
hl = args.hl
batchsize = args.batch_size
lr = args.learning_rate
project = args.project
cve = args.cve_id
epoch = args.num_epoch

# 如果高亮方案通过扩充维度执行
if GRAPH_MODE == "single":
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

# if GRAPH_MODE == 'single':
#     if hl:
#         edge_feature_dim = 1
#         # edge_feature_dim = 1
#     else:
#         edge_feature_dim = 1
# else:
#     if hl:
#         edge_feature_dim = 1
#     else:
#         edge_feature_dim = 1
# 如果高亮方案没有扩充维度执行
# if GRAPH_MODE == 'single':
#     edge_feature_dim = 1
# else:
#     edge_feature_dim = len(GRAPH_TYPES)


if GRAPH_MODE == "disjoint":
    # config = get_disjoint_config(hl, edge_feature_dim, batchsize, lr, project, cve, epoch, GRAPH_MODE)
    # for k, v in config.items():
    #     print("%s= %s" % (k, v))
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

# 返回的是类
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
# edge_feature_dim = first_batch_graphs.edge_features.shape[-1] - 2
# 边特征维度：3
# edge_feature_dim = 1
# torch.autograd.set_detect_anomaly(True)
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
# ###############################################

# accumulated_metrics = collections.defaultdict(list)

training_n_graphs_in_batch = config["training"]["batch_size"]
if config["training"]["mode"] == "pair":
    training_n_graphs_in_batch *= 2
    num_samples_batch_in_training_set = num_pairs_batch_in_training_set
else:
    raise ValueError("Unknown training mode: %s" % config["training"]["mode"])
# print('num_samples_batch_in_training_set = {}'.format(num_samples_batch_in_training_set))
# print('num_pairs_batch_in_validation_set = {}'.format(num_pairs_batch_in_validation_set))

epoch_idx = 0
info_str = ""
t_start = time.time()

# ! change
avg_loss_50 = 0

data, labels = training_set.generate_all_data

data_list = []
print(data[0][0].shape,data[0][1].shape, data[0][2].shape, data[0][3].shape,  )
for idx, pair in enumerate(data):
    data_list.append(Data(x=pair[0], edge_index=pair[2][..., :2].t(), edge_attr=pair[2][..., 2:], y=labels[idx]))
    data_list.append(Data(x=pair[1], edge_index=pair[3][..., :2].t(), edge_attr=pair[3][..., 2:], y=labels[idx]))

train_loader = DataLoader(data_list, batch_size=config["training"]["batch_size"] * 2, drop_last=True, follow_batch=['x'])

# # data是列表存储tensor形式
# # print("data", data[0][0], data[1][0])

#  # 统一tensor的形状
# def same_shape(tensor_groups):
#     # 保存原始形状
#     # original_shapes = [[tensor.shape for tensor in group] for group in tensor_groups]
#     num_tensors = len(tensor_groups[0])  # 获取每组中tensor的数量
#     original_shapes = [[] for _ in range(num_tensors)]  # 初始化对应位置的形状列表

#     # 遍历每个tensor组
#     for group in tensor_groups:
#         for position, tensor in enumerate(group):
#             # 将每个位置的tensor形状添加到对应的列表中
#             original_shapes[position].append(tensor.shape)

#     # 找出每个位置tensor的最大形状
#     max_shapes = []
#     for i in range(4):  # 每组有4个tensor
#         max_shape = torch.Size([0, 0])  # 初始化最大形状
#         for group in tensor_groups:
#             max_shape = torch.Size([max(max_shape[0], group[i].shape[0]), max(max_shape[1], group[i].shape[1])])
#             # print("MAX_SHAPE", max_shape)
#         max_shapes.append(max_shape)

#     # 填充每个tensor到统一形状
#     padded_tensor_groups = []
#     for group in tensor_groups:
#         padded_group = []
#         for i, tensor in enumerate(group):
#             # pad = (0, max_shapes[i][1] - tensor.shape[1], 0, max_shapes[i][0] - tensor.shape[0])
#             # padded_tensor = F.pad(tensor, pad)
#             #在右侧和下侧填充
#             pad = (max_shapes[i][1] - tensor.shape[1], 0, 0, max_shapes[i][0] - tensor.shape[0])
#             padded_tensor = F.pad(tensor, pad)
#             padded_group.append(padded_tensor)
#         padded_tensor_groups.append(padded_group)
#     return original_shapes, padded_tensor_groups

# original_shapes, data_same_shape = same_shape(data)

# # 假设你已经有了一个Dataset对象，它返回你需要的训练数据
# train_dataset = CustomDataset(data_same_shape, labels)
# train_loader = DataLoader(train_dataset, batch_size=config["training"]["batch_size"], num_workers=4, pin_memory=True, drop_last=True)


# print("train_loader", type(train_loader))
model.train(mode=True)
model.to(device)

# epoch_idx为每个batch被分配的索引
for train_batch_idx, x in enumerate(train_loader):
    # print(type(x), len(x), type(x[0]), len(x[0]), type(x[0][0]), x[0][0].shape, x[0][1].shape, x[0][2].shape, x[0][3].shape)
    # print("x[1]", x[1].shape, x[1])
    # print(x[0][0])

    batch_idx_of_epoch = train_batch_idx - epoch_idx * num_samples_batch_in_training_set

    # start_idx = train_batch_idx * train_loader.batch_size
    # print("start_idx", start_idx)

    # end_idx = start_idx + train_loader.batch_size
    # print("end_idx", end_idx)

    # # 提取出当前批次中每个样本的原始形状
    # batch_original_shapes = []
    # for each in original_shapes:
    #     batch_original_shapes.append(each[start_idx:end_idx])
    # print("batch_original_shapes", len(batch_original_shapes[0]))
    # batch_shapes = [list(t) for t in zip(*batch_original_shapes)]
    # reorganized_list = [[sublist[i] for sublist in batch_shapes] for i in range(4)]
    # batch_shapes = reorganized_list
    # # print("batch_shapes", batch_shapes)

    # # 还原成之前的形状
    # restored_tensor_groups = []
    # restored_group_nm = []
    # restored_group_ad = []
    # # for padded_group, orig_shapes in zip(x[0], batch_shapes):
    # for padded_group, each_shapes in zip(x[0], batch_shapes):
    #     for padded_tensor, orig_shape in zip(padded_group, each_shapes):
    #         if orig_shape[0] <= padded_tensor.size(0) and orig_shape[1] <= padded_tensor.size(1):
    #             if orig_shape[1] == 56:
    #                 restored_group_nm.append(padded_tensor[: orig_shape[0], : orig_shape[1]])
    #             else:
    #                 restored_group_ad.append(padded_tensor[: orig_shape[0], : orig_shape[1]])

    # graphs = pack_batch_torch(restored_group_nm, restored_group_ad)

    # # 根据pack_batch_torch组织成graph，然后投入
    # node_features, edge_features, from_idx, to_idx, graph_idx = get_graph_torch(graphs)
    # print(node_features.shape)
    # print(edge_features.shape)
    
    node_features = x.x.float()
    edge_features = x.edge_attr.float()
    from_idx = x.edge_index.t()[..., 0].long()
    to_idx = x.edge_index.t()[..., 1].long()
    labels = x.y[:config["training"]["batch_size"]].to(device)
    graph_idx = x.x_batch

    node_features = node_features.to(device)
    edge_features = edge_features.to(device)
    from_idx = from_idx.to(device)
    to_idx = to_idx.to(device)
    graph_idx = graph_idx.to(device)
    # labels = x[1].to(device)

    # print(node_features.shape)
    # print(edge_features.shape)
    # print(from_idx.shape, to_idx.shape)
    # print(graph_idx.shape)
    # print(training_n_graphs_in_batch)

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
        # 样本标签为1的数量
        is_pos = (labels == torch.ones(labels.shape).long().to(device)).float()
        # 样本标签为-1的数量
        is_neg = 1 - is_pos
        n_pos = torch.sum(is_pos)
        n_neg = torch.sum(is_neg)
        sim = compute_similarity(config, x, y)
        sim_pos = torch.sum(sim * is_pos) / (n_pos + 1e-8)
        sim_neg = torch.sum(sim * is_neg) / (n_neg + 1e-8)
        # print(f"========================\nsim_pos:{sim_pos}\nsim_neg: {sim_neg}\n======================")
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

    sim_diff = sim_pos - sim_neg  # 正数减去负数
    batch_loss = torch.sum(loss)
    # print("batch_loss", batch_loss)

    avg_loss_50 += batch_loss
    print("train_batch_idx", train_batch_idx, (train_batch_idx + 1) % 50)
    if (train_batch_idx + 1) % 50 == 0:
        # if (epoch_idx + 1) % 50 == 0:
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
        # with open(config["training_log_path"], "a") as info_logger:
        #     info_logger.write(info_str)
        #     info_str = ""

    # if (train_batch_idx + 1) % config['training']['step_per_train_epoch'] == 0:
    # 一个epoch，使用验证集检查一下效果
    if (train_batch_idx + 1) % num_samples_batch_in_training_set == 0:
        t_end = time.time()
        t_diff = t_end - t_start
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
                new_info = "batch_{}_of_validation_epoch_{}(pair): pair_auc = {}\n".format(vali_pair_batch_idx, epoch_idx, pair_auc)
                # print(new_info)
                info_str += new_info
                if (vali_pair_batch_idx + 1) % 50 == 0:
                    with open(config["training_log_path"], "a") as info_logger:
                        info_logger.write(info_str)
                        info_str = ""

            # accumulated_triplet_acc = []
            # for vali_triplet_batch_idx in range(config['training']['step_per_vali_epoch']):
            #     batch = next(validation_triplet_iter)
            #     node_features, edge_features, from_idx, to_idx, graph_idx = get_graph(batch)
            #     eval_triplets = model(node_features.to(device), edge_features.to(device), from_idx.to(device),
            #                           to_idx.to(device),
            #                           graph_idx.to(device),
            #                           config['evaluation']['batch_size'] * 4)
            #     x_1, y, x_2, z = reshape_and_split_tensor(eval_triplets, 4)
            #     # print('x1 = {}\n x2 = {}'.format(x_1, x_2))
            #     sim_1 = compute_similarity(config, x_1, y)
            #     sim_2 = compute_similarity(config, x_2, z)
            #     # print('sim_1(triplet) = {}; sim_2(triplet) = {}'.format(sim_1, sim_2))
            #     triplet_acc = torch.mean((sim_1 > sim_2).float())
            #     accumulated_triplet_acc.append(triplet_acc.cpu().numpy())
            #     new_info = 'batch_{}_of_validation_epoch_{}(triplet): triplet_acc = {}\n'.format(
            #         vali_triplet_batch_idx, epoch_idx, triplet_acc)
            #     print(new_info)
            #     info_str += new_info
            #     if (vali_triplet_batch_idx + 1) % 50 == 0:
            #         with open(config['training_log_path'], 'a') as info_logger:
            #             info_logger.write(info_str)
            #             info_str = ''

            info_str += "validation_epoch_{}: mean_accumulated_pair_auc = {}\n".format(epoch_idx, np.mean(accumulated_pair_auc))
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
        # t_start = time.time()

# time_end = time.time()
# time_cost = time_end - time_start
# print("traing_timeCost", time_cost)

if hl:
    model_path = os.path.join("../../data/", args.project, args.cve_id, "model_hl", GRAPH_MODE) + "/"
else:
    model_path = os.path.join("../../data/", args.project, args.cve_id, "model", GRAPH_MODE) + "/"


os.makedirs(model_path, exist_ok=True)
torch.save(model, model_path + args.cve_id + ".pkl")
