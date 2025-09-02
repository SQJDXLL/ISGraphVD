import os
import sys


def get_yzd_config():
    node_state_dim = 56   
    graph_rep_dim = 128
    graph_embedding_net_config = dict(
        node_state_dim=node_state_dim,
        edge_hidden_sizes=[node_state_dim * 2, node_state_dim * 2],
        node_hidden_sizes=[node_state_dim * 2],   
        n_prop_layers=5,
        share_prop_params=True,
        edge_net_init_scale=0.1,
        node_update_type="gru",
        use_reverse_direction=True,
        reverse_dir_param_different=True,
        layer_norm=False,
    )
    graph_matching_net_config = graph_embedding_net_config.copy()
    graph_matching_net_config["similarity"] = "dotproduct"
    batch_size = 32
    learning_rate = 1e-4
    training_indicator = 'b{}_lr{}_{}_log'.format(batch_size, learning_rate ,"CVE-2015-6031_new_detect_layer5_single")
    ckpt_save_path = 'saved_ckpt/{}_ckpt'.format(training_indicator)
    training_log_path = 'training_logs/{}_log.txt'.format(training_indicator)
    if os.path.isfile(training_log_path):
        os.system('rm {}'.format(training_log_path))
    os.system('touch {}'.format(training_log_path))
    return dict(
        encoder=dict(
            node_hidden_sizes=[node_state_dim],
            node_feature_dim=63,
            edge_hidden_sizes=None),
        aggregator=dict(
            node_hidden_sizes=[graph_rep_dim],
            graph_transform_sizes=[graph_rep_dim],
            input_size=[node_state_dim],
            gated=True,
            aggregation_type="sum",
        ),
        graph_embedding_net=graph_embedding_net_config,
        graph_matching_net=graph_matching_net_config,
         
        model_type="matching",   
        data=dict(
            problem="malicious_detection",
            dataset_params=dict(

                training_dataset_dir_vul_nm='../graph_matrix/output/curl/CVE-2010-3482/train/vul/node',
                training_dataset_dir_vul_am='../graph_matrix/output/curl/CVE-2010-3482/train/vul/adj',
                training_dataset_dir_fix_nm='../graph_matrix/output/curl/CVE-2010-3482/train/fix/node',
                training_dataset_dir_fix_am='../graph_matrix/output/curl/CVE-2010-3482/train/fix/adj',
                validation_dataset_dir_vul_nm='../graph_matrix/output/curl/CVE-2010-3482/validate/vul/node',
                validation_dataset_dir_vul_am='../graph_matrix/output/curl/CVE-2010-3482/validate/vul/adj',
                validation_dataset_dir_fix_nm='../graph_matrix/output/curl/CVE-2010-3482/validate/fix/node',
                validation_dataset_dir_fix_am='../graph_matrix/output/curl/CVE-2010-3482/validate/fix/adj',
                train_dataset_dir='../graph_matrix/output/miniupnpc/CVE-2015-6031/train',
                vali_dataset_dir='../graph_matrix/output/miniupnpc/CVE-2015-6031/validate/',   
                eval_dataset_dir='../graph_matrix/output/miniupnpc/CVE-2015-6031/test/',   
                max_num_node_of_one_graph=50000
            ),
        ),
        training=dict(
            batch_size = batch_size, 
            learning_rate=learning_rate,
            mode="pair",
            loss="margin",
            margin=1.0,
            graph_vec_regularizer_weight=1e-6,
            clip_value=10.0,
            num_epoch = 10,
            print_after=10,
            eval_after=50,
            step_per_train_epoch=10000,
            step_per_vali_epoch=1000,
            num_validation_pairs=3000,
            num_validation_triplets=3000
        ),
        evaluation=dict(
            batch_size=batch_size
        ),
        seed=8,
        ckpt_save_path=ckpt_save_path,
        training_log_path=training_log_path,
        if_sampling=False
    )

def get_zyl_config():
    node_state_dim = 56   
    graph_rep_dim = 128
    graph_embedding_net_config = dict(
        node_state_dim=node_state_dim,
        edge_hidden_sizes=[node_state_dim * 2, node_state_dim * 2],
        node_hidden_sizes=[node_state_dim * 2],   
        n_prop_layers=5,
        share_prop_params=True,
        edge_net_init_scale=0.1,
        node_update_type="gru",
        use_reverse_direction=True,
        reverse_dir_param_different=True,
        layer_norm=False,
    )
    graph_matching_net_config = graph_embedding_net_config.copy()
    graph_matching_net_config["similarity"] = "dotproduct"
     
    batch_size = 32
    learning_rate = 1e-4
    training_indicator = 'b{}_lr{}_{}_log'.format(batch_size, learning_rate ,"CVE-2021-22901_diff_new_layer5_single")
    ckpt_save_path = 'saved_ckpt/{}_ckpt'.format(training_indicator)
    training_log_path = 'training_logs/{}_log.txt'.format(training_indicator)
    if os.path.isfile(training_log_path):
        os.system('rm {}'.format(training_log_path))
    os.system('touch {}'.format(training_log_path))

    return dict(
        encoder=dict(
            node_hidden_sizes=[node_state_dim],
            node_feature_dim=63,
            edge_hidden_sizes=None),
         
        aggregator=dict(
            node_hidden_sizes=[graph_rep_dim],
            graph_transform_sizes=[graph_rep_dim],
            input_size=[node_state_dim],
            gated=True,
            aggregation_type="sum",
        ),
        graph_embedding_net=graph_embedding_net_config,
        graph_matching_net=graph_matching_net_config,
         
        model_type="matching",   
        data=dict(
            problem="malicious_detection",
            dataset_params=dict(
                 
                train_dataset_dir = '../dataset_diff_matrix/curl/CVE-2021-22901/train',
                eval_dataset_dir='../dataset_diff_matrix/curl/CVE-2021-22901/validate', 
                vali_dataset_dir='../dataset_diff_matrix/curl/CVE-2021-22901/test',   
                max_num_node_of_one_graph=50000
            ),
        ),
        training=dict(
            batch_size=batch_size,
             
            learning_rate=learning_rate,
            mode="pair",
            loss="margin",
            margin=1.0,
            graph_vec_regularizer_weight=1e-6,     
            clip_value=10.0,
            num_epoch = 50,
            print_after=10,
            eval_after=50,
            step_per_train_epoch=10000,
            step_per_vali_epoch=1000,
            num_validation_pairs=3000,
            num_validation_triplets=3000
        ),
        evaluation=dict(
            batch_size=batch_size
        ),
        seed=8,
        ckpt_save_path=ckpt_save_path,
        training_log_path=training_log_path,
        if_sampling=False
    )

 
def get_zyl_disjoint_config(edge_state_dim: int = 3):
    node_state_dim = 56   
     
    graph_rep_dim = 128
    graph_embedding_net_config = dict(
        node_state_dim=node_state_dim,
        edge_hidden_sizes=[node_state_dim * 2 + edge_state_dim, node_state_dim * 2],
        node_hidden_sizes=[node_state_dim * 2],   
        n_prop_layers=5,
        share_prop_params=True,
        edge_net_init_scale=0.1,
        node_update_type="gru",
        use_reverse_direction=True,
        reverse_dir_param_different=True,
        layer_norm=False,
    )
    graph_matching_net_config = graph_embedding_net_config.copy()
    graph_matching_net_config["similarity"] = "dotproduct"
    batch_size = 32
    learning_rate = 1e-4
    training_indicator = 'b{}_lr{}_{}_log'.format(batch_size, learning_rate ,"CVE-2018-20679_diff_disjoint_new_layer5_single")
    ckpt_save_path = 'saved_ckpt/{}_ckpt'.format(training_indicator)
    training_log_path = 'training_logs/{}_log.txt'.format(training_indicator)
    if os.path.isfile(training_log_path):
        os.system('rm {}'.format(training_log_path))
    os.system('touch {}'.format(training_log_path))
     
    return dict(
        encoder=dict(
            node_hidden_sizes=[node_state_dim], 
            node_feature_dim=63,
            edge_hidden_sizes=None),
         
        aggregator=dict(
            node_hidden_sizes=[graph_rep_dim], 
            graph_transform_sizes=[graph_rep_dim],
            input_size=[node_state_dim], 
            gated=True,
            aggregation_type="sum",
        ),
         
        graph_embedding_net = graph_embedding_net_config,
        graph_matching_net = graph_matching_net_config,
         
        model_type="matching",   
        data=dict(
            problem="malicious_detection",
            dataset_params=dict(
                 
                train_dataset_dir = '../dataset_diff_matrix/busybox/CVE-2018-20679/train',
                eval_dataset_dir='../dataset_diff_matrix/busybox/CVE-2018-20679/validate', 
                vali_dataset_dir='../dataset_diff_matrix/busybox/CVE-2018-20679/test',   
                max_num_node_of_one_graph=50000
            ),
        ),
        training=dict(
            batch_size=batch_size,    
            learning_rate=learning_rate,
            mode="pair",
            loss="margin",
            margin=1.0,
            graph_vec_regularizer_weight=1e-6,  
            clip_value=10.0,
            num_epoch = 20, 
            print_after=10,
            eval_after=50,
            step_per_train_epoch=10000,
            step_per_vali_epoch=1000,
            num_validation_pairs=3000,
            num_validation_triplets=3000
        ),
        evaluation=dict(
            batch_size=batch_size
        ),
        seed=8,
        ckpt_save_path=ckpt_save_path,
        training_log_path=training_log_path,
        if_sampling=False
    )



def get_disjoint_config(hl, edge_state_dim: int = 3, batchsize: int = 32, lr: float = 1e-4, project: str = "curl", cve: str = "CVE-2021-22901", epoch: int = 10, graphMode: str = "disjoint"):
    node_state_dim = 56   
     
    graph_rep_dim = 128
    graph_embedding_net_config = dict(
        node_state_dim=node_state_dim,
        edge_hidden_sizes=[node_state_dim * 2 + edge_state_dim, node_state_dim * 2],
        node_hidden_sizes=[node_state_dim * 2],   
        n_prop_layers=5,
        share_prop_params=True,
        edge_net_init_scale=0.1,
        node_update_type="gru",
        use_reverse_direction=True,
        reverse_dir_param_different=True,
        layer_norm=False,
    )
    graph_matching_net_config = graph_embedding_net_config.copy()
    graph_matching_net_config["similarity"] = "dotproduct"
    batch_size = batchsize
    learning_rate = lr
    training_indicator = 'b{}_lr{}_{}_log'.format(batch_size, learning_rate, cve)

    if hl:
        ckpt_save_path = '../../data/{}/{}/model_hl/{}/saved_ckpt/{}_ckpt'.format(project, cve, graphMode, training_indicator)
    else:
        ckpt_save_path = '../../data/{}/{}/model/{}/saved_ckpt/{}_ckpt'.format(project, cve, graphMode, training_indicator)
     
    if hl:
        training_log_path = '../../data/{}/{}/model_hl/{}/training_logs/{}_log.txt'.format(project, cve, graphMode, training_indicator)
    else:
        training_log_path = '../../data/{}/{}/model/{}/training_logs/{}_log.txt'.format(project, cve, graphMode, training_indicator)

    if os.path.isfile(training_log_path):
        os.system('rm {}'.format(training_log_path))
    os.system('touch {}'.format(training_log_path))

    if hl:
        path_dataset = os.path.join("../../data/", project, cve, "matrix_{}_divide_hl".format(graphMode))
    else:
        path_dataset = os.path.join("../../data/", project, cve, "matrix_{}_divide".format(graphMode))

    if hl:
        edge_hidden_sizes_value = [edge_state_dim]
    else:
        if graphMode == "single":
            edge_hidden_sizes_value = None
        else:
            edge_hidden_sizes_value = [edge_state_dim]

    return dict(
        encoder=dict(
            node_hidden_sizes=[node_state_dim], 
            node_feature_dim=63,
             
            edge_hidden_sizes=edge_hidden_sizes_value),  
         
        aggregator=dict(
            node_hidden_sizes=[graph_rep_dim], 
            graph_transform_sizes=[graph_rep_dim],
            input_size=[node_state_dim], 
            gated=True,
            aggregation_type="sum",
        ),
         
        graph_embedding_net = graph_embedding_net_config,
        graph_matching_net = graph_matching_net_config,
         
        model_type="matching",   
        
        data=dict(
            problem="malicious_detection",
            dataset_params=dict(
                train_dataset_dir = os.path.join(path_dataset, "train"),
                vali_dataset_dir = os.path.join(path_dataset, "validate"),
                eval_dataset_dir = os.path.join(path_dataset, "test"),
                max_num_node_of_one_graph = 50000
            ),
        ),
        training=dict(
            batch_size=batch_size,
            learning_rate=learning_rate,
            mode="pair",
            loss="margin",
            margin=1.0,
            graph_vec_regularizer_weight=1e-6,
            clip_value=10.0,
            num_epoch = epoch, 
            print_after=10, 
            eval_after=50,
            step_per_train_epoch=10000,
            step_per_vali_epoch=1000,
            num_validation_pairs=3000,
            num_validation_triplets=3000
        ),
        evaluation=dict(
            batch_size=batch_size
        ),
        seed=8,
        ckpt_save_path=ckpt_save_path,
        training_log_path=training_log_path,
        if_sampling=False
    )


def get_single_config(hl, edge_state_dim: int = 1, batchsize: int = 32, lr: float = 1e-4, project: str = "curl", cve: str = "CVE-2021-22901", epoch: int = 10, graphMode: str = "disjoint"):
    node_state_dim = 56   
     
    graph_rep_dim = 128
    graph_embedding_net_config = dict(
        node_state_dim=node_state_dim,
        edge_hidden_sizes=[node_state_dim * 2 + edge_state_dim, node_state_dim * 2],
        node_hidden_sizes=[node_state_dim * 2],   
        n_prop_layers=5,   
        share_prop_params=True,
        edge_net_init_scale=0.1,
        node_update_type="gru",
        use_reverse_direction=True,
        reverse_dir_param_different=True,
        layer_norm=False,
    )
    graph_matching_net_config = graph_embedding_net_config.copy()
    graph_matching_net_config["similarity"] = "dotproduct"
    batch_size = batchsize
    learning_rate = lr
    training_indicator = 'b{}_lr{}_{}_log'.format(batch_size, learning_rate, cve)

    if hl:
        ckpt_save_path = '../../data/{}/{}/model_hl/{}/saved_ckpt/{}_ckpt'.format(project, cve, graphMode, training_indicator)
    else:
        ckpt_save_path = '../../data/{}/{}/model/{}/saved_ckpt/{}_ckpt'.format(project, cve, graphMode, training_indicator)
     
    if hl:
        training_log_path = '../../data/{}/{}/model_hl/{}/training_logs/{}_log.txt'.format(project, cve, graphMode, training_indicator)
    else:
        training_log_path = '../../data/{}/{}/model/{}/training_logs/{}_log.txt'.format(project, cve, graphMode, training_indicator)

    if os.path.isfile(training_log_path):
        os.system('rm {}'.format(training_log_path))
    os.system('touch {}'.format(training_log_path))

    if hl:
        path_dataset = os.path.join("../../data/", project, cve, "matrix_{}_divide_hl".format(graphMode))
    else:
        path_dataset = os.path.join("../../data/", project, cve, "matrix_{}_divide".format(graphMode))

    if hl:
        edge_hidden_sizes_value = [edge_state_dim]
    else:
        edge_hidden_sizes_value = None
        

    return dict(
        encoder=dict(
            node_hidden_sizes=[node_state_dim],
            node_feature_dim=63,
            edge_hidden_sizes=edge_hidden_sizes_value),
         
        aggregator=dict(
            node_hidden_sizes=[graph_rep_dim],
            graph_transform_sizes=[graph_rep_dim],
            input_size=[node_state_dim],
            gated=True,
            aggregation_type="sum",
        ),
         
        graph_embedding_net = graph_embedding_net_config,
        graph_matching_net = graph_matching_net_config,
        model_type="matching",   
        data=dict(
            problem="malicious_detection",
            dataset_params=dict(
                train_dataset_dir = os.path.join(path_dataset, "train"),
                vali_dataset_dir = os.path.join(path_dataset, "validate"),
                eval_dataset_dir = os.path.join(path_dataset, "test"),
                max_num_node_of_one_graph = 50000
            ),
        ),
        training=dict(
            batch_size=batch_size,
             
            learning_rate=learning_rate,
            mode="pair",
            loss="margin",
            margin=1.0,
            graph_vec_regularizer_weight=1e-6,
            clip_value=10.0,
            num_epoch = epoch,
            print_after=10,
            eval_after=50,
            step_per_train_epoch=10000,
            step_per_vali_epoch=1000,
            num_validation_pairs=3000,
            num_validation_triplets=3000
        ),
        evaluation=dict(
            batch_size=batch_size
        ),
        seed=8,
        ckpt_save_path=ckpt_save_path,
        training_log_path=training_log_path,
        if_sampling=False
    )


