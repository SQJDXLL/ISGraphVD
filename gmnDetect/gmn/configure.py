import os
import sys


def get_yzd_config():
    node_state_dim = 56  # not set yet, but it gotta be the dim of original node feature (from yzd)
    graph_rep_dim = 128
    graph_embedding_net_config = dict(
        node_state_dim=node_state_dim,
        edge_hidden_sizes=[node_state_dim * 2, node_state_dim * 2],
        node_hidden_sizes=[node_state_dim * 2],  # this setting is based on appendix of the paper (from yzd)
        n_prop_layers=5,
        # set to False to not share parameters across message passing layers
        share_prop_params=True,
        # initialize message MLP with small parameter weights to prevent
        # aggregated message vectors blowing up, alternatively we could also use
        # e.g. layer normalization to keep the scale of these under control.
        edge_net_init_scale=0.1,
        # other types of update like `mlp` and `residual` can also be used here.
        node_update_type="gru",
        # set to False if your graph already contains edges in both directions.
        use_reverse_direction=True,
        # set to True if your graph is directed
        reverse_dir_param_different=True,
        # we didn't use layer norm in our experiments but sometimes this can help.
        layer_norm=False,
    )
    graph_matching_net_config = graph_embedding_net_config.copy()
    graph_matching_net_config["similarity"] = "dotproduct"
    # batch_size = 32
    batch_size = 32
    learning_rate = 1e-4
    # learning_rate = 5  # 1e-3 actually (from yzd)
    training_indicator = 'b{}_lr{}_{}_log'.format(batch_size, learning_rate ,"CVE-2015-6031_new_detect_layer5_single")
    #training_indicator = 'debug_log'
    ckpt_save_path = 'saved_ckpt/{}_ckpt'.format(training_indicator)
    #ckpt_save_path = 'debug_ckpt'
    training_log_path = 'training_logs/{}_log.txt'.format(training_indicator)
    if os.path.isfile(training_log_path):
        os.system('rm {}'.format(training_log_path))
    os.system('touch {}'.format(training_log_path))
    # armhf_gcc-8_default_hostapd_fix_hostapd_notif_node.npy

    return dict(
        encoder=dict(
            node_hidden_sizes=[node_state_dim],
            node_feature_dim=63,
            edge_hidden_sizes=None),
        # encoder=dict(node_hidden_sizes=[node_state_dim], edge_hidden_sizes=None),
        aggregator=dict(
            node_hidden_sizes=[graph_rep_dim],
            graph_transform_sizes=[graph_rep_dim],
            input_size=[node_state_dim],
            gated=True,
            aggregation_type="sum",
        ),
        graph_embedding_net=graph_embedding_net_config,
        graph_matching_net=graph_matching_net_config,
        # Set to `embedding` to use the graph embedding net.
        model_type="matching",  # this project provides both of the 2 model_types: graph embedding & graph matching
        data=dict(
            problem="malicious_detection",
            dataset_params=dict(
                # always generate graphs with 20 nodes and p_edge=0.2.

                training_dataset_dir_vul_nm='../graph_matrix/output/curl/CVE-2010-3482/train/vul/node',
                training_dataset_dir_vul_am='../graph_matrix/output/curl/CVE-2010-3482/train/vul/adj',
                training_dataset_dir_fix_nm='../graph_matrix/output/curl/CVE-2010-3482/train/fix/node',
                training_dataset_dir_fix_am='../graph_matrix/output/curl/CVE-2010-3482/train/fix/adj',
                validation_dataset_dir_vul_nm='../graph_matrix/output/curl/CVE-2010-3482/validate/vul/node',
                validation_dataset_dir_vul_am='../graph_matrix/output/curl/CVE-2010-3482/validate/vul/adj',
                validation_dataset_dir_fix_nm='../graph_matrix/output/curl/CVE-2010-3482/validate/fix/node',
                validation_dataset_dir_fix_am='../graph_matrix/output/curl/CVE-2010-3482/validate/fix/adj',
                # train_dataset_dir='../graph_matrix/output/dnsmasq/CVE-2015-8899/train',
                # vali_dataset_dir='../graph_matrix/output/dnsmasq/CVE-2015-8899/validate/', 
                # #for test.py
                # eval_dataset_dir='../graph_matrix/output/dnsmasq/CVE-2015-8899/test/',  # including other projects

                train_dataset_dir='../graph_matrix/output/miniupnpc/CVE-2015-6031/train',
                vali_dataset_dir='../graph_matrix/output/miniupnpc/CVE-2015-6031/validate/', 
                #for test.py
                eval_dataset_dir='../graph_matrix/output/miniupnpc/CVE-2015-6031/test/',  # including other projects
                
                # training & validation dataset dir should be set here (from yzd)
                max_num_node_of_one_graph=50000
            ),
        ),
        training=dict(
            batch_size = batch_size,
            # learning_rate=eval('1e-{}'.format(learning_rate)),
            learning_rate=learning_rate,
            mode="pair",
            loss="margin",
            margin=1.0,
            # A small regularizer on the graph vector scales to avoid the graph
            # vectors blowing up.  If numerical issues is particularly bad in the
            # model we can add `snt.LayerNorm` to the outputs of each layer, the
            # aggregated messages and aggregated node representations to
            # keep the network activation scale in a reasonable range.
            graph_vec_regularizer_weight=1e-6,
            # Add gradient clipping to avoid large gradients.
            clip_value=10.0,
            # Increase this to train longer.
            #num_epoch=10,  # I prefer num_epoch than total training_steps (yzd)
            num_epoch = 10,
            # n_training_steps=10000,
            # Print training information every this many training steps.
            print_after=10,
            # Evaluate on validation set every `eval_after * print_after` steps.
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



def get_anonymous_config():
    node_state_dim = 56  # not set yet, but it gotta be the dim of original node feature (from yzd)
    graph_rep_dim = 128
    graph_embedding_net_config = dict(
        node_state_dim=node_state_dim,
        edge_hidden_sizes=[node_state_dim * 2, node_state_dim * 2],
        node_hidden_sizes=[node_state_dim * 2],  # this setting is based on appendix of the paper (from yzd)
        n_prop_layers=5,
        # set to False to not share parameters across message passing layers
        share_prop_params=True,
        # initialize message MLP with small parameter weights to prevent
        # aggregated message vectors blowing up, alternatively we could also use
        # e.g. layer normalization to keep the scale of these under control.
        edge_net_init_scale=0.1,
        # other types of update like `mlp` and `residual` can also be used here.
        node_update_type="gru",
        # set to False if your graph already contains edges in both directions.
        use_reverse_direction=True,
        # set to True if your graph is directed
        reverse_dir_param_different=True,
        # we didn't use layer norm in our experiments but sometimes this can help.
        layer_norm=False,
    )
    graph_matching_net_config = graph_embedding_net_config.copy()
    graph_matching_net_config["similarity"] = "dotproduct"
    # batch_size = 32
    batch_size = 32
    learning_rate = 1e-4
    # learning_rate = 5  # 1e-3 actually (from yzd)
    training_indicator = 'b{}_lr{}_{}_log'.format(batch_size, learning_rate ,"CVE-2021-22901_diff_new_layer5_single")
    #training_indicator = 'debug_log'
    ckpt_save_path = 'saved_ckpt/{}_ckpt'.format(training_indicator)
    #ckpt_save_path = 'debug_ckpt'
    training_log_path = 'training_logs/{}_log.txt'.format(training_indicator)
    if os.path.isfile(training_log_path):
        os.system('rm {}'.format(training_log_path))
    os.system('touch {}'.format(training_log_path))
    # armhf_gcc-8_default_hostapd_fix_hostapd_notif_node.npy

    return dict(
        encoder=dict(
            node_hidden_sizes=[node_state_dim],
            node_feature_dim=63,
            edge_hidden_sizes=None),
        # encoder=dict(node_hidden_sizes=[node_state_dim], edge_hidden_sizes=None),
        aggregator=dict(
            node_hidden_sizes=[graph_rep_dim],
            graph_transform_sizes=[graph_rep_dim],
            input_size=[node_state_dim],
            gated=True,
            aggregation_type="sum",
        ),
        graph_embedding_net=graph_embedding_net_config,
        graph_matching_net=graph_matching_net_config,
        # Set to `embedding` to use the graph embedding net.
        model_type="matching",  # this project provides both of the 2 model_types: graph embedding & graph matching
        data=dict(
            problem="malicious_detection",
            dataset_params=dict(
                # always generate graphs with 20 nodes and p_edge=0.2.
                train_dataset_dir = '../dataset_diff_matrix/curl/CVE-2021-22901/train',
                eval_dataset_dir='../dataset_diff_matrix/curl/CVE-2021-22901/validate', 
                #for test.py
                vali_dataset_dir='../dataset_diff_matrix/curl/CVE-2021-22901/test',  # including other projects
                # training & validation dataset dir should be set here (from yzd)
                max_num_node_of_one_graph=50000
            ),
        ),
        training=dict(
            batch_size=batch_size,
            # learning_rate=eval('1e-{}'.format(learning_rate)),
            learning_rate=learning_rate,
            mode="pair",
            loss="margin",
            margin=1.0,
            # A small regularizer on the graph vector scales to avoid the graph
            # vectors blowing up.  If numerical issues is particularly bad in the
            # model we can add `snt.LayerNorm` to the outputs of each layer, the
            # aggregated messages and aggregated node representations to
            # keep the network activation scale in a reasonable range.
            graph_vec_regularizer_weight=1e-6,
            # Add gradient clipping to avoid large gradients.
            clip_value=10.0,
            # Increase this to train longer.
            #num_epoch=10,  # I prefer num_epoch than total training_steps (yzd)
            num_epoch = 50,
            # n_training_steps=10000,
            # Print training information every this many training steps.
            print_after=10,
            # Evaluate on validation set every `eval_after * print_after` steps.
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


def get_anonymous_disjoint_config(edge_state_dim: int = 3):
    node_state_dim = 56  # not set yet, but it gotta be the dim of original node feature (from yzd)
    # The dimension of graph representation is 128.
    graph_rep_dim = 128
    graph_embedding_net_config = dict(
        node_state_dim=node_state_dim,
        edge_hidden_sizes=[node_state_dim * 2 + edge_state_dim, node_state_dim * 2],
        node_hidden_sizes=[node_state_dim * 2],  # this setting is based on appendix of the paper (from yzd)
        n_prop_layers=5,
        # set to False to not share parameters across message passing layers
        share_prop_params=True,
        # initialize message MLP with small parameter weights to prevent
        # aggregated message vectors blowing up, alternatively we could also use
        # e.g. layer normalization to keep the scale of these under control.
        edge_net_init_scale=0.1,
        # other types of update like `mlp` and `residual` can also be used here.
        node_update_type="gru",
        # set to False if your graph already contains edges in both directions.
        use_reverse_direction=True,
        # set to True if your graph is directed
        reverse_dir_param_different=True,
        # we didn't use layer norm in our experiments but sometimes this can help.
        layer_norm=False,
    )
    graph_matching_net_config = graph_embedding_net_config.copy()
    graph_matching_net_config["similarity"] = "dotproduct"
    # batch_size = 32
    batch_size = 32
    learning_rate = 1e-4
    # learning_rate = 5  # 1e-3 actually (from yzd)
    training_indicator = 'b{}_lr{}_{}_log'.format(batch_size, learning_rate ,"CVE-2018-20679_diff_disjoint_new_layer5_single")
    #training_indicator = 'debug_log'
    ckpt_save_path = 'saved_ckpt/{}_ckpt'.format(training_indicator)
    #ckpt_save_path = 'debug_ckpt'
    training_log_path = 'training_logs/{}_log.txt'.format(training_indicator)
    if os.path.isfile(training_log_path):
        os.system('rm {}'.format(training_log_path))
    os.system('touch {}'.format(training_log_path))
    # armhf_gcc-8_default_hostapd_fix_hostapd_notif_node.npy

    return dict(
        encoder=dict(
            node_hidden_sizes=[node_state_dim],#56
            node_feature_dim=63,
            # edge_hidden_sizes=[edge_state_dim]),
            edge_hidden_sizes=None),
        # encoder=dict(node_hidden_sizes=[node_state_dim], edge_hidden_sizes=None),
        aggregator=dict(
            node_hidden_sizes=[graph_rep_dim],#128
            graph_transform_sizes=[graph_rep_dim],
            input_size=[node_state_dim],#56
            gated=True,
            aggregation_type="sum",
        ),
        # config文件是一样的
        graph_embedding_net = graph_embedding_net_config,
        graph_matching_net = graph_matching_net_config,
        # Set to `embedding` to use the graph embedding net.
        model_type="matching",  # this project provides both of the 2 model_types: graph embedding & graph matching
        data=dict(
            problem="malicious_detection",
            dataset_params=dict(
                # always generate graphs with 20 nodes and p_edge=0.2.
                train_dataset_dir = '../dataset_diff_matrix/busybox/CVE-2018-20679/train',
                eval_dataset_dir='../dataset_diff_matrix/busybox/CVE-2018-20679/validate', 
                #for test.py
                vali_dataset_dir='../dataset_diff_matrix/busybox/CVE-2018-20679/test',  # including other projects
                # training & validation dataset dir should be set here (from yzd)
                max_num_node_of_one_graph=50000
            ),
        ),
        training=dict(
            batch_size=batch_size,
            # learning_rate=eval('1e-{}'.format(learning_rate)),
            learning_rate=learning_rate,
            mode="pair",
            loss="margin",
            margin=1.0,
            # A small regularizer on the graph vector scales to avoid the graph
            # vectors blowing up.  If numerical issues is particularly bad in the
            # model we can add `snt.LayerNorm` to the outputs of each layer, the
            # aggregated messages and aggregated node representations to
            # keep the network activation scale in a reasonable range.
            graph_vec_regularizer_weight=1e-6,
            # Add gradient clipping to avoid large gradients.
            clip_value=10.0,
            # Increase this to train longer.
            #num_epoch=10,  # I prefer num_epoch than total training_steps (yzd)
            num_epoch = 20,
            # n_training_steps=10000,
            # Print training information every this many training steps.
            print_after=10,
            # Evaluate on validation set every `eval_after * print_after` steps.
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
    node_state_dim = 56  # not set yet, but it gotta be the dim of original node feature (from yzd)
    # The dimension of graph representation is 128.
    graph_rep_dim = 128
    graph_embedding_net_config = dict(
        node_state_dim=node_state_dim,
        edge_hidden_sizes=[node_state_dim * 2 + edge_state_dim, node_state_dim * 2],
        node_hidden_sizes=[node_state_dim * 2],  # this setting is based on appendix of the paper (from yzd)
        n_prop_layers=5,
        # set to False to not share parameters across message passing layers
        share_prop_params=True,
        # initialize message MLP with small parameter weights to prevent
        # aggregated message vectors blowing up, alternatively we could also use
        # e.g. layer normalization to keep the scale of these under control.
        edge_net_init_scale=0.1,
        # other types of update like `mlp` and `residual` can also be used here.
        node_update_type="gru",
        # set to False if your graph already contains edges in both directions.
        use_reverse_direction=True,
        # set to True if your graph is directed
        reverse_dir_param_different=True,
        # we didn't use layer norm in our experiments but sometimes this can help.
        layer_norm=False,
    )
    graph_matching_net_config = graph_embedding_net_config.copy()
    graph_matching_net_config["similarity"] = "dotproduct"
    # batch_size = 32
    batch_size = batchsize
    learning_rate = lr
    # learning_rate = 5  # 1e-3 actually (from yzd)
    # training_indicator = 'b{}_lr{}_{}_log'.format(batch_size, learning_rate ,"CVE-2019-12110_disjoint_new_layer5_single")
    training_indicator = 'b{}_lr{}_{}_log'.format(batch_size, learning_rate, cve)
    #training_indicator = 'debug_log'

    if hl:
        ckpt_save_path = '../../data/{}/{}/model_hl/{}/saved_ckpt/{}_ckpt'.format(project, cve, graphMode, training_indicator)
    else:
        ckpt_save_path = '../../data/{}/{}/model/{}/saved_ckpt/{}_ckpt'.format(project, cve, graphMode, training_indicator)
    #ckpt_save_path = 'debug_ckpt'
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
            node_hidden_sizes=[node_state_dim],#56
            node_feature_dim=63,
            # edge_hidden_sizes = None),
            edge_hidden_sizes=edge_hidden_sizes_value),
        # encoder=dict(node_hidden_sizes=[node_state_dim], edge_hidden_sizes=None),
        aggregator=dict(
            node_hidden_sizes=[graph_rep_dim],#128
            graph_transform_sizes=[graph_rep_dim],
            input_size=[node_state_dim],#56
            gated=True,
            aggregation_type="sum",
        ),
        # config文件是一样的
        graph_embedding_net = graph_embedding_net_config,
        graph_matching_net = graph_matching_net_config,
        # Set to `embedding` to use the graph embedding net.
        model_type="matching",  # this project provides both of the 2 model_types: graph embedding & graph matching
        
        data=dict(
            problem="malicious_detection",
            
            dataset_params=dict(
                # always generate graphs with 20 nodes and p_edge=0.2.
               
                train_dataset_dir = os.path.join(path_dataset, "train"),
                vali_dataset_dir = os.path.join(path_dataset, "validate"),
                eval_dataset_dir = os.path.join(path_dataset, "test"),
                
                # train_dataset_dir ='../graph_matrix/output_disjoint/miniupnpc/CVE-2019-12110/train',
                # vali_dataset_dir ='../graph_matrix/output_disjoint/miniupnpc/CVE-2019-12110/validate/', 
                # #for test.py
                # eval_dataset_dir ='../graph_matrix/output_disjoint/miniupnpc/CVE-2019-12110/test/', # training & validation dataset dir should be set here (from yzd)
                max_num_node_of_one_graph = 50000
            ),
        ),
        training=dict(
            batch_size=batch_size,
            # learning_rate=eval('1e-{}'.format(learning_rate)),
            learning_rate=learning_rate,
            mode="pair",
            loss="margin",
            margin=1.0,
            # A small regularizer on the graph vector scales to avoid the graph
            # vectors blowing up.  If numerical issues is particularly bad in the
            # model we can add `snt.LayerNorm` to the outputs of each layer, the
            # aggregated messages and aggregated node representations to
            # keep the network activation scale in a reasonable range.
            graph_vec_regularizer_weight=1e-6,
            # Add gradient clipping to avoid large gradients.
            clip_value=10.0,
            # Increase this to train longer.
            #num_epoch=10,  # I prefer num_epoch than total training_steps (yzd)
            num_epoch = epoch,
            # n_training_steps=10000,
            # Print training information every this many training steps.
            print_after=10,
            # Evaluate on validation set every `eval_after * print_after` steps.
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
    node_state_dim = 56  # not set yet, but it gotta be the dim of original node feature (from yzd)
    # The dimension of graph representation is 128.
    graph_rep_dim = 128
    graph_embedding_net_config = dict(
        node_state_dim=node_state_dim,
        edge_hidden_sizes=[node_state_dim * 2 + edge_state_dim, node_state_dim * 2],
        node_hidden_sizes=[node_state_dim * 2],  # this setting is based on appendix of the paper (from yzd)
        n_prop_layers=5,
        # set to False to not share parameters across message passing layers
        share_prop_params=True,
        # initialize message MLP with small parameter weights to prevent
        # aggregated message vectors blowing up, alternatively we could also use
        # e.g. layer normalization to keep the scale of these under control.
        edge_net_init_scale=0.1,
        # other types of update like `mlp` and `residual` can also be used here.
        node_update_type="gru",
        # set to False if your graph already contains edges in both directions.
        use_reverse_direction=True,
        # set to True if your graph is directed
        reverse_dir_param_different=True,
        # we didn't use layer norm in our experiments but sometimes this can help.
        layer_norm=False,
    )
    graph_matching_net_config = graph_embedding_net_config.copy()
    graph_matching_net_config["similarity"] = "dotproduct"
    # batch_size = 32
    batch_size = batchsize
    learning_rate = lr
    # learning_rate = 5  # 1e-3 actually (from yzd)
    # training_indicator = 'b{}_lr{}_{}_log'.format(batch_size, learning_rate ,"CVE-2019-12110_disjoint_new_layer5_single")
    training_indicator = 'b{}_lr{}_{}_log'.format(batch_size, learning_rate, cve)
    #training_indicator = 'debug_log'

    if hl:
        ckpt_save_path = '../../data/{}/{}/model_hl/{}/saved_ckpt/{}_ckpt'.format(project, cve, graphMode, training_indicator)
    else:
        ckpt_save_path = '../../data/{}/{}/model/{}/saved_ckpt/{}_ckpt'.format(project, cve, graphMode, training_indicator)
    #ckpt_save_path = 'debug_ckpt'
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
        # encoder=dict(node_hidden_sizes=[node_state_dim], edge_hidden_sizes=None),
        aggregator=dict(
            node_hidden_sizes=[graph_rep_dim],
            graph_transform_sizes=[graph_rep_dim],
            input_size=[node_state_dim],
            gated=True,
            aggregation_type="sum",
        ),
        # config文件是一样的
        graph_embedding_net = graph_embedding_net_config,
        graph_matching_net = graph_matching_net_config,
        # Set to `embedding` to use the graph embedding net.
        model_type="matching",  # this project provides both of the 2 model_types: graph embedding & graph matching
        
        data=dict(
            problem="malicious_detection",
            
            dataset_params=dict(
                # always generate graphs with 20 nodes and p_edge=0.2.
               
                train_dataset_dir = os.path.join(path_dataset, "train"),
                vali_dataset_dir = os.path.join(path_dataset, "validate"),
                eval_dataset_dir = os.path.join(path_dataset, "test"),
                
                # train_dataset_dir ='../graph_matrix/output_disjoint/miniupnpc/CVE-2019-12110/train',
                # vali_dataset_dir ='../graph_matrix/output_disjoint/miniupnpc/CVE-2019-12110/validate/', 
                # #for test.py
                # eval_dataset_dir ='../graph_matrix/output_disjoint/miniupnpc/CVE-2019-12110/test/', # training & validation dataset dir should be set here (from yzd)
                max_num_node_of_one_graph = 50000
            ),
        ),
        training=dict(
            batch_size=batch_size,
            # learning_rate=eval('1e-{}'.format(learning_rate)),
            learning_rate=learning_rate,
            mode="pair",
            loss="margin",
            margin=1.0,
            # A small regularizer on the graph vector scales to avoid the graph
            # vectors blowing up.  If numerical issues is particularly bad in the
            # model we can add `snt.LayerNorm` to the outputs of each layer, the
            # aggregated messages and aggregated node representations to
            # keep the network activation scale in a reasonable range.
            graph_vec_regularizer_weight=1e-6,
            # Add gradient clipping to avoid large gradients.
            clip_value=10.0,
            # Increase this to train longer.
            #num_epoch=10,  # I prefer num_epoch than total training_steps (yzd)
            num_epoch = epoch,
            # n_training_steps=10000,
            # Print training information every this many training steps.
            print_after=10,
            # Evaluate on validation set every `eval_after * print_after` steps.
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


