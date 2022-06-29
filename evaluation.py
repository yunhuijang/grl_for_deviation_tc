from statistics import mean
from trace_clustering import trace_clustering, filter_log_with_cluster
from pm4py.algo.evaluation.replay_fitness import algorithm as replay_fitness_evaluator
from pm4py.algo.evaluation.precision import algorithm as precision_evaluator
from pm4py.algo.evaluation.simplicity.algorithm import apply as simplicity_evaluator
from pm4py.algo.evaluation.generalization.evaluator import apply as generalization_evaluator
import pickle
from pm4py.objects.log.obj import EventLog
from feature_extraction import feature_extraction_ae
from transform_event_log import transform_event_log, split_train_test, train_test_loader, run, map_graph_embedding, \
    get_data_list, GraphDataset, get_instance_graph, trace2vec_event_log, case_dict_gram, sequence_alignment_embedding, \
    alignment_subnet_trace_clustering
from process_discovery import process_discovery, make_deviation_label
import numpy as np
import pandas as pd
import random
from sklearn.metrics import silhouette_score
from pm4py.algo.filtering.log.attributes import attributes_filter
import multiprocessing
from discover_expressive_process_models import discover_expressive_process_models
from graph2vec import edge2vec_embedding, node2vec_embedding, get_model_embedding_and_variants
from set_gnn_param import get_best_gnn_param
from distance_graph_model import map_distance_graph


def cluster_write_evaluate(log, filename, eval_parameter, algorithm, conformance_type, sample_size, random_state,
                           embedding, eval_type):
    cluster_result = trace_clustering(embedding)
    filtered_log_list = filter_log_with_cluster(log, cluster_result)
    evaluation = evaluate_process_mining(filtered_log_list,
                                         evaluation_parameter=eval_parameter,
                                         algorithm=algorithm,
                                         conformance_type=conformance_type)

    write_file(filename=filename, eval_parameter=eval_parameter, algorithm=algorithm,
               eval_type=eval_type, eval_result=evaluation,
               filtered_log=filtered_log_list,
               conformance_type=conformance_type, sample_size=sample_size,
               random_state=random_state)


def silhouette_evaluate(embedding):
    cluster_result = trace_clustering(embedding)
    result = silhouette_score(list(embedding.values()), list(cluster_result.values()))
    return result


def evaluate_clustering(log, filename, eval_parameter, algorithm, sample_size, random_state, evaluation_type):
    act_freq_result, act_set_result, act_gram_result, t2v_ae_result, t2v_result, gnn_result = (
    None, None, None, None, None, None)
    clustering_result = {'act_freq': act_freq_result, 'act_set': act_set_result,
                         'act_gram': act_gram_result, 't2v_ae': t2v_ae_result,
                         't2v': t2v_result, 'gnn': gnn_result}
    try:
        with open(
                f'''./output/evaluation/{filename}/{sample_size}/clustering_evaluation_{algorithm}_{eval_parameter}_{random_state}_result.txt''',
                'rb') as f:
            clustering_result = pickle.load(f)
    except FileNotFoundError:
        with open(
                f'''./output/evaluation/{filename}/{sample_size}/clustering_evaluation_{algorithm}_{eval_parameter}_{random_state}_result.txt''',
                'wb') as f:
            pickle.dump(clustering_result, f)

    act_freq_result = clustering_result['act_freq']
    act_set_result = clustering_result['act_set']
    act_gram_result = clustering_result['act_gram']
    t2v_result = clustering_result['t2v']
    t2v_ae_result = clustering_result['t2v_ae']
    gnn_result = clustering_result['gnn']

    # traditional approach
    case_dict = transform_event_log(log)
    case_act_freq = {key: value['activity_freq_list'] for key, value in
                     case_dict.items()}
    case_act_list = {key: value['activity_list'] for key, value in case_dict.items()}

    if 'act_freq' in evaluation_type:
        # activitiy frequency
        print("Activity frequency")
        act_freq_result = silhouette_evaluate(case_act_freq)

    if 'act_set' in evaluation_type:
        # activity 유무
        print("Activity list")
        act_set_result = silhouette_evaluate(case_act_list)

    if 'act_gram' in evaluation_type:
        # n-gram
        print('n-gram')
        case_gram = case_dict_gram(log)
        act_gram_result = silhouette_evaluate(case_gram)

    if 't2v' in evaluation_type:
        # trace2vec
        case_dict_trace2vec = trace2vec_event_log(f'data/sampled/{filename}_{sample_size}_{random_state}_t2v.xes', 16)
        t2v_result = silhouette_evaluate(case_dict_trace2vec)

    if 'gnn' in evaluation_type:

        # graph embedding
        print("Graph embedding")
        instance_graph = get_instance_graph(log)
        data_list, _, _ = get_data_list(instance_graph, log, filename, sample_size, random_state, parameters=eval_parameter,
                                  algorithm=algorithm)
        dataset = GraphDataset('data/', data_list, instance_graph=instance_graph,
                               log=log)
        if data_list[0].x.shape != dataset[0].x.shape:
            raise Exception('dataset 형태 오류')

        train_dataset, test_dataset = split_train_test(dataset)
        train_loader, test_loader = train_test_loader(train_dataset, test_dataset)
        model, train_acc, test_acc = run(train_loader, test_loader, dataset)
        case_dict = transform_event_log(log)
        graph_embedding = map_graph_embedding(case_dict, model, dataset)
        gnn_result = silhouette_evaluate(graph_embedding)

    clustering_result = {'act_freq': act_freq_result, 'act_set': act_set_result,
                         'act_gram': act_gram_result, 't2v_ae': t2v_ae_result,
                         't2v': t2v_result, 'gnn': gnn_result}
    with open(
            f'''./output/evaluation/{filename}/{sample_size}/clustering_evaluation_{algorithm}_{eval_parameter}_{random_state}_result.txt''',
            'wb') as f:
        pickle.dump(clustering_result, f)


def count_dimension(embedding):
    for embed in embedding.values():
        return len(embed)


def evaluate_dimension(log, filename, eval_parameter, algorithm, conformance_type,
                       sample_size, random_state, evaluation_type):

    dim_result = {'act_freq': 0, 'act_set': 0,
                         'act_gram': 0, 't2v_ae': 0,
                         't2v': 0, 'gnn': 0, 'sa': 0, 'gnn_mid': 0}

    # traditional approach
    case_dict = transform_event_log(log)
    case_act_freq = {key: value['activity_freq_list'] for key, value in
                     case_dict.items()}
    case_act_list = {key: value['activity_list'] for key, value in case_dict.items()}

    if 'act_freq' in evaluation_type:
        # activitiy frequency
        print("Activity frequency")
        dim_result['act_freq'] = count_dimension(case_act_freq)

    if 'act_set' in evaluation_type:
        # activity 유무
        print("Activity list")
        dim_result['act_set'] = count_dimension(case_act_list)

    if 'act_gram' in evaluation_type:
        # n-gram
        print('n-gram')
        case_gram = case_dict_gram(log)
        dim_result['act_gram'] = count_dimension(case_gram)

    if ('dev_list' in evaluation_type) or ('dev_onehot' in evaluation_type):
        net, im, fm = process_discovery(log, discovery_parameter=eval_parameter, algorithm=algorithm)
        deviation_list, deviation_onehot = make_deviation_label(log, net, im, fm)
        if 'dev_list' in evaluation_type:
            dim_result['dev_list'] = count_dimension(deviation_list)
        if 'dev_onehot' in evaluation_type:
            dim_result['dev_onehot'] = count_dimension(deviation_onehot)

    if ('t2v_ae' in evaluation_type) or ('t2v' in evaluation_type):
        # trace2vec
        case_dict_trace2vec = trace2vec_event_log(f'data/sampled/{filename}_{sample_size}_{random_state}_t2v.xes', 16)

        if 't2v_ae' in evaluation_type:
            # t2v with ae
            print("Trace2vec with AE")
            case_dict_ae = feature_extraction_ae(case_dict_trace2vec)
            print(count_dimension(case_dict_ae))

        if 't2v' in evaluation_type:
            # t2v without ae
            print("Trace2vec without AE")
            print(count_dimension(case_dict_trace2vec))

    if 'gnn' in evaluation_type or 'gnn_mid' in evaluation_type:

        # graph embedding
        print("Graph embedding")
        instance_graph = get_instance_graph(log)
        data_list, _, _ = get_data_list(instance_graph, log, filename, sample_size, random_state, parameters=eval_parameter,
                                  algorithm=algorithm)
        dataset = GraphDataset('data/', data_list, instance_graph=instance_graph,
                               log=log)
        if data_list[0].x.shape != dataset[0].x.shape:
            raise Exception('dataset 형태 오류')

        if 'gnn' in evaluation_type:
            print(dataset[0].y.shape)

    if 'sa' in evaluation_type:
        print('sa')
        sa_embedding = sequence_alignment_embedding(log)
        dim_result['sa'] = count_dimension(sa_embedding)

    return dim_result


def evaluate(log, filename, eval_parameter, algorithm, conformance_type, sample_size, random_state, evaluation_type):
    '''
    input: log, parameters(for process discovery), algorithm(for process discovery)
    output: evaluation result list (activity freq / activity 유무 / t2v / t2v with ae / graph embedding) 
            - fitness, precision, simplicity, generalization
    '''
    print("Evaluation Start")
    act_freq_result, act_set_result, act_gram_result, t2v_ae_result, t2v_result, gnn_result, sa_result, gnn_mid_result = (
    None, None, None, None, None, None, None, None)
    clustering_result = {'act_freq': act_freq_result, 'act_set': act_set_result,
                         'act_gram': act_gram_result, 't2v_ae': t2v_ae_result,
                         't2v': t2v_result, 'gnn': gnn_result, 'sa': sa_result, 'gnn_mid': gnn_mid_result}
    try:
        with open(
                f'''./output/evaluation/{filename}/{sample_size}/clustering_evaluation_{algorithm}_{eval_parameter}_{random_state}_result.txt''',
                'rb') as f:
            clustering_result = pickle.load(f)
    except FileNotFoundError:
        with open(
                f'''./output/evaluation/{filename}/{sample_size}/clustering_evaluation_{algorithm}_{eval_parameter}_{random_state}_result.txt''',
                'wb') as f:
            pickle.dump(clustering_result, f)

    # traditional approach
    case_dict = transform_event_log(log)
    case_act_freq = {key: value['activity_freq_list'] for key, value in
                     case_dict.items()}
    case_act_list = {key: value['activity_list'] for key, value in case_dict.items()}
    if 'dg' in evaluation_type:
        print('Distance graph')
        dg = map_distance_graph(log)
        cluster_write_evaluate(log, filename, eval_parameter, algorithm,
                               conformance_type, sample_size, random_state,
                               embedding=dg, eval_type='dg')

    if 'act_freq' in evaluation_type:
        # activitiy frequency
        print("Activity frequency")
        cluster_write_evaluate(log, filename, eval_parameter, algorithm,
                               conformance_type, sample_size, random_state,
                               embedding=case_act_freq, eval_type='act_freq')

        act_freq_result = silhouette_evaluate(case_act_freq)

    if 'act_set' in evaluation_type:
        # activity 유무
        print("Activity list")
        cluster_write_evaluate(log, filename, eval_parameter, algorithm,
                               conformance_type, sample_size, random_state,
                               embedding=case_act_list, eval_type='act_set')
        act_set_result = silhouette_evaluate(case_act_list)
    if 'gram_set' in evaluation_type:
        case_gram = case_dict_gram(log)
        case_gram_only_one = {key: list(map((lambda x: 0 if x == 0 else 1), value)) for key, value in case_gram.items()}
        gram_set_dict = {key: np.concatenate((case_act_list[key], case_gram_only_one[key]), axis=None)
                             for key in case_act_list.keys()}
        cluster_write_evaluate(log, filename, eval_parameter, algorithm,
                               conformance_type, sample_size, random_state,
                               embedding=gram_set_dict, eval_type='gram_set')

    if 'act_gram' in evaluation_type:
        # n-gram
        print('n-gram')
        case_gram = case_dict_gram(log)
        cluster_write_evaluate(log, filename, eval_parameter, algorithm,
                               conformance_type, sample_size, random_state,
                               embedding=case_gram, eval_type='act_gram')
        act_gram_result = silhouette_evaluate(case_gram)
    if ('dev_list' in evaluation_type) or ('dev_onehot' in evaluation_type) or ('dev' in evaluation_type)\
            or ('dev_act_freq' in evaluation_type) or ('dev_act_set' in evaluation_type)\
            or ('dev_act_gram' in evaluation_type) or ('dev_ae' in evaluation_type) or ('dev_t2v' in evaluation_type)\
            or ('dev_n2v' in evaluation_type) or ('dev_e2v' in evaluation_type) or ('dev_gram_set' in evaluation_type)\
            or ('dev_gram_set_ae' in evaluation_type):
        net, im, fm = process_discovery(log, discovery_parameter=eval_parameter, algorithm=algorithm)
        # try:
        #     with open(
        #             f'''./output/evaluation/{filename}/{sample_size}/dev_onehot_{algorithm}_{eval_parameter}_{random_state}_result.txt''',
        #             'rb') as f:
        #         deviation_onehot = pickle.load(f)
        #     with open(
        #             f'''./output/evaluation/{filename}/{sample_size}/dev_list_{algorithm}_{eval_parameter}_{random_state}_result.txt''',
        #             'rb') as f:
        #         deviation_list = pickle.load(f)
        # except FileNotFoundError:
        deviation_list, deviation_onehot = make_deviation_label(log, net, im, fm)
        with open(
                f'''./output/evaluation/{filename}/{sample_size}/dev_onehot_{algorithm}_{eval_parameter}_{random_state}_result.txt''',
                'wb') as f:
            pickle.dump(deviation_onehot, f)
        with open(
                f'''./output/evaluation/{filename}/{sample_size}/dev_list_{algorithm}_{eval_parameter}_{random_state}_result.txt''',
                'wb') as f:
            pickle.dump(deviation_list, f)
        if ('dev_gram_set' in evaluation_type) or ('dev_gram_set_ae' in evaluation_type):
            case_gram = case_dict_gram(log)
            case_gram_only_one = {key: list(map((lambda x: 0 if x == 0 else 1), value)) for key, value in
                                  case_gram.items()}
            dev_gram_set_dict = {key: np.concatenate((case_act_list[key], case_gram_only_one[key], deviation_onehot[key]), axis=None)
                             for key in case_act_list.keys()}
            if 'dev_gram_set' in evaluation_type:
                cluster_write_evaluate(log, filename, eval_parameter, algorithm,
                                       conformance_type, sample_size, random_state,
                                       embedding=dev_gram_set_dict, eval_type='dev_gram_set')
            if 'dev_gram_set_ae' in evaluation_type:
                dev_gram_set_ae = feature_extraction_ae(dev_gram_set_dict)
                cluster_write_evaluate(log, filename, eval_parameter, algorithm,
                                       conformance_type, sample_size, random_state,
                                       embedding=dev_gram_set_ae, eval_type='dev_gram_set_ae')
        if 'dev_ae' in evaluation_type:
            case_dict_dev_ae = feature_extraction_ae(deviation_onehot)
            cluster_write_evaluate(log, filename, eval_parameter, algorithm,
                                   conformance_type, sample_size, random_state,
                                   embedding=case_dict_dev_ae, eval_type='dev_ae')
        if 'dev_act_freq' in evaluation_type:
            dev_act_freq_dict = {key: np.concatenate((case_act_freq[key], deviation_onehot[key]), axis=None)
                                 for key in case_act_freq.keys()}
            cluster_write_evaluate(log, filename, eval_parameter, algorithm,
                                   conformance_type, sample_size, random_state,
                                   embedding=dev_act_freq_dict, eval_type='dev_act_freq')

        if 'dev_act_set' in evaluation_type:
            dev_act_set_dict = {key: np.concatenate((case_act_list[key], deviation_onehot[key]), axis=None)
                                 for key in case_act_list.keys()}
            cluster_write_evaluate(log, filename, eval_parameter, algorithm,
                                   conformance_type, sample_size, random_state,
                                   embedding=dev_act_set_dict, eval_type='dev_act_set')

        if 'dev_act_gram' in evaluation_type:
            case_gram = case_dict_gram(log)
            dev_act_gram_dict = {key: np.concatenate((case_gram[key], deviation_onehot[key]), axis=None)
                                 for key in case_gram.keys()}
            cluster_write_evaluate(log, filename, eval_parameter, algorithm,
                                   conformance_type, sample_size, random_state,
                                   embedding=dev_act_gram_dict, eval_type='dev_act_gram')
        if ('dev_n2v' in evaluation_type) or ('dev_e2v' in evaluation_type):
            variants, model_embedding = get_model_embedding_and_variants(log, n=64)
            if 'dev_n2v' in evaluation_type:
                node2vec_mean, node2vec_max = node2vec_embedding(log, variants, model_embedding)
                dev_n2v_dict = {key: np.concatenate((node2vec_mean[key], deviation_onehot[key]), axis=None)
                                    for key in node2vec_mean.keys()}
                cluster_write_evaluate(log, filename, eval_parameter, algorithm,
                                       conformance_type, sample_size, random_state,
                                       embedding=dev_n2v_dict, eval_type='dev_n2v_64')
            if 'dev_e2v' in evaluation_type:
                edge2vec_mean_avg, edge2vec_max_avg, edge2vec_mean_mul, edge2vec_max_mul = edge2vec_embedding(log,
                                                                                                             variants,
                                                                                                             model_embedding,
                                                                                                             n=16)
                dev_e2v_dict = {key: np.concatenate((edge2vec_mean_avg[key], deviation_onehot[key]), axis=None)
                                    for key in edge2vec_mean_avg.keys()}
                cluster_write_evaluate(log, filename, eval_parameter, algorithm,
                                       conformance_type, sample_size, random_state,
                                       embedding=dev_e2v_dict, eval_type='dev_e2v_64')
        if 'dev_t2v' in evaluation_type:
            case_dict_trace2vec = trace2vec_event_log(f'data/sampled/{filename}_{sample_size}_{random_state}_t2v.xes',
                                                      64)
            dev_t2v_dict = {key: np.concatenate((case_dict_trace2vec[key], deviation_onehot[key]), axis=None)
                                    for key in case_dict_trace2vec.keys()}
            cluster_write_evaluate(log, filename, eval_parameter, algorithm,
                                   conformance_type, sample_size, random_state,
                                   embedding=dev_t2v_dict, eval_type='dev_t2v')
        if 'dev_list' in evaluation_type:
            cluster_write_evaluate(log, filename, eval_parameter, algorithm,
                                   conformance_type, sample_size, random_state,
                                   embedding=deviation_list, eval_type='dev_list')
        if 'dev_onehot' in evaluation_type:
            cluster_write_evaluate(log, filename, eval_parameter, algorithm,
                                   conformance_type, sample_size, random_state,
                                   embedding=deviation_onehot, eval_type='dev_onehot')

        if 'dev_dg' in evaluation_type:
            dg = map_distance_graph(log)
            dev_dg_dict = {key: np.concatenate((dg[key], deviation_onehot[key]), axis=None)
                            for key in dg.keys()}
            cluster_write_evaluate(log, filename, eval_parameter, algorithm,
                                   conformance_type, sample_size, random_state,
                                   embedding=dev_dg_dict, eval_type='dev_dg')
        if 'dev_only' in evaluation_type:
            for value in deviation_onehot.values():
                num_deviations = len(value)
                break
            deviation_caseid_dict = dict.fromkeys(range(num_deviations))
            for key, value in deviation_onehot.items():
                dev = np.where(value == 1)[0][0]
                if not deviation_caseid_dict[dev]:
                    deviation_caseid_dict[dev] = set(key)
                else:
                    deviation_caseid_dict[dev].add(key)
            filtered_log_list = []
            for value in deviation_caseid_dict.values():
                cluster_caseid_set = value
                filtered_log_cases = attributes_filter.apply_trace_attribute(log, cluster_caseid_set,
                                                                             parameters={
                                                                                 attributes_filter.Parameters.ATTRIBUTE_KEY: 'concept:name'})
                filtered_log_list.append(filtered_log_cases)
            evaluation = evaluate_process_mining(filtered_log_list,
                                                 evaluation_parameter=eval_parameter,
                                                 algorithm=algorithm,
                                                 conformance_type=conformance_type)
            write_file(filename=filename, eval_parameter=eval_parameter, algorithm=algorithm,
                       eval_type='dev', eval_result=evaluation,
                       filtered_log=filtered_log_list,
                       conformance_type=conformance_type, sample_size=sample_size,
                       random_state=random_state)

    if ('t2v_ae' in evaluation_type) or ('t2v' in evaluation_type):
        # trace2vec
        case_dict_trace2vec = trace2vec_event_log(f'data/sampled/{filename}_{sample_size}_{random_state}_t2v.xes', 64)

        if 't2v_ae' in evaluation_type:
            # t2v with ae
            print("Trace2vec with AE")
            case_dict_ae = feature_extraction_ae(case_dict_trace2vec)
            cluster_write_evaluate(log, filename, eval_parameter, algorithm,
                                   conformance_type, sample_size, random_state,
                                   embedding=case_dict_ae, eval_type='t2v_ae')
            t2v_ae_result = silhouette_evaluate(case_dict_ae)

        if 't2v' in evaluation_type:
            # t2v without ae
            print("Trace2vec without AE")
            cluster_write_evaluate(log, filename, eval_parameter, algorithm,
                                   conformance_type, sample_size, random_state,
                                   embedding=case_dict_trace2vec, eval_type='t2v_64')
            t2v_result = silhouette_evaluate(case_dict_trace2vec)

    if 'gnn' in evaluation_type or 'gnn_mid' in evaluation_type or 'gat' in evaluation_type \
            or 'dev_gnn' in evaluation_type or 'gnn_ft' in evaluation_type or 'gin' in evaluation_type:

        # graph embedding
        print("Graph embedding")
        instance_graph = get_instance_graph(log)
        data_list, dev_list, dev_onehot = get_data_list(instance_graph, log, filename, sample_size, random_state,
                                                        parameters=eval_parameter, algorithm=algorithm)
        with open(
                f'''./output/evaluation/{filename}/{sample_size}/dev_onehot_{algorithm}_{eval_parameter}_{random_state}_result.txt''',
                'wb') as f:
            pickle.dump(dev_onehot, f)
        with open(
                f'''./output/evaluation/{filename}/{sample_size}/dev_list_{algorithm}_{eval_parameter}_{random_state}_result.txt''',
                'wb') as f:
            pickle.dump(dev_list, f)
        dataset = GraphDataset('data/', data_list, instance_graph=instance_graph,
                               log=log)
        if data_list[0].x.shape != dataset[0].x.shape:
            raise Exception('dataset 형태 오류')
        if 'gin' in evaluation_type:
            model_param = 'gin'
        if 'gat' in evaluation_type:
            model_param = 'gat'
        else:
            model_param = 'gcn'
        train_dataset, test_dataset = split_train_test(dataset)
        train_loader, test_loader = train_test_loader(train_dataset, test_dataset)
        if 'gnn_ft' in evaluation_type:
            gnn_param = get_best_gnn_param(train_loader, test_loader, dataset)
        else:
            gnn_param = {'hidden_channels': 128, 'dropout_rate': 0.5,
                         'learning_rate': 0.01}
        model, train_acc, test_acc = run(train_loader, test_loader, dataset, gnn_param, model_param)
        case_dict = transform_event_log(log)
        if 'gnn' in evaluation_type or 'gnn_ft' in evaluation_type or 'gin' in evaluation_type or 'gat' in evaluation_type:
            graph_embedding = map_graph_embedding(case_dict, model, dataset, option='last')
            # graph embedding clustering
            if 'gnn_ft' in evaluation_type:
                cluster_write_evaluate(log, filename, eval_parameter, algorithm,
                                       conformance_type, sample_size, random_state,
                                       embedding=graph_embedding, eval_type='gnn_ft')
            if 'gin' in evaluation_type:
                cluster_write_evaluate(log, filename, eval_parameter, algorithm,
                                       conformance_type, sample_size, random_state,
                                       embedding=graph_embedding, eval_type='gin')
            if 'gat' in evaluation_type:
                cluster_write_evaluate(log, filename, eval_parameter, algorithm,
                                       conformance_type, sample_size, random_state,
                                       embedding=graph_embedding, eval_type='gat')
            else:
                cluster_write_evaluate(log, filename, eval_parameter, algorithm,
                                       conformance_type, sample_size, random_state,
                                       embedding=graph_embedding, eval_type='gnn')
                gnn_result = silhouette_evaluate(graph_embedding)
            with open(
                    f'''./output/evaluation/{filename}/{sample_size}/gnn_accuracy_{algorithm}_{eval_parameter}_{random_state}_result.txt''',
                    'wb') as f:
                pickle.dump([train_acc, test_acc], f)

        if 'gnn_mid' in evaluation_type:
            graph_embedding = map_graph_embedding(case_dict, model, dataset, option='mid')
            # graph embedding clustering
            cluster_write_evaluate(log, filename, eval_parameter, algorithm,
                                   conformance_type, sample_size, random_state,
                                   embedding=graph_embedding, eval_type='gnn_mid_32')
            gnn_mid_result = silhouette_evaluate(graph_embedding)
            with open(
                    f'''./output/evaluation/{filename}/{sample_size}/gnn_mid_32_accuracy_{algorithm}_{eval_parameter}_{random_state}_result.txt''',
                    'wb') as f:
                pickle.dump([train_acc, test_acc], f)
        if 'dev_gnn' in evaluation_type:
            graph_embedding = map_graph_embedding(case_dict, model, dataset, option='last')
            with open(
                    f'''./output/evaluation/{filename}/{sample_size}/dev_onehot_{algorithm}_{eval_parameter}_{random_state}_result.txt''',
                    'rb') as f:
                deviation_onehot = pickle.load(f)
            dev_gnn_dict = {key: np.concatenate((graph_embedding[key], deviation_onehot[key]), axis=None)
                            for key in graph_embedding.keys()}
            cluster_write_evaluate(log, filename, eval_parameter, algorithm,
                                   conformance_type, sample_size, random_state,
                                   embedding=dev_gnn_dict, eval_type='dev_gnn')

    if 'sa' in evaluation_type:
        sa_embedding = sequence_alignment_embedding(log)
        cluster_write_evaluate(log, filename, eval_parameter, algorithm,
                               conformance_type, sample_size, random_state,
                               embedding=sa_embedding, eval_type='sa')
        sa_result = silhouette_evaluate(sa_embedding)

    if 'amstc' in evaluation_type:
        filtered_log_list = alignment_subnet_trace_clustering(log, eval_parameter, algorithm)
        evaluation = evaluate_process_mining(filtered_log_list,
                                             evaluation_parameter=eval_parameter,
                                             algorithm=algorithm,
                                             conformance_type=conformance_type)
        write_file(filename=filename, eval_parameter=eval_parameter, algorithm=algorithm,
                   eval_type='amstc', eval_result=evaluation,
                   filtered_log=filtered_log_list,
                   conformance_type=conformance_type, sample_size=sample_size,
                   random_state=random_state)

    if 'depm' in evaluation_type:
        filtered_log_list = discover_expressive_process_models(log, algorithm=algorithm, discovery_parameter=eval_parameter)
        evaluation = evaluate_process_mining(filtered_log_list,
                                             evaluation_parameter=eval_parameter,
                                             algorithm=algorithm,
                                             conformance_type=conformance_type)
        write_file(filename=filename, eval_parameter=eval_parameter, algorithm=algorithm,
                   eval_type='depm', eval_result=evaluation,
                   filtered_log=filtered_log_list,
                   conformance_type=conformance_type, sample_size=sample_size,
                   random_state=random_state)

    if ('n2v' in evaluation_type) or ('e2v' in evaluation_type):
        variants, model_embedding = get_model_embedding_and_variants(log, n=64)
        if 'n2v' in evaluation_type:
            node2vec_mean, node2vec_max = node2vec_embedding(log, variants, model_embedding)
            cluster_write_evaluate(log, filename, eval_parameter, algorithm,
                                   conformance_type, sample_size, random_state,
                                   embedding=node2vec_mean, eval_type='n2v_mean_64')
            # cluster_write_evaluate(log, filename, eval_parameter, algorithm,
            #                        conformance_type, sample_size, random_state,
            #                        embedding=node2vec_mean, eval_type='n2v_max')
        if 'e2v' in evaluation_type:
            edge2vec_mean_avg, edge2vec_max_avg, edge2vec_mean_mul, edge2vec_max_mul = edge2vec_embedding(log, variants,
                                                                                                      model_embedding, n=16)
            cluster_write_evaluate(log, filename, eval_parameter, algorithm,
                                   conformance_type, sample_size, random_state,
                                   embedding=edge2vec_mean_avg, eval_type='e2v_mean_avg_64')
            # cluster_write_evaluate(log, filename, eval_parameter, algorithm,
            #                        conformance_type, sample_size, random_state,
            #                        embedding=edge2vec_max_avg, eval_type='e2v_max_avg')

    clustering_result = {'act_freq': act_freq_result, 'act_set': act_set_result,
                         'act_gram': act_gram_result, 't2v_ae': t2v_ae_result,
                         't2v': t2v_result, 'gnn': gnn_result, 'sa': sa_result,
                         'gnn_mid': gnn_mid_result}
    with open(
            f'''./output/evaluation/{filename}/{sample_size}/clustering_evaluation_{algorithm}_{eval_parameter}_{random_state}_result.txt''',
            'wb') as f:
        pickle.dump(clustering_result, f)

    print("Trace clustering and evaluation completed")
    return


def sample_event_log(log, sample_size, random_state=1):
    n = len(log)
    random.seed(random_state)
    sampled_log = EventLog(random.sample(log, sample_size))
    return sampled_log


def read_event_log(filename):
    with open(filename, 'rb') as f:
        data = pickle.load(f)
        log = EventLog(data)
        return log


def write_file(filename, eval_parameter, algorithm, eval_type, eval_result, filtered_log, conformance_type, sample_size,
               random_state):
    with open(
            f'''./output/evaluation/{filename}/{sample_size}/{algorithm}_{eval_parameter}_{eval_type}_{conformance_type}_{random_state}_result.txt''',
            'wb') as f:
        pickle.dump(eval_result, f)
    with open(
            f'''./output/evaluation/{filename}/{sample_size}/{algorithm}_{eval_parameter}_{eval_type}_{conformance_type}_{random_state}_log.txt''',
            'wb') as f:
        pickle.dump(filtered_log, f)


def set_value(label, d, metric, value):
    a = d[label]
    a[metric] = value
    d[label] = a


def calculate_alignments(label, filtered_log, evaluation_dict, evaluation_parameter, algorithm, conformance_type):
    print(f'Event log {label} processed: {len(filtered_log)} traces')

    net, im, fm = (None, None, None)
    try:
        net, im, fm = process_discovery(filtered_log, discovery_parameter=evaluation_parameter, algorithm=algorithm)
    except AttributeError as e:
        print(f"No net for label {label} -> all measure 0")
        for metric in ['fitness', 'simplicity', 'precision', 'generalization']:
            set_value(label, evaluation_dict, metric, None)
        return

    try:
        if conformance_type == 'alignment':
            fitness = replay_fitness_evaluator.apply(filtered_log, net, im, fm,
                                                 variant=replay_fitness_evaluator.Variants.ALIGNMENT_BASED)
            set_value(label, evaluation_dict, 'fitness', fitness['averageFitness'])
            set_value(label, evaluation_dict, 'precision', precision_evaluator.apply(filtered_log, net, im, fm,
                                                                            variant=precision_evaluator.Variants.ALIGN_ETCONFORMANCE))
        else:
            fitness = replay_fitness_evaluator.apply(filtered_log, net, im, fm,
                                                     variant=replay_fitness_evaluator.Variants.TOKEN_BASED)
            set_value(label, evaluation_dict, 'fitness', fitness['log_fitness'])
            set_value(label, evaluation_dict, 'precision', precision_evaluator.apply(filtered_log, net, im, fm,
                                                                            variant=precision_evaluator.Variants.ETCONFORMANCE_TOKEN))
    except (Exception) as e:
        print(f'Unsound net -> fitness & precision: 0 for label {label}')
        for metric in ['fitness', 'precision']:
            set_value(label, evaluation_dict, metric, None)

    set_value(label, evaluation_dict, 'simplicity', simplicity_evaluator(net))
    set_value(label, evaluation_dict, 'generalization', generalization_evaluator(filtered_log, net, im, fm))


def evaluate_process_mining(filtered_log_list, evaluation_parameter, algorithm, conformance_type):
    '''
    input: filtered log lilst splitted by cluster labels
    output: evaluation_dict (key: cluster label / value: evaluation metrics(fitness, precision, 
    simplicity, generalization)
    '''
    manager = multiprocessing.Manager()
    evaluation_dict = manager.dict()
    for i in range(len(filtered_log_list)):
        evaluation_dict[i] = {'fitness': 0, 'precision': 0, 'simplicity':0, 'generalization':0}
    processes = []
    for label, filtered_log in enumerate(filtered_log_list):
        p = multiprocessing.Process(target=calculate_alignments, args=(label, filtered_log, evaluation_dict,
                                                                       evaluation_parameter, algorithm,
                                                                       conformance_type))
        processes.append(p)
        p.start()

    for process in processes:
        process.join()

    return dict(evaluation_dict)


def get_average_metric(evaluation_dict):
    '''
    return mean values of evluation metrics
    input: evaluation_dict
    output: mean of fitness, precision, simplicity, generalization
    '''
    fitness_list = [value['fitness'] if value != None else None for value in evaluation_dict.values()]
    precision_list = [value['precision'] if value != None else None for value in evaluation_dict.values()]
    simplicity_list = [value['simplicity'] if value != None else None for value in evaluation_dict.values()]
    generalization_list = [value['generalization'] if value != None else None for value in evaluation_dict.values()]

    dicts = {'fitness': fitness_list, 'precision': precision_list,
             'simplicity': simplicity_list, 'generalization': generalization_list}

    for key, value in dicts.items():
        n = 0
        while (None in value):
            value.remove(None)
            n += 1
        print(f'{n} None in {key}')

    return mean(fitness_list), mean(precision_list), mean(simplicity_list), mean(generalization_list)


def get_weighted_sum(metric, data, log_len):
    metric_list = [value[metric] if value != None else None for value in data.values()]
    selected_indices = [i for i, x in enumerate(metric_list) if x != None]
    n = sum(np.array(log_len)[selected_indices])
    metric_list = list(map(lambda x: 0 if (x == None) else x, metric_list))
    weighted_metric_list = [metric * length / n for metric, length in zip(metric_list, log_len)]
    return sum(weighted_metric_list)


def get_evaluation_df(df):
    df = pd.DataFrame(df)
    df.columns = ['f', 'p', 's', 'g']
    df['f1'] = 2 * df['f'] * df['p'] / (df['f'] + df['p'])
    df = df[['f', 'p', 's', 'g']]
    return df


def get_random_state_average(df, methods, n):
    result_df = pd.DataFrame()
    method_dfs = []
    for i, method in enumerate(methods):
        indices = np.arange(i * n, (i + 1) * n)
        method_df = pd.DataFrame([df.iloc[indices,].mean(axis=0),
                                  df.iloc[indices,].std(axis=0)])
        method_df = method_df.set_index(pd.Series([method, method]))
        result_df = pd.concat([result_df, method_df])
        method_dfs.append(df.iloc[indices,])
    result_df = result_df.apply(lambda x: round(x, 3))[['f', 'p', 's', 'g']]
    result_df['avg'] = result_df.mean(axis=1)
    return result_df, method_dfs


def draw_result_table(final_df):
    final_df = final_df.apply(lambda x: round(x, 3))
    new_df = final_df.reset_index().astype(str)
    enter = '\n'
    for metric in ['f', 'p', 's', 'g']:
        for index in np.arange(0, len(new_df), 2):
            new_df.loc[index, metric] += f' ({new_df.loc[index + 1, metric]})'
    new_df = new_df.loc[np.arange(0, len(new_df), 2)].set_index('index').T
    result_df = pd.DataFrame(index=range(8), columns=new_df.columns)
    for idx, row in enumerate(new_df.iterrows()):
        if idx == 4:
            break
        for column in result_df.columns:
            result_df.loc[idx * 2, column] = row[1][column].split('(')[0]
            result_df.loc[idx*2+1, column] = row[1][column].split('(')[1].split(')')[0]
    return new_df, result_df


def read_evaluation_final_file():
    df = pd.read_csv('evaluation_final_2.csv', index_col=[0, 1],
                     header=[0, 1, 2])
    df = df.reset_index()
    df['index'] = df.index.astype(str) + '-' + df['level_0'].astype(str) + '-' + df['level_1'].astype(str)
    df = df.set_index(df['index'])
    df = df.drop([col for col in df.columns if ('level_0' in col) or ('level_1' in col) or ('index' in col)], axis=1)
    df = df.applymap(lambda x: abs(x) if type(x) != str else x)

    new_df = df.shift(-1)
    final_df = df.astype(str) + '\n(' + new_df.astype(str) + ')'
    final_df.loc[[idx for idx in final_df.index if 'avg' in idx]] = final_df.loc[
        [idx for idx in final_df.index if 'avg' in idx]].applymap(lambda x: x.split('\n')[0])
    final_df.loc[[idx for idx in final_df.index if 'rank' in idx]] = final_df.loc[
        [idx for idx in final_df.index if 'rank' in idx]].applymap(lambda x: x.split('\n')[0].split('.')[0])
    final_df = final_df.loc[[idx for idx in final_df.index if 'dim' not in idx]]
    final_df = final_df.loc[[idx for idx in final_df.index if 'nan' not in idx]]
    final_df = final_df.loc[[idx for idx in final_df.index if '-g' not in idx]]
    final_df.to_csv('result_df_2.csv')
    return final_df