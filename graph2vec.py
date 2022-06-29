from pm4py.algo.discovery.dfg import algorithm as dfg_discovery
import networkx as nx
from karateclub.node_embedding.neighbourhood import Node2Vec
from pm4py.statistics.traces.generic.log.case_statistics import get_variant_statistics, index_log_caseid
from pm4py.statistics.variants.log.get import get_variants
import numpy as np


def get_model_embedding_and_variants(log, n):
    graph = nx.Graph()

    dfg = dfg_discovery.apply(log)
    for edge in dfg:
        graph.add_weighted_edges_from([(edge[0], edge[1], dfg[edge])])
    mapping = dict(zip(graph.nodes(), [i for i in range(len(graph.nodes()))]))
    node_label_dict = {value: key for key, value in mapping.items()}
    graph = nx.relabel_nodes(graph, mapping)

    # get model embedding (each node's embedding)
    model = Node2Vec(dimensions=n)
    model.fit(graph)
    model_embedding = {node_label_dict[i]: model.get_embedding()[i] for i in range(len(graph.nodes()))}
    variants = get_variants(log)
    return variants, model_embedding


def get_case_dict(log, variants, embedding_dict):
    caseid = index_log_caseid(log)
    case_dict = dict.fromkeys(caseid.keys())
    # make case_dict
    for variant, value in variants.items():
        for case in value:
            case_dict[case.attributes['concept:name']] = embedding_dict[variant]
    return case_dict


def node2vec_embedding(log, variants, model_embedding):

    variant_embedding_dict = dict.fromkeys(variants.keys())
    for variant in variant_embedding_dict.keys():
        variant_embedding = [model_embedding[activity] for activity in variant.split(',')]
        mean_embedding = np.array(variant_embedding).mean(axis=0)
        max_embedding = np.array(variant_embedding).max(axis=0)
        variant_embedding_dict[variant] = {'mean': mean_embedding, 'max': max_embedding}
    case_dict = get_case_dict(log, variants, variant_embedding_dict)
    case_node2vec_mean = {key: value['mean'] for key, value in case_dict.items()}
    case_node2vec_max = {key: value['max'] for key, value in case_dict.items()}
    return case_node2vec_mean, case_node2vec_max


def edge2vec_embedding(log, variants, model_embedding, n):
    variant_embedding_dict = dict.fromkeys(variants.keys())
    for variant in variant_embedding_dict.keys():
        splitted_variants = variant.split(',')
        variant_embedding = [(model_embedding[act_1], model_embedding[act_2]) for act_1, act_2
                             in zip(splitted_variants[1:], splitted_variants[:-1])]
        if len(variant_embedding) == 0:
            zero_vector = np.zeros(n)
            variant_embedding_dict[variant] = {'mean_avg': zero_vector, 'max_avg': zero_vector,
                                           'mean_mul': zero_vector, 'max_mul': zero_vector}
            continue
        variant_embedding_avg = [(act_1 + act_2)/2 for act_1, act_2 in variant_embedding]
        variant_embedding_mul = [np.multiply(act_1, act_2) for act_1, act_2 in variant_embedding]
        mean_embedding_avg = np.array(variant_embedding_avg).mean(axis=0)
        mean_embedding_mul = np.array(variant_embedding_mul).mean(axis=0)
        max_embedding_avg = np.array(variant_embedding_avg).max(axis=0)
        max_embedding_mul = np.array(variant_embedding_mul).max(axis=0)
        variant_embedding_dict[variant] = {'mean_avg': mean_embedding_avg, 'max_avg': max_embedding_avg,
                                           'mean_mul': mean_embedding_mul, 'max_mul': max_embedding_mul}
    case_dict = get_case_dict(log, variants, variant_embedding_dict)
    case_edge2vec_mean_avg = {key: value['mean_avg'] for key, value in case_dict.items()}
    case_edge2vec_max_avg = {key: value['max_avg'] for key, value in case_dict.items()}
    case_edge2vec_mean_mul = {key: value['mean_mul'] for key, value in case_dict.items()}
    case_edge2vec_max_mul = {key: value['max_mul'] for key, value in case_dict.items()}

    return case_edge2vec_mean_avg, case_edge2vec_max_avg, case_edge2vec_mean_mul, case_edge2vec_max_mul
