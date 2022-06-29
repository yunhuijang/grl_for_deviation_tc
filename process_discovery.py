import pandas as pd
import numpy as np
from pm4py.objects.log.importer.xes import importer as xes_importer
from pm4py.algo.discovery.inductive import algorithm as inductive_miner
from pm4py.algo.discovery.alpha import algorithm as alpha_miner
from pm4py.algo.conformance.alignments.petri_net import algorithm as alignments
from pm4py.algo.discovery.inductive.algorithm import Variants
from pm4py.algo.discovery.heuristics import algorithm as heuristics_miner


def import_log(path):
    log = xes_importer.apply(path)
    return log

def process_discovery(log, discovery_parameter, algorithm):
    if algorithm=='alpha':
        net, initial_marking, final_marking = alpha_miner.apply(log)
    elif algorithm=='in_f':
        net, initial_marking, final_marking = inductive_miner.apply(log, variant=Variants.IMf,
                                                                   parameters=discovery_parameter)
    elif algorithm=='heu':
        net, initial_marking, final_marking = heuristics_miner.apply(log)
    elif algorithm=='in_d':
        net, initial_marking, final_marking = inductive_miner.apply(log, variant=Variants.IMd)
    else:
        net, initial_marking, final_marking = inductive_miner.apply(log)
    return net, initial_marking, final_marking

def make_deviation_label(log, net, initial_marking, final_marking):
    
    # alignment-based conformance checking
    aligned_traces = alignments.apply_log(log, net, initial_marking, final_marking)

    # make caseid set
    caseid_list = []
    for trace in log:
        caseid = trace.attributes['concept:name']
        caseid_list.append(caseid)

    # make fitness sheet for each case
    case_fitness = dict.fromkeys(caseid_list)
    for caseid, aligned_trace in zip(caseid_list, aligned_traces):
        case_fitness[caseid] = aligned_trace['fitness']

    # make deviation sheet for each case
    case_deviation = dict.fromkeys(caseid_list)
    for caseid, aligned_trace in zip(caseid_list, aligned_traces):
        deviation_set = set()
        for move in aligned_trace['alignment']:
            # log move + model move
            if (move[0] != move[1]) and ((move[0]!='>>' and move[1]!=None) or (move[1]!= '>>')):
            # log move
            # if (move[0] != move[1]) and (move[0] != '>>' and move[1] != None):
                deviation_set.add(move)
        case_deviation[caseid] = deviation_set

    # make all deviation set
    all_deviation_set = set.union(*case_deviation.values())

    # make case_deviation_df (row: caseid / columns: deviation)

    case_deviation_df = pd.DataFrame(index=caseid_list, columns =all_deviation_set)
    for key, values in case_deviation.items():
        for deviation in all_deviation_set:
            if deviation in values:
                case_deviation_df.loc[key, deviation] = 1
    case_deviation_df.fillna(0, inplace=True)

    # make case_deviation_dict (key: caseid / value: matrix of deviation)
    case_deviation_dict = dict.fromkeys(caseid_list)
    for key in case_deviation_dict.keys():
        case_deviation_dict[key] = case_deviation_df.loc[key].values
        
    # map one hot rows to each deviation
    deviation_set = {tuple(value) for value in case_deviation_dict.values()}
    deviation_dict = dict.fromkeys(deviation_set)
    n = len(deviation_set)
    for idx, deviation in enumerate(deviation_set):
        deviation_dict[deviation] = np.eye(n)[idx]
    new_dict = {key: deviation_dict[tuple(value)] for key, value in case_deviation_dict.items()}
    
    return case_deviation_dict, new_dict