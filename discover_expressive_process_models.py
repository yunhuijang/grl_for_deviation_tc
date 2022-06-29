from process_discovery import process_discovery
from pm4py.algo.conformance.tokenreplay import algorithm as token_replay
from trace_clustering import trace_clustering, filter_log_with_cluster
from collections import Counter
from pm4py.statistics.traces.generic.log.case_statistics import get_variant_statistics
from nltk import ngrams


def mine_workflow(log, algorithm, discovery_parameter):
    net, im, fm = process_discovery(log, algorithm, discovery_parameter)
    return net, im, fm


def maximum_process_discovery(log, algorithm, discovery_parameter, m=6):
    net, im, fm = mine_workflow(log, algorithm, discovery_parameter)
    workflow_set = [[net, im, fm, log]]
    no_refinement = False
    sigma = 0.3
    gamma = 0.3
    l = 5
    max_f = 3
    while (len(workflow_set) < m) and not no_refinement:
        # ws: [net, im, fm, log]
        ws = select_schema(workflow_set)
        old_size = len(workflow_set)
        workflow_set.extend(refine_workflow(ws, sigma, gamma, l, max_f, algorithm, discovery_parameter))
        workflow_set.remove(ws)
        if len(workflow_set) == old_size:
            no_refinement = True
            workflow_set.append(ws)
    return workflow_set


def calculate_soundness(workflow_list):
    net, im, fm, log = workflow_list
    replayed_traces = token_replay.apply(log, net, im, fm)
    result_list = [trace['trace_is_fit'] for trace in replayed_traces]
    return Counter(result_list)[True]/len(result_list)


def select_schema(ws_set):
    if len(ws_set) == 1:
        return ws_set[0]
    soundness_list = [calculate_soundness(ws) for ws in ws_set]
    min_value = min(soundness_list)
    min_ws = ws_set[soundness_list.index(min_value)]
    return min_ws


def get_total_gram_counter(log, gram_size=2):
    statistics = get_variant_statistics(log)
    total_grams = Counter()
    for variant_dict in statistics:
        variant = variant_dict['variant']
        count = variant_dict['count']
        gram_zip = ngrams(variant.split(','), gram_size)
        variant_grams = Counter(gram_zip)
        for key in variant_grams.keys():
            variant_grams[key] = variant_grams[key]*count
        total_grams += variant_grams
    return total_grams


def get_frequent_grams(total_grams, gamma, log_size):
    frequent_grams = {key: value for key, value in total_grams.items() if value > gamma*log_size}
    return frequent_grams


def find_features(ws, sigma, gamma, l, max_f):
    # ws: [net, im, fm, log]
    log = ws[3]
    log_size = len(log)
    total_grams = get_total_gram_counter(log, gram_size=2)
    frequent_grams_2 = get_frequent_grams(total_grams, sigma, log_size)
    length = 3
    feature_set = []
    while(length <= l):
        cand_len = set()
        feature_set_len = []
        total_grams = get_total_gram_counter(log, gram_size=length-1)
        total_grams_len = get_total_gram_counter(log, gram_size=length)
        frequent_grams_len_1 = get_frequent_grams(total_grams, sigma, log_size)
        frequent_grams_len_gamma = get_frequent_grams(total_grams_len, gamma, log_size)
        for gram in frequent_grams_len_1.keys():
            for two_gram in frequent_grams_2.keys():
                if gram[-1] == two_gram[0]:
                    cand_len.add(gram+(two_gram[-1],))
        gamma_log = {key: value for key, value in frequent_grams_len_gamma.items() if key in cand_len}
        candidates = cand_len - set(gamma_log.keys())
        for cand in candidates:
            a = cand[-1]
            seq = cand[:-1]
            flag = True
            for feature in feature_set:
                if a == feature[1] and feature[0] <= seq:
                    flag = False
                    break
                for cand_2 in candidates:
                    if (cand[-1], cand_2[-1]) in frequent_grams_2.keys():
                        flag = False
                        break
            if flag:
                feature_set_len.append([cand[:-1], cand[-1]])
            else:
                continue
        feature_set.extend(feature_set_len)
        length += 1
    return feature_set


def project(log, features):
    caseids = [trace.attributes['concept:name'] for trace in log]
    case_dict = dict.fromkeys(caseids)
    # feature: tuple (sequence, activity)
    for trace in log:
        caseid = trace.attributes['concept:name']
        activity_set = set([event['concept:name'] for event in trace])
        mapped_list = []
        for feature in features:
            n = len(feature[0])
            if feature[1] in activity_set:
                value = 0
            elif (feature[1] not in activity_set) and (len(activity_set.intersection(set(feature[0]))) == n):
                value = 1
            else:
                sequence_list = [1 if activity in activity_set else 0 for activity in feature[0]]
                a = sum([b*(n**power) for b, power in zip(sequence_list, range(n))])
                b = sum([n**power for power in range(n)])
                value = a/b
            mapped_list.append(value)
        case_dict[caseid] = mapped_list
    return case_dict


def refine_workflow(ws, sigma, gamma, l, max_f, algorithm, discovery_parameter):
    features = find_features(ws, sigma, gamma, l, max_f)
    log = ws[3]
    if len(features) >= 1:
        mapped_log = project(log, features)
        cluster_result = trace_clustering(mapped_log, clusterType='kmeans', num_clusters=3)
        filtered_log_list = filter_log_with_cluster(log, cluster_result)
        new_ws_list = []
        for log in filtered_log_list:
            try:
                net, im, fm = process_discovery(log, algorithm=algorithm, discovery_parameter=discovery_parameter)
                new_ws_list.append([net, im, fm, log])
            except:
                continue
    else:
        new_ws_list = [ws]
        print("No change")
    return new_ws_list


def discover_expressive_process_models(log, algorithm, discovery_parameter):
    ws_set = maximum_process_discovery(log, algorithm, discovery_parameter)
    filtered_log_list = [ws[3] for ws in ws_set]
    return filtered_log_list