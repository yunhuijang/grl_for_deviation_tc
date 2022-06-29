from sklearn.cluster import KMeans
from sklearn.cluster import AgglomerativeClustering
from sklearn.cluster import DBSCAN
import numpy as np
from pm4py.algo.filtering.log.attributes import attributes_filter


def trace_clustering(case_dict, clusterType='kmeans', num_clusters=6, random_state=1):
    '''
    do clsutering on trace feature vectors
    input: case_dict (key: caseid / value: feature)
    output: cluster_result_dict (key: caseid / value: cluster label)
    '''
    clustering_target = np.array([*case_dict.values()])
    if clusterType == 'kmeans':
        clustering = KMeans(num_clusters, random_state=1).fit(clustering_target)
    elif clusterType == 'hier':
        clustering = AgglomerativeClustering(num_clusters, random_state=random_state).fit(clustering_target)
    elif clusterType == 'dbscan':
        clustering = DBSCAN(min_samples=10).fit(clustering_target)
    else:
        print("Clustering Type Error")
    assigned_clusters = clustering.labels_
    cluster_result_dict = dict(zip(case_dict.keys(), clustering.labels_))

    return cluster_result_dict


def filter_log_with_cluster(log, cluster_result):
    '''
    filter logs based on cluster labels
    input: cluster_result dictionary (key: caseid / value: cluster label)
    output: list of logs splitted with cluster labels
    '''
    filtered_log_list = []
    for label in set(cluster_result.values()):
        cluster_caseid_set = {k for k, v in cluster_result.items() if v == label}
        filtered_log_cases = attributes_filter.apply_trace_attribute(log, cluster_caseid_set,
                                                                     parameters={attributes_filter.Parameters.ATTRIBUTE_KEY: 'concept:name'})
        filtered_log_list.append(filtered_log_cases)

    return filtered_log_list
