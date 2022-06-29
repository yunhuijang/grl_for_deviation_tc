from pm4py.objects.log.importer.xes import importer as xes_importer
from torch import Tensor
from torch import LongTensor
from torch import FloatTensor
from torch_geometric.data import Data
from torch_geometric.data import Dataset
import os
import numpy as np
import torch
from torch.nn import Linear
import torch.nn.functional as F
from torch_geometric.nn import GCNConv, GINConv, GATConv
from torch_geometric.nn import global_mean_pool
from torch_geometric.loader import DataLoader
from grm.grm.util import create_pig
from grm.grm.preprocessing import preprocess
from pm4py.objects.log.util.prefix_matrix import get_activities_list
from pm4py.statistics.traces.generic.log.case_statistics import get_variant_statistics, index_log_caseid
from pm4py.statistics.variants.log.get import get_variants
from nltk import ngrams
from collections import Counter
import pandas as pd
from process_discovery import process_discovery, make_deviation_label
from act2vec import Trace2Vec
from act2vec.loadXES import get_doc_XES_tagged
from sklearn.manifold import MDS
from pm4py.algo.conformance.alignments.edit_distance.algorithm import apply as get_alignment
from tqdm import tqdm
from da4py.main.analytics.amstc import samplingVariantsForAmstc
import pickle


def case_dict_gram(log):
    '''
    transform event log into case_dict with n-gram
    input: log
    output: case_dict (key - caseid / value: n-grams)
    '''
    variants = get_variants(log)
    statistics = get_variant_statistics(log)
    caseid = index_log_caseid(log)
    case_dict = dict.fromkeys(caseid.keys())
    # make case_dict
    for variant, value in variants.items():
        for case in value:
            case_dict[case.attributes['concept:name']] = variant

    gram_set = set()
    # make k-gram for each variant
    for variant_dict in statistics:
        variant = variant_dict['variant']
        gram_zip = ngrams(variant.split(','), 2)
        variant_grams = Counter(gram_zip)
        variants[variant] = variant_grams
        gram_set.update(set(variant_grams.keys()))
    # generate case_dict (key: caseid / value: n-grams)
    for key, value in case_dict.items():
        case_dict[key] = variants[value]

    # make case_df_gram
    case_df_gram = pd.DataFrame(index=case_dict.keys(), columns=gram_set)
    for caseid, value in case_dict.items():
        for gram, freq in value.items():
            case_df_gram.loc[caseid, gram] = freq
    case_df_gram.fillna(0, inplace=True)

    # put array into case_dict_gram
    case_dict_gram = dict.fromkeys(case_dict.keys())
    for key in case_dict_gram.keys():
        case_dict_gram[key] = case_df_gram.loc[key].values

    return case_dict_gram


def transform_event_log(log):
    '''
    transform event log in path to case_dict
    input: path of event log
    output: case_dict (key - caseid / value: variants / attributes)
    '''

    # variants: key - variant / value: attributes/events of each case
    variants = get_variants(log)

    # variant_dict: key - variant / value: caseid set for each variants
    variant_dict = dict.fromkeys(variants.keys())
    for key, value in variants.items():
        caseid_set = set()
        for case in value:
            caseid = case.attributes['concept:name']
            caseid_set.add(caseid)
        variant_dict[key] = caseid_set

    # case_dict = key - caseid / value: variants
    case_dict = dict.fromkeys(set.union(*variant_dict.values()))
    for key, value in variants.items():
        for case in value:
            case_dict[case.attributes['concept:name']] = {'variant': key}

    # make bag-of-activities
    for key, value in case_dict.items():
        case_dict[key]['activity_freq_set'] = Counter(value['variant'].split(','))
        case_dict[key]['activity_set'] = set(value['variant'].split(','))

    # activity_set: 모든 activity set
    activity_set = set(get_activities_list(log))

    # case_df_freq: feature를 matrix 형태로 표현 (freq 고려)
    # case_df_set: feature를 matrix 형태로 표현 (freq 고려하지 않고 존재 여부만 고려)
    case_df_freq = pd.DataFrame(index=case_dict.keys(), columns=activity_set)
    case_df_set = pd.DataFrame(index=case_dict.keys(), columns=activity_set)

    # generate case_df set / case_df_freq
    for key, value in case_dict.items():
        for act in value['activity_set']:
            case_df_set.loc[key, act] = 1
        for act in value['activity_freq_set']:
            case_df_freq.loc[key, act] = value['activity_freq_set'][act]
    case_df_set.fillna(0, inplace=True)
    case_df_freq.fillna(0, inplace=True)

    # put array into case_dict
    for key in case_dict.keys():
        case_dict[key]['activity_freq_list'] = case_df_freq.loc[key].values
        case_dict[key]['activity_list'] = case_df_set.loc[key].values

    return case_dict


def get_caseid_list(log):
    caseid_list = []
    for trace in log:
        caseid_list.append(trace.attributes['concept:name'])
    return caseid_list


def trace2vec_event_log(data, vectorsize):
    # input: file name of event log / vectorsize of trace2vec
    # output: case_dict (keys: caseid / values: trace2vec)
    trace2vec = Trace2Vec.learn(data, vectorsize)
    log = xes_importer.apply(data)
    caseid_list = get_caseid_list(log)

    corpus = get_doc_XES_tagged(data)
    print('Data Loading finished, ', str(len(corpus)), ' traces found.')
    # get trace vector for each case
    model = trace2vec
    # vectors: vector list for each case
    vectors = []
    print("inferring vectors")
    for doc_id in range(len(corpus)):
        inferred_vector = model.infer_vector(corpus[doc_id].words)
        vectors.append(inferred_vector)

    # make case_dict_trace2vec (key: caseid / value: trace2vec vector)
    case_dict_trace2vec = dict(zip(caseid_list, vectors))

    return case_dict_trace2vec


def get_instance_graph(log):
    '''
    input: log
    output: graph embeddings (target: timestamp / graph: edges (source node, edge type, target node) 
            / node_features: node feature / caseid: caseid)
    '''

    mappings, numeric_log = preprocess(log, get_activities_list(log))
    instance_graph = create_pig(numeric_log, len(mappings['activities']))
    for caseid, graph in zip(mappings['cases'], instance_graph):
        graph['caseid'] = caseid
    return instance_graph


def get_edge_index(graph):
    edge_list = []
    for edge in graph['graph']:
        edge_list.append([int(edge[0]), int(edge[2])])
    edge_index = LongTensor(edge_list).t()
    return edge_index


def get_edge_type_num(instance_graph):
    edge_set = set()
    for graph in instance_graph:
        for edge in graph['graph']:
            edge_set.add(edge[1])
    return len(edge_set)


def get_edge_type(graph, edge_type_num):
    edge_attribute = np.zeros((len(graph['graph']), edge_type_num))
    for index, edge in enumerate(graph['graph']):
        edge_type = int(edge[1])
        edge_attribute[index, edge_type - 1] = 1
    return edge_attribute


def get_data_list(instance_graph, log, filename, sample_size, random_state, parameters, algorithm):
    data_list = []
    n = get_edge_type_num(instance_graph)
    net, im, fm = process_discovery(log, parameters, algorithm)
    try:
        with open(
                f'''./output/evaluation/{filename}/{sample_size}/dev_onehot_{algorithm}_{parameters}_{random_state}_result.txt''', 'rb') as f:
            deviation_onehot = pickle.load(f)
        with open(
                f'''./output/evaluation/{filename}/{sample_size}/dev_list_{algorithm}_{parameters}_{random_state}_result.txt''', 'rb') as f:
            deviation_list = pickle.load(f)
    except FileNotFoundError:
        deviation_list, deviation_onehot = make_deviation_label(log, net, im, fm)
    for graph in instance_graph:
        # get node feature
        node_feature = FloatTensor(graph['node_features'])
        # get edge index
        edge_index = get_edge_index(graph)
        # get edge attribute
        edge_attribute = FloatTensor(get_edge_type(graph, n))
        # get caseid
        caseid = graph['caseid']
        # get y (deviation list)
        y = FloatTensor(deviation_onehot[graph['caseid']])
        data = Data(x=node_feature, edge_index=edge_index, edge_attr=edge_attribute, caseid=caseid,
                    y=y)
        data_list.append(data)
    return data_list, deviation_list, deviation_onehot


class GraphDataset(Dataset):
    def __init__(self, root, data_list, instance_graph, log, transform=None):
        self.instance_graph = instance_graph
        self.log = log
        self.data_list = data_list
        super().__init__(root, transform)

    @property
    def processed_file_names(self):
        return [f'{self.processed_dir}/data_{idx}.pt' for idx in range(len(self.data_list))]

    def process(self):
        for idx, data in enumerate(self.data_list):
            torch.save(data, f'{self.processed_dir}/data_{idx}.pt')

    def len(self):
        return len(self.data_list)

    def get(self, idx):
        data = torch.load(os.path.join(self.processed_dir, f'data_{idx}.pt'))
        return data


def split_train_test(dataset):
    torch.manual_seed(2022)
    dataset = dataset.shuffle()
    # split train set and test set
    train_dataset = dataset[int(len(dataset) / 5):]
    test_dataset = dataset[:int(len(dataset) / 5)]

    print(f'Number of training graphs: {len(train_dataset)}')
    print(f'Number of test graphs: {len(test_dataset)}')

    return train_dataset, test_dataset


def train_test_loader(train_dataset, test_dataset):
    train_loader = DataLoader(train_dataset, batch_size=128, shuffle=True)
    test_loader = DataLoader(test_dataset, batch_size=128, shuffle=False)
    for step, data in enumerate(train_loader):
        print(f'Step {step + 1}:')
        print('=======')
        print(f'Number of graphs in the current batch: {data.num_graphs}')
        print(data)
        print()
    return train_loader, test_loader


class GAT(torch.nn.Module):
    def __init__(self, dataset, hidden_channels, dropout_rate):
        super(GAT, self).__init__()
        torch.manual_seed(12345)
        self.conv1 = GATConv(dataset[0].x.shape[0], hidden_channels)
        self.conv2 = GATConv(hidden_channels, hidden_channels)
        self.conv3 = GATConv(hidden_channels, hidden_channels)
        self.lin = Linear(hidden_channels, dataset[0].y.shape[0])
        self.dropout_rate = dropout_rate

    def forward(self, x, edge_index, batch):
        # 1. Obtain node embeddings
        x = self.conv1(x, edge_index)
        x = x.relu()
        x = self.conv2(x, edge_index)
        x = x.relu()
        x = self.conv3(x, edge_index)
        # 2. Readout layer
        x = global_mean_pool(x, batch)  # [batch_size, hidden_channels]

        # 3. Apply a final classifier
        x = F.dropout(x, p=self.dropout_rate, training=self.training)
        x_mid_layer = x.clone()
        x = self.lin(x)

        return x, x_mid_layer


class GIN(torch.nn.Module):
    def __init__(self, dataset, hidden_channels, dropout_rate):
        super(GIN, self).__init__()
        torch.manual_seed(12345)
        self.conv1 = GINConv(dataset[0].x.shape[0], hidden_channels)
        self.conv2 = GINConv(hidden_channels, hidden_channels)
        self.conv3 = GINConv(hidden_channels, hidden_channels)
        self.lin = Linear(hidden_channels, dataset[0].y.shape[0])
        self.dropout_rate = dropout_rate

    def forward(self, x, edge_index, batch):
        # 1. Obtain node embeddings
        x = self.conv1(x, edge_index)
        x = x.relu()
        x = self.conv2(x, edge_index)
        x = x.relu()
        x = self.conv3(x, edge_index)
        # 2. Readout layer
        x = global_mean_pool(x, batch)  # [batch_size, hidden_channels]

        # 3. Apply a final classifier
        x = F.dropout(x, p=self.dropout_rate, training=self.training)
        x_mid_layer = x.clone()
        x = self.lin(x)

        return x, x_mid_layer


class GCN(torch.nn.Module):
    def __init__(self, dataset, hidden_channels, dropout_rate):
        super(GCN, self).__init__()
        torch.manual_seed(12345)
        self.conv1 = GCNConv(dataset[0].x.shape[0], hidden_channels)
        self.conv2 = GCNConv(hidden_channels, hidden_channels)
        self.conv3 = GCNConv(hidden_channels, hidden_channels)
        self.lin = Linear(hidden_channels, dataset[0].y.shape[0])
        self.dropout_rate = dropout_rate

    def forward(self, x, edge_index, batch):
        # 1. Obtain node embeddings
        x = self.conv1(x, edge_index)
        x = x.relu()
        x = self.conv2(x, edge_index)
        x = x.relu()
        x = self.conv3(x, edge_index)
        # 2. Readout layer
        x = global_mean_pool(x, batch)  # [batch_size, hidden_channels]

        # 3. Apply a final classifier
        x = F.dropout(x, p=self.dropout_rate, training=self.training)
        x_mid_layer = x.clone()
        x = self.lin(x)

        return x, x_mid_layer


def run(train_loader, test_loader, dataset, gnn_param, model_param):
    if model_param == 'gin':
        model = GIN(dataset, hidden_channels=gnn_param['hidden_channels'],
                    dropout_rate=gnn_param['dropout_rate'])
    elif model_param == 'gat':
        model = GAT(dataset, hidden_channels=gnn_param['hidden_channels'],
                    dropout_rate=gnn_param['dropout_rate'])
    else:
        model = GCN(dataset, hidden_channels=gnn_param['hidden_channels'],
                    dropout_rate=gnn_param['dropout_rate'])

    optimizer = torch.optim.Adam(model.parameters(), lr=gnn_param['learning_rate'])
    criterion = torch.nn.CrossEntropyLoss()

    def train():
        model.train()
        for data in train_loader:  # Iterate in batches over the training dataset.
            data = data
            out, _ = model(data.x, data.edge_index, data.batch)  # Perform a single forward pass.
            # loss = criterion(out, data.y.reshape(out.shape))  # Compute the loss.
            loss = criterion(out, data.y.reshape(out.shape[0], -1))
            loss.backward()  # Derive gradients.
            optimizer.step()  # Update parameters based on gradients.
            optimizer.zero_grad()  # Clear gradients.

    def test(loader):
        model.eval()
        correct = 0
        for data in loader:  # Iterate in batches over the training/test dataset.
            data = data
            out, _ = model(data.x, data.edge_index, data.batch)
            pred = Tensor(np.eye(data[0].y.shape[0])[out.argmax(dim=1)])  # Use the class with highest probability.
            correct += ((pred == data.y.reshape(pred.shape[0], -1)).all(dim=1).sum())
            # correct += ((pred == data.y.reshape(out.shape)).all(dim=1).sum())  # Check against ground-truth labels.
        return correct / len(loader.dataset)  # Derive ratio of correct predictions.

    train_acc_result = []
    test_acc_result = []
    for epoch in range(1, 40):
        train()
        train_acc = test(train_loader)
        test_acc = test(test_loader)
        print(f'Epoch: {epoch:03d}, Train Acc: {train_acc:.4f}, Test Acc: {test_acc:.4f}')
        train_acc_result.append(train_acc)
        test_acc_result.append(test_acc)

    return model, train_acc_result, test_acc_result


def map_graph_embedding(case_dict, model, dataset, option='last'):
    graph_embedding_dict = dict.fromkeys(case_dict.keys())
    for data in dataset:
        out, x_mid_layer = model(data.x, data.edge_index, data.batch)
        if option == 'last':
            graph_embedding_dict[str(data.caseid)] = np.array([tensor.detach().numpy() for tensor in out][0])
        if option == 'mid':
            graph_embedding_dict[str(data.caseid)] = np.array([tensor.detach().numpy() for tensor in x_mid_layer][0])
    return graph_embedding_dict


def upp2sym_inplace(a):
    for i in range(len(a)):
        for j in range(i):
            a[i][j] = a[j][i]

    return a


def sequence_alignment_embedding(log):
    similarity_matrix = []
    for trace in tqdm(log):
        aligned_traces = get_alignment(log, [trace])
        similarity_array = [trace['cost'] for trace in aligned_traces]
        similarity_matrix.append(similarity_array)

    similarity_matrix_sym = upp2sym_inplace(similarity_matrix)

    mds = MDS(n_components=int(len(log) ** 0.5), dissimilarity='precomputed')
    log_transformed = mds.fit_transform(np.array(similarity_matrix_sym))
    sa_embedding = {trace.attributes['concept:name']: trace_transformed for trace, trace_transformed in
                    zip(log, log_transformed)}

    return sa_embedding


def alignment_subnet_trace_clustering(log, eval_parameter, algorithm):
    '''
    alignment subnet trace clsutering (ASMTC)
    returns filtered log list (no embedding)
    '''
    # process model
    net, im, fm = process_discovery(log, discovery_parameter=eval_parameter, algorithm=algorithm)

    # sampleSize : number of traces that are used in the sampling method
    sampleSize = 5

    # sizeOfRun : maximal length requested to compute alignment
    sizeOfRun = 8

    # maxNbC : maximal number of transitions per cluster to avoid to get a unique centroid
    maxNbC = 5

    # m : number of cluster that will be searching at each AMSTC of the sampling method. Understand that more than
    # m cluster can be returned.
    m = 6
    # maxCounter : as this is a sampling method, maxCounter is the number of fails of AMSTC before the sampling
    # method stops
    # silent_label : every transition that contains this string will not cost in alignment
    clustering = samplingVariantsForAmstc(net, im, fm, log, sampleSize, sizeOfRun, 8, maxNbC, m, maxCounter=1,
                                          silent_label="tau")
    filtered_log_list = []
    for cluster in clustering:
        filtered_log_list.append(cluster[1])
    return filtered_log_list
