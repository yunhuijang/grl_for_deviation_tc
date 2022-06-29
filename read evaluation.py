import pandas as pd
from evaluation import evaluate, get_evaluation_df, draw_result_table, get_random_state_average
import pickle
from evaluation import evaluate_clustering, read_event_log, get_average_metric, get_weighted_sum
from pm4py.algo.discovery.inductive.variants.im_f.algorithm import Parameters
from itertools import product


# read result
filenames = ['BPI_Challenge_2019']
filenames = ['BPI_Challenge_2012', 'BPI_Challenge_2017', 'BPI_Challenge_2019']
algo_params = [('in_f', {Parameters.NOISE_THRESHOLD: 0.9})]
methods = ['dev_gram_set', 'gnn']
conformance = ['alignment']
random_states = [1,2,3,4]
sample_size = 1000
n = len(random_states)
result_total_df = pd.DataFrame()
final_total_df = pd.DataFrame()
for filename in filenames:
    file_data_list = []
    final_df = pd.DataFrame()
    for algo_param in algo_params:
        file_weighted_list = []
        avg_list = []
        weighted_list = []
        algo = algo_param[0]
        parameter = algo_param[1]
        for method in methods:
            print(method)
            for random_state in random_states:
                with open(
                        f'output/evaluation/{filename}/{sample_size}/{algo}_{parameter}_{method}_alignment_{random_state}_result.txt',
                        'rb') as f:
                    data = pickle.load(f)
                    avg = get_average_metric(data)
                    avg_list.append([*avg])
                with open(
                        f'output/evaluation/{filename}/{sample_size}/{algo}_{parameter}_{method}_alignment_{random_state}_log.txt',
                        'rb') as f_log:
                    log_data = pickle.load(f_log)
                    log_len = [len(sub_log) for sub_log in log_data]
                    print(f'# of clusters: {len(log_len)}')
                    weighted_fitness = get_weighted_sum('fitness', data, log_len)
                    weighted_precision = get_weighted_sum('precision', data, log_len)
                    weighted_simplicity = get_weighted_sum('simplicity', data, log_len)
                    weighted_generalization = get_weighted_sum('generalization', data, log_len)
                    weighted_list = [weighted_fitness, weighted_precision, weighted_simplicity, weighted_generalization]
                    file_weighted_list.append(weighted_list)
        file_data_list.append(avg_list)
        weighted_df = get_evaluation_df(file_weighted_list)
        result_df, method_dfs = get_random_state_average(weighted_df, methods, len(random_states))
        df, df_new = draw_result_table(result_df)
        result_total_df = pd.concat([result_total_df, df_new], axis=0)
        total_method_df = pd.DataFrame()
        for method, method_df in zip(methods, method_dfs):
            new_method_df = method_df.T
            final_method_df = pd.DataFrame(new_method_df.to_numpy().flatten(),
                                           index=list(product(*[['f', 'p', 's', 'g'], random_states])),
                                           columns=[method])
            total_method_df = pd.concat([total_method_df, final_method_df], axis=1)
        final_df = pd.concat([final_df, total_method_df], axis=0)
    final_total_df = pd.concat([final_total_df, final_df], axis=1)
final_total_df = final_total_df.applymap(lambda x: round(x, 3))
# final_total_df.to_csv('output/result/total_df.csv')
df_new.to_csv('output/result/result_df.csv')
final_total_df.to_csv('output/result/total_df.csv')
styler = df.style.highlight_max(axis='index')
print(df_new.to_string())