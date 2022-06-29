from collections import Counter
from nltk import ngrams
from pm4py.statistics.traces.generic.log.case_statistics import get_variant_statistics, index_log_caseid
from pm4py.statistics.variants.log.get import get_variants
import pandas as pd


def map_distance_graph(log):
    variants = get_variants(log)
    statistics = get_variant_statistics(log)
    caseid = index_log_caseid(log)
    case_dict = dict.fromkeys(caseid.keys())
    # make case_dict
    for variant, value in variants.items():
        for case in value:
            case_dict[case.attributes['concept:name']] = variant

    order_0_set = set()
    order_1_set = set()
    order_2_set = set()
    for variant_dict in statistics:
        variant = variant_dict['variant']
        gram_zip = ngrams(variant.split(','), 1)
        order_0 = Counter(gram_zip)
        order_0_set.update(set(order_0.keys()))
        gram_zip_2 = ngrams(variant.split(','), 2)
        order_1 = Counter(gram_zip_2)
        order_1_set.update(set(order_1.keys()))
        gram_zip_3 = ngrams(variant.split(','), 3)
        gram_3 = Counter(gram_zip_3)
        order_2 = {(key[0], key[2]): value for key, value in gram_3.items()}
        order_2_set.update(set(order_2.keys()))
        variant_embedding = order_0+order_1+Counter(order_2)
        variants[variant] = variant_embedding

    for key, value in case_dict.items():
        case_dict[key] = variants[value]

    df_column = [*order_0_set, *order_1_set, *order_2_set]
    df_column = list(set(map(str, df_column)))

    # make case_df_gram
    case_df_gram = pd.DataFrame(index=case_dict.keys(), columns=df_column)
    for caseid, value in case_dict.items():
        for gram, freq in value.items():
            case_df_gram.loc[caseid, str(gram)] = freq
    case_df_gram.fillna(0, inplace=True)
    case_df_gram = case_df_gram[[col for col in case_df_gram.columns if col.split(',')[0][2:-1] != col.split(',')[1][2:-2]]]

    # put array into case_dict_gram
    case_dict_gram = dict.fromkeys(case_dict.keys())
    for key in case_dict_gram.keys():
        case_dict_gram[key] = case_df_gram.loc[key].values

    return case_dict_gram
