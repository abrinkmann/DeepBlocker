# GiG
import json
import os
import time
from datetime import datetime

import numpy as np
import pandas as pd
from pathlib import Path

from deep_blocker import DeepBlocker
from tuple_embedding_models import AutoEncoderTupleEmbedding, CTTTupleEmbedding, HybridTupleEmbedding
from vector_pairing_models import ExactTopKVectorPairing
import blocking_utils


def do_blocking(folder_root, left_table_fname, right_table_fname, cols_to_block, tuple_embedding_model,
                vector_pairing_model, with_train, dataset_name):
    folder_root = Path(folder_root)
    left_df = pd.read_csv(folder_root / left_table_fname)
    right_df = pd.read_csv(folder_root / right_table_fname)

    if dataset_name in ['abt-buy', 'amazon-google', 'dblp-acm_1', 'dblp-acm_2',
                        'dblp-googlescholar_1', 'dblp-googlescholar_2',
                        'walmart-amazon_1', 'walmart-amazon_2']:
        left_df = left_df.rename(columns={"title": "name"})
        right_df = right_df.rename(columns={"title": "name"})
    elif dataset_name in ['itunes-amazon_1', 'itunes-amazon_2']:
        left_df = left_df.rename(columns={"Song_Name": "name"})
        right_df = right_df.rename(columns={"Song_Name": "name"})


    train_df = pd.read_csv(Path(folder_root) / "train.csv")
    valid_df = pd.read_csv(Path(folder_root) / "valid.csv")
    test_df = pd.read_csv(Path(folder_root) / "test.csv")
    golden_df = pd.concat([train_df, valid_df, test_df]).drop_duplicates()
    golden_df = golden_df[golden_df['label'] == 1]

    train_pairs_df = pd.concat([train_df, valid_df]).drop_duplicates()

    db = DeepBlocker(tuple_embedding_model, vector_pairing_model)
    if with_train:
        db.prepocess_tuple_embeddings(left_df, right_df, cols_to_block, train_pairs_df, dataset_name)
    else:
        db.prepocess_tuple_embeddings(left_df, right_df, cols_to_block, None, dataset_name)

    statistics_dict = {}
    for k in range(1, 21):
        candidate_set_df = db.block_datasets(k)
        statistics_dict[k] = blocking_utils.compute_blocking_statistics(candidate_set_df, golden_df, left_df, right_df)

    return statistics_dict

def save_aggregated_result(aggregated_result, file_name):
    path_to_results = 'result/{}'.format(aggregated_result['schema_org_class'])
    if not os.path.isdir(path_to_results):
        os.makedirs(path_to_results)

    path_to_results = '{}/aggregated_{}'.format(path_to_results, file_name)

    with open(path_to_results, 'a+', encoding='utf-8') as f:
        json.dump(aggregated_result, f)
        f.write('\n')


if __name__ == "__main__":

    datasets = ['abt-buy', 'amazon-google', 'dblp-acm_1', 'dblp-acm_2', 'dblp-googlescholar_1', 'dblp-googlescholar_2',
                'itunes-amazon_1', 'itunes-amazon_2', 'walmart-amazon_1', 'walmart-amazon_2']
    cols_to_block_per_ds = {'abt-buy': ["name", "description", "price"],
                            'amazon-google': ["name", "manufacturer","price"],
                            'dblp-acm_1': ["name", "authors", "venue", "year"],
                            'dblp-acm_2': ["name", "authors", "venue", "year"],
                            'dblp-googlescholar_1': ["name", "authors", "venue", "year"],
                            'dblp-googlescholar_2': ["name", "authors", "venue", "year"],
                            'itunes-amazon_1': ["name", "Artist_Name", "Album_Name", "Genre", "Price", "CopyRight",
                                                "Time", "Released"],
                            'itunes-amazon_2': ["name", "Artist_Name", "Album_Name", "Genre", "Price", "CopyRight",
                                                "Time", "Released"],
                            'walmart-amazon_1': ["name", "category", "brand", "modelno", "price"],
                            'walmart-amazon_2': ["name", "category", "brand", "modelno", "price"]
                            }

    file_name = 'results_{}.json'.format(datetime.now().strftime('%Y-%m-%d_%H-%M-%S'))

    for dataset in datasets:
        folder_root = "{}data/{}".format(os.environ['DATA_DIR'], dataset)
        left_table_fname, right_table_fname = "tableA.csv", "tableB.csv"
        cols_to_block = cols_to_block_per_ds[dataset]

        # print("using AutoEncoder embedding")
        # tuple_embedding_model = AutoEncoderTupleEmbedding()
        # topK_vector_pairing_model = ExactTopKVectorPairing(K=50) # K is irrelevant since we always need k 1-20
        # start_time = time.time()
        # statistics_dict = do_blocking(folder_root, left_table_fname, right_table_fname, cols_to_block, tuple_embedding_model, topK_vector_pairing_model, False, dataset)
        # execution_time = time.time() - start_time
        # for k in statistics_dict:
        #     statistics_dict[k]['k'] = k
        #     statistics_dict[k]['model_name'] = 'AutoEncoder'
        #     statistics_dict[k]['retrieval_strategy'] = 'DL-Block'
        #     statistics_dict[k]['schema_org_class'] = dataset
        #     statistics_dict[k]['execution_time'] = execution_time
        #     statistics_dict[k]['recall_'] = statistics_dict[k]['recall']
        #     del statistics_dict[k]['recall']
        #     save_aggregated_result(statistics_dict[k], file_name)
        #     print(statistics_dict[k])

        # print("using CTT embedding with training data")
        # tuple_embedding_model = CTTTupleEmbedding()
        # topK_vector_pairing_model = ExactTopKVectorPairing(K=50)  # K is irrelevant since we always need k 1-20
        # statistics_dict = do_blocking(folder_root, left_table_fname, right_table_fname, cols_to_block,
        #                               tuple_embedding_model, topK_vector_pairing_model, True)
        # for k in statistics_dict:
        #     statistics_dict[k]['k'] = k
        #     statistics_dict[k]['model_name'] = 'CTT-existing_train'
        #     statistics_dict[k]['retrieval_strategy'] = 'DL-Block'
        #     statistics_dict[k]['schema_org_class'] = dataset
        #     statistics_dict[k]['recall_'] = statistics_dict[k]['recall']
        #     del statistics_dict[k]['recall']
        #     save_aggregated_result(statistics_dict[k], file_name)
        #     print(statistics_dict[k])

        print("using CTT embedding with synthetic training data")
        tuple_embedding_model = CTTTupleEmbedding()
        topK_vector_pairing_model = ExactTopKVectorPairing(K=50)  # K is irrelevant since we always need k 1-20
        start_time = time.time()
        statistics_dict = do_blocking(folder_root, left_table_fname, right_table_fname, cols_to_block,
                                      tuple_embedding_model, topK_vector_pairing_model, False, dataset)
        execution_time = time.time() - start_time
        for k in statistics_dict:
            statistics_dict[k]['k'] = k
            statistics_dict[k]['model_name'] = 'CTT-synthetic_train'
            statistics_dict[k]['retrieval_strategy'] = 'DL-Block'
            statistics_dict[k]['schema_org_class'] = dataset
            statistics_dict[k]['recall_'] = statistics_dict[k]['recall']
            statistics_dict[k]['execution_time'] = execution_time
            del statistics_dict[k]['recall']
            save_aggregated_result(statistics_dict[k], file_name)
            print(statistics_dict[k])

    # print("using Hybrid embedding")
    # tuple_embedding_model = CTTTupleEmbedding()
    # topK_vector_pairing_model = ExactTopKVectorPairing(K=50)
    # statistics_dict = do_blocking(folder_root, left_table_fname, right_table_fname, cols_to_block, tuple_embedding_model, topK_vector_pairing_model)
    # for k in statistics_dict:
    #     statistics_dict[k]['k'] = k
    #     print(statistics_dict[k])
