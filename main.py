#GiG
import sys
import numpy as np
import pandas as pd
from pathlib import Path
import pprint

from deep_blocker import DeepBlocker
from tuple_embedding_models import  AutoEncoderTupleEmbedding, CTTTupleEmbedding, HybridTupleEmbedding
from vector_pairing_models import ExactTopKVectorPairing, ThresholdVectorPairing
import blocking_utils

# Removed cols_to_block
def do_blocking(folder_root, left_table_fname, right_table_fname, tuple_embedding_model, vector_pairing_model):
    folder_root = Path(folder_root)
    left_df = pd.read_csv(folder_root / left_table_fname)
    right_df = pd.read_csv(folder_root / right_table_fname)
    # left_df = pd.read_csv(folder_root / left_table_fname, index_col='unique_id_number')
    # right_df = pd.read_csv(folder_root / right_table_fname, index_col='unique_id_number')

    db = DeepBlocker(tuple_embedding_model, vector_pairing_model)
    candidate_set_df = db.block_datasets(left_df, right_df)

    # golden_df = pd.read_csv(Path(folder_root) /  "matches.csv")
    # golden_df = pd.read_csv(f"data/Structured/nyc_matches/matches_{left_table_fname.split('.')[0]}_{right_table_fname.split('.')[0]}.csv")

    # statistics_dict = blocking_utils.compute_blocking_statistics(candidate_set_df, golden_df, left_df, right_df)
    statistics_dict = blocking_utils.compute_join_percentage(candidate_set_df, left_df, right_df)
    return statistics_dict


if __name__ == "__main__":
    # folder_root = "data/Structured/Amazon-Google"
    # left_table_fname, right_table_fname = "tableA.csv", "tableB.csv"
    # cols_to_block = ["title", "manufacturer", "price"]

    folder_root = "data/Structured/nyc_cleaned"
    left_table_fname = sys.argv[1] + ".csv"
    right_table_fname = sys.argv[2] + ".csv"

    print("using AutoEncoder embedding")
    tuple_embedding_model = AutoEncoderTupleEmbedding()
    # topK_vector_pairing_model = ExactTopKVectorPairing(K=10)
    topK_vector_pairing_model = ThresholdVectorPairing(threshold=0.8)
    # Removed cols_to_block
    statistics_dict = do_blocking(folder_root, left_table_fname, right_table_fname, tuple_embedding_model, topK_vector_pairing_model)
    pprint.pprint(statistics_dict, sort_dicts=False)

    # print("using CTT embedding")
    # tuple_embedding_model = CTTTupleEmbedding()
    # topK_vector_pairing_model = ExactTopKVectorPairing(K=50)
    # statistics_dict = do_blocking(folder_root, left_table_fname, right_table_fname, cols_to_block, tuple_embedding_model, topK_vector_pairing_model)
    # print(statistics_dict)
    
    # print("using Hybrid embedding")
    # tuple_embedding_model = HybridTupleEmbedding()
    # topK_vector_pairing_model = ExactTopKVectorPairing(K=50)
    # statistics_dict = do_blocking(folder_root, left_table_fname, right_table_fname, cols_to_block, tuple_embedding_model, topK_vector_pairing_model)
    # print(statistics_dict)
