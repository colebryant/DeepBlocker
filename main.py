#GiG
import sys
import numpy as np
import pandas as pd
from pathlib import Path
import pprint
from scipy.spatial import distance

from deep_blocker import DeepBlocker
from tuple_embedding_models import  AutoEncoderTupleEmbedding, CTTTupleEmbedding, HybridTupleEmbedding
from vector_pairing_models import ExactTopKVectorPairing, ThresholdVectorPairing
import blocking_utils

# Removed cols_to_block
def do_blocking(folder_root, left_table_fname, right_table_fname, tuple_embedding_model, vector_pairing_model):
    folder_root = Path(folder_root)
    left_df = pd.read_csv(folder_root / left_table_fname)
    right_df = pd.read_csv(folder_root / right_table_fname)

    db = DeepBlocker(tuple_embedding_model, vector_pairing_model)
    candidate_set_df = db.block_datasets(left_df, right_df)

    # golden_df = pd.read_csv(Path(folder_root) /  "matches.csv")
    # golden_df = pd.read_csv(f"data/Structured/nyc_matches/matches_{left_table_fname.split('.')[0]}_{right_table_fname.split('.')[0]}.csv")

    # statistics_dict = blocking_utils.compute_blocking_statistics(candidate_set_df, golden_df, left_df, right_df)
    statistics_dict = blocking_utils.compute_join_percentage(candidate_set_df, left_df, right_df)
    return statistics_dict


def baseline_one(folder_root, left_table_fname, right_table_fname, tuple_embedding_model):
    """ Method which uses existing code to produce tuple embeddings, and then runs a cosine similarity on all possible pairs. Thresholds the similarity to determine valid pairs
    """
    # topK_vector_pairing_model = ExactTopKVectorPairing(K=10)
    vector_pairing_model = ThresholdVectorPairing(threshold=0.75)
    # Removed cols_to_block
    statistics_dict = do_blocking(folder_root, left_table_fname, right_table_fname, tuple_embedding_model, vector_pairing_model)
    pprint.pprint(statistics_dict, sort_dicts=False)


def baseline_three(folder_root, left_table_fname, right_table_fname, tuple_embedding_model):
    """ Method which uses existing code to produce tuple embeddings, and then creates an average embedding for each table. Finally, runs cosine similarity on the two vectors and thresholds to determine join prediction
    """

    folder_root = Path(folder_root)
    left_df = pd.read_csv(folder_root / left_table_fname)
    right_df = pd.read_csv(folder_root / right_table_fname)

    # Get tuple embeddings
    db = DeepBlocker(tuple_embedding_model, None)
    db.get_table_embeddings(left_df, right_df)

    # Compute average vector embedding for each table
    avg_left = np.average(db.left_tuple_embeddings, axis=0) 
    avg_right = np.average(db.right_tuple_embeddings, axis=0) 
    # print(avg_left)
    # print(avg_right)

    cosine_similarity = 1 - distance.cdist(avg_left.reshape(1,-1), avg_right.reshape(1,-1), metric="cosine")
    print(cosine_similarity)



if __name__ == "__main__":
    # folder_root = "data/Structured/Amazon-Google"
    # left_table_fname, right_table_fname = "tableA.csv", "tableB.csv"
    # cols_to_block = ["title", "manufacturer", "price"]

    folder_root = "data/Structured/nyc_cleaned"
    left_table_fname = sys.argv[1] + ".csv"
    right_table_fname = sys.argv[2] + ".csv"
    method_choice = sys.argv[3]

    print("using AutoEncoder embedding")
    tuple_embedding_model = AutoEncoderTupleEmbedding()
    # print("using CTT embedding")
    # tuple_embedding_model = CTTTupleEmbedding()
    # print("using Hybrid embedding")
    # tuple_embedding_model = HybridTupleEmbedding()

    if method_choice == "1":
        baseline_one(folder_root, left_table_fname, right_table_fname, tuple_embedding_model)
    else:
        baseline_three(folder_root, left_table_fname, right_table_fname, tuple_embedding_model)

   