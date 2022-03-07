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
import baseline_fasttext
import pickle


def do_blocking(folder_root, left_table_fname, right_table_fname, tuple_embedding_model, vector_pairing_model):
    folder_root = Path(folder_root)
    left_df = pd.read_csv(folder_root / left_table_fname)
    right_df = pd.read_csv(folder_root / right_table_fname)

    db = DeepBlocker(tuple_embedding_model, vector_pairing_model)
    candidate_set_df = db.block_datasets(left_df, right_df)

    statistics_dict = blocking_utils.compute_join_percentage(candidate_set_df, left_df, right_df)
    return statistics_dict

def do_blocking_cols(folder_root, left_table_fname, right_table_fname, tuple_embedding_model, vector_pairing_model):
    folder_root = Path(folder_root)
    left_df = pd.read_csv(folder_root / left_table_fname)
    right_df = pd.read_csv(folder_root / right_table_fname)

    db = DeepBlocker(tuple_embedding_model, vector_pairing_model)
    candidate_set_df = db.block_cols(left_df, right_df)

    golden_df = get_golden_set(left_table_fname,right_table_fname)

    statistics_dict = blocking_utils.compute_column_statistics((left_table_fname.split('.')[0],right_table_fname.split('.')[0]), candidate_set_df, golden_df, left_df, right_df)

    return statistics_dict


def baseline_one(folder_root, left_table_fname, right_table_fname, tuple_embedding_model):
    """ Method which uses existing code to produce tuple embeddings, and then runs a cosine similarity on all possible pairs. Thresholds the similarity to determine valid pairs
    """
    # topK_vector_pairing_model = ExactTopKVectorPairing(K=10)
    vector_pairing_model = ThresholdVectorPairing(threshold=0.75)
    # Removed cols_to_block
    statistics_dict = do_blocking(folder_root, left_table_fname, right_table_fname, tuple_embedding_model, vector_pairing_model)
    pprint.pprint(statistics_dict, sort_dicts=False)

def baseline_two(args):
    """ Method which does not use tuple embedding, but instead does an average of word embeddings on each column
    and runs cosine similarity between column pairs to determine join / no join.
    """
    baseline_fasttext.main(args)

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


def column_embedding_method(folder_root, left_table_fname, right_table_fname, tuple_embedding_model):
    """ Method which generates tuple embeddings for each column using self-supervision   
    """

    vector_pairing_model = ThresholdVectorPairing(threshold=0.95)
    statistics_dict = do_blocking_cols(folder_root, left_table_fname, right_table_fname, tuple_embedding_model, vector_pairing_model)
    pprint.pprint(statistics_dict, sort_dicts=False)
    return statistics_dict


def column_embedding_method_loop():
    output_file = 'nyc_output/aurum-output.txt'
    with open(output_file) as f:
        lines = f.readlines()
    line_df = pd.DataFrame(lines,columns=['full'])
    line_df = line_df['full'].str.split("JOIN", n = 1, expand = True)
    line_df = line_df.replace('\n',' ', regex=True)
    line_df.columns = ['ltable_id','rtable_id']

    dataset_pairs_dict = {}

    left_datasets = list(line_df['ltable_id'].apply(lambda x: x.split('.')[0]).unique())

    for dataset in left_datasets:
        filtered_df = line_df[line_df['ltable_id'].apply(lambda x: x.split('.')[0]) == dataset]
        dataset_pairs_dict[dataset] = list(filtered_df['rtable_id'].apply(lambda x: x.split('.')[0]).unique())


    agg_stats = {
        "num_experiments": 0,
        "golden_set_length": 0,
        "candidate_set_length": 0,
        "merged_set_length": 0,
        "false_positives_length": 0
    }
    folder_root = "data/Structured/nyc_cleaned"
    # Run model on each of the dataset pairs and aggregate results
    for left_dataset in dataset_pairs_dict.keys():
        for right_dataset in dataset_pairs_dict[left_dataset]:

            # Skip if either dataframe is too small
            if len(pd.read_csv(folder_root + "/" + left_dataset.strip() + ".csv")) < 10 or len(pd.read_csv(folder_root + "/" + right_dataset.strip() + ".csv")) < 10:
                continue

            print("")
            print(f"Run model on {left_dataset.strip()} and {right_dataset.strip()}")
            # tuple_embedding_model = CTTTupleEmbedding(synth_tuples_per_tuple=100)
            tuple_embedding_model = AutoEncoderTupleEmbedding()
            statistics_dict = column_embedding_method(folder_root, left_dataset.strip() + ".csv", right_dataset.strip() + ".csv", tuple_embedding_model)

            agg_stats["num_experiments"] += 1
            agg_stats["golden_set_length"] += statistics_dict["golden_set_length"]
            agg_stats["candidate_set_length"] += statistics_dict["candidate_set_length"]
            agg_stats["merged_set_length"] += statistics_dict["merged_set_length"]
            agg_stats["false_positives_length"] += statistics_dict["false_positives_length"]

            with open("indv_output.pickle", "wb") as f:
                pickle.dump(statistics_dict, f)
            # pprint.pprint(statistics_dict, sort_dicts=False)

    # Compute aggregated statistics
    agg_stats["recall"] = float(agg_stats["merged_set_length"]) / agg_stats["golden_set_length"]
    agg_stats["precision"] = float(agg_stats["merged_set_length"]) / (agg_stats["merged_set_length"] + agg_stats["false_positives_length"])

    with open("agg_output.pickle", "wb") as f:
        pickle.dump(agg_stats, f)
    pprint.pprint(agg_stats, sort_dicts=False)


def get_golden_set(left_table_fname, right_table_fname,):
    # output_file = 'nyc_output/'+ left_table_fname.split('.')[0] + '-output.txt'
    output_file = 'nyc_output/aurum-output.txt'
    with open(output_file) as f:
        lines = f.readlines()
    line_df = pd.DataFrame(lines,columns=['full'])
    line_df = line_df['full'].str.split("JOIN", n = 1, expand = True)
    line_df = line_df.replace('\n',' ', regex=True)
    line_df.columns = ['ltable_id','rtable_id']
    golden_df = line_df[line_df['ltable_id'].str.contains(left_table_fname.split('.')[0])]
    golden_df = golden_df[line_df['rtable_id'].str.contains(right_table_fname.split('.')[0])]
    golden_df.ltable_id = golden_df.ltable_id.str.strip()
    golden_df.rtable_id = golden_df.rtable_id.str.strip()

    golden_df['ltable_id'] = golden_df['ltable_id'].astype('str')

    return golden_df


if __name__ == "__main__":
    # folder_root = "data/Structured/Amazon-Google"
    # left_table_fname, right_table_fname = "tableA.csv", "tableB.csv"
    # cols_to_block = ["title", "manufacturer", "price"]

    usage = """
    Usage: python main.py [left_table] [right_table] [method_choice] [embedding_choice]

    left_table = name of left table from nyc dataset
    right_table = name of right table from nyc dataset
    method_choice = choice of method to run dataset/column join prediction on:
        1 = baseline method 1
        2 = baseline method 2
        3 = baseline method 3
        4 = column embedding method
    tuple_embedding = choice of tuple embedding model:
        AE = Autoencoder Embedding
        CTT = CTT Tuple Embedding
        Hybrid = Hybrid Embedding
        * Note: currently only need to specify for baseline models 1 and 3 
     
    Example: python main.py myrx-addi 8vqd-3345 1 AE 
    """
    # print(usage)

    if len(sys.argv) < 4:
        column_embedding_method_loop()
    else:

        folder_root = "data/Structured/nyc_cleaned"
        left_table_fname = sys.argv[1] + ".csv"
        right_table_fname = sys.argv[2] + ".csv"
        method_choice = sys.argv[3]

        if method_choice in ["1", "3", "4"]:

            embedding_choice = sys.argv[4]

            if embedding_choice == "AE":
                print("using AutoEncoder embedding")
                tuple_embedding_model = AutoEncoderTupleEmbedding()
            elif embedding_choice == "CTT":
                print("using CTT embedding")
                tuple_embedding_model = CTTTupleEmbedding()
            elif embedded_choice == "Hybrid":
                print("using Hybrid embedding")
                tuple_embedding_model = HybridTupleEmbedding()
            else:
                print("Invalid tuple embedding choice")

            if method_choice == "1":
                baseline_one(folder_root, left_table_fname, right_table_fname, tuple_embedding_model)
            elif method_choice == "3":
                baseline_three(folder_root, left_table_fname, right_table_fname, tuple_embedding_model)
            else:
                column_embedding_method(folder_root, left_table_fname, right_table_fname, tuple_embedding_model)

        else:
            baseline_two(sys.argv[1:])

   