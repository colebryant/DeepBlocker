import pandas as pd
import numpy as np

def topK_neighbors_to_candidate_set(topK_neighbors):
    #We create a data frame corresponding to topK neighbors.
    # We are given a 2D matrix of the form 1: [a1, a2, a3], 2: [b1, b2, b3]
    # where a1, a2, a3 are the top-3 neighbors for tuple 1 and so on.
    # We will now create a two column DF fo the form (1, a1), (1, a2), (1, a3), (2, b1), (2, b2), (2, b3)
    topK_df = pd.DataFrame(topK_neighbors)
    topK_df["ltable_id"] = topK_df.index
    melted_df = pd.melt(topK_df, id_vars=["ltable_id"])
    melted_df["rtable_id"] = melted_df["value"]
    candidate_set_df = melted_df[["ltable_id", "rtable_id"]]
    return candidate_set_df

def thresholded_pairs_to_candidate_set(thresholded_pairs):
    # Merge record pair arrays to create DataFrame of candidate pairs
    merged_arr = np.vstack((thresholded_pairs[0], thresholded_pairs[1])).T
    candidate_set_df = pd.DataFrame(merged_arr, columns=["ltable_id", "rtable_id"])
    return candidate_set_df

#This accepts four inputs:
# data frames for candidate set and ground truth matches
# left and right data frames
def compute_blocking_statistics(candidate_set_df, golden_df, left_df, right_df):
    #Now we have two data frames with two columns ltable_id and rtable_id
    # If we do an equi-join of these two data frames, we will get the matches that were in the top-K

    merged_df = pd.merge(candidate_set_df, golden_df, on=['ltable_id', 'rtable_id'])
    # Added to calculate total false positives
    false_pos = candidate_set_df[~candidate_set_df['ltable_id'].isin(merged_df['ltable_id'])|(~candidate_set_df['rtable_id'].isin(merged_df['rtable_id']))]

    left_num_tuples = len(left_df)
    right_num_tuples = len(right_df)
    statistics_dict = {
        "left_num_tuples": left_num_tuples,
        "right_num_tuples": right_num_tuples,
        "candidate_set_length": len(candidate_set_df),
        "golden_set_length": len(golden_df),
        "merged_set_length": len(merged_df),
        "false_positives_length": len(false_pos),
        "precision": len(merged_df) / (len(merged_df) + len(false_pos)) if len(golden_df) > 0 else "N/A",
        "recall": len(merged_df) / len(golden_df) if len(golden_df) > 0 else "N/A",
        "cssr": len(candidate_set_df) / (left_num_tuples * right_num_tuples)
        }

    return statistics_dict

def compute_join_percentage(candidate_set_df, left_df, right_df):

    THRESHOLD = 20

    left_num_tuples = len(left_df)
    right_num_tuples = len(right_df)

    left_percent_join = 100 * round(candidate_set_df['ltable_id'].unique().shape[0] / left_num_tuples, 3)
    right_percent_join = 100 * round(candidate_set_df['rtable_id'].unique().shape[0] / right_num_tuples, 3)
    total_percent_join = 100 * round((candidate_set_df['ltable_id'].unique().shape[0] + candidate_set_df['rtable_id'].unique().shape[0]) / (left_num_tuples + right_num_tuples), 3)

    statistics_dict = {
        "left_num_tuples": left_num_tuples,
        "right_num_tuples": right_num_tuples,
        "candidate_set_length": len(candidate_set_df),
        "left_percent_join": f"{left_percent_join}%",
        "right_percent_join": f"{right_percent_join}%",
        "right_percent_join": f"{right_percent_join}%",
        "total_percent_join": f"{total_percent_join}%",
        "prediction": "JOIN" if max(left_percent_join, right_percent_join) > THRESHOLD else "NO JOIN", 
        "cssr": len(candidate_set_df) / (left_num_tuples * right_num_tuples)
        }

    return statistics_dict


def compute_column_statistics(table_names,candidate_set_df, golden_df,left_df, right_df):

    candidate_set_df = candidate_set_df.astype('str')

    candidate_set_df['ltable_id_table'] = candidate_set_df['ltable_id'].apply(lambda x: left_df.columns[int(x)])
    candidate_set_df['ltable_id_table'] = table_names[0] + '.' + candidate_set_df['ltable_id_table']
    candidate_set_df['rtable_id_table'] = candidate_set_df['rtable_id'].apply(lambda x: right_df.columns[int(x)])
    candidate_set_df['rtable_id_table'] = table_names[1] + '.' + candidate_set_df['rtable_id_table']

    candidate_set_df = candidate_set_df[['ltable_id_table','rtable_id_table']].rename(columns={'ltable_id_table':'ltable_id','rtable_id_table':'rtable_id'})

    merged_df = pd.merge(candidate_set_df, golden_df, on=['ltable_id', 'rtable_id'])
    
    # Added to calculate total false positives
    false_pos = candidate_set_df[~candidate_set_df['ltable_id'].isin(merged_df['ltable_id'])|(~candidate_set_df['rtable_id'].isin(merged_df['rtable_id']))]
    if len(golden_df) > 0 and (len(merged_df) + len(false_pos)) > 0:
    	fp = float(len(merged_df)) / (len(merged_df) + len(false_pos))
    else:
    	fp = "N/A"

    left_num_columns = len(left_df.columns)
    right_num_columns = len(right_df.columns)
    statistics_dict = {
    	"left_table": table_names[0],
    	"right_table": table_names[1],
        "left_num_columns": left_num_columns,
        "right_num_columns": right_num_columns,
        "candidate_set_length": len(candidate_set_df),
        "candidate_set": candidate_set_df,
        "golden_set_length": len(golden_df),
        "golden_set": golden_df,
        "merged_set_length": len(merged_df),
        "merged_set": merged_df,
        "false_positives_length": len(false_pos),
        "false_positives": false_pos,
        "precision": fp,
        "recall": float(len(merged_df)) / len(golden_df) if len(golden_df) > 0 else "N/A",
        "cssr": len(candidate_set_df) / (left_num_columns * right_num_columns)
        }

    return statistics_dict


#This function is useful when you download the preprocessed data from DeepMatcher dataset
# and want to convert to matches format.
#It loads the train/valid/test files, filters the duplicates,
# and saves them to a new file called matches.csv
def process_files(folder_root):
    df1 = pd.read_csv(folder_root + "/train.csv")
    df2 = pd.read_csv(folder_root + "/valid.csv")
    df3 = pd.read_csv(folder_root + "/test.csv")

    df1 = df1[df1["label"] == 1]
    df2 = df2[df2["label"] == 1]
    df3 = df3[df3["label"] == 1]

    df = pd.concat([df1, df2, df3], ignore_index=True)

    df[["ltable_id","rtable_id"]].to_csv(folder_root + "/matches.csv", header=True, index=False)
