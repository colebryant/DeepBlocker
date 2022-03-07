"""
WIP method to predict column embeddings using remainder of row as context
"""
import pandas as pd
import fasttext
from scipy.spatial import distance
from itertools import product
import sys
import numpy as np
from configurations import *
from tuple_embedding_models import AutoEncoderTupleEmbedding, AutoEncoderTupleEmbeddingAdjusted


def produce_column_embeddings(table1, table2):
    """
    Loop through each column in each table and generate dictionaries containing
    columns (keys) and embedded representations of columns (values)
    """
    table1_dict = {}
    for col in table1.columns:
        embedded_values = predict_column_embedding(table1, col)
        embed_avg = np.average(np.array(embedded_values), axis=0)
        table1_dict[col] = embed_avg

    table2_dict = {}
    for col in table2.columns:
        embedded_values = predict_column_embedding(table2, col)
        embed_avg = np.average(np.array(embedded_values), axis=0)
        table2_dict[col] = embed_avg

    return table1_dict, table2_dict


def predict_column_embedding(table, column):
    """
    
    """
    print(f"Creating embedding for column: {column}")

    tuple_embedding_model = AutoEncoderTupleEmbedding()
    column_embedding_model = AutoEncoderTupleEmbeddingAdjusted()

    # Drop rows which have nan values in the column we are embedding
    table = table.dropna(subset=column)
    # Drop rows which have all nan values in remaining rows other than column
    table = table.dropna(how='all', subset=[col for col in table.columns if col != column])
    # Fill remainder of nan values with empty spaces
    table = table.fillna(' ')
    # Convert to all string values
    table = table.astype(str)
    # Merge words in all rows except column we are holding out
    table["_merged_text"] = table[[col for col in table.columns if col != column]].agg(' '.join, axis=1)
    # Extract held out column
    held_out_col = table[column]
    # Drop columns not in merged text
    table = table.drop(columns=[col for col in table.columns if col != "_merged_text"])
    # Preprocess tuple embedding model
    tuple_embedding_model.preprocess(table["_merged_text"])
    # Get tuple embeddings
    print("Getting tuple embeddings")
    tuple_embeddings = tuple_embedding_model.get_tuple_embedding(table["_merged_text"])

    # Now, predict column embeddings by using column embedding model
    column_embedding_model.preprocess(tuple_embeddings, held_out_col)
    print("Getting column embeddings")
    column_embeddings = column_embedding_model.get_tuple_embedding(tuple_embeddings)

    return column_embeddings
    

def cos_sim(col1_embed, col2_embed):

    cosine_similarity = 1 - distance.cdist(col1_embed.reshape(1,-1), col2_embed.reshape(1,-1), metric="cosine")
#     print(cosine_similarity)
    return cosine_similarity


def generate_column_similarity(table1,table2):

    # Delete below when done
    table1 = table1.iloc[:, :2]
    table2 = table2.iloc[:, :2]
    ###
    column_compare_combos = list(product(table1.columns, table2.columns))
    # Goal to retrieve column embeddings below
    table1_dict, table2_dict = produce_column_embeddings(table1, table2)

    cs_list = []
    for item in column_compare_combos:
        cs = cos_sim(table1_dict[item[0]], table2_dict[item[1]])
#         print(cs)
        cs_list.append(cs[0][0])
    return cs_list,column_compare_combos

def make_prediction_df(table_names, table_files):
    table1_name = table_names[0]
    table2_name = table_names[1]
    dfa = pd.read_csv(table_files[0])
    dfb = pd.read_csv(table_files[1])
    if len(dfb) > 5000:
        return None
    print(len(dfa))
    print(len(dfb))
    cs_list, column_compare_combos = generate_column_similarity(dfa,dfb)

    test = pd.DataFrame(column_compare_combos,cs_list,columns=['ltable_id','rtable_id']).reset_index()
    test = test.rename(columns={'index':'score'})
#     test['join'] = test.apply(lambda x: 1 if x.score > 0.7 else 0,axis=1)
    test['ltable_id'] = table1_name + '.' + test['ltable_id']
    test['rtable_id'] = table2_name + '.' + test['rtable_id']
    return test

def compute_blocking_statistics(table_names,candidate_set_df, golden_df,left_df, right_df):
    #Now we have two data frames with two columns ltable_id and rtable_id
    # If we do an equi-join of these two data frames, we will get the matches that were in the top-K

    merged_df = pd.merge(candidate_set_df, golden_df, on=['ltable_id', 'rtable_id'])
    
    # Added to calculate total false positives
    false_pos = candidate_set_df[~candidate_set_df['ltable_id'].isin(merged_df['ltable_id'])&(~candidate_set_df['rtable_id'].isin(merged_df['rtable_id']))]
    if len(golden_df) > 0 and (len(merged_df) + len(false_pos)) > 0:
        fp = float(len(merged_df)) / (len(merged_df) + len(false_pos))
    else:
        fp = "N/A"

    left_num_tuples = len(left_df)
    right_num_tuples = len(right_df)
    statistics_dict = {
        "left_table": table_names[0],
        "right_table": table_names[1],
        "left_num_tuples": left_num_tuples,
        "right_num_tuples": right_num_tuples,
        "candidate_set_length": len(candidate_set_df),
        "golden_set_length": len(golden_df),
        "merged_set_length": len(merged_df),
        "false_positives_length": len(false_pos),
        "precision": fp,
        "recall": float(len(merged_df)) / len(golden_df) if len(golden_df) > 0 else "N/A",
        "cssr": len(candidate_set_df) / (left_num_tuples * right_num_tuples)
        }

    return statistics_dict

def main():
    # usage: python baseline_fasttext.py kvhd-5fmu 2j8u-wtju
    args = sys.argv[1:]

    # table_names = ('kvhd-5fmu','2j8u-wtju')
    # table_files = ('nyc_cleaned/kvhd-5fmu.csv','nyc_cleaned/2j8u-wtju.csv')


    output_file = 'nyc_output/'+ args[0] + '-output.txt'
    with open(output_file) as f:
        lines = f.readlines()
    line_df = pd.DataFrame(lines,columns=['full'])
    line_df = line_df['full'].str.split("JOIN", n = 1, expand = True)
    line_df = line_df.replace('\n',' ', regex=True)
    line_df.columns = ['ltable_id','rtable_id']

    if len(args )== 2:
        joining_tables = [args[1]]
    else:
        joining_tables = line_df['rtable_id'].str.split('.').apply(lambda x: x[0].strip()).unique()

    table_file1 = 'data/Structured/nyc_cleaned/' + args[0] + '.csv'
    stats_list = []
    for table in joining_tables:
        print(table)
        table_file2 = 'data/Structured/nyc_cleaned/' + table + '.csv'
        table_names = (args[0],table)
        table_files = (table_file1,table_file2)

        print("Getting cosine_similarity")
        test = make_prediction_df(table_names, table_files)
        if test is None:
            continue
        print("Compute stats")
        print(test)
        candidate_set_df = test[test['score']> 0.95] #change to top 10? 
        golden_df = line_df[line_df['ltable_id'].str.contains(table_names[0])]
        golden_df = line_df[line_df['rtable_id'].str.contains(table_names[1])]
        golden_df.ltable_id = golden_df.ltable_id.str.strip()
        golden_df.rtable_id = golden_df.rtable_id.str.strip()
        candidate_set_df['ltable_id'] = candidate_set_df['ltable_id'].astype('str')
        golden_df['ltable_id'] = golden_df['ltable_id'].astype('str')
        stats = compute_blocking_statistics(table_names,candidate_set_df, golden_df,test['ltable_id'].unique(), test['rtable_id'].unique())
        print(stats)
        stats_list.append(stats)

    print(stats_list)
main()