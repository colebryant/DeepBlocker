#GiG
import numpy as np
import pandas as pd
from pathlib import Path
import blocking_utils
from configurations import *

class DeepBlocker:
    def __init__(self, tuple_embedding_model, vector_pairing_model):
        self.tuple_embedding_model = tuple_embedding_model
        self.vector_pairing_model = vector_pairing_model

    # def validate_columns(self):
    #     #Assumption: id column is named as id
    #     if "id" not in self.cols_to_block:
    #         self.cols_to_block.append("id")
    #     self.cols_to_block_without_id = [col for col in self.cols_to_block if col != "id"]
    
    #     #Check if all required columns are in left_df
    #     check = all([col in self.left_df.columns for col in self.cols_to_block])
    #     if not check:
    #         raise Exception("Not all columns in cols_to_block are present in the left dataset")
    
    #     #Check if all required columns are in right_df
    #     check = all([col in self.right_df.columns for col in self.cols_to_block])
    #     if not check:
    #         raise Exception("Not all columns in cols_to_block are present in the right dataset")


    def preprocess_datasets(self):
        # self.left_df = self.left_df[self.cols_to_block]
        # self.right_df = self.right_df[self.cols_to_block]

        self.left_df.fillna(' ', inplace=True)
        self.right_df.fillna(' ', inplace=True)

        self.left_df = self.left_df.astype(str)
        self.right_df = self.right_df.astype(str)

        # self.left_df["_merged_text"] = self.left_df[[col for col in self.left_df.columns if col != "unique_id_number"]].agg(' '.join, axis=1)
        # self.right_df["_merged_text"] = self.right_df[[col for col in self.right_df.columns if col != "unique_id_number"]].agg(' '.join, axis=1)
        self.left_df["_merged_text"] = self.left_df.agg(' '.join, axis=1)
        self.right_df["_merged_text"] = self.right_df.agg(' '.join, axis=1)

        #Drop the other columns
        no_drop_cols = ["_merged_text"]
        self.left_df = self.left_df.drop(columns=[col for col in self.left_df.columns if col not in no_drop_cols])
        self.right_df = self.right_df.drop(columns=[col for col in self.right_df.columns if col not in no_drop_cols])

    
    def preprocess_columns(self):

        if IGNORE_NUMERIC_COLS:
            # DROP all columns in both tables which are numeric
            self.left_df = self.left_df.drop(columns=list(self.left_df.select_dtypes(include='number').columns))
            self.right_df = self.right_df.drop(columns=list(self.right_df.select_dtypes(include='number').columns))
        self.left_df = self.left_df.astype('str')
        self.right_df = self.right_df.astype('str')
        self.left_df.fillna(' ', inplace=True)
        self.right_df.fillna(' ', inplace=True)
        
        # n=1 # length of dataset
        # k = TUPLE_SAMPLE_SIZE #length of tuple

        # set sample size to min of dataframe lengths
        k = min(len(self.left_df), len(self.right_df))

        left_list = []
        for cols in self.left_df.columns:
            # left_df_tuple = [self.left_df.sample(k)[cols].T for x in range(0,n)]
            left_df_tuple = [self.left_df.sample(k)[cols].T]
            left_tuple_list = [','.join(i) for i in left_df_tuple]
            left_list.extend(left_tuple_list)
        right_list = [] 
        for cols in self.right_df.columns:
            # right_df_tuple = [self.right_df.sample(k)[cols].T for x in range(0,n)]
            right_df_tuple = [self.right_df.sample(k)[cols].T]
            right_df_tuple_list = [','.join(i) for i in right_df_tuple]
            right_list.extend(right_df_tuple_list)

        self.left_df = pd.DataFrame(left_list,columns=['_merged_text'])
        self.left_df['id'] = np.arange(len(self.left_df))
        self.right_df = pd.DataFrame(right_list,columns=['_merged_text'])
        self.right_df['id'] = np.arange(len(self.right_df))
        

    # Removed cols_to_block
    def block_datasets(self, left_df, right_df):
        self.left_df = left_df
        self.right_df = right_df
        # self.cols_to_block = cols_to_block

        # No need to block specific columns
        # self.validate_columns()
        self.preprocess_datasets()

        print("Performing pre-processing for tuple embeddings ")
        all_merged_text = pd.concat([self.left_df["_merged_text"], self.right_df["_merged_text"]], ignore_index=True)
        self.tuple_embedding_model.preprocess(all_merged_text)

        print("Obtaining tuple embeddings for left table")
        self.left_tuple_embeddings = self.tuple_embedding_model.get_tuple_embedding(self.left_df["_merged_text"])
        print("Obtaining tuple embeddings for right table")
        self.right_tuple_embeddings = self.tuple_embedding_model.get_tuple_embedding(self.right_df["_merged_text"])


        print("Indexing the embeddings from the right dataset")
        self.vector_pairing_model.index(self.right_tuple_embeddings)

        print("Querying the embeddings from left dataset")
        # topK_neighbors = self.vector_pairing_model.query(self.left_tuple_embeddings)
        thresholded_pairs = self.vector_pairing_model.query(self.left_tuple_embeddings)

        # self.candidate_set_df = blocking_utils.topK_neighbors_to_candidate_set(topK_neighbors)
        self.candidate_set_df = blocking_utils.thresholded_pairs_to_candidate_set(thresholded_pairs)

        return self.candidate_set_df


    def block_cols(self, left_df, right_df):
        self.left_df = left_df
        self.right_df = right_df

        self.preprocess_columns()

        print("Performing pre-processing for tuple embeddings ")
        all_merged_text = pd.concat([self.left_df["_merged_text"], self.right_df["_merged_text"]], ignore_index=True)
        self.tuple_embedding_model.preprocess(all_merged_text)

        print("Obtaining tuple embeddings for left table")
        self.left_tuple_embeddings = self.tuple_embedding_model.get_tuple_embedding(self.left_df["_merged_text"])
        print("Obtaining tuple embeddings for right table")
        self.right_tuple_embeddings = self.tuple_embedding_model.get_tuple_embedding(self.right_df["_merged_text"])


        print("Indexing the embeddings from the right dataset")
        self.vector_pairing_model.index(self.right_tuple_embeddings)

        print("Querying the embeddings from left dataset")
        thresholded_pairs = self.vector_pairing_model.query(self.left_tuple_embeddings)

        self.candidate_set_df = blocking_utils.thresholded_pairs_to_candidate_set(thresholded_pairs)

        return self.candidate_set_df


    def get_table_embeddings(self, left_df, right_df):
        self.left_df = left_df
        self.right_df = right_df
        self.preprocess_datasets()

        print("Performing pre-processing for tuple embeddings ")
        all_merged_text = pd.concat([self.left_df["_merged_text"], self.right_df["_merged_text"]], ignore_index=True)
        self.tuple_embedding_model.preprocess(all_merged_text)

        print("Obtaining tuple embeddings for left table")
        self.left_tuple_embeddings = self.tuple_embedding_model.get_tuple_embedding(self.left_df["_merged_text"])
        print("Obtaining tuple embeddings for right table")
        self.right_tuple_embeddings = self.tuple_embedding_model.get_tuple_embedding(self.right_df["_merged_text"])
