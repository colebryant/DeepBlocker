import sys
import pandas as pd
import pprint

def output_percentages(left_table_name, right_table_name, left_join_col, right_join_col):

    left_df = pd.read_csv(f'data/Structured/nyc_cleaned/{left_table_name}.csv')
    left_df['ltable_id'] = left_df.index
    right_df = pd.read_csv(f'data/Structured/nyc_cleaned/{right_table_name}.csv')
    right_df['rtable_id'] = right_df.index
    merged_df = left_df.merge(right_df, left_on=left_join_col, right_on=right_join_col, how='inner')

    left_percent_join = 100 * round(merged_df['ltable_id'].unique().shape[0] / len(left_df), 3)
    right_percent_join = 100 * round(merged_df['rtable_id'].unique().shape[0] / len(right_df), 3)
    total_percent_join = 100 * round((merged_df['ltable_id'].unique().shape[0] + merged_df['rtable_id'].unique().shape[0]) / (len(left_df) + len(right_df)), 3)

    ground_truth_dict = {
        "left_percent_join": f"{left_percent_join}%",
        "right_percent_join": f"{right_percent_join}%",
        "total_percent_join": f"{total_percent_join}%",
        }

    pprint.pprint(ground_truth_dict)    

if __name__ == "__main__":

    # Ex. kvhd-5fmu 
    left_table_name = sys.argv[1]
    # Ex. 2j8u-wtju
    right_table_name = sys.argv[2]
    # Ex. unique_id_number
    left_join_col = sys.argv[3]
    # Ex. unique_id_number
    right_join_col = sys.argv[4]

    output_percentages(left_table_name, right_table_name, left_join_col, right_join_col)