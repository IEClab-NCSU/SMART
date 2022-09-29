"""
Create a Datashop KC Model Import file for a dataset based on a Datashop KC model from a different dataset.
"""

import os
from pathlib import Path
import pandas as pd
import numpy as np


if __name__ == '__main__':
    # Define path to two files: (1) the KC model export of all models in DataShop for the dataset that you would like to
    # compare SMART with and (2) the KC model produced by SMART.
    datashop_model_path = os.path.join(str(Path.home()), 'OneDrive/SMART/DataShop Import/gen_chem/gen_chem1-2.3_import.txt')  # gen-chem1-2.3
    smart_model_path = os.path.join(str(Path.home()), 'OneDrive/SMART/SMART output/gen_chem/SMART_models/ds4650_kcm69136_2021_0721_1643101_assessment_skill_first_tfidf_112_assessment.txt') # SMART champion

    # read the KC Models into a pandas dataframe
    datashop_model = pd.read_csv(datashop_model_path, sep='\t')
    smart_model = pd.read_csv(smart_model_path, sep='\t')

    # store the column names for the columns to merge on (data common to all KC models for the dataset).
    smart_columns = list(smart_model.columns.values)
    merge_on = smart_columns[0:len(smart_columns)-1]

    # merge the dataframes to obtain a single set with all KC models (merge on the model with fewest observations).
    updated_df = pd.merge(datashop_model, smart_model, how='left', on=merge_on)

    # view the count of entries in each column
    copy_df = updated_df.copy()
    print(copy_df.count(axis=0))

    # output both KC models to file (multiple models can be imported to DataShop in one file).
    updated_df.to_csv('gen_chem_final_import.txt', sep='\t', index=False)
