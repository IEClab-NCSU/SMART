"""
Create a Datashop KC Model Import file for a dataset based on a Datashop skill to problem mapping.
"""

import os
from pathlib import Path
import pandas as pd
import numpy as np


if __name__ == '__main__':
    # Define path to two files: (1) the mapping of skills to problems from DataShop for a KC model (2) an exported KC model to be compared with the KC model.
    assessment_skill_file = os.path.join(str(Path.home()), 'OneDrive/SMART/DataShop Export/OLI General Chemistry I/chem1_skill_map_2.3_5_20_20-Problems.csv') # gen_chem1-2.3
    DataShopExport = os.path.join(str(Path.home()), 'OneDrive/SMART/DataShop Export/OLI General Chemistry I/ds4650_kcm69136_2021_0721_164310.txt') # Chemistry base

    # read the mapping and KC model into pandas dataframes
    skills = pd.read_csv(assessment_skill_file, sep='\t')
    new_model = pd.read_csv(DataShopExport, sep='\t')

    # in the mapping dataframe, rename Resource column to Problem Name to match convention in the KC model.
    skills = skills.rename(columns={'Resource':'Problem Name'})
    # delete the Step column since it is not used
    del skills['Step']

    # Add a column for the KC model to capture the Problem name in a manner that matches the skill to problem mapping.
    new_model['Problem'] = new_model['Step Name'].str.split('_').str[0]

    # merge the dataframes to label the steps with skills from the skill to problem mapping
    updated_df = pd.merge(new_model, skills, how='left', on=['Problem Name', 'Problem'])

    # obtain the names of the columns that are no longer needed (columns for old KC model and the problem column).
    updated_columns = list(updated_df.columns.values)
    kc1 = updated_columns[16]
    kc2 = updated_columns[17]
    kc3 = updated_columns[18]
    prob = updated_columns[19]

    # remove unnecessary columns
    updated_col_list = [kc1, kc2, kc3, prob]
    updated_df = updated_df.drop(updated_col_list, axis=1)

    # output the count of valid entries for each column
    copy_df = updated_df.copy()
    copy_df = copy_df.dropna(subset=['Skill1'])
    print(copy_df.count(axis=0))

    # remove rows with a missing value for the skill in the KC model
    updated_df = updated_df.dropna(subset=['Skill1'])

    # output the KC model to file for import into DataShop
    updated_df.to_csv('gen_chem1-2.3_import.txt', sep='\t', index=False)
