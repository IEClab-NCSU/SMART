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
    datashop_models_path = os.path.join(str(Path.home()), 'OneDrive/SMART/DataShop Export/OLI Biology/ALMAP spring 2014 DS 960 (Problem View fixed and Custom Field fixed)/ds1934_kcm_2021_0726_123337.txt')
    smart_model_path = os.path.join(str(Path.home()), 'OneDrive/SMART/SMART output/intro_bio/SMART_models/Model1_clst100w_nm_nmfC10old-PV-models2_assessment_skill_first_tfidf_167_assessment.txt')  # SMART champion

    # read the KC Models into a pandas dataframe
    datashop_models = pd.read_csv(datashop_models_path, sep='\t')
    smart_model = pd.read_csv(smart_model_path, sep='\t')

    # store the column names for the columns to merge on (data common to all KC models for the dataset).
    smart_columns = list(smart_model.columns.values)
    merge_on = smart_columns[0:len(smart_columns) - 1]

    # merge the dataframes to obtain a single set with all KC models
    updated_df = pd.merge(datashop_models, smart_model, how='left', on=merge_on)
    updated_df = updated_df.replace(' ', np.nan)

    # obtain the column names for all KC models
    updated_columns = list(updated_df.columns.values)
    m1 = updated_columns[20]
    m2 = updated_columns[17]
    m3 = updated_columns[16]
    m4_1 = updated_columns[23]
    m4_2 = updated_columns[24]
    m4_3 = updated_columns[25]
    m4_4 = updated_columns[26]
    m4_5 = updated_columns[27]
    m4_6 = updated_columns[28]
    m4_7 = updated_columns[29]
    m4_8 = updated_columns[30]
    m4_9 = updated_columns[31]
    m4_10 = updated_columns[32]
    m4_11 = updated_columns[33]
    m4_12 = updated_columns[34]
    m4_13 = updated_columns[35]
    m4_14 = updated_columns[36]
    m4_15 = updated_columns[37]
    m4_16 = updated_columns[38]
    m4_17 = updated_columns[39]
    m4_18 = updated_columns[40]
    m4_19 = updated_columns[41]
    m4_20 = updated_columns[42]
    m4_21 = updated_columns[43]
    m4_22 = updated_columns[44]
    m4_23 = updated_columns[45]
    m4_24 = updated_columns[46]
    m4_25 = updated_columns[47]
    m4_26 = updated_columns[48]
    m4_27 = updated_columns[49]
    m4_28 = updated_columns[50]
    m4_29 = updated_columns[51]
    m4_30 = updated_columns[52]
    m5 = updated_columns[21]
    m6_1 = updated_columns[18]
    m6_2 = updated_columns[19]
    m7_1 = updated_columns[53]
    m7_2 = updated_columns[54]
    m7_3 = updated_columns[55]
    m7_4 = updated_columns[56]
    m7_5 = updated_columns[57]
    m7_6 = updated_columns[58]
    m8 = updated_columns[22]
    new_model = updated_columns[59]
    useless = updated_columns[60]
    smart = updated_columns[61]
    m4 = [m4_1, m4_2, m4_3, m4_4, m4_5, m4_6, m4_7, m4_8, m4_9, m4_10, m4_11, m4_12, m4_13, m4_14, m4_15, m4_16, m4_17, m4_18, m4_19, m4_20, m4_21, m4_22, m4_23, m4_24, m4_25, m4_26, m4_27, m4_28, m4_29, m4_30]
    m6 = [m6_1, m6_2]
    m7 = [m7_1, m7_2, m7_3, m7_4, m7_5, m7_6]

    # drop any rows with Nan values in the column with the fewest entries (KC model with fewest observations).
    updated_df = updated_df.dropna(subset=[m8])

    # retrieve the correct columns for each KC model and output each to file separately.
    drop_list_1 = [m2] + [m3] + m4 + [m5] + m6 + m7 + [m8] + [new_model] + [smart] + [useless]
    model_1_df = updated_df.drop(drop_list_1, axis=1)
    model_1_df = model_1_df.rename(columns={m1: 'KC (model_1)'})
    model_1_df.to_csv('model_1_import.txt', sep='\t', index=False)

    drop_list_2 = [m1] + [m3] + m4 + [m5] + m6 + m7 + [m8] + [new_model] + [smart] + [useless]
    model_2_df = updated_df.drop(drop_list_2, axis=1)
    model_2_df = model_2_df.rename(columns={m2: 'KC (model_2)'})
    model_2_df.to_csv('model_2_import.txt', sep='\t', index=False)

    drop_list_3 = [m1] + [m2] + m4 + [m5] + m6 + m7 + [m8] + [new_model] + [smart] + [useless]
    model_3_df = updated_df.drop(drop_list_3, axis=1)
    model_3_df = model_3_df.rename(columns={m3: 'KC (model_3)'})
    model_3_df.to_csv('model_3_import.txt', sep='\t', index=False)

    drop_list_4 = [m1] + [m2] + [m3] + [m5] + m6 + m7 + [m8] + [new_model] + [smart] + [useless]
    model_4_df = updated_df.drop(drop_list_4, axis=1)
    model_4_df = model_4_df.rename(columns={m4_1: 'KC (model_4)'})
    model_4_df.to_csv('model_4_import.txt', sep='\t', index=False)

    drop_list_5 = [m1] + [m2] + [m3] + m4 + m6 + m7 + [m8] + [new_model] + [smart] + [useless]
    model_5_df = updated_df.drop(drop_list_5, axis=1)
    model_5_df = model_5_df.rename(columns={m5: 'KC (model_5)'})
    model_5_df.to_csv('model_5_import.txt', sep='\t', index=False)

    drop_list_6 = [m1] + [m2] + [m3] + m4 + [m5] + m7 + [m8] + [new_model] + [smart] + [useless]
    model_6_df = updated_df.drop(drop_list_6, axis=1)
    model_6_df = model_6_df.rename(columns={m6_1: 'KC (model_6)'})
    model_6_df.to_csv('model_6_import.txt', sep='\t', index=False)

    drop_list_7 = [m1] + [m2] + [m3] + m4 + [m5] + m6 + [m8] + [new_model] + [smart] + [useless]
    model_7_df = updated_df.drop(drop_list_7, axis=1)
    model_7_df = model_7_df.rename(columns={m7_1: 'KC (model_7)'})
    model_7_df.to_csv('model_7_import.txt', sep='\t', index=False)

    drop_list_8 = [m1] + [m2] + [m3] + m4 + [m5] + m6 + m7 + [new_model] + [smart] + [useless]
    model_8_df = updated_df.drop(drop_list_8, axis=1)
    model_8_df = model_8_df.rename(columns={m8: 'KC (model_8)'})
    model_8_df.to_csv('model_8_import.txt', sep='\t', index=False)

    drop_list_s = [m1] + [m2] + [m3] + m4 + [m5] + m6 + m7 + [m8] + [new_model] + [useless]
    model_s_df = updated_df.drop(drop_list_s, axis=1)
    model_s_df = model_s_df.rename(columns={smart: 'KC (SMART_1)'})
    model_s_df.to_csv('smart_import.txt', sep='\t', index=False)
