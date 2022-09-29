import os
import shutil
import pandas as pd
from sklearn.metrics.cluster import adjusted_mutual_info_score, normalized_mutual_info_score
import csv


def main():

    if os.path.isdir('final_nmi_comparison'):
        shutil.rmtree('final_nmi_comparison')
        os.mkdir('final_nmi_comparison')
    else:
        os.mkdir('final_nmi_comparison')

    # retrieve and store the human-generated model for the OLI Biology course
    human_bio_df = pd.read_csv('./models/model_8_import.txt', sep='\t')
    print(list(human_bio_df.columns))
    human_bio_labels = human_bio_df['KC (model_8)'].values.tolist()

    # human_bio_df = pd.melt(human_bio_df, )

    # retrieve and store the champion SMART skill model for the OLI Biology course
    smart_bio_df = pd.read_csv('./models/intro_bio_smart_nm_325_named_import.txt', sep='\t')
    smart_bio_labels = smart_bio_df['KC(SMART_1_nm_named_325)'].values.tolist()

    # retrieve and store the human-generated model for the OLI Chemistry course
    human_chem_df = pd.read_csv('./models/gen_chem_final_import.txt', sep='\t')
    human_chem_labels = human_chem_df['KC(gen-chem1-2_3)'].values.tolist()

    # retrieve and store the champion SMART skill model for the OLI Chemistry course
    smart_chem_df = pd.read_csv('./models/gen_chem_smart_nm_215_named_import.txt', sep='\t')
    smart_chem_labels = smart_chem_df['KC(SMART_1_nm_named_215)'].values.tolist()

    # calculate the NMI metric between champion SMART and human-model for OLI Biology
    smart_human_bio_nmi = normalized_mutual_info_score(smart_bio_labels, human_bio_labels)

    # calculate the NMI metric between champion SMART and human-model for OLI Chemistry
    smart_human_chem_nmi = normalized_mutual_info_score(smart_chem_labels, human_chem_labels)

    heading = ['Course', 'Clustering A', 'Clustering B', 'NMI_Score']
    with open('./final_nmi_comparison/final_nmi_values.csv', 'w') as file:
        writer = csv.writer(file, delimiter=',')
        writer.writerow(heading)
    file.close()

    smart_human_bio = ['oli_biology', 'SMART', 'Human', smart_human_bio_nmi]
    smart_human_chem = ['oli_chemistry', 'SMART', 'Human', smart_human_chem_nmi]
    
    with open('./final_nmi_comparison/final_nmi_values.csv', 'a') as file:
        writer = csv.writer(file, delimiter=',')
        writer.writerow(smart_human_bio)
        writer.writerow(smart_human_chem)
    file.close()

if __name__ == '__main__':
    main()
