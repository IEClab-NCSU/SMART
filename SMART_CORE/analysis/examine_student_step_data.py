import numpy as np
import pandas as pd
from pathlib import Path
from mod_performance_v5 import stem, calc_oli_stats, calc_public_stats, remove_stopwords
import shutil
import os
import sys


def main():

    prepare_file_system()

    dataset_n_dict = {'inspec': 10, 'kdd': 5, 'oli-intro-bio': 1, 'oli-gen-chem': 1}

    oli_datasets = ['oli-intro-bio', 'oli-gen-chem']
    public_datasets = ['inspec', 'kdd']
    datasets = oli_datasets + public_datasets

    for dataset in datasets:
        best_f1_indices = []
        worst_f1_indices = []

        dataset_n = dataset_n_dict[dataset]

        keybert_results_path = f'OneDrive/SMART/Jesse_2022/skill_labeling/AKE_study_2_27_2022/output/ake_results/{dataset}_ngrams_sbert-mean-pooling_cos-doc-cand_top-n-mss_{dataset_n}_ae_off_trial_1_output.csv'
        keybert_results_path = os.path.join(str(Path.home()), keybert_results_path)
        # embedding_based_results_path = f'OneDrive/SMART/Jesse_2022/skill_labeling/Comparison of Results/{dataset}/{dataset}_parsing_sbert-mean-pooling_cos-doc-cand_top-n_{dataset_n}_ae_off_trial_1_output.csv'
        # embedding_based_results_path = os.path.join(str(Path.home()), embedding_based_results_path)
        rake_results_path = f'OneDrive/SMART/Jesse_2022/skill_labeling/AKE_study_2_27_2022/output/ake_results/{dataset}_RAKE_{dataset_n}_trial_1_output.csv'
        rake_results_path = os.path.join(str(Path.home()), rake_results_path)
        yake_results_path = f'OneDrive/SMART/Jesse_2022/skill_labeling/AKE_study_2_27_2022/output/ake_results/{dataset}_YAKE_{dataset_n}_trial_1_output.csv'
        yake_results_path = os.path.join(str(Path.home()), yake_results_path)
        textrank_results_path = f'OneDrive/SMART/Jesse_2022/skill_labeling/AKE_study_2_27_2022/output/ake_results/{dataset}_TextRank_{dataset_n}_trial_1_output.csv'
        textrank_results_path = os.path.join(str(Path.home()), textrank_results_path)
        singlerank_results_path = f'OneDrive/SMART/Jesse_2022/skill_labeling/AKE_study_2_27_2022/output/ake_results/{dataset}_SingleRank_{dataset_n}_trial_1_output.csv'
        singlerank_results_path = os.path.join(str(Path.home()), singlerank_results_path)
        topicrank_results_path = f'OneDrive/SMART/Jesse_2022/skill_labeling/AKE_study_2_27_2022/output/ake_results/{dataset}_TopicRank_{dataset_n}_trial_1_output.csv'
        topicrank_results_path = os.path.join(str(Path.home()), topicrank_results_path)
        multipartiterank_results_path = f'OneDrive/SMART/Jesse_2022/skill_labeling/AKE_study_2_27_2022/output/ake_results/{dataset}_MultipartiteRank_{dataset_n}_trial_1_output.csv'
        multipartiterank_results_path = os.path.join(str(Path.home()), multipartiterank_results_path)
        # Add GPT-2 Results

        keybert_results_df = pd.read_csv(keybert_results_path, index_col=0)
        # embedding_based_results_df = pd.read_csv(embedding_based_results_path, index_col=0)
        rake_results_df = pd.read_csv(rake_results_path, index_col=0)
        yake_results_df = pd.read_csv(yake_results_path, index_col=0)
        textrank_results_df = pd.read_csv(textrank_results_path, index_col=0)
        singlerank_results_df = pd.read_csv(singlerank_results_path, index_col=0)
        topicrank_results_df = pd.read_csv(topicrank_results_path, index_col=0)
        multipartite_results_df = pd.read_csv(multipartiterank_results_path, index_col=0)

        dataframes = [keybert_results_df, rake_results_df, yake_results_df, textrank_results_df, singlerank_results_df, topicrank_results_df, multipartite_results_df]

        for df in dataframes:
            list_pred_kps = df.Keyphrases.values
            list_pred_kps = [text.split(',') for text in list_pred_kps]
            list_pred_kps = [[text.strip("' [].\u2026") for text in kps] for kps in list_pred_kps]
            print(list_pred_kps)
            list_gold_kps = df.Gold_Keyphrases.values
            list_gold_kps = [text.split(',') for text in list_gold_kps]
            list_gold_kps = [[text.strip("' [].\u2026") for text in kps] for kps in list_gold_kps]
            print(list_gold_kps)

            best_f1_index, worst_f1_index = find_best_worst_f1(list_pred_kps, list_gold_kps)

            best_f1_indices.append(best_f1_index)
            worst_f1_indices.append(worst_f1_index)

        keybert_results_df = keybert_results_df.rename(
            columns={'Keyphrases': 'KeyBERT_Keyphrases',
                     'Keyphrase_Scores': 'KeyBERT_Keyphrase_Scores'})
        # embedding_based_results_df = embedding_based_results_df.rename(columns={'Keyphrases': 'Embedding_Based_Keyphrases', 'Keyphrase_Scores': 'Embedding_Based_Keyphrase_Scores'})

        rake_results_df = rake_results_df.rename(columns={'Keyphrases': 'RAKE_Keyphrases', 'Keyphrase_Scores': 'RAKE_Keyphrase_Score'})
        yake_results_df = yake_results_df.rename(
            columns={'Keyphrases': 'YAKE_Keyphrases', 'Keyphrase_Scores': 'YAKE_Keyphrase_Score'})
        textrank_results_df = textrank_results_df.rename(
            columns={'Keyphrases': 'TextRank_Keyphrases', 'Keyphrase_Scores': 'TextRank_Keyphrase_Score'})
        singlerank_results_df = singlerank_results_df.rename(
            columns={'Keyphrases': 'SingleRank_Keyphrases', 'Keyphrase_Scores': 'SingleRank_Keyphrase_Score'})
        topicrank_results_df = topicrank_results_df.rename(
            columns={'Keyphrases': 'TopicRank_Keyphrases', 'Keyphrase_Scores': 'TopicRank_Keyphrase_Score'})
        multipartite_results_df = multipartite_results_df.rename(
            columns={'Keyphrases': 'MultipartiteRank_Keyphrases', 'Keyphrase_Scores': 'MultipartiteRank_Keyphrase_Score'})

        # full_df = keybert_results_df.merge(embedding_based_results_df, on=["Original_Document", "Gold_Keyphrases"])
        full_df = keybert_results_df.merge(rake_results_df, on=["Original_Document", "Gold_Keyphrases"])
        full_df = full_df.merge(yake_results_df, on=["Original_Document", "Gold_Keyphrases"])
        full_df = full_df.merge(textrank_results_df, on=["Original_Document", "Gold_Keyphrases"])
        full_df = full_df.merge(singlerank_results_df, on=["Original_Document", "Gold_Keyphrases"])
        full_df = full_df.merge(topicrank_results_df, on=["Original_Document", "Gold_Keyphrases"])
        full_df = full_df.merge(multipartite_results_df, on=["Original_Document", "Gold_Keyphrases"])

        flag = np.zeros(full_df.shape[0])

        for index in best_f1_indices:
            flag[index] = 1
        for index in worst_f1_indices:
            flag[index] = 1

        full_df['Flag'] = flag

        final_df = full_df.loc[full_df['Flag'] == 1]

        # output final_df to file
        all_df = full_df[['Gold_Keyphrases', 'KeyBERT_Keyphrases',
                          'RAKE_Keyphrases', 'YAKE_Keyphrases', 'TextRank_Keyphrases',
                          'SingleRank_Keyphrases', 'TopicRank_Keyphrases', 'MultipartiteRank_Keyphrases']]
        all_outfile = f'keyphrase_comparison/{dataset}_keyphrase_comparison_full.csv'
        all_df.to_csv(all_outfile)

        output_filepath = f'keyphrase_comparison/{dataset}_keyphrase_comparison_filtered.csv'
        final_df.to_csv(output_filepath)

        flag_df = pd.DataFrame()
        flag_df['Algorithm'] = ['KeyBERT', 'RAKE', 'YAKE', 'TextRank', 'SingleRank', 'TopicRank', 'MultipartiteRank']
        flag_df['Best Prediction Index'] = best_f1_indices
        flag_df['Worst Prediction Index'] = worst_f1_indices

        outfile = f'keyphrase_comparison/{dataset}_best_worst_predictions.csv'
        flag_df.to_csv(outfile)


if __name__ == '__main__':
    main()
