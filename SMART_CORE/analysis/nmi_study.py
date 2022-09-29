from sklearn.cluster import KMeans
from collections import defaultdict
from keyword_extraction import *
from sklearn.feature_extraction.text import TfidfVectorizer
from create_input_lists import create_input_lists_from_csv
from get_KCModel_import import addSkillColumn
from datetime import datetime
from sklearn.metrics import normalized_mutual_info_score
from summa.keywords import keywords
from custom_stopwords import get_custom_stopwords
import csv
import numpy as np
import os
import shutil
import pandas as pd
import matplotlib.pyplot as plt

plt.style.use('seaborn-whitegrid')


def main():

    if os.path.isdir('nmi_study'):
        shutil.rmtree('nmi_study')
        os.mkdir('nmi_study')
    else:
        os.mkdir('nmi_study')

    # obtain and pre-process the texts for the OLI Biology course
    paragraph_path_bio = './OneDrive-2020-12-04/intro_bio (with periods)_labelled/paragraphs.csv'
    assessment_path_bio = './OneDrive-2020-12-04/intro_bio (with periods)_labelled/assessments.csv'

    # obtain and pre-process the texts for the OLI Chemistry course
    paragraph_path_chem = './OneDrive-2020-12-04/gen_chem/paragraphs.csv'
    assessment_path_chem = './OneDrive-2020-12-04/gen_chem/assessments.csv'

    para_ids_bio, lemma_para_bio, orig_para_bio, assess_ids_bio, lemma_assess_bio, orig_assess_bio = \
        create_input_lists_from_csv(paragraph_path_bio, assessment_path_bio)

    para_ids_chem, lemma_para_chem, orig_para_chem, assess_ids_chem, lemma_assess_chem, orig_assess_chem = \
        create_input_lists_from_csv(paragraph_path_chem, assessment_path_chem)

    # generate the TF-IDF vectorization of the texts
    vectorizer_bio = TfidfVectorizer(use_idf=True, stop_words='english')
    para_vectors_bio = vectorizer_bio.fit_transform(lemma_para_bio)
    assess_vectors_bio = vectorizer_bio.fit_transform(lemma_assess_bio)

    vectorizer_chem = TfidfVectorizer(use_idf=True, stop_words='english')
    para_vectors_chem = vectorizer_chem.fit_transform(lemma_para_chem)
    assess_vectors_chem = vectorizer_chem.fit_transform(lemma_assess_chem)

    print('Shape of OLI Biology paragraph vectors (first):', para_vectors_bio.shape)
    print('Shape of OLI Biology assessment vectors (first):', assess_vectors_bio.shape)

    print('Shape of OLI Chemistry paragraph vectors (first):', para_vectors_chem.shape)
    print('Shape of OLI Chemistry assessment vectors (first):', assess_vectors_chem.shape)

    # retrieve and store the set of labels for the human-generated model for the OLI Biology course
    human_bio_df = pd.read_csv('/home/jwood/OneDrive/SMART/DataShop Import/intro_bio/model_8_import.txt', sep='\t')
    human_bio_labels = human_bio_df['KC (model_8)'].values.tolist()

    # retrieve and store the human-generated model for the OLI Chemistry course
    human_chem_df = pd.read_csv('/home/jwood/OneDrive/SMART/DataShop Import/gen_chem/gen_chem_final_import.txt', sep='\t')
    human_chem_labels = human_chem_df['KC(gen-chem1-2_3)'].values.tolist()

    # def functions as necessary
    def perform_clustering(k, assess_vectors, assess_ids, orig_assess):

        km = KMeans(n_clusters=k, max_iter=50000, init='k-means++')
        cluster_assignment = km.fit(assess_vectors).labels_

        # Map the cluster index to the original text
        clusterIndex_to_clusteredText = defaultdict(lambda: " ")	
        text_id_to_clusterIndex = dict()

        for i, clusterIndex in enumerate(cluster_assignment):  # len(cluster_assignment) is equal to len(vectors)
            clusterIndex_to_clusteredText[clusterIndex] += orig_assess[i] + ". " 
            text_id_to_clusterIndex[assess_ids[i]] = clusterIndex
  
        return clusterIndex_to_clusteredText, text_id_to_clusterIndex

    def extract_keywords_with_merge(clusterIndex_to_clusteredText):
        clusteredText1_to_skill = dict()
        clusterIndex_to_skill = dict()

        for clusterIndex, clusteredText in clusterIndex_to_clusteredText.items():
            if not clusteredText.encode('utf-8'):
                skill_names = 'UNKNOWN'
            else:
                skill_names = keywords(clusteredText, additional_stopwords=get_custom_stopwords())  # by default: ratio
                # (default = 0.2)

            top_skill_name = skill_names.split('\n')[0]
            if len(top_skill_name) == 0:
                # modified on 5/19/2021: changed to select most frequent lemma instead of most frequent original word
                words_freq = Counter(lemmatize(clusteredText).split())
                maxFreq = 0
                for word, freq in words_freq.items():
                    if freq > maxFreq:
                        top_skill_name = word
                        maxFreq = freq

            if not top_skill_name:
                top_skill_name = 'UNKNOWN'

            clusteredText1_to_skill[clusteredText] = top_skill_name
            clusterIndex_to_skill[clusterIndex] = top_skill_name

        distinct_skill_count = len(set(clusterIndex_to_skill.values()))  # new_num_clusters

        return clusteredText1_to_skill, clusterIndex_to_skill, distinct_skill_count

    def output_assess_cluster(text_id_to_clusterIndex, clusterIndex_to_skill, filepath):

        text_to_clusterindex_df = pd.DataFrame.from_dict(text_id_to_clusterIndex, orient='index',
                                                         columns=['clusterIndex'])
        text_to_clusterindex_df.index.name = 'Assessment ID'
        text_to_clusterindex_df.reset_index(inplace=True)
        clusterindex_to_skill_df = pd.DataFrame.from_dict(clusterIndex_to_skill, orient='index', columns=['skill'])
        clusterindex_to_skill_df.index.name = 'clusterIndex'
        clusterindex_to_skill_df.reset_index(inplace=True)
        final_df = pd.merge(text_to_clusterindex_df, clusterindex_to_skill_df)
        final_df = final_df.drop(['clusterIndex'], axis=1)
        final_df.to_csv(filepath, index=False)

        return

    def transform_model_and_retrieve_labels(filepath, k, iteration, representation, merge_status, course):

        today_date = datetime.today().strftime('%m_%d_%Y')  # today's date for naming convention
        if course == 'bio':
            DataShopExport = '/home/jwood/OneDrive/SMART/DataShop Import/intro_bio/model_8_import.txt'
            KC_model_name = 'KC(SMART_bio_' + representation + '_' + merge_status + '_' + str(k) + '_' + str(iteration)
        elif course == 'chem':
            DataShopExport = '/home/jwood/OneDrive/SMART/DataShop Import/gen_chem/gen_chem_final_import.txt'
            KC_model_name = 'KC(SMART_chem_' + representation + '_' + merge_status + '_' + str(k) + '_' + str(iteration)
        else:
            DataShopExport = './None/'
            KC_model_name = 'None'
        
        outputFolder = 'SMART_models_' + today_date  # create a new folder with today's date
        addSkillColumn(filepath, DataShopExport, KC_model_name, outputFolder)

        export_base = os.path.basename(DataShopExport)[:-4]
        assessment_skill_base = os.path.basename(filepath)[:-4]
        import_filepath = os.path.join(outputFolder, export_base + assessment_skill_base + '.txt')
        smart_df = pd.read_csv(import_filepath, sep='\t')
        smart_labels = smart_df[KC_model_name].values.tolist()

        return smart_labels

    # define variables for running clustering and storing metrics
    num_samples = 25

    nmi_headers = []
    for index in range(1, num_samples+1):
        header = 'nmi_' + str(index)
        nmi_headers.append(header)
    headers = ['course', 'representation_level', 'merge_operation', 'num_clusters'] + nmi_headers + ['mean', 'std']

    with open('./nmi_study/nmi_scores.csv', 'w') as file:
        writer = csv.writer(file, delimiter=',')
        writer.writerow(headers)
    file.close()
    
    # k_values = [10, 50, 100, 150, 200]
    k_values = [10, 50, 100, 150, 200, 250, 300, 350, 400, 450, 500, 550, 600, 650, 700, 750, 800, 850, 900, 950, 1000]

    mean_bio_first_nm_list = []
    mean_bio_first_merge_list = []

    mean_chem_first_nm_list = []
    mean_chem_first_merge_list = []

    std_bio_first_nm_list = []
    std_bio_first_merge_list = []

    std_chem_first_nm_list = []
    std_chem_first_merge_list = []

    for k in k_values:

        iteration = 1

        nmi_bio_first_nm_scores = []
        nmi_bio_first_merge_scores = []

        nmi_chem_first_nm_scores = []
        nmi_chem_first_merge_scores = []

        while iteration <= num_samples:
            # perform k-means clustering of assessment items
            clusterIndex_to_clusteredText_bio_first, text_id_to_clusterIndex_bio_first = perform_clustering(k, assess_vectors_bio, assess_ids_bio, orig_assess_bio)

            clusterIndex_to_clusteredText_chem_first, text_id_to_clusterIndex_chem_first = perform_clustering(k, assess_vectors_chem, assess_ids_chem, orig_assess_chem)

            # perform TextRank keyword extraction of assessment items
            clusteredText_to_skill_bio_first, clusterIndex_to_skill_bio_first, new_num_clusters_bio_first = extract_keywords(clusterIndex_to_clusteredText_bio_first, 'non-iterative')

            clusteredText_to_skill_chem_first, clusterIndex_to_skill_chem_first, new_num_clusters_chem_first = extract_keywords(clusterIndex_to_clusteredText_chem_first, 'non-iterative')

            # for clustering with merge, merge clusters with the same TextRank keyword
            clusteredText_to_skill_bio_first_merge, clusterIndex_to_skill_bio_first_merge, new_num_clusters_bio_first_merge = extract_keywords_with_merge(clusterIndex_to_clusteredText_bio_first)

            clusteredText_to_skill_chem_first_merge, clusterIndex_to_skill_chem_first_merge, new_num_clusters_chem_first_merge = extract_keywords_with_merge(clusterIndex_to_clusteredText_chem_first)

            # output assessment item, cluster name pairs to file
            filepath_bio_first_nm = './nmi_study/_assessment_to_skill_SMART_bio_first_nm_' + str(k) + '_' + str(iteration) + '.csv'
            output_assess_cluster(text_id_to_clusterIndex_bio_first, clusterIndex_to_skill_bio_first, filepath_bio_first_nm)

            filepath_bio_first_merge = './nmi_study/_assessment_to_skill_SMART_bio_first_merge_' + str(k) + '_' + str(iteration) + '.csv'
            output_assess_cluster(text_id_to_clusterIndex_bio_first, clusterIndex_to_skill_bio_first_merge, filepath_bio_first_merge)

            filepath_chem_first_nm = './nmi_study/_assessment_to_skill_SMART_chem_first_nm_' + str(k) + '_' + str(iteration) + '.csv'
            output_assess_cluster(text_id_to_clusterIndex_chem_first, clusterIndex_to_skill_chem_first, filepath_chem_first_nm)

            filepath_chem_first_merge = './nmi_study/_assessment_to_skill_SMART_chem_first_merge_' + str(k) + '_' + str(iteration) + '.csv'
            output_assess_cluster(text_id_to_clusterIndex_chem_first, clusterIndex_to_skill_chem_first_merge, filepath_chem_first_merge)

            # transform smart assessment item names to the DataShop KC model format and store the set of labels for the smart model
            smart_bio_first_nm_labels = transform_model_and_retrieve_labels(filepath_bio_first_nm, k, iteration, 'first', 'no_merge', 'bio')
            smart_bio_first_merge_labels = transform_model_and_retrieve_labels(filepath_bio_first_merge, k, iteration, 'first', 'merge', 'bio')

            smart_chem_first_nm_labels = transform_model_and_retrieve_labels(filepath_chem_first_nm, k, iteration, 'first', 'no_merge', 'chem')
            smart_chem_first_merge_labels = transform_model_and_retrieve_labels(filepath_chem_first_merge, k, iteration, 'first', 'merge', 'chem')

            # calculate the nmi between the human model and smart model
            nmi_bio_first_nm = normalized_mutual_info_score(human_bio_labels, smart_bio_first_nm_labels)
            nmi_bio_first_nm_scores.append(nmi_bio_first_nm)

            nmi_bio_first_merge = normalized_mutual_info_score(human_bio_labels, smart_bio_first_merge_labels)
            nmi_bio_first_merge_scores.append(nmi_bio_first_merge)

            nmi_chem_first_nm = normalized_mutual_info_score(human_chem_labels, smart_chem_first_nm_labels)
            nmi_chem_first_nm_scores.append(nmi_chem_first_nm)

            nmi_chem_first_merge = normalized_mutual_info_score(human_chem_labels, smart_chem_first_merge_labels)
            nmi_chem_first_merge_scores.append(nmi_chem_first_merge)

            # increment the iteration
            iteration += 1

        # calculate the mean and standard deviation for each set of nmi scores and output nmi, mean, std to file
        mean_bio_first_nm = np.mean(nmi_bio_first_nm_scores)
        mean_bio_first_merge = np.mean(nmi_bio_first_merge_scores)

        mean_chem_first_nm = np.mean(nmi_chem_first_nm_scores)
        mean_chem_first_merge = np.mean(nmi_chem_first_merge_scores)

        mean_bio_first_nm_list.append(mean_bio_first_nm)
        mean_bio_first_merge_list.append(mean_bio_first_merge)

        mean_chem_first_nm_list.append(mean_chem_first_nm)
        mean_chem_first_merge_list.append(mean_chem_first_merge)

        std_bio_first_nm = np.std(nmi_bio_first_nm_scores, ddof=1)
        std_bio_first_merge = np.std(nmi_bio_first_merge_scores, ddof=1)

        std_chem_first_nm = np.std(nmi_chem_first_nm_scores, ddof=1)
        std_chem_first_merge = np.std(nmi_chem_first_merge_scores, ddof=1)

        std_bio_first_nm_list.append(std_bio_first_nm)
        std_bio_first_merge_list.append(std_bio_first_merge)

        std_chem_first_nm_list.append(std_chem_first_nm)
        std_chem_first_merge_list.append(std_chem_first_merge)

        bio_first_nm_output = ['oli_biology', 'first', 'no_merge', k] + nmi_bio_first_nm_scores + [mean_bio_first_nm, std_bio_first_nm]
        bio_first_merge_output = ['oli_biology', 'first', 'merge', k] + nmi_bio_first_merge_scores + [mean_bio_first_merge, std_bio_first_merge]

        chem_first_nm_output = ['oli_chemistry', 'first', 'no_merge', k] + nmi_chem_first_nm_scores + [mean_chem_first_nm, std_chem_first_nm]
        chem_first_merge_output = ['oli_chemistry', 'first', 'merge', k] + nmi_chem_first_merge_scores + [mean_chem_first_merge, std_chem_first_merge]

        with open('./nmi_study/nmi_scores.csv', 'a') as file:
            writer = csv.writer(file, delimiter=',')
            writer.writerow(bio_first_nm_output)
            writer.writerow(bio_first_merge_output)
            writer.writerow(chem_first_nm_output)
            writer.writerow(chem_first_merge_output)
        file.close()

    # plot the mean and standard deviation for each set of samples
    plt.errorbar(k_values, mean_bio_first_nm_list, yerr=std_bio_first_nm_list, fmt='.k', color='blue', label='SMART (no merge)')
    plt.errorbar(k_values, mean_bio_first_merge_list, yerr=std_bio_first_merge_list, fmt='.k', color='green', label='SMART (merge)')
    plt.xlabel('Number of Clusters')
    plt.ylabel('NMI')
    plt.title('NMI Between SMART and Human model [OLI Biology]')
    plot_filepath = 'nmi_study/OLI_Biology_first_nm_nmi_error_plot.pdf'
    plt.savefig(plot_filepath)
    plt.close()

    plt.errorbar(k_values, mean_chem_first_nm_list, yerr=std_chem_first_nm_list, fmt='.k', color='blue', label='SMART (no merge)')
    plt.errorbar(k_values, mean_chem_first_merge_list, yerr=std_chem_first_merge_list, fmt='.k', color='green', label='SMART (merge)')
    plt.xlabel('Number of Clusters')
    plt.ylabel('NMI')
    plt.title('NMI Between SMART and Human model [OLI Chemistry]')
    plot_filepath = 'nmi_study/OLI_Chemistry_first_nm_nmi_error_plot.pdf'
    plt.savefig(plot_filepath)
    plt.close()


if __name__ == '__main__':
    main()
