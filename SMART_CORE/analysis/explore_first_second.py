import csv
import numpy as np
from numpy import dot
from numpy.linalg import norm
from sklearn.feature_extraction.text import TfidfVectorizer
from create_input_lists import create_input_lists_from_csv, lemmatize
from statistics import mean
from sklearn.metrics.pairwise import cosine_similarity, euclidean_distances
from statistics import mean, median, mode
import matplotlib.pyplot as plt
import math
from sklearn.cluster import KMeans
from collections import defaultdict, Counter
from sklearn.metrics import silhouette_score
from numpy import var
from math import sqrt
from keyword_extraction import *
import os
import shutil
import pandas as pd

def main():

    if os.path.isdir('representation_level'):
        shutil.rmtree('representation_level')
        os.mkdir('representation_level')
    else:
        os.mkdir('representation_level')

    # obtain and pre-process the texts for the OLI Biology course
    # paragraph_path = './OneDrive-2020-12-04/intro_bio (with periods)_labelled/paragraphs.csv'
    # assessment_path = './OneDrive-2020-12-04/intro_bio (with periods)_labelled/assessments.csv'

    # obtain and pre-process the texts for the OLI Chemistry course
    paragraph_path = './OneDrive-2020-12-04/gen_chem/paragraphs.csv'
    assessment_path = './OneDrive-2020-12-04/gen_chem/assessments.csv'

    para_ids, lemma_para, orig_para, assess_ids, lemma_assess, orig_assess = \
        create_input_lists_from_csv(paragraph_path, assessment_path)

    # generate the TF-IDF vectorization of the texts
    vectorizer = TfidfVectorizer(use_idf=True, stop_words='english')
    para_vectors = vectorizer.fit_transform(lemma_para)
    assess_vectors = vectorizer.fit_transform(lemma_assess)

    second_para_vectors = cosine_similarity(para_vectors, para_vectors)
    second_assess_vectors = cosine_similarity(assess_vectors, assess_vectors)

    print('Shape of paragraph vectors (first):', para_vectors.shape)
    print('Shape of assessment vectors (first):', assess_vectors.shape)
    print('Shape of paragraph vectors (second):', second_para_vectors.shape)
    print('Shape of assessment vectors (second):', second_assess_vectors.shape, '\n')

    # count the non-zero entries in the vectors and output summary statistics
    para_nonzero_counts = []
    for vector in para_vectors:
        para_nonzero_counts.append(vector.count_nonzero())

    assess_nonzero_counts = []
    for vector in assess_vectors:
        assess_nonzero_counts.append(vector.count_nonzero())

    second_para_nonzero_counts = []
    for vector in second_para_vectors:
        vector_count = 0
        for value in vector:
            if value == 0:
                vector_count += 1
        second_para_nonzero_counts.append(vector_count)

    second_assess_nonzero_counts = []
    for vector in second_assess_vectors:
        vector_count = 0
        for value in vector:
            if value == 0:
                vector_count += 1
        second_assess_nonzero_counts.append(vector_count)

    print('TF-IDF for Paragraphs:\n')
    print('Average Number of Non-Zeros (First):', mean(para_nonzero_counts))
    print('Total # of Dimensions (First):', para_vectors.shape[1])
    print('Average Number of Non-Zeros (Second):', mean(second_para_nonzero_counts))
    print('Total # of Dimensions (Second):', second_para_vectors.shape[1])

    print('\n\nTF-IDF for Assessments:\n')
    print('Average Number of Non-Zeros (First):', mean(assess_nonzero_counts))
    print('Total # of Dimensions (First):', assess_vectors.shape[1])
    print('Average Number of Non-Zeros (Second):', mean(second_assess_nonzero_counts))
    print('Total # of Dimensions (Second):', second_assess_vectors.shape[1], '\n')

    # define functions to obtain within cluster, between cluster pairs; calculate pairwise metrics, and print the
    # pairwise distance statistics
    def get_between_within_pairs(item_ids, pairs, cluster_index):
        pair_ids = [[a, b] for idx, a in enumerate(item_ids) for b in item_ids[idx + 1:]]

        between_pairs = []
        within_pairs = []
        for index in range(0, len(pairs)):
            a, b = pairs[index]
            a_id, b_id = pair_ids[index]
            a_cluster = cluster_index[a_id]
            b_cluster = cluster_index[b_id]
            if a_cluster != b_cluster:
                between_pairs.append([a, b])
            else:
                within_pairs.append([a, b])
        return between_pairs, within_pairs

    def plot_distances(distances_list_between, distances_list_within, name):
        plt.hist(distances_list_between, rwidth=0.8,
                 bins=np.arange(min(distances_list_between), max(distances_list_between) + 0.01, 0.01), color='b',
                 label='Between Cluster Pairs')
        plt.hist(distances_list_within, rwidth=0.8,
                 bins=np.arange(min(distances_list_within), max(distances_list_within) + 0.01, 0.01), color='g',
                 label='Within Cluster Pairs')
        plt.xlabel('Pairwise Distance', fontsize=10)
        plt.ylabel('Frequency', fontsize=10)
        plt.xticks(fontsize=6)
        plt.yticks(fontsize=6)
        plt.legend()
        plt.grid()
        plot_filepath = 'representation_level/' + name + '_plot.pdf'
        plt.savefig(plot_filepath)
        plt.close()

    def calculate_pairwise_distances(pairs):
        dist_list = []
        for pair in pairs:
            a, b = pair
            a = np.array(a)
            b = np.array(b)
            dist = norm(a - b)
            dist_list.append(dist)

        return dist_list

    def print_distance_statistics(dist_list):
        dist_list = [x for x in dist_list if math.isnan(x) is False]
        print('Euclidean Distance Mean:', mean(dist_list))
        print('Euclidean Distance Median:', median(dist_list))
        # print('Euclidean Distance Mode:', mode(dist_list))
        print('Euclidean Distance Min:', min(dist_list))
        print('Euclidean Distance Max:', max(dist_list))

    def calc_d(dist1, dist2):
        # calculate the size of the samples
        n1, n2 = len(dist1), len(dist2)
        # calculate the variance of the samples
        s1, s2 = var(dist1, ddof=1), var(dist2, ddof=1)
        # calculate the pooled standard deviation
        std = sqrt(((n1 - 1) * s1 + (n2 - 1) * s2) / (n1 + n2 - 2))
        # calculate the means of the samples
        u1, u2 = mean(dist1), mean(dist2)
        # calculate the effect size
        return (u1 - u2) / std

    assess_vectors = assess_vectors.toarray().tolist()
    assess_pairs = [[a, b] for idx, a in enumerate(assess_vectors) for b in assess_vectors[idx + 1:]]

    second_assess_vectors = second_assess_vectors.tolist()
    second_assess_pairs = [[a, b] for idx, a in enumerate(second_assess_vectors) for b in second_assess_vectors[idx + 1:]]

    k_values = [10, 50, 100, 150, 200, 250, 300, 350, 400, 450, 500]
    s_scores = []
    s_second_scores = []
    d_first_prior_values = []
    d_second_prior_values = []
    d_first_post_values = []
    d_second_post_values = []

    for k in k_values:
        iteration = 1
        df1 = []
        ds1 = []
        df2 = []
        ds2 = []
        while iteration <= 5:
            # perform k-means clustering for first and second level representation
            km = KMeans(n_clusters=k, max_iter=50000, init='k-means++')
            cluster_assignment = km.fit(assess_vectors).labels_
            s = silhouette_score(assess_vectors, cluster_assignment, metric='euclidean')
            s_scores.append(s)

            km_second = KMeans(n_clusters=k, max_iter=50000, init='k-means++')
            cluster_assignment_second = km_second.fit(second_assess_vectors).labels_
            s_second = silhouette_score(second_assess_vectors, cluster_assignment_second, metric='euclidean')
            s_second_scores.append(s_second)

            clusterIndex_to_clusteredText1 = defaultdict(lambda: " ")
            text_id1_to_clusterIndex = dict()

            for i, clusterIndex in enumerate(cluster_assignment):  # len(cluster_assignment) is equal to len(vectors)
                clusterIndex_to_clusteredText1[clusterIndex] += orig_assess[i] + ". "
                text_id1_to_clusterIndex[assess_ids[i]] = clusterIndex

            clusterIndex_to_clusteredText1_second = defaultdict(lambda: " ")
            text_id1_to_clusterIndex_second = dict()

            for i, clusterIndex in enumerate(cluster_assignment_second):  # len(cluster_assignment) is equal to len(vectors)
                clusterIndex_to_clusteredText1_second[clusterIndex] += orig_assess[i] + ". "
                text_id1_to_clusterIndex_second[assess_ids[i]] = clusterIndex

            # obtain the between cluster and within cluster pairs for first and second level representation
            bw_pairs, with_pairs = get_between_within_pairs(assess_ids, assess_pairs, text_id1_to_clusterIndex)
            bw_pairs_second, with_pairs_second = get_between_within_pairs(assess_ids, second_assess_pairs,
                                                                          text_id1_to_clusterIndex_second)

            # obtain the pairwise distances for between cluster pairs and within cluster pairs for both representations.
            bw_pair_distances, with_pair_distances = calculate_pairwise_distances(bw_pairs),\
                                                     calculate_pairwise_distances(with_pairs)
            bw_pair_distances_second, with_pair_distances_second = calculate_pairwise_distances(bw_pairs_second), \
                                                                   calculate_pairwise_distances(with_pairs_second)

            # plot the distances for both representations
            name1 = 'first_level_1_k' + str(k) + '_iteration_' + str(iteration)
            plot_distances(bw_pair_distances, with_pair_distances, name1)
            print('Statistics for First Level, Between Cluster Pairs (Prior to Merge):\n')
            print_distance_statistics(bw_pair_distances)
            print('\nStatistics for First Level, Within Cluster Pairs (Prior to Merge):\n')
            print_distance_statistics(with_pair_distances)
            name2 = 'second_level_1_k' + str(k) + '_iteration_' + str(iteration)
            plot_distances(bw_pair_distances_second, with_pair_distances_second, name2)
            print('\nStatistics for Second Level, Between Cluster Pairs (Prior to Merge):\n')
            print_distance_statistics(bw_pair_distances_second)
            print('\nStatistics for Second Level, Within Cluster Pairs (Prior to Merge):\n')
            print_distance_statistics(with_pair_distances_second)

            # calculate the effect size among between cluster pairs and within cluster pairs for first and second level
            # representation

            d_first_1 = calc_d(bw_pair_distances, with_pair_distances)
            d_second_1 = calc_d(bw_pair_distances_second, with_pair_distances_second)

            df1.append(d_first_1)
            ds1.append(d_second_1)

            print('Initial Effect Size (First Level):', d_first_1)
            print('Initial Effect Size (Second Level):', d_second_1)

            # perform TextRank on the clusters and merge any clusters with the same keyword for both representations.
            clusteredText1_to_skill, clusterIndex_to_skill, new_num_clusters = extract_keywords(
                clusterIndex_to_clusteredText1)
            clusteredText1_to_skill_second, clusterIndex_to_skill_second, new_num_clusters_second = extract_keywords(
                clusterIndex_to_clusteredText1_second)

            text_id1_to_skill = {}
            for text_id, clusterIndex in text_id1_to_clusterIndex.items():
                text_id1_to_skill[text_id] = clusterIndex_to_skill[clusterIndex]

            text_id1_to_skill_second = {}
            for text_id, clusterIndex in text_id1_to_clusterIndex_second.items():
                text_id1_to_skill_second[text_id] = clusterIndex_to_skill_second[clusterIndex]

            # obtain the between cluster and within cluster pairs for first and second level representation after merging
            # clusters
            bw_pairs_2, with_pairs_2 = get_between_within_pairs(assess_ids, assess_pairs, text_id1_to_skill)
            bw_pairs_second_2, with_pairs_second_2 = get_between_within_pairs(assess_ids, second_assess_pairs,
                                                                              text_id1_to_skill_second)

            # obtain the pairwise distances for between cluster pairs and within cluster pairs for both representations.
            bw_pair_distances_2, with_pair_distances_2 = calculate_pairwise_distances(bw_pairs_2), \
                                                     calculate_pairwise_distances(with_pairs_2)
            bw_pair_distances_second_2, with_pair_distances_second_2 = calculate_pairwise_distances(bw_pairs_second_2), \
                                                                   calculate_pairwise_distances(with_pairs_second_2)

            # plot the distances for both representations
            name_1 = 'first_level_2_k' + str(k) + '_iteration_' + str(iteration)
            plot_distances(bw_pair_distances_2, with_pair_distances_2, name_1)
            print('\nStatistics for First Level, Between Cluster Pairs (After Merge):\n')
            print_distance_statistics(bw_pair_distances_2)
            print('\nStatistics for First Level, Within Cluster Pairs (After Merge):\n')
            print_distance_statistics(with_pair_distances_2)
            name_2 = 'second_level_2_k' + str(k) + '_iteration_' + str(iteration)
            plot_distances(bw_pair_distances_second_2, with_pair_distances_second_2, name_2)
            print('\nStatistics for Second Level, Between Cluster Pairs (After Merge):\n')
            print_distance_statistics(bw_pair_distances_second_2)
            print('\nStatistics for Second Level, Within Cluster Pairs (After Merge):\n')
            print_distance_statistics(with_pair_distances_second_2)

            # calculate the effect size among between cluster pairs and within cluster pairs for first and second level
            # representation
            d_first_2 = calc_d(bw_pair_distances_2, with_pair_distances_2)
            d_second_2 = calc_d(bw_pair_distances_second_2, with_pair_distances_second_2)

            df2.append(d_first_2)
            ds2.append(d_second_2)

            print('Post-Merge Effect Size (First Level):', d_first_2)
            print('Post-Merge Effect Size (Second Level):', d_second_2)

            iteration += 1

        d_first_prior_values.append(mean(df1))
        d_second_prior_values.append(mean(ds1))
        d_first_post_values.append(mean(df2))
        d_second_post_values.append(mean(ds2))

    plt.plot(k_values, d_first_prior_values, 'b', label='First Level-Prior to Merge')
    plt.plot(k_values, d_first_post_values, 'b--', label='First Level - After Merge')
    plt.plot(k_values, d_second_prior_values, 'g', label='Second Level - Prior to Merge')
    plt.plot(k_values, d_second_post_values, 'g--', label='Second Level - After Merge')
    plt.xlabel('k Value')
    plt.ylabel('Effect Size')
    # plt.title('Comparison of Intra/Inter Cluster Pair Effect Size (OLI Biology, Assessments)')
    plt.title('Comparison of Intra/Inter Cluster Pair Effect Size (OLI Chemistry, Assessments)')
    plt.legend()
    prior_filepath = 'representation_level/effect-size_comparison.pdf'
    plt.savefig(prior_filepath)
    plt.close()

    df = pd.DataFrame()
    df['d_first_prior'] = d_first_prior_values
    df['d_first_post'] = d_first_post_values
    df['d_second_prior'] = d_second_prior_values
    df['d_second_post'] = d_second_post_values
    df.to_csv('representation_level/effect_sizes.csv')


if __name__ == '__main__':
    main()
