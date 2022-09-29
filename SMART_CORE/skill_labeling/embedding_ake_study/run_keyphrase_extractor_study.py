from keyphrase_extractor import *
import argparse
import csv
import warnings
warnings.filterwarnings('ignore')


def output_ake_results(original_document, gold_keyphrases, keyphrases, keyphrase_scores, filepath):
    results_df = pd.DataFrame()

    results_df['Original_Document'] = original_document
    results_df['Gold_Keyphrases'] = gold_keyphrases
    results_df['Keyphrases'] = keyphrases
    if len(keyphrase_scores) != 0:
        results_df['Keyphrase_Scores'] = keyphrase_scores

    results_df.to_csv(filepath)


def run_baseline_ake(keyphrase_extractor, dataset_name, algorithm, top_n, trial):
    if algorithm == 'TextRank':
        keyphrase_extractor.run_TextRank(top_n, verbose=True)
    elif algorithm == 'SingleRank':
        keyphrase_extractor.run_SingleRank(top_n, verbose=True)
    elif algorithm == 'TopicRank':
        keyphrase_extractor.run_TopicRank(top_n, verbose=True)
    elif algorithm == 'MultipartiteRank':
        keyphrase_extractor.run_MultipartiteRank(top_n, verbose=True)
    elif algorithm == 'RAKE':
        keyphrase_extractor.run_RAKE(top_n, verbose=True)
    elif algorithm == 'YAKE':
        keyphrase_extractor.run_YAKE(top_n, verbose=True)
    else:
        print("Invalid input for baseline algorithm.")
        sys.exit()
    keyphrase_extractor.evaluate(verbose=True)

    filepath = './output/ake_results/{}_{}_{}_trial_{}_output.csv'.format(dataset_name, algorithm, top_n, trial)
    output_ake_results(keyphrase_extractor.original_documents_whole, keyphrase_extractor.document_labels,
                       keyphrase_extractor.final_keyphrases, keyphrase_extractor.final_keyphrase_scores, filepath)

    evaluation_filepath = './output/evaluation/{}_evaluation.csv'.format(dataset_name)
    row_data = [dataset_name, algorithm, 'n/a', 'n/a', 'n/a', 'n/a', top_n, 'n/a', trial]
    row_data += [keyphrase_extractor.overall_precision_exact, keyphrase_extractor.overall_recall_exact,
                 keyphrase_extractor.overall_f1_exact]
    row_data += [keyphrase_extractor.overall_precision_partial, keyphrase_extractor.overall_recall_partial,
                 keyphrase_extractor.overall_f1_partial]

    with open(evaluation_filepath, 'a', newline='') as file:
        write = csv.writer(file)
        write.writerow(row_data)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('-datasetName', default='inspec', type=str)
    parser.add_argument('-candidateMethod', default='parsing', type=str)
    parser.add_argument('-numberTrials', default=25, type=int)
    parser.add_argument('-runBaseline', default=True, type=bool)

    args = parser.parse_args()

    dataset_name = args.datasetName.lower()  # string
    candidate_method = args.candidateMethod.lower()  # string
    num_trials = args.numberTrials
    run_baseline = args.runBaseline

    data_summary_df = pd.DataFrame()

    keyphrase_extractor = KeyphraseExtractor(dataset_name, verbose=True)
    keyphrase_extractor.get_candidate_keyphrases(candidate_method, verbose=True)

    # output dataset summaries to file
    new_data_entry = {'Dataset': dataset_name,
                      'Candidate Generation Method': candidate_method,
                      '# Documents': keyphrase_extractor.num_docs,
                      '# Tokens per Document': keyphrase_extractor.tokens_per_doc,
                      '# Gold KP per Document': keyphrase_extractor.gold_kp_per_doc,
                      '% Stopwords in Gold KP': keyphrase_extractor.percent_stopwords_gold_kp,
                      '% Gold KP missing from Document': keyphrase_extractor.percent_kp_words_missing_from_doc,
                      '# Candidates per Document': keyphrase_extractor.cand_per_doc,
                      '% Gold KP missing from Candidates': keyphrase_extractor.percent_kp_missing_in_cand}
    new_data_df = pd.DataFrame(new_data_entry, index=[1])
    data_summary_df = data_summary_df.append(new_data_df, ignore_index=True)

    dataset_summary_path = './output/temp/AKE_Dataset_Summary_' + dataset_name + '_' + candidate_method + '.csv'
    data_summary_df.to_csv(dataset_summary_path)

    dataset_n_dict = {'inspec': 10, 'kdd': 5, 'oli-intro-bio': 1, 'oli-gen-chem': 1}

    for trial in range(1, num_trials + 1):
        for ae_status in ['ae_off']:  # 'ae_on'
            for embedding_method in ['w2v', 'glove', 'd2v', 'sbert-mean-pooling']:  # 'w2v', 'glove', 'd2v', bert-cls, bert-sum-last4, bert-concat-last4
                keyphrase_extractor.get_embeddings(embedding_method, verbose=True)
                if ae_status == 'ae_on':
                    filepath = './output/ae_learning_curves/{}_{}_trial{}.pdf'.format(dataset_name, candidate_method,
                                                                                      embedding_method)
                    keyphrase_extractor.get_latent_representation(filepath, verbose=True)
                for ranking_and_selection_method in [('cos-doc-cand', 'top-n'), ('cos-doc-cand', 'top-n-mmr'), ('cos-doc-cand', 'top-n-mss'), ('cos-then-sum', 'top-n')]:
                    for top_n in [dataset_n_dict[dataset_name]]:
                        if trial % 1 == 0:
                            print('Beginning trial #{} for {}/{}/{}/{}/{}/{}'.format(trial, dataset_name, candidate_method,
                                                                                     embedding_method,
                                                                                     ranking_and_selection_method[0],
                                                                                     ranking_and_selection_method[1],
                                                                                     top_n))

                        if ranking_and_selection_method[1] == 'top-n':
                            keyphrase_extractor.rank_candidates(ranking_and_selection_method[0], verbose=True)
                            keyphrase_extractor.select_keyphrase(ranking_and_selection_method[1], top_n, verbose=True)
                        else:
                            keyphrase_extractor.select_keyphrase(ranking_and_selection_method[1], top_n, verbose=True)

                        filepath = './output/ake_results/{}_{}_{}_{}_{}_{}_{}_trial_{}_output.csv'.format(dataset_name, candidate_method, embedding_method, ranking_and_selection_method[0], ranking_and_selection_method[1], top_n, ae_status, trial)
                        output_ake_results(keyphrase_extractor.original_documents_whole, keyphrase_extractor.document_labels, keyphrase_extractor.final_keyphrases, keyphrase_extractor.final_keyphrase_scores, filepath)

                        keyphrase_extractor.evaluate(verbose=True)

                        evaluation_filepath = './output/evaluation/{}_evaluation.csv'.format(dataset_name)
                        row_data = [dataset_name, 'Embedding-based', candidate_method, embedding_method, ranking_and_selection_method[0], ranking_and_selection_method[1], top_n, ae_status, trial]
                        row_data += [keyphrase_extractor.overall_precision_exact, keyphrase_extractor.overall_recall_exact, keyphrase_extractor.overall_f1_exact]
                        row_data += [keyphrase_extractor.overall_precision_partial, keyphrase_extractor.overall_recall_partial, keyphrase_extractor.overall_f1_partial]

                        with open(evaluation_filepath, 'a', newline='') as file:
                            write = csv.writer(file)
                            write.writerow(row_data)

                        keyphrase_extractor.reinitialize()

    if candidate_method == 'parsing' and run_baseline:
        for algorithm in ['TextRank', 'SingleRank', 'TopicRank', 'MultipartiteRank', 'RAKE', 'YAKE']:
            for top_n in [dataset_n_dict[dataset_name]]:
                for trial in range(1, num_trials + 1):
                    if trial % 1 == 0:
                        print('Beginning trial #{} for {}/{}/{})'.format(trial, dataset_name, algorithm, top_n))
                    run_baseline_ake(keyphrase_extractor, dataset_name, algorithm, top_n, trial)
                    keyphrase_extractor.reinitialize()


if __name__ == '__main__':
    main()
    # test()
    # test_baseline()
