from transformers import AutoTokenizer, AutoModelForCausalLM
from process_data import preprocess
from keyphrase_generation_gpt_oli import evaluate, generate_keyphrases
import pandas as pd
import shutil
import os


def prep_evaluation_folders():
    output_path = 'gpt_output_oli_test/'
    results_path = 'gpt_evaluation_oli_test/'

    if os.path.exists(output_path):
        shutil.rmtree(output_path)
    if os.path.exists(results_path):
        shutil.rmtree(results_path)

    os.mkdir(output_path)
    os.mkdir(results_path)


def main():

    prep_evaluation_folders()

    special_token = '<|endoftext|>'

    test_part_prec_list = []
    test_part_recall_list = []
    test_part_f1_list = []
    test_prec_list = []
    test_recall_list = []
    test_f1_list = []

    best_model_list = []

    best_num_epochs_dict = {'bio': 70, 'chem': 40}
    
    for course in ['bio', 'chem']:
        # load the test dataset
        # load train and validation data
        test_dataset_path = f'ValidationDatasets/test_{course}_10_percent.txt'

        with open(test_dataset_path, 'r') as filepath:
            test_data = filepath.read()

        test_data = test_data.split(special_token)

        test_df_path = f'Dataframes/test_{course}_10_percent.pkl'
        test_df = pd.read_pickle(test_df_path)
        test_gold_kps = list(test_df.processed_keyphrases)

        # evaluate best model on the test dataset
        best_model_path = f'FineTunedModels/{course}_10_percent_epochs_{best_num_epochs_dict[course]}'
        best_model_list.append(best_model_path)

        tokenizer = AutoTokenizer.from_pretrained(best_model_path, use_fast=False)
        model = AutoModelForCausalLM.from_pretrained(best_model_path)

        test_generated_keyphrases = []
        for data_index in range(0, len(test_data) - 1):
            # generate keyphrase(s)
            input_data = test_data[data_index]
            list_result = generate_keyphrases(input_data, tokenizer, model, special_token, 10, 0.8)
            test_generated_keyphrases.append(list_result)

        test_generated_keyphrases = [[preprocess(kp) for kp in kp_list] for kp_list in test_generated_keyphrases]

        test_part_prec, test_part_recall, test_part_f1, test_prec, test_recall, test_f1 = evaluate(
            test_generated_keyphrases,
            test_gold_kps, verbose=True)

        # Output generated keyphrases to file
        test_output_df = pd.DataFrame()
        test_output_df['whole_document'] = test_df['whole_document']
        test_output_df['keyphrases'] = test_df['keyphrases']
        test_output_df['generated_keyphrases'] = test_generated_keyphrases

        filename = f'test_{course}_epochs_{best_num_epochs_dict[course]}_output.csv'
        gpt_output_path = 'gpt_output_oli_test/' + filename
        test_output_df.to_csv(gpt_output_path)

        test_part_prec_list.append(test_part_prec)
        test_part_recall_list.append(test_part_recall)
        test_part_f1_list.append(test_part_f1)
        test_prec_list.append(test_prec)
        test_recall_list.append(test_recall)
        test_f1_list.append(test_f1)

    evaluation_df = pd.DataFrame()
    evaluation_df['Model'] = best_model_list
    evaluation_df['Dataset'] = ['oli-intro-bio', 'oli-gen-chem']

    evaluation_df['Test Exact Match Precision'] = test_prec_list
    evaluation_df['Test Exact Match Recall'] = test_recall_list
    evaluation_df['Test Exact Match F1 Score'] = test_f1_list
    evaluation_df['Test Partial Precision'] = test_part_prec_list
    evaluation_df['Test Partial Recall'] = test_part_recall_list
    evaluation_df['Test Partial F1 Score'] = test_part_f1_list

    gpt_evaluation_path = 'gpt_evaluation_oli_test/GPT2-finetuned-models-evaluation-num-epochs-oli.csv'
    evaluation_df.to_csv(gpt_evaluation_path)


if __name__ == '__main__':
    main()
