from keyphrase_generation_gpt_oli_v2 import evaluate_public, generate_keyphrases
import matplotlib.pyplot as plt
from transformers import AutoTokenizer, AutoModelForCausalLM
from process_data import preprocess
import pandas as pd
import shutil
import os


def prep_evaluation_folders_public():
    output_path = 'gpt_output_public/'
    results_path = 'gpt_evaluation_public/'

    if os.path.exists(output_path):
        shutil.rmtree(output_path)
    if os.path.exists(results_path):
        shutil.rmtree(results_path)

    os.mkdir(output_path)
    os.mkdir(results_path)


def main():

    prep_evaluation_folders_public()

    special_token = '<|endoftext|>'

    test_part_prec_list = []
    test_part_recall_list = []
    test_part_f1_list = []

    best_model_list = []
    
    train_part_prec_list = []
    train_part_recall_list = []
    train_part_f1_list = []

    val_part_prec_list = []
    val_part_recall_list = []
    val_part_f1_list = []

    model_list = []
    dataset_list = []
    num_step_list = []
    for steps in range(910, 18201, 910):
        num_step_list.append(steps)

    best_partial_f1 = 0.0
    best_num_steps = 0

    for num_steps in num_step_list:
        # load model and tokenizer
        model_path = f'FineTunedModels/inspec_kdd_first_only_80_percent_epochs_200/checkpoint-{num_steps}'

        tokenizer = AutoTokenizer.from_pretrained(model_path, use_fast=False)
        model = AutoModelForCausalLM.from_pretrained(model_path)

        # load train and validation data
        train_dataset_path = f'TrainingDatasets/train_inspec_kdd_first_only_80_percent.txt'
        val_dataset_path = f'ValidationDatasets/val_inspec_kdd_first_only_10_percent.txt'

        with open(train_dataset_path, 'r') as filepath:
            train_data = filepath.read()

        with open(val_dataset_path, 'r') as filepath:
            val_data = filepath.read()

        train_data = train_data.split(special_token)
        val_data = val_data.split(special_token)

        train_df_path = f'Dataframes/train_inspec_kdd_first_only.pkl'
        val_df_path = f'Dataframes/val_inspec_kdd_first_only.pkl'
        train_df = pd.read_pickle(train_df_path)
        val_df = pd.read_pickle(val_df_path)
        train_gold_kps = list(train_df.eval_processed_keyphrases)
        val_gold_kps = list(val_df.eval_processed_keyphrases)

        # generate keyphrases
        train_generated_keyphrases = []
        for data_index in range(0, len(train_data) - 1):
            # generate keyphrase(s)
            input_data = train_data[data_index]
            list_result = generate_keyphrases(input_data, tokenizer, model, special_token, 10, 0.8)
            train_generated_keyphrases.append(list_result)

        train_generated_keyphrases = [[preprocess(kp) for kp in kp_list] for kp_list in train_generated_keyphrases]

        val_generated_keyphrases = []
        for data_index in range(0, len(val_data) - 1):
            # generate keyphrase(s)
            input_data = val_data[data_index]
            list_result = generate_keyphrases(input_data, tokenizer, model, special_token, 10, 0.8)
            val_generated_keyphrases.append(list_result)

        val_generated_keyphrases = [[preprocess(kp) for kp in kp_list] for kp_list in val_generated_keyphrases]

        # training and validation evaluation
        train_part_prec, train_part_recall, train_part_f1 = evaluate_public(train_generated_keyphrases, train_gold_kps)
        val_part_prec, val_part_recall, val_part_f1 = evaluate_public(val_generated_keyphrases, val_gold_kps)

        train_part_prec_list.append(train_part_prec)
        train_part_recall_list.append(train_part_recall)
        train_part_f1_list.append(train_part_f1)

        val_part_prec_list.append(val_part_prec)
        val_part_recall_list.append(val_part_recall)
        val_part_f1_list.append(val_part_f1)

        model_list.append(model_path)
        dataset_list.append(val_dataset_path)

        if val_part_f1 > best_partial_f1:
            best_partial_f1 = val_part_f1
            best_num_steps = num_steps

        # Output generated keyphrases to file
        train_output_df = pd.DataFrame()
        train_output_df['whole_document'] = train_df['whole_document']
        train_output_df['keyphrases'] = train_df['keyphrases']
        train_output_df['processed_keyphrases'] = train_df['eval_processed_keyphrases']
        train_output_df['generated_keyphrases'] = train_generated_keyphrases

        filename = f'train_inspec_kdd_first_only_steps_{num_steps}_output.csv'
        gpt_output_path = 'gpt_output_public/' + filename
        train_output_df.to_csv(gpt_output_path)

        val_output_df = pd.DataFrame()
        val_output_df['whole_document'] = val_df['whole_document']
        val_output_df['keyphrases'] = val_df['keyphrases']
        val_output_df['processed_keyphrases'] = val_df['eval_processed_keyphrases']
        val_output_df['generated_keyphrases'] = val_generated_keyphrases

        filename = f'val_inspec_kdd_first_only_steps_{num_steps}_output.csv'
        gpt_output_path = 'gpt_output_public/' + filename
        val_output_df.to_csv(gpt_output_path)

    # plot the learning curves
    plt.plot(num_step_list, train_part_prec_list, label='Training')
    plt.plot(num_step_list, val_part_prec_list, label='Validation')
    plt.xlabel('Number of steps')
    plt.ylabel('Partial Precision')
    plt.title('Learning Curve (Partial Precision)')
    plt.legend()
    plt.savefig(f'gpt_evaluation_public/inspec_kdd_first_only_precision_learning_curve.pdf')
    plt.close()

    plt.plot(num_step_list, train_part_recall_list, label='Training')
    plt.plot(num_step_list, val_part_recall_list, label='Validation')
    plt.xlabel('Number of Steps')
    plt.ylabel('Partial Recall')
    plt.title('Learning Curve (Partial Recall)')
    plt.legend()
    plt.savefig(f'gpt_evaluation_public/inspec_kdd_first_only_recall_learning_curve.pdf')
    plt.close()

    plt.plot(num_step_list, train_part_f1_list, label='Training')
    plt.plot(num_step_list, val_part_f1_list, label='Validation')
    plt.xlabel('Number of Steps')
    plt.ylabel('Partial F1')
    plt.title('Learning Curve (Partial F1)')
    plt.legend()
    plt.savefig(f'gpt_evaluation_public/inspec_kdd_first_only_f1_learning_curve.pdf')
    plt.close()

    # load the test dataset
    # load train and validation data
    test_dataset_path = f'ValidationDatasets/test_inspec_kdd_first_only_10_percent.txt'

    with open(test_dataset_path, 'r') as filepath:
        test_data = filepath.read()

    test_data = test_data.split(special_token)

    test_df_path = f'Dataframes/test_inspec_kdd_first_only.pkl'
    test_df = pd.read_pickle(test_df_path)
    test_gold_kps = list(test_df.eval_processed_keyphrases)

    # evaluate best model on the test dataset
    best_model_path = f'FineTunedModels/inspec_kdd_first_only_80_percent_steps_{best_num_steps}'
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

    test_part_prec, test_part_recall, test_part_f1 = evaluate_public(test_generated_keyphrases, test_gold_kps)

    # Output generated keyphrases to file
    test_output_df = pd.DataFrame()
    test_output_df['whole_document'] = test_df['whole_document']
    test_output_df['keyphrases'] = test_df['keyphrases']
    test_output_df['processed_keyphrases'] = test_df['eval_processed_keyphrases']
    test_output_df['generated_keyphrases'] = test_generated_keyphrases

    filename = f'test_inspec_kdd_first_only_steps_{best_num_steps}_output.csv'
    gpt_output_path = 'gpt_output_public/' + filename
    test_output_df.to_csv(gpt_output_path)

    test_part_prec_list.append(test_part_prec)
    test_part_recall_list.append(test_part_recall)
    test_part_f1_list.append(test_part_f1)

    evaluation_df = pd.DataFrame()
    evaluation_df['Model'] = best_model_list
    evaluation_df['Dataset'] = ['oli-intro-bio', 'oli-gen-chem']

    evaluation_df['Test Partial Precision'] = test_part_prec_list
    evaluation_df['Test Partial Recall'] = test_part_recall_list
    evaluation_df['Test Partial F1 Score'] = test_part_f1_list

    gpt_evaluation_path = 'gpt_evaluation_public/GPT2-finetuned-models-evaluation-num-steps-public.csv'
    evaluation_df.to_csv(gpt_evaluation_path)


if __name__ == '__main__':
    main()
