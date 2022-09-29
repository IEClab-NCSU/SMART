import matplotlib.pyplot as plt
import torch
from transformers import AutoTokenizer, AutoModelForCausalLM
from process_data import stem, lemmatize, remove_stopwords, preprocess
import pandas as pd
import shutil
import os
import re
from nltk import word_tokenize


def prep_evaluation_folders_oli():
    output_path = 'gpt_output_oli/'
    results_path = 'gpt_evaluation_oli/'

    if os.path.exists(output_path):
        shutil.rmtree(output_path)
    if os.path.exists(results_path):
        shutil.rmtree(results_path)

    os.mkdir(output_path)
    os.mkdir(results_path)


def calc_partial_match_stats(predicted_keyphrase_list, gold_keyphrase_list):
    predicted_keyphrases_words = [word_tokenize(text) for text in predicted_keyphrase_list]
    predicted_keyphrases_words = [item for sublist in predicted_keyphrases_words for item in sublist]
    set_pred_kp_words = set(predicted_keyphrases_words)

    target_keyphrases_words = [word_tokenize(text) for text in gold_keyphrase_list]
    target_keyphrases_words = [item for sublist in target_keyphrases_words for item in sublist]
    set_tgt_kp_words = set(target_keyphrases_words)

    common_tokens = len(list(set_pred_kp_words.intersection(set_tgt_kp_words)))

    num_predicted_words = len(list(set_pred_kp_words))
    num_target_words = len(list(set_tgt_kp_words))

    return num_predicted_words, num_target_words, common_tokens


def evaluate(pred_keyphrases, target_keyphrases, verbose=False):
    total_pred_words = 0
    total_tgt_words = 0
    total_common_words = 0

    for index in range(0, len(pred_keyphrases)):
        predicted_keyphrase_list = pred_keyphrases[index]
        predicted_keyphrase_list = [remove_stopwords(keyphrase) for keyphrase in predicted_keyphrase_list]
        predicted_keyphrase_list = [stem(keyphrase) for keyphrase in predicted_keyphrase_list]

        gold_keyphrase_list = target_keyphrases[index]
        gold_keyphrase_list = [remove_stopwords(keyphrase) for keyphrase in gold_keyphrase_list]
        gold_keyphrase_list = [stem(keyphrase) for keyphrase in gold_keyphrase_list]
        gold_keyphrase_list = [re.sub(r"[!.?]", "", keyphrase) for keyphrase in gold_keyphrase_list]

        num_pred_words, num_tgt_words, num_common_words = calc_partial_match_stats(predicted_keyphrase_list,
                                                                                   gold_keyphrase_list)
        total_pred_words += num_pred_words
        total_tgt_words += num_tgt_words
        total_common_words += num_common_words

    overall_precision_partial = total_common_words / total_pred_words
    overall_recall_partial = total_common_words / total_tgt_words
    if overall_precision_partial + overall_recall_partial > 0:
        overall_f1_partial = 2 * (overall_precision_partial * overall_recall_partial) / (overall_precision_partial
                                                                                         + overall_recall_partial)
    else:
        overall_f1_partial = 0.0

    if verbose:
        print('Overall Precision, Recall, and F1 (Partial Match): {:.4f} {:.4f} {:.4f}'.format(
            overall_precision_partial, overall_recall_partial, overall_f1_partial))
        print('AKE algorithm evaluation is complete.')

    return overall_precision_partial, overall_recall_partial, overall_f1_partial


def generate_keyphrases(input_data, tokenizer, model, special_token, beams, div_penalty):
    prompt_text = input_data.split('Keyphrases:')[0]
    encoded_prompt_begin = tokenizer.encode(prompt_text, max_length=967, add_special_tokens=False, return_tensors='pt')
    encoded_prompt_end = tokenizer.encode('\nKeyphrases:\n', max_length=7, add_special_tokens=False,
                                          return_tensors='pt')
    encoded_prompt = torch.concat((encoded_prompt_begin, encoded_prompt_end), axis=1)
    output_sequence = model.generate(
        input_ids=encoded_prompt,
        max_new_tokens=15,
        early_stopping=True,
        repetition_penalty=1.01,
        length_penalty=0.99,
        num_beams=beams,
        do_sample=False,
        num_beam_groups=beams,
        diversity_penalty=div_penalty,
        num_return_sequences=1,
        pad_token_id=50256)
    result = tokenizer.decode(output_sequence[0])
    decoded_prompt = tokenizer.decode(encoded_prompt[0])
    result = result.replace(decoded_prompt, '')
    if special_token in result:
        print('\n------\n')
        result = result.replace(special_token, '')

    return result


def main():

    prep_evaluation_folders_oli()

    special_token = '\n<|endoftext|>\n'

    test_prec_list = []
    test_recall_list = []
    test_f1_list = []

    best_model_list = []
    
    for course in ['bio', 'chem']:
        train_prec_list = []
        train_recall_list = []
        train_f1_list = []

        val_prec_list = []
        val_recall_list = []
        val_f1_list = []

        model_list = []
        dataset_list = []
        num_steps_list = []

        if course == 'bio':
            for steps in range(10, 201, 10):
                num_steps_list.append(steps)
        elif course == 'chem':
            for steps in range(20, 401, 20):
                num_steps_list.append(steps)

        best_f1 = 0.0
        best_num_steps = 0

        for num_steps in num_steps_list:
            # load model and tokenizer
            model_path = f'FineTunedModels/{course}_10_percent_epochs_200/checkpoint-{num_steps}'

            tokenizer = AutoTokenizer.from_pretrained(model_path, use_fast=False)
            model = AutoModelForCausalLM.from_pretrained(model_path)

            # load train and validation data
            train_dataset_path = f'TrainingDatasets/train_{course}_10_percent.txt'
            val_dataset_path = f'ValidationDatasets/val_{course}_10_percent.txt'

            with open(train_dataset_path, 'r') as filepath:
                train_data = filepath.read()

            with open(val_dataset_path, 'r') as filepath:
                val_data = filepath.read()

            train_data = train_data.split(special_token)
            val_data = val_data.split(special_token)

            train_df_path = f'Dataframes/train_{course}_10_percent.pkl'
            val_df_path = f'Dataframes/val_{course}_10_percent.pkl'
            train_df = pd.read_pickle(train_df_path)
            # print(train_df.head())
            val_df = pd.read_pickle(val_df_path)
            # print(val_df.head())
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
            train_prec, train_recall, train_f1 = evaluate(train_generated_keyphrases, train_gold_kps)
            val_prec, val_recall, val_f1 = evaluate(val_generated_keyphrases, val_gold_kps)

            train_prec_list.append(train_prec)
            train_recall_list.append(train_recall)
            train_f1_list.append(train_f1)

            val_prec_list.append(val_prec)
            val_recall_list.append(val_recall)
            val_f1_list.append(val_f1)

            model_list.append(model_path)
            dataset_list.append(val_dataset_path)

            if val_f1 > best_f1:
                best_f1 = val_f1
                best_num_steps = num_steps

            # Output generated keyphrases to file
            train_output_df = pd.DataFrame()
            train_output_df['whole_document'] = train_df['whole_document']
            train_output_df['keyphrases'] = train_df['keyphrases']
            train_output_df['processed_keyphrases'] = train_df['eval_processed_keyphrases']
            train_output_df['generated_keyphrases'] = train_generated_keyphrases

            filename = f'train_{course}_steps_{num_steps}_output.csv'
            gpt_output_path = 'gpt_output_oli/' + filename
            train_output_df.to_csv(gpt_output_path)

            val_output_df = pd.DataFrame()
            val_output_df['whole_document'] = val_df['whole_document']
            val_output_df['keyphrases'] = val_df['keyphrases']
            val_output_df['processed_keyphrases'] = val_df['eval_processed_keyphrases']
            val_output_df['generated_keyphrases'] = val_generated_keyphrases

            filename = f'val_{course}_steps_{num_steps}_output.csv'
            gpt_output_path = 'gpt_output_oli/' + filename
            val_output_df.to_csv(gpt_output_path)

        # plot the learning curves
        plt.plot(num_steps_list, train_prec_list, label='Training')
        plt.plot(num_steps_list, val_prec_list, label='Validation')
        plt.xlabel('Number of Steps')
        plt.ylabel('Partial Precision')
        plt.title('Learning Curve (Partial Precision)')
        plt.legend()
        plt.savefig(f'gpt_evaluation_oli/{course}_precision_learning_curve.pdf')
        plt.close()

        plt.plot(num_steps_list, train_recall_list, label='Training')
        plt.plot(num_steps_list, val_recall_list, label='Validation')
        plt.xlabel('Number of Steps')
        plt.ylabel('Partial Recall')
        plt.title('Learning Curve (Partial Recall)')
        plt.legend()
        plt.savefig(f'gpt_evaluation_oli/{course}_recall_learning_curve.pdf')
        plt.close()

        plt.plot(num_steps_list, train_f1_list, label='Training')
        plt.plot(num_steps_list, val_f1_list, label='Validation')
        plt.xlabel('Number of Steps')
        plt.ylabel('Partial F1')
        plt.title('Learning Curve (Partial F1)')
        plt.legend()
        plt.savefig(f'gpt_evaluation_oli/{course}_f1_learning_curve.pdf')
        plt.close()

        # load the test dataset
        # load train and validation data
        test_dataset_path = f'ValidationDatasets/test_{course}_80_percent.txt'

        with open(test_dataset_path, 'r') as filepath:
            test_data = filepath.read()

        test_data = test_data.split(special_token)

        test_df_path = f'Dataframes/test_{course}_80_percent.pkl'
        test_df = pd.read_pickle(test_df_path)
        test_gold_kps = list(test_df.eval_processed_keyphrases)

        # evaluate best model on the test dataset
        best_model_path = f'FineTunedModels/{course}_10_percent_epochs_200/checkpoint-{best_num_steps}'
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

        test_prec, test_recall, test_f1 = evaluate(test_generated_keyphrases, test_gold_kps)

        # Output generated keyphrases to file
        test_output_df = pd.DataFrame()
        test_output_df['whole_document'] = test_df['whole_document']
        test_output_df['keyphrases'] = test_df['keyphrases']
        test_output_df['processed_keyphrases'] = test_df['eval_processed_keyphrases']
        test_output_df['generated_keyphrases'] = test_generated_keyphrases

        filename = f'test_{course}_steps_{best_num_steps}_output.csv'
        gpt_output_path = 'gpt_output_oli/' + filename
        test_output_df.to_csv(gpt_output_path)

        test_prec_list.append(test_prec)
        test_recall_list.append(test_recall)
        test_f1_list.append(test_f1)

    evaluation_df = pd.DataFrame()
    evaluation_df['Model'] = best_model_list
    evaluation_df['Dataset'] = ['oli-intro-bio', 'oli-gen-chem']

    evaluation_df['Test Precision'] = test_prec_list
    evaluation_df['Test Recall'] = test_recall_list
    evaluation_df['Test F1 Score'] = test_f1_list

    gpt_evaluation_path = 'gpt_evaluation_oli/GPT2-finetuned-models-evaluation-num-steps-oli.csv'
    evaluation_df.to_csv(gpt_evaluation_path)


if __name__ == '__main__':
    main()
