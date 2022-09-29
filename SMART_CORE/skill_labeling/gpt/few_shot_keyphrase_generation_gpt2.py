from process_data import preprocess
from keyphrase_generation_gpt_oli import evaluate
from transformers import AutoTokenizer, AutoModelForCausalLM
import pandas as pd
import shutil
import os
import random


def prep_evaluation_folders():
    output_path = 'gpt_j_output_few_shot/'
    results_path = 'gpt_j_evaluation_few_shot/'

    if os.path.exists(output_path):
        shutil.rmtree(output_path)
    if os.path.exists(results_path):
        shutil.rmtree(results_path)

    os.mkdir(output_path)
    os.mkdir(results_path)


def generate_keyphrase(input_data, model, tokenizer, beams, div_penalty):
    special_token = '<|endoftext|>'

    encoded_prompt = tokenizer.encode(input_data, add_special_tokens=False, return_tensors='pt')

    # print(f'Encoded Prompt:\n{tokenizer.decode(encoded_prompt[0])}')

    output_sequence = model.generate(
        input_ids=encoded_prompt,
        max_new_tokens=15,
        early_stopping=True,
        num_beams=beams,
        do_sample=False,
        num_beam_groups=beams,
        diversity_penalty=div_penalty,
        num_return_sequences=1,
        pad_token_id=50256)
    result = tokenizer.decode(output_sequence[0])
    # print(f'Result: {result}')
    decoded_prompt = tokenizer.decode(encoded_prompt[0])
    result = result.replace(decoded_prompt, '')
    if special_token in result:
        # print('\n------\n')
        result = result.replace(special_token, '')
    # print(f'Processed Result: {result}')
    result = preprocess(result)

    return result


def main():

    prep_evaluation_folders()

    # 'EleutherAI/gpt-j-6B'
    model = AutoModelForCausalLM.from_pretrained('EleutherAI/gpt-j-6B')
    tokenizer = AutoTokenizer.from_pretrained('EleutherAI/gpt-j-6B')

    special_token = '\n<|endoftext|>\n'

    for course in ['bio', 'chem']:

        # retrieve the dataframe
        if course == 'bio':
            df = pd.read_pickle('Dataframes/oli_intro_bio_dataset_preprocessed.pkl')
        else:
            df = pd.read_pickle('Dataframes/oli_gen_chem_dataset_preprocessed.pkl')

        # load data
        dataset_path = f'ValidationDatasets/test_{course}_full.txt'

        with open(dataset_path, 'r') as filepath:
            data = filepath.read()

        data = data.split(special_token)
        # data = data[:-1]

        gold_kps = list(df.eval_processed_keyphrases)

        # For 10 iterations, randomly select a sample for one-shot example, generate the keyphrases, and evaluate
        # performance (average across iterations)
        precision_list = []
        recall_list = []
        f1_list = []
        course_list = []
        example = ''

        for num_shots in [3, 2]:
            random_numbers = random.sample(range(0, len(data)), num_shots)
            # print(random_numbers)
            for sample_index in random_numbers:
                # get the one-shot example
                example += data[sample_index] + special_token

            # for all others, craft input and generate the associated keyphrase
            new_data = data.copy()
            new_gold_kps = gold_kps.copy()
            for index in sorted(random_numbers, reverse=True):
                del new_data[index]
                del new_gold_kps[index]
            new_df = df.drop(index=random_numbers)

            generated_keyphrases = []

            # print(f'Example Keyphrases: {example}')
            print(f'Beginning {num_shots}-shot skill label generation for {course}.')
            ten_percent = int(len(new_data)/10)

            for data_index in range(0, len(new_data)):
                if data_index % ten_percent == 0:
                    print(f'{data_index/len(new_data) * 100}% complete.')
                # craft the input
                input_data = new_data[data_index]
                prompt_text = input_data.split('\nKeyphrases:\n')[0] + '\nKeyphrases:\n'
                # gold = input_data.split('\nKeyphrases:\n')[1]

                total_text = example + prompt_text
                # print(f'Total Text:\n{total_text}')

                # print(f'Keyphrase to Generate: {prompt_text}')

                result = generate_keyphrase(total_text, model, tokenizer, 10, 0.8)
                # print(f'Generated Keyphrase: {result}')
                # print(f'Target Keyphrase: {gold}')
                generated_keyphrases.append([result])
                # time.sleep(15)

            generated_keyphrases = [[preprocess(kp) for kp in kp_list] for kp_list in generated_keyphrases]

            # Output generated keyphrases to file
            output_df = pd.DataFrame()
            output_df['whole_document'] = new_df['whole_document']
            output_df['keyphrases'] = new_df['keyphrases']
            output_df['processed_keyphrases'] = new_df['eval_processed_keyphrases']
            output_df['generated_keyphrases'] = generated_keyphrases

            filename = f'few_shot_gpt_j_{course}_{num_shots}_shots_output.csv'
            gpt_output_path = 'gpt_j_output_few_shot/' + filename
            output_df.to_csv(gpt_output_path)

            # evaluate the performance
            prec, recall, f1 = evaluate(generated_keyphrases, gold_kps)

            precision_list.append(prec)
            recall_list.append(recall)
            f1_list.append(f1)
            course_list.append(course)

            if course == 'bio':
                course_name = 'OLI Introduction to Biology'
            else:
                course_name = 'OLI General Chemistry 1'

            print(f'Overall Partial Precision for the {course_name} course with {num_shots} shots: {prec}')
            print(f'Overall Partial Recall for the {course_name} course with {num_shots} shots: {recall}')
            print(f'Overall Partial F1 Score for the {course_name} course with {num_shots} shots: {f1}')

    '''# output the results
    overall_precision = np.mean(np.array(precision_list))
    overall_recall = np.mean(np.array(recall_list))
    overall_f1 = np.mean(np.array(f1_list))

    evaluation_df = pd.DataFrame()
    evaluation_df['Dataset'] = course_list
    evaluation_df['Example Index'] = random_numbers
    evaluation_df['Test Precision'] = precision_list
    evaluation_df['Test Recall'] = recall_list
    evaluation_df['Test F1 Score'] = f1_list

    evaluation_path = f'gpt_evaluation_oli_one_shot/GPT2-one-shot-evaluation-{course}.csv'
    evaluation_df.to_csv(evaluation_path)

    if course == 'bio':
        course_name = 'OLI Introduction to Biology'
    else:
        course_name = 'OLI General Chemistry 1'
    print(f'Overall Partial Precision for the {course_name} course: {overall_precision}')
    print(f'Overall Partial Recall for the {course_name} course: {overall_recall}')
    print(f'Overall Partial F1 Score for the {course_name} course: {overall_f1}')'''


if __name__ == '__main__':
    main()
