import shutil
import os
import csv


def prep_folders():
    output_path = './output/'
    temp_path = output_path + 'temp/'
    results_path = output_path + 'ake_results/'
    evaluation_path = output_path + 'evaluation/'
    ae_path = output_path + 'ae_learning_curves/'
    if os.path.exists(output_path):
        shutil.rmtree(output_path)

    os.mkdir(output_path)
    os.mkdir(temp_path)
    os.mkdir(results_path)
    os.mkdir(evaluation_path)
    os.mkdir(ae_path)
    for dataset_name in ['inspec', 'kdd', 'oli-intro-bio', 'oli-gen-chem']:
        evaluation_filepath = './output/evaluation/{}_evaluation.csv'.format(dataset_name)
        header = ['Dataset', 'Algorithm', 'Candidate Keyphrase Generation', 'Embedding', 'Ranking', 'Selection', 'N', 'Autoencoder Status', 'Trial']
        header += ['Precision (Exact)', 'Recall (Exact)', 'F1 (Exact)', 'Precision (Partial)', 'Recall (Partial)', 'F1 (Partial)']

        with open(evaluation_filepath, 'w', newline='') as file:
            write = csv.writer(file)
            write.writerow(header)


if __name__ == '__main__':
    prep_folders()


