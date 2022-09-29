# CROSS VALIDATION
# Creating folds
import math
from read_data import read_data
from bert_finetuning_classification import get_accuracy

def cross_validation():
	texts, labels, target_names = read_data()

	n = len(texts)
	n_folds = 5
	split_size = math.ceil(n/n_folds)
	splits = []
	for i in range(n_folds):
	fold = [list(texts[i*split_size:i*split_size+split_size]), labels[i*split_size:i*split_size+split_size]]
	splits.append(fold)


	#Training n_folds times
	cv_accuracy = 0
	for i in range(n_folds):
	train_texts, valid_texts, train_labels, valid_labels = [], [], [], []
	for j in range(n_folds):
		fold_texts, fold_labels = splits[i]
		if i == j:
		valid_texts = fold_texts
		valid_labels = fold_labels
		else:
		train_texts.extend(fold_texts)
		train_labels.extend(fold_labels)

	train_labels = np.array(train_labels)
	valid_labels = np.array(valid_labels)
	cv_accuracy += get_accuracy(train_texts, valid_texts, train_labels, valid_labels)

	print('Cross-validation accuracy = ', cv_accuracy/5)

if __name__ == '__main__':
	print(cross_validation())