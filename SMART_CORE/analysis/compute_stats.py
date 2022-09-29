from statistics import mean, median, mode	
import matplotlib.pyplot as plt
from collections import Counter
import numpy as np
import csv
import matplotlib.pyplot as plt
from sklearn.metrics.pairwise import cosine_similarity, euclidean_distances


def zeros(vectors):
	total_non_0s = 0
	total_0s = 0
	value_count = 0
	average_value = 0

	for value in np.nditer(vectors):
		if  value != 0:
			total_non_0s += 1
			average_value += value
		else:
			total_0s += 1
		value_count += 1

	print('Total non-zeros', total_non_0s)
	print('Total zeros', total_0s)
	print('Average non-zeros per vector= ', float(total_non_0s)/len(vectors))
	print('Average value of the non-zero elements', float(average_value)/total_non_0s)
	print('value_count', value_count)

def distancesAmongVectors(vectors, encodingType, clusteringType):
	print('\nDistances...')
	#distances = cosine_similarity(vectors)
	distances = euclidean_distances(vectors)
	#print(distances)
	#print(type(distances)) #<type 'numpy.ndarray'>
	print('len(distances)', len(distances)) #5461

	print('saving euclidean matrix...')
	filename = 'euclideanDistances_'+encodingType+'_'+clusteringType+'.csv'
	np.savetxt(filename, distances, delimiter=",")
	print(filename)
	
	distances_list = []
	for i in range(distances.shape[0]):
		for j in range(i+1, distances.shape[1]):
			distances_list.append(distances[i][j])
	
	print('len(distances_list) = ', len(distances_list))

	print('saving distances_uniquePair...')
	# filename = 'distances_uniquePair_'+encodingType+'_'+clusteringType
	# with open(filename+'.csv', 'w') as f:
	# 	writer = csv.writer(f)
	# 	# for distance in distances_list:
	# 	# 	writer.writerow([distance])
	# 	writer.writerow(distances_list)

	# print(filename)

	#counter = Counter(distances_list)
	#print(counter)

	print('mean = ', mean(distances_list))
	print('median = ', median(distances_list))
	print('mode = ', mode(distances_list))
	print('min = ', min(distances_list))
	print('max = ', max(distances_list))

	plt.hist(distances_list, rwidth = 0.8, bins=np.arange(min(distances_list), max(distances_list) + 0.01, 0.01))
	# plt.title(filename,fontsize=12)
	plt.xlabel('Pairwise distance',fontsize=10)
	plt.ylabel('Frequency',fontsize=10)
	plt.xticks(fontsize=6)
	plt.yticks(fontsize=6)
	#if encodingType == 'bert':
	plt.xticks(np.arange(min(distances_list), max(distances_list)+0.1, 0.1))
	plt.grid()
	plt.show()

def pca_analysis(vectors, encodingType, clusteringType):
	print('encodingType = {} clusteringType = {}'.format(encodingType, clusteringType))
	from sklearn.decomposition import PCA
	pca = PCA()
	transformed_vectors = pca.fit_transform(vectors)

	components = pca.components_.T
	print('components\n', components)
	
	print('explained_variance_ratio\n', pca.explained_variance_ratio_)
	print('SUM OF EXPLAINED VARIANCE RATIO: ', np.sum(pca.explained_variance_ratio_))
	
	print('No. of components = ',len(pca.explained_variance_))

	# print('Variance of components', np.var(components, axis=0)) 	#Does not make sense
	
	# variance = np.var(transformed_vectors, axis = 0)
	# print('Variance', variance)
	# print('Variance percentage: ',variance/np.sum(variance))
	# cum_variance = np.cumsum(variance)/np.sum(variance)
	# print('Cumulative variance percentage: ', cum_variance)

	cumulative_explained_variance_ratio = 0
	for i, var in enumerate(pca.explained_variance_ratio_):
		if i == 100: break
		cumulative_explained_variance_ratio += var
		print(i+1, cumulative_explained_variance_ratio)

	
	plt.plot(pca.explained_variance_ratio_)
	plt.title('Elbow plot')
	plt.ylabel('explained_variance')
	plt.xlabel('Component')
	plt.show()

	#print('\nPCA (variance retained = {})\n{}'.format(percentage,transformed_vectors))
	
def compute_stats(vectors, encodingType, clusteringType):

	print('Running compute_stats()')

	# if encodingType != 'bert' and clusteringType == 'first':
	# 	vectors = vectors.todense()
		
	print('len(vectors)', len(vectors))
	print('vectors.shape',vectors.shape)
	
	zeros(vectors)
	distancesAmongVectors(vectors, encodingType, clusteringType)
	# pca_analysis(vectors, encodingType, clusteringType)
