#from sentence_transformers import SentenceTransformer
#from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.decomposition import PCA
from numpy import linalg as lg
import numpy as np
import matplotlib.pyplot as plt

np.random.seed(10)

#vectors = np.random.randint(15, size=(3,5))
#print('Input (3 x 5) => 3 samples, 5 dimensions\n',vectors)

vectors = np.random.randint(40, size=(10,4))
#print('Input (10 x 4) => 10 samples, 4 dimensions\n',vectors)

def library_pca(A):
	percentage = None	
	pca = PCA(n_components = percentage)
	#transformed_vectors = pca.fit(vectors)
	transformed_vectors = pca.fit_transform(vectors)
	
	components = pca.components_.T
	print('components\n', components)
	
	explained_variance_ratio = pca.explained_variance_ratio_
	print('explained_variance_ratio\n', explained_variance_ratio)
	
	# print('Variance of components', np.var(components, axis=0)) 	#Does not make sense
	
	variance = np.var(transformed_vectors, axis = 0)
	print('\nPCA (variance retained = {})\n{}'.format(percentage,transformed_vectors))
	print('Variance', variance)
	print('Variance percentage: ',variance/np.sum(variance))
	print('Cumulative variance percentage: ', np.cumsum(variance)/np.sum(variance))



	# pca = PCA(n_components = 2)
	# transformed_vectors = pca.fit_transform(vectors)
	# print('\nOutput (for n_components = 2)\n',transformed_vectors)

	# pca = PCA(n_components = 1)
	# transformed_vectors = pca.fit_transform(vectors)
	# print('\nOutput (for n_components = 1)\n',transformed_vectors)

	return transformed_vectors


def manual_pca(X):
	# 0-centering X
	X = X - np.mean(X, axis=0)

	# Covariance matrix
	C = np.dot(np.transpose(X), X)/(X.shape[0] - 1)

	# Eigendecomposition of covariance matrix
	values, vectors = lg.eig(C)

	# Sorting eigenvectors based on corresponding eigenvalues (decreasing order)
	val_vecs = sorted(list(zip(values, vectors.T)), reverse = True)
	values, vectors = zip(*val_vecs)

	# project data
	P = np.dot(X, np.transpose(vectors))

	print('\nManual PCA\n', P)
	print('Variance',np.var(P, axis=0))
	print('eigenvalues', values)
	# plt.plot(values)
	# plt.title('Elbow plot')
	# plt.ylabel('Value')
	# plt.xlabel('k')
	# plt.show()

	
	return P

def manual_pca_svd(X):
	X = X-np.mean(X, axis=0)

	# Compute full SVD
	U, Sigma, Vh = np.linalg.svd(X, 
	  full_matrices=False, # It's not necessary to compute the full matrix of U or V
	  compute_uv=True)

	# Transform X with SVD components
	X_svd = np.dot(U, np.diag(Sigma))

	print('\nManual PCA (with SVD)\n', X_svd)
	print('Variance', np.var(X_svd, axis=0))
	print('sigma^2/(n-1)',Sigma**2/(X.shape[0]-1))
	
	return X_svd

#driver code

a=library_pca(vectors)

b= manual_pca(vectors)

c=manual_pca_svd(vectors)
