import csv 
from sklearn.metrics.pairwise import cosine_similarity
import sys

def interClusterSimilarity(encodingType, clusteringType, current_num_clusters, cluster_assignment, vectors):
	clusterGroups = [[] for _ in range(current_num_clusters)]
	for i, clusterIndex in enumerate(cluster_assignment):
		clusterGroups[clusterIndex].append(i)

	interCluster_distanceMatrix = [[None]*current_num_clusters for _ in range(current_num_clusters)]

	for X in range(len(clusterGroups)):
		clusX = clusterGroups[X]
		for Y in range(X, len(clusterGroups)):
			clusY = clusterGroups[Y]

			count = 0
			total = 0

			for v_i in clusX:				
				for v_j in clusY:
					total += cosine_similarity([vectors[v_i]], [vectors[v_j]])[0][0]
					count += 1

			interCluster_distanceMatrix[X][Y] = total/count
			print('Done',X,Y)

	import csv
	file = 'interCluster_distanceMatrix_{}_{}_{}.csv'.format(encodingType, clusteringType, current_num_clusters)
	with open(file, 'w') as f:
		writer = csv.writer(f)
		writer.writerows(interCluster_distanceMatrix) 

	sys.exit()
