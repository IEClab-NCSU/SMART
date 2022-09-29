from collections import defaultdict
import csv
def variance_analysis(vectors, cluster_assignment, clusterIndex_to_skill):
    # print(type(vectors), vectors.shape)
    # print(type(cluster_assignment), len(cluster_assignment))
    # print(cluster_assignment)

    clusterIndex_to_vectorIndexes = defaultdict(list)
    for i, clusterIndex in enumerate(cluster_assignment):
        clusterIndex_to_vectorIndexes[clusterIndex].append(i)        

    clusteredIdx_to_vectors = dict()
    for clusterIndex, vectorIndexes in clusterIndex_to_vectorIndexes.items():
        # print(k, len(v))
        clusteredIdx_to_vectors[clusterIndex] = vectors[vectorIndexes]

    variances = vectors.var(0).flatten().tolist()
    print('type(variances)', type(variances))
    print('len(variances)', len(variances))
    # print(variances)

    # print('variances.shape', variances.shape)
    # print(variannces)
    # print('variances[0].shape', variances[0].shape)
    # print(variances[0])
    
    least10_variances_columnIdx = sorted(range(vectors.shape[1]), key=lambda i: variances[i])[:10]
    # print(least10_variances_columnIdx)

    result_matrix = []
    row0 = [None]*(len(clusterIndex_to_skill)+2)
    row0[0], row0[1]= 'dim#', 'All vectors - ' + str(len(vectors))
    for clusterIdx, skill in clusterIndex_to_skill.items():
        row0[clusterIdx+2] = skill + ' - ' + str(len(clusteredIdx_to_vectors[clusterIdx]))

    result_matrix.append(row0)

    for i in range(10):
        row = [None]*(len(clusterIndex_to_skill)+2)

        columnIdx = least10_variances_columnIdx[i]
        row[0] = columnIdx

        allVectors_var = variances[columnIdx]
        
        row[1] = allVectors_var
        for clusterIdx,cluster_vectors in clusteredIdx_to_vectors.items():
            row[clusterIdx+2] = cluster_vectors.var(0).flatten().tolist()[columnIdx]
            
        result_matrix.append(row)

    file = 'variance_analysis.csv' 
    with open(file, 'w') as f_write:
        writer = csv.writer(f_write)
        writer.writerows(result_matrix)

    

    


