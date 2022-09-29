with open('../Results/clusters/clusters.txt', 'r') as f:
    content = f.read()
    clusters = content.split('\n\n')
    for index, cluster in enumerate(clusters):
        if cluster == '':
            continue
        with open('../Results/clusters/cluster'+str(index+1)+'.txt', 'w') as obj:
            obj.write(cluster)
