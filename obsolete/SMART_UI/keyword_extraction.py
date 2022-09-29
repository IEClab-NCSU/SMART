"""
Extract Keywords from Clusters

"""

from summa import keywords


def extract_keywords(clusters):
    cluster_labels = []
    for cluster in clusters:
        if not cluster.encode('utf-8') or len(cluster.split()) < 20:
            cluster_labels.append([''])
            continue
        words = keywords.keywords(cluster.encode("utf-8"))
        keyword = words.split("\n")
        cluster_labels.append(keyword)

    print "The cluster labels are..."

    for index, label in enumerate(cluster_labels):
        print str(index+1) + ". " + str(label[:2])

    merged = []
    for index, label in enumerate(cluster_labels):
        if index in merged:
            continue
        if label[0] == '':
            merged.append(index)
        for index2, label2 in enumerate(cluster_labels[index+1:]):
            if index2 in merged:
                continue
            if len(set(label[:2]).intersection(label2[:2])) >= 2:
                merged.append(index2)

    if len(merged) > 0:
        print "\n"+str(len(merged)) + " clusters are being merged..."

    new_num_clusters = len(cluster_labels)-len(merged)

    return new_num_clusters, cluster_labels
