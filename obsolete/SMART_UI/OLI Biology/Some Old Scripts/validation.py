from sklearn.metrics.pairwise import euclidean_distances
import csv
from collections import defaultdict
import pprint
import re
import numpy as np

def printall():
    print "The number of skills is .... ", len(set(skills))
    print "The number of clusters is ....", len(set(clusters))
    print "Size of Skill to assessment matrix is ... ", np.array(skill_to_assessment_matrix).shape
    print "Size of Cluster to assessment matrix is ... ", np.array(cluster_to_assessment_matrix).shape
    #print cluster_to_assessment_matrix[103]

pattern = 'q([0-9])+_'

skill_to_assessment_dict = defaultdict(lambda: [])
skills = []
count = 0

list_of_all_assessments = []

with open('Results/cluster_assessment_matches.csv', 'rU') as csvfile:
    csvreader = csv.reader(csvfile)
    for index, row in enumerate(csvreader):
        list_of_all_assessments.append(row[0])


with open('KC Models/Spring 2014 DS 960/cls100tx.csv', 'rU') as csvfile:
    csvreader = csv.reader(csvfile)
    for index, row in enumerate(csvreader):
        if index == 0:
            continue
        question_no = re.findall(pattern, row[4])
        if re.match('k[0-9]+', row[16]) and question_no != []:
            assessment_item = row[2] + '_Q' + question_no[0] + '.txt'
            if assessment_item in list_of_all_assessments:
                skills.append(row[16])
                skill_to_assessment_dict[row[16]].append(assessment_item)



index_of_assessment = defaultdict()
index_of_skill = defaultdict()
index = 0
sindex = 0
for skill in skill_to_assessment_dict:
    index_of_skill[skill]=sindex
    for a in skill_to_assessment_dict[skill]:
        if a not in index_of_assessment:
            index_of_assessment[a] = index
            index += 1
    sindex += 1


skill_to_assessment_matrix = [[0 for x in range(len(index_of_assessment))] for y in range(len(set(skills)))]

for skill in skill_to_assessment_dict:
    for assessment_item in skill_to_assessment_dict[skill]:
        skill_to_assessment_matrix[index_of_skill[skill]][index_of_assessment[assessment_item]] = 1

clusters = []
cluster_to_assessment_dict = defaultdict(lambda: [])
with open('Results/cluster_assessment_matches.csv', 'rU') as csvfile:
    csvreader = csv.reader(csvfile)
    for index, row in enumerate(csvreader):
        clusters.append(row[1])
        assessment_item = row[0]
        cluster_to_assessment_dict[row[1]].append(assessment_item)

index_of_cluster=defaultdict()
sindex = 0
for cluster in cluster_to_assessment_dict:
    index_of_cluster[cluster]=sindex
    sindex += 1
cluster_to_assessment_matrix = [[0 for x in range(len(index_of_assessment))] for y in range(len(set(clusters)))]

for cluster in cluster_to_assessment_dict:
    for assessment_item in cluster_to_assessment_dict[cluster]:
        if assessment_item in index_of_assessment:
            cluster_to_assessment_matrix[index_of_cluster[cluster]][index_of_assessment[assessment_item]] = 1

printall()

for i in range(np.array(cluster_to_assessment_matrix).shape[0]):
    for j in range(np.array(cluster_to_assessment_matrix).shape[1]):
        if j == 0 and cluster_to_assessment_matrix[i][j] == 1:
            print "This cluster", i


ED = euclidean_distances(cluster_to_assessment_matrix, skill_to_assessment_matrix)

#if cluster_to_assessment_matrix[130] == cluster_to_assessment_matrix[131]:
    #print "yes"
print skill_to_assessment_matrix[1]
print cluster_to_assessment_matrix[1]
#print ED
skill_list = []
for i in range(ED.shape[0]):
    distance_of_this_cluster_to_closest_skill = np.min(ED[i])
    print distance_of_this_cluster_to_closest_skill
    closest_skill = np.argmin(ED[i])
    if distance_of_this_cluster_to_closest_skill == 0:
        print "Perfect match" , i, closest_skill
    skill_list.append(closest_skill)
print skill_list