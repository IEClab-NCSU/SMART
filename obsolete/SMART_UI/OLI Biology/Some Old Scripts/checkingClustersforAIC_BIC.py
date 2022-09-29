from sklearn.metrics.pairwise import euclidean_distances
import csv
from collections import defaultdict
import pprint
import re
import numpy as np


assessment_to_cluster = defaultdict()
pattern = '([\w]+)_i1'
list_of_all_assessments = []

with open('Corrected Discrepancies in OLI Data/cluster_assessment_matches.csv', 'rU') as csvfile:
    csvreader = csv.reader(csvfile)
    for index, row in enumerate(csvreader):
        list_of_all_assessments.append(row[20])
        assessment_to_cluster[row[20]]=row[21]

print len(assessment_to_cluster)

problem_matches = 0
problem_no_matches = []
## list of problem names that are not matching
problem_name_no_match  = []

with open('KC Models/Spring 2014 DS 960/myKC.csv', 'rU') as csvfile:

    csvreader = csv.reader(csvfile)

    for index, row in enumerate(csvreader):
        if index == 0:
            continue

        ## If the row is skewed
        if row[4] == ' ("':
            if re.match(pattern, row[7]):
                problem_step_concat = row[5]+'_'+re.findall(pattern, row[7])[0]
                if problem_step_concat in assessment_to_cluster:
                    problem_matches += 1
                else:
                    print problem_step_concat
                    problem_name_no_match.append(row[5])

        ## If the row is not skewed
        else:
            if re.match(pattern, row[4]):
                problem_step_concat = row[2]+'_'+re.findall(pattern, row[4])[0]
                if problem_step_concat in assessment_to_cluster:
                    problem_matches += 1
                else:
                    print problem_step_concat
                    problem_name_no_match.append(row[2])

print set(problem_name_no_match)
print len(problem_name_no_match)
'''
still_no_matches = []
with open('KC Models/Spring 2014 DS 960/myKC.csv', 'rU') as csvfile:
    csvreader = csv.reader(csvfile)
    for index, row in enumerate(csvreader):
        #print "yes"
        if index in problem_no_matches:
            #print "yes"
            ## If the row is skewed
            if row[4] == ' ("':
                if re.match(pattern, row[7]):
                    problem_step_concat = re.findall(pattern, row[7])[0]
                    if problem_step_concat == 'q3':
                        continue
                    for file in assessment_to_cluster:
                        if problem_step_concat in file:
                            problem_matches += 1
                            #print file, problem_step_concat


            ## If the row is not skewed
            else:
                if re.match(pattern, row[4]):
                    problem_step_concat = re.findall(pattern, row[4])[0]
                    if problem_step_concat == 'q3':
                       continue
                    for file in assessment_to_cluster:
                        if problem_step_concat in file:
                            #print file, problem_step_concat
                            problem_matches += 1
'''
print problem_matches

