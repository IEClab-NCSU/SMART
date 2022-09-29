import csv

l4088 = []
with open('../Results/Archives/70cluster_assessment_matches.csv', 'r') as f:
    csvreader = csv.reader(f)
    for index, row in enumerate(csvreader):
        l4088.append(row[0])

import os
l4078 = []
count = 0
for root, directories, filenames in os.walk('../'):
    for directory in directories:
        if directory == "Assessments":
            workbookcontents = os.listdir(os.path.join(root, directory))
            for file in workbookcontents:
                filepath = os.path.join(root, directory, file)
                l4078.append(os.path.basename(os.path.splitext(filepath)[0]))

print l4078
print l4088
print len(set(l4088))

import collections
print [item for item, count in collections.Counter(l4088).items() if count > 1]