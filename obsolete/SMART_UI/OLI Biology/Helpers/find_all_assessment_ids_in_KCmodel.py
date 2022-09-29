import os
import csv
import re
count = 0

pattern = '([\w]+)_'
list_of_all_assessment_items_in_KCmodel = []
with open('../KC Models/Spring 2014 DS 960/myKC.csv', 'rU') as csvfile:
    csvreader = csv.reader(csvfile)
    for index, row in enumerate(csvreader):
        if index == 0:
            continue
        if re.findall(pattern, row[4]):
            list_of_all_assessment_items_in_KCmodel.append(re.findall(pattern, row[4])[0])
        else:
            if re.findall(pattern, row[7]):
                list_of_all_assessment_items_in_KCmodel.append(re.findall(pattern, row[7])[0])
            else:
                print row[4]




list_of_all_assessment_items_in_content = []

for root, directories, filenames in os.walk('../content/'):

    for directory in directories:

        if directory == "x-oli-assessment2" or directory == "x-oli-assessment2-pool" or \
                        directory == "x-oli-inline-assessment" or directory == "high_stakes":

            workbookcontents = os.listdir(os.path.join(root, directory))

            for file in workbookcontents:

                filepath = os.path.join(root, directory, file)

                if os.path.isfile(filepath) and file.endswith('.txt'):
                    #print os.path.splitext(os.path.basename(filepath))[0]
                    list_of_all_assessment_items_in_content.append(os.path.splitext(os.path.basename(filepath))[0])

print len(list_of_all_assessment_items_in_KCmodel)
print list_of_all_assessment_items_in_KCmodel

print len(list_of_all_assessment_items_in_content)
print list_of_all_assessment_items_in_content

#print len(list(list_of_all_assessment_items_in_content & list_of_all_assessment_items_in_KCmodel))

intersection  = []
for l in list_of_all_assessment_items_in_content:
    if l in list_of_all_assessment_items_in_KCmodel:
        intersection.append(l)

print len(intersection)
print intersection