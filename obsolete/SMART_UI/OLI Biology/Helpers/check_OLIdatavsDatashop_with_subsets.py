import csv
from collections import defaultdict

def correctstepname(matchfile):
    stepnamesinOLI = []
    clustername = defaultdict()
    with open('../Results/Sentence_70cluster_assessment_matches_corrected_problemname.csv', 'rU') as csvfile:
        csvreader = csv.reader(csvfile)
        for index, row in enumerate(csvreader):
            stepnamesinOLI.append(row[0])
            clustername[row[0]] = row[1]


    stepnameinDS = []
    with open('../Corrected Discrepancies in OLI Data/Model1_clst100w_nm_nmfC10old-PV-models.csv', 'rU') as csvfile:
        csvreader = csv.reader(csvfile)
        for index, row in enumerate(csvreader):
            if index == 0:
                continue
            if index == 3666:
                break
            stepnameinDS.append(row[6])



    count = 0
    print len(stepnameinDS)

    clustername_Datashop = defaultdict()
    #flag = False
    result = []
    for step in stepnameinDS:
        flag = False
        for s in stepnamesinOLI:
            if s == step:
                count += 1
                clustername_Datashop[step]=clustername[s]
                result.append([step, clustername[s]])
                flag = True
                break

        if flag == False:
            for s in stepnamesinOLI:
                if s in step:
                    count += 1
                    # print s, step
                    clustername_Datashop[step] = clustername[s]
                    result.append([step, clustername[s]])
                    flag = True
                    break

    print count
    print len(result)

    with open('../Results/Sentence_70cluster_assessment_matches_corrected_stepname.csv', 'wb') as csvfile:
        writer = csv.writer(csvfile)
        writer.writerows(result)