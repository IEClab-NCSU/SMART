import csv
from collections import defaultdict
import os

def correctstepname(matchfile):
    stepnamesinOLI = []
    clustername = defaultdict()
    with open(matchfile, 'rU') as csvfile:
        csvreader = csv.reader(csvfile)
        for index, row in enumerate(csvreader):
            stepnamesinOLI.append(row[0])
            clustername[row[0]] = row[1]

    stepnameinDS = []
    with open('Corrected Discrepancies in OLI Data/Model1_clst100w_nm_nmfC10old-PV-models.csv', 'rU') as csvfile:
        csvreader = csv.reader(csvfile)
        for index, row in enumerate(csvreader):
            if index == 0:
                continue
            if index == 3666:
                break
            stepnameinDS.append(row[6])

    print len(stepnameinDS)

    clustername_Datashop = defaultdict()
    flag = False
    result = []
    for step in stepnameinDS:
        flag = False
        for s in stepnamesinOLI:
            if s == step:
                clustername_Datashop[step]=clustername[s]
                result.append([step, clustername[s]])
                flag = True
                break

        if not flag:
            for s in stepnamesinOLI:
                if s in step:
                    # print s, step
                    clustername_Datashop[step] = clustername[s]
                    result.append([step, clustername[s]])
                    flag = True
                    break

    print "The number of assessment items in OLI that could not be matched is .. ", len(stepnameinDS)-len(result)
    with open('Results/'+os.path.basename(matchfile)+'_corrected.csv', 'wb') as csvfile:
        writer = csv.writer(csvfile)
        writer.writerows(result)
