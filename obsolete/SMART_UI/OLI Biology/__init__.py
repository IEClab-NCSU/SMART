"""
Run Local Python Files

"""

from parsingWorkbook_by_paragraph import parse as parse_workbook
from parsingAssessmentsbyid import parse as parse_assessments
from createMapping import cluster_assessment_mapping

create_clusters= raw_input("Do you want to create clusters??\n")

if create_clusters == "yes":
    create_clusters = True
    num_clusters = raw_input("How many clusters do you want???\n")
    create_cluster_file = raw_input("Do you want to create file to store clusters\n")
else:
    create_clusters = False
    num_clusters = 0
    create_cluster_file = False

create_file = raw_input("Do you want to save the mappings into a file??\n")
use_sentence = raw_input("Do you want to use Sentence Based Clustering?? (Type 'yes' or 'no')\n")

if use_sentence == "yes":
    use_sentence = True
else:
    use_sentence = False

if create_file == "yes":
    create_file = True
else:
    create_file = False

if create_cluster_file == "yes":
    create_cluster_file = True
else:
    create_cluster_file = False

#parse_workbook()
#parse_assessments()

cluster_assessment_mapping(use_sentence, int(num_clusters), create_file, create_cluster_file, create_clusters)
