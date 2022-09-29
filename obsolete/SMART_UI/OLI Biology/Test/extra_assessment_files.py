with open('assessment_names', 'r') as f:
    name1 = f.read()
    name1_list = name1.split("\n")
    print name1_list
with open('assessment_names2', 'r') as f:
    name2 = f.read()
    name2_list = name2.split("\n")
    print name2_list

print set([x for x in name1_list if name1_list.count(x) > 1])