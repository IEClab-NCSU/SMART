import re

my_file = open('unique_skills_list.txt')
all_the_lines = my_file.readlines()
items = []
for i in all_the_lines:
    i = re.sub(r'\n', '', i)
    i = re.sub(r'\r', '', i)
    if i not in items:
    	items.append(i)

print "Unique KCs"
print len(items)
combined_words = []

for j in items:
	if " " in j:
		combined_words.append(j)

print "Unique Compound KCs"
print len(combined_words)

map = {}
new_list = []
for k in items:
	for l in combined_words:
		if k != l and k != '' and k in l:
			x = re.search(" " + k + "$", l)
			y = re.search(k + " ", l)
			z = re.search(" " + k + " ", l)
			new_list = []
			if (x != None or y != None or z != None):
				#print k
				if l in map.keys():
					new_list = map[l]
					if (k not in new_list):
						new_list.append(k)
						map[l] = new_list
				else:
					new_list.append(k)
					map[l] = new_list


#print items
#print map
print "Unique Compound KCs with Partial KCs"
print len(map)
