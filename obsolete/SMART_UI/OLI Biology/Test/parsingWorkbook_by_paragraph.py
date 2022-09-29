from bs4 import BeautifulSoup
with open('membrane_structure.xml') as f:
    soup = BeautifulSoup(f, "xml")

for script in soup.find_all(['param', 'author']):
    script.extract()

result = ""
for index, child in enumerate(soup.findAll(['p', 'ol'])):
    result = child.get_text().encode('utf-8')

    with open('membrane_structure' + str(index + 1) + '.txt', 'w+') as w:
        w.write(result)