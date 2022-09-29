from bs4 import BeautifulSoup, Comment
import os

count = 0

with open('the_cell.xml', 'r') as f:
    soup = BeautifulSoup(f, "xml")


for script in soup.find_all(['param', 'author']):
    script.extract()

bo = soup.findAll(['head', 'body'])

result = ""
for child in bo:
    result+= child.get_text().encode('utf-8')

with open('membrane_structure.txt', 'w+') as w:
    w.write(result)
    count += 1
