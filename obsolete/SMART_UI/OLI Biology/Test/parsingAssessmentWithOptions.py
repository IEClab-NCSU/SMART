from bs4 import BeautifulSoup#, Comment
import os

#filepath='content/x-oli-assessment2-pool/a2_Diffusion.xml'

for root, directories, filenames in os.walk('content/'):
    for directory in directories:
        if directory == "x-oli-assessment2" or directory == "x-oli-assessment2-pool" or \
                        directory == "x-oli-inline-assessment":

            workbookcontents = os.listdir(os.path.join(root, directory))

            for file in workbookcontents:

                filepath = os.path.join(root, directory, file)

                if os.path.isfile(filepath) and file.endswith('.xml'):

                    with open(filepath) as f:
                        soup = BeautifulSoup(f, "xml")
                    result = ''

                    x = soup.findAll('body')

                    for x1 in x:
                        y = x1.get_text().encode('utf-8')
                        result += y
                        result += '\n\n'

                    with open(os.path.splitext(filepath)[0] + '.txt', 'w+') as w:
                        w.write(result)

                else:
                    if file.endswith('.txt'):
                        continue
                    else:
                        print filepath