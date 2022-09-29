from bs4 import BeautifulSoup#, Comment
import os

def parseAssessmentwithoutOptions():
    count = 0
    for root, directories, filenames in os.walk('content/'):
        for directory in directories:
            if directory == "x-oli-assessment2" or directory == "x-oli-assessment2-pool" or \
                            directory == "x-oli-inline-assessment" or directory == "high_stakes":

                workbookcontents = os.listdir(os.path.join(root, directory))

                for file in workbookcontents:

                    filepath = os.path.join(root, directory, file)

                    if os.path.isfile(filepath) and file.endswith('.xml'):

                        with open(filepath) as f:
                            soup = BeautifulSoup(f, "xml")
                        result = ''

                        all_questions = soup.findAll('body')

                        for index, question in enumerate(all_questions):
                            que = question.get_text().encode('utf-8')
                            #with open(os.path.splitext(filepath)[0] + '_Q' + str(index+1) + '.txt', 'w+') as w:
                            #    w.write(que)
                            count+=1
                        #os.remove(os.path.splitext(filepath)[0] + '.txt')
    print count

                    #else:
                        #if file.endswith('.txt'):
                            #os.remove(filepath)
                        #    else:
                    #        print filepath


parseAssessmentwithoutOptions()
print "Parsing Assessments Done..."