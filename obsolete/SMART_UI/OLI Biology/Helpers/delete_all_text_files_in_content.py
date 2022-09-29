import os
count=0
for root, directories, filenames in os.walk('../'):

    for directory in directories:
        if directory == "Assessments":
            workbookcontents = os.listdir(os.path.join(root, directory))

            for file in workbookcontents:
                filepath = os.path.join(root, directory, file)
                if file.endswith('.txt'):
                    os.remove(filepath)
                    count+=1
print count
