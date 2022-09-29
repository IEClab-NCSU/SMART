import os

count = 0
for root, directories, filenames in os.walk('../'):
    for directory in directories:
        if directory == "Assessments":
            workbookcontents = os.listdir(os.path.join(root, directory))
            for file in workbookcontents:
                filepath = os.path.join(root, directory, file)
                if os.path.isfile(filepath) and file.endswith('.txt'):
                    print os.path.basename(os.path.splitext(filepath)[0])
                    count += 1
print count
