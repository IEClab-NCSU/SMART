import argparse
import csv
import os
import re

def folderToCsv(folder_name, filename):
	print "\nReading folder: ", folder_name
	with open (filename, 'wb') as f:
		writer = csv.writer(f)
		textId_text = []
		directory = folder_name + '/'
		folder_contents = os.listdir(directory)
		for file in folder_contents:
			filepath = os.path.join(directory, file)
			if os.path.isfile(filepath) and file.endswith('.txt'):
				with open(filepath, 'r') as f:
					content = f.read()

				content = content.decode('utf-8')
				content = re.sub(r'\n', ' ', content) #replace newline with a space (' ')
				content = re.sub(r'\t', ' ', content) #replace tab with a space (' ')
				content = re.sub(r' +', ' ', content) #replace one or more spaces with a single space(' ')
				content = re.sub(r'[^\w\d\s]', ' ', content) #replaces anything which is not [a-zA-Z0-9_], [0-9] or [ \t\n\r\f\v] with a single space (' ')
				
				textId = os.path.splitext(os.path.basename(filepath))[0]
				textId_text.append([textId, content])
				
		writer.writerows(textId_text)
	print"Created", filename

if __name__ == '__main__':
	parser = argparse.ArgumentParser()
	nerve = '/home/ieclab/OneDrive/PASTEL/SMART/Data'
	parser.add_argument('-dataFolder', default = nerve, type=str)

	args = parser.parse_args()
	dataFolder = args.dataFolder

	folderToCsv(os.path.join(dataFolder, 'Inputs/Assessments'), 'assessments.csv')
	folderToCsv(os.path.join(dataFolder, 'Inputs/Paragraphs'), 'paragraphs.csv')