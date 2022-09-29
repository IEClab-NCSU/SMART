"""Parse iteratively for all unit ids
"""

from bs4 import BeautifulSoup
import csv
import os
import re

from parse_instruction import parse_instruction
from safe_open import safe_open_w


def parse_idrefs(organization_file, xml_path_mapping, outputFolder):
	with open(organization_file) as f:
		soup = BeautifulSoup(f, "xml")

	units = soup.findAll('unit')
	for unit in units:
		resourcerefs = unit.findAll('resourceref')
		unit_id = unit['id']
		for resourceref in resourcerefs:
			filename = resourceref['idref'] + '.xml'
			parse_instruction(unit_id, filename, outputFolder, xml_path_mapping)

#Driver (to test the code)
if __name__ == '__main__':
	filename = 'input_here'
	parse_idrefs(filename)