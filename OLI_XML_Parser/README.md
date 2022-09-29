# OLI XML Parser
    
## Overview
This parser extracts the xml files present in the courseware repository. [BeautifulSoup](#https://www.crummy.com/software/BeautifulSoup/) library has been used  for this parser.  

## Table of Contents
- [OLI XML Parser](#oli-xml-parser)
  * [Overview](#overview)
  * [How to run?](#how-to-run-)
    + [Input](#input)
    + [Output](#output)
  * [Dataset](#dataset)
    + [DataShop - OLI data mapping](#datashop---oli-data-mapping)
    + [Structure of XML files in OLI's intro_bio course](#structure-of-xml-files-in-oli-s-intro-bio-course)
      - [Instructional paragraphs](#instructional-paragraphs)
      - [Insturctional assessments](#insturctional-assessments)
      - [Referencing](#referencing)
  * [How does it work?](#how-does-it-work-)

## How to run?
To run the parser, simply run the `main_labelled.py` script from terminal as follows:
```
python main_labelled.py
```
It can take following optional argument:
`-outputFolder <a string>`: (default = `OneDrive/PASTEL/SMART/assessment_paragraph data/intro_bio/`)

### Input
It takes the courseware SVN repository as the input.

### Output
It generates two output files:
1. assessments.csv
2. paragraphs.csv

The output files contain 3 columns each: `<unit id, assessment id, assessment>`

The output of OLI XML Parser is fed to [SMART - CORE](SMART%20-%20CORE)

## Dataset
To validate our system, we used data from 
[Introduction to Biology](https://oli.cmu.edu/jcourse/lms/students/syllabus.do?section=df3e23850a0001dc518491159056b43c)
by [Open Learning Initiative(OLI)](https://oli.cmu.edu/). The location of the dataset stored on OneDrive can be found [here](https://docs.google.com/document/d/1yh_n4Y6sa8fauFSLaoUGJeLwNitquqzp-G_mG4Mote8/edit#heading=h.bo7bt5ec741c). This dataset was extracted from the OLI SVN repository. More details about the SVN repository has been explained [here](https://docs.google.com/document/d/1yh_n4Y6sa8fauFSLaoUGJeLwNitquqzp-G_mG4Mote8/edit#heading=h.7edis7b0707l).

According to their website,
```
The Open Learning Initiative (OLI) is a grant-funded group at Carnegie Mellon University, offering innovative online courses to anyone who wants to learn or teach. Our aim is to create high-quality courses and contribute original research to improve learning and transform higher education.
```
Anyone can take this course on oli.cmu.edu for free without registration.
Since this database is a private repository, we cannot share the data on Github. 

### DataShop - OLI data mapping
When applying SMART to OLI data, we often want to compare its model fit to existing data on DataShop.  For this reason, it becomes crucial to have an appropriate version of OLI course content data that corresponds with the learning data on DataShop. The following table shows this mapping. 

For the DataShop data, the value in a cell shows <project>/<dataset>, whereas for the corresponding OLI data show<course name>/<version>. 

| DataShop data | OLI data |
|-|-|
| OLI Biology/ALMAP spring 2014 DS 960 (Problem View fixed and Custom Field fixed) | intro_bio / v_1_0-prod-2014-03-06 |
| General Chemistry I/AHA Chemistry 1 v2.3 Fall 2020 | chemistry_general1/branches/v2_3 |
|  |  |

The DataShop data can be found on https://pslcdatashop.web.cmu.edu/. The OLI data access for intro_bio is explained [here](#bookmark=id.je49wc6trd4w). The OLI data access for chemistry_general is https://svn.oli.cmu.edu/svn/content/chemistry_general/chemisty_general1/.

### Structure of XML files in OLI
The parser uses following types of tags 

#### Instructional paragraphs
An xml file with `<workbook_page>` tag is identified as the file with instructional paragraphs.

Tags used to parse paragraphs: `<p>`, `<ol>`, `<ul>`

#### Insturctional assessments
If the xml file does not have the `<workbook_page>` tag, it is identified as a file with instructional assessments.

Tags used parse assessments (intro_bio): `<multiple_choice>`, `<question>`, `<ordering>`, `<short_answer>`
Additional tags used to parse assessments (chemistry_general): `<fill_in_the_blank>`, `<numeric>`, `<text>`

The assessment texts are appended with correct choices, the corresponding feedback, hint messages, and expanations.

Occasionally, some assessment xml files do not contain any assessments directly in the files. Instead, they contain references to other xml files with assessments. Such references are done using the `<pool_ref>` tags.

#### Referencing
Following tags were identified that refer to other xml files: `<xref>`, `<inline>`, `<activity>`, `<activity_link>`

## How does it work?
Unlike `main.py` that sequentially visits all the xml files in the course folder, the `main_labelled.py` uses references among xml files to reach different xml files. As  such, the obsolete and irrelevant xml files may not be reached. We can ensure that all the files for a particular course version have been parsed by running `checkMappings.py`. It will generate a csv file with a TRUE label for each `<problemName - stepName pair>` in the Student Step dataset if it is present in the generated assessment dataset.

It takes `organizations/introucdavis/organization.xml` as the version of the courseware for intro_bio and `organizations/default/organization.xml` for chemistry_general. Based on the organization in this version, the assessments and paragraphs are parsed recursively based on the references to other xml files from a given xml file.



