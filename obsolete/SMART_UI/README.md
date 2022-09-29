# SMART (Skill Model mining with Automated detection of Resemblance among Texts)


### Table of Contents

* [Overview](#overview)
* [Website](#website)
* [Technical Details](#technical-details)
* [Data](#data)
* [Python Libraries](#python-libraries)
* [Acknowledgments](#acknowledgments)

### Overview

The goal of this project is to extract a Skill Model from Massive Open Online Courses (MOOC) and to map
a skill with each assessment item (question) as well as a text item (paragraph) in the course.
Nowadays, there are a lot of MOOCs available for learners to consider and learn from. Therefore,
it becomes very important to let the machines themselves learn
important skills associated with each assessment item, without any 
supervision from the tutor. This application can be used to provide
automatic feedback and hints to the learner of the course. Hence, 
propose a new system which creates associations between assessment
items and workbook contents (topics).

### Website


### Technical Details
1. **Parsing XML files**: First, the xml files from the data is parsed and converted into .txt files. [BeautifulSoup](#https://www.crummy.com/software/BeautifulSoup/) library has been used  for this purpose.  
2. **Lower casing and removing punctuations**: All the words in the text files is converted into lower case so 
same words are not treated differently because of their casing.
2. **Term Frequency**: Term Frequency is calculated for all the words within each text item as well as assessment item.
3. **Cosine Similarity**: Cosine Similarity is then calculated for each text item and an assessment item. The pairings with maximum cosine 
similarity means they are most associated with each other.
4. **K Means Clustering**: Instead of using each text item separately, we can also cluster
multiple text items using [K means clustering](#https://en.wikipedia.org/wiki/K-means_clustering).
These clusters are hypothesised to be the skills associated with the online course. 

5. **Keyword Extraction using TextRank**: [Keyword Extraction](#https://en.wikipedia.org/wiki/Keyword_extraction) is performed using [TextRank](#https://web.eecs.umich.edu/~mihalcea/papers/mihalcea.emnlp04.pdf). These 
keywords are hypothesised to represent the name of the skill (cluster) that they represent.
6. **Merging Clusters based on Common Keywords**: To find the optimal number of skills taught in the course, skills (clusters) are merged together based on the common 
keywords extracted from each. So, if two clusters have the same two keywords, then they are merged together. 

### Data
To validate our system, we used data from 
[Introduction to Biology](https://oli.cmu.edu/jcourse/lms/students/syllabus.do?section=df3e23850a0001dc518491159056b43c)
by [Open Learning Initiative(OLI)](https://oli.cmu.edu/). According to their website (insert link here), 
"The Open Learning Initiative (OLI) is a grant-funded group at Carnegie Mellon University, offering innovative online 
courses to anyone who wants to learn or teach. Our aim is to create high-quality courses and contribute original 
research to improve learning and transform higher education." 
Anyone can take this course on oli.cmu.edu for free without registration.
Since this database is a private repository, we 
cannot share the data on Github. 

### Python Libraries

1. [BeautifulSoup](https://pypi.python.org/pypi/beautifulsoup4) (Parsing XML Files)
```
pip install beautifulsoup4
```
2. [lxml](https://pypi.python.org/pypi/lxml/3.8.0) (Processing XML Files)
```
pip install lxml
```
3. [Sklearn](https://pypi.python.org/pypi/scikit-learn/0.18.1) (TF-IDF Vectorizer, K Means Clustering)
```
pip install -U scikit-learn
```
4. [summa](https://pypi.python.org/pypi/summa/0.0.7) (Keyword Extraction using TextRank)
```
pip install summa
```
5. [Networkx](https://pypi.python.org/pypi/networkx/) (Building graph for TextRank)
```
pip install networkx
```
6. [Pattern](https://pypi.python.org/pypi/Pattern) (Part-of-speech Tagging for Keyword Extraction)
```
pip install pattern
```

