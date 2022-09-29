This implementation of TextRank Algorithm has been taken from the source -

https://github.com/JRC1995/TextRank-Keyword-Extraction/blob/master/README.md

In SMART_CORE, we are using the summa library for the performing the text ranking of cluster
names. However, we need to understand the actual working of the TextRank algorithm and 
so, this implementation can help.

Refer for more details about the TextRank algorithm -
https://web.eecs.umich.edu/~mihalcea/papers/mihalcea.emnlp04.pdf


Setup instructions -
1. Install pip
sudo apt update
sudo apt install python-pip

2. Install nltk
sudo pip install nltk
sudo python -m nltk.downloader punkt

-----------------------------------------------------
Run on console
python text_rank.py

Output -
Keywords with the correct ranking from left to right for the text provided in the text_rank.py