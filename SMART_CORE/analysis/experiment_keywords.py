"""
Using local (modified) summa package
"""
from .summa import keywords
from .summa.preprocessing.textcleaner import clean_text_by_word as _clean_text_by_word
from .summa.preprocessing.textcleaner import tokenize_by_word as _tokenize_by_word

def getText_Textname(text):

	tokens = _clean_text_by_word(text, "english") #tokens (a Syntactic object) is a mapping (dict) of original_word to [original_word, lemma]
	lemmatized_text = []
	for word in list(_tokenize_by_word(text)):
		if word in tokens:
			lemmatized_text.append(tokens[word].token)

	lemmatized_text = (' ').join(lemmatized_text)

	return lemmatized_text, tokens

if __name__ == '__main__':
	#text = "Forty didn't Coronaviruses: are be+ alpha & bravo 2*3 = 5 a group of related R.N.A. viruses that cause diseases in 2 mammals and birds. In humans, these viruses cause respiratory tract infections that can range from mild to lethal. Mild illnesses include some cases of the common cold (which is also caused by other viruses, predominantly rhinoviruses), while more lethal varieties can cause SARS, MERS, and COVID-19. Symptoms in other species vary: in chickens, they cause an upper respiratory tract disease, while in cows and pigs they cause diarrhea. There are as yet no vaccines or antiviral drugs to prevent or treat human coronavirus infections."
	text = 'partnessying go went gone writing answering answered'
	print(text)
	print(getText_Textname(text))
	#print(keywords.keywords(text))