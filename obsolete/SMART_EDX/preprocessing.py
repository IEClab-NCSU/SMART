"""
Preprocessing of the text (questions and paragraphs)
"""

import re


def removeIfStartsWithNumber(textunit):

    if len(textunit) >= 3 and \
            textunit[0].isdigit() \
            and textunit[1] == '.':
        textunit = textunit[3:]

    return textunit


def cleanText(text):

    newcontent = []

    for textunit in text:

        if textunit[1] \
                and not (textunit[1].startswith("Answer"))\
                and textunit[1] != "":
            newtext = removeIfStartsWithNumber(textunit[1])
            newtext = newtext.replace("_", "")
            newcontent.append((textunit[0], newtext))

    return newcontent


def removeTags(text):

    newcontent = []
    for textunit in text:
        newtext = re.sub(r'\<[^>]*\>', '', textunit[1])
        newtext = unicode(newtext, errors='replace')
        newcontent.append((textunit[0], newtext))
    return newcontent


def preprocess(text):

    text = cleanText(text)
    text = removeTags(text)
    return text
