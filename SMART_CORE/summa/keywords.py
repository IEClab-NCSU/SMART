from itertools import combinations as _combinations
from queue import Queue

from .pagerank_weighted import pagerank_weighted_scipy as _pagerank
from .preprocessing.textcleaner import clean_text_by_word as _clean_text_by_word
from .preprocessing.textcleaner import tokenize_by_word as _tokenize_by_word
from .commons import build_graph as _build_graph
from .commons import remove_unreachable_nodes as _remove_unreachable_nodes

WINDOW_SIZE = 2

"""Check tags in http://www.clips.ua.ac.be/pages/mbsp-tags and use only first two letters
Example: filter for nouns and adjectives:
INCLUDING_FILTER = ['NN', 'JJ']"""
INCLUDING_FILTER = ['NN', 'JJ']
EXCLUDING_FILTER = []


def _get_pos_filters():
    return frozenset(INCLUDING_FILTER), frozenset(EXCLUDING_FILTER)


def _get_words_for_graph(tokens):
    include_filters, exclude_filters = _get_pos_filters()
    if include_filters and exclude_filters:
        raise ValueError("Can't use both include and exclude filters, should use only one")

    result = []
    for word, unit in tokens.items():
        if exclude_filters and unit.tag in exclude_filters:
            continue
        if (include_filters and unit.tag in include_filters) or not include_filters or not unit.tag:
            result.append(unit.token)
    return result


def _get_first_window(split_text):
    return split_text[:WINDOW_SIZE]


def _set_graph_edge(graph, tokens, word_a, word_b):
    if word_a in tokens and word_b in tokens:
        lemma_a = tokens[word_a].token
        lemma_b = tokens[word_b].token
        edge = (lemma_a, lemma_b)

        if graph.has_node(lemma_a) and graph.has_node(lemma_b) and not graph.has_edge(edge):
            graph.add_edge(edge)


def _process_first_window(graph, tokens, split_text):
    first_window = _get_first_window(split_text)
    for word_a, word_b in _combinations(first_window, 2):
        _set_graph_edge(graph, tokens, word_a, word_b)


def _init_queue(split_text):
    queue = Queue()
    first_window = _get_first_window(split_text)
    for word in first_window[1:]:
        queue.put(word)
    return queue


def _process_word(graph, tokens, queue, word):
    for word_to_compare in _queue_iterator(queue):
        _set_graph_edge(graph, tokens, word, word_to_compare)


def _update_queue(queue, word):
    queue.get()
    queue.put(word)
    assert queue.qsize() == (WINDOW_SIZE - 1)


def _process_text(graph, tokens, split_text):
    queue = _init_queue(split_text)
    for i in range(WINDOW_SIZE, len(split_text)):
        word = split_text[i]
        _process_word(graph, tokens, queue, word)
        _update_queue(queue, word)


def _queue_iterator(queue):
    iterations = queue.qsize()
    for i in range(iterations):
        var = queue.get()
        yield var
        queue.put(var)


def _set_graph_edges(graph, tokens, split_text):
    _process_first_window(graph, tokens, split_text)
    _process_text(graph, tokens, split_text)


# modified 5/24/2021: retain the highest scoring keyword, at a minimum
def _extract_tokens(lemmas, scores, ratio, words):
    lemmas.sort(key=lambda s: scores[s], reverse=True)

    # If no "words" option is selected, the number of sentences is
    # reduced by the provided ratio, else, the ratio is ignored.
    length = len(lemmas) * ratio if words is None else words
    if int(length) == 0:
        length = 1
    return [(scores[lemmas[i]], lemmas[i],) for i in range(int(length))]


def _lemmas_to_words(tokens):
    lemma_to_word = {}
    for word, unit in tokens.items():
        lemma = unit.token
        if lemma in lemma_to_word:
            lemma_to_word[lemma].append(word)
        else:
            lemma_to_word[lemma] = [word]
    return lemma_to_word


# modified 5/19/2021: removed translation to original text
def _get_keywords_with_score(extracted_lemmas):
    """
    :param extracted_lemmas:list of tuples
    :return: dict of {keyword:score}
    """
    keywords = {}
    for score, lemma in extracted_lemmas:
        keywords[lemma] = score
    return keywords


def _strip_word(word):
    stripped_word_list = list(_tokenize_by_word(word))
    return stripped_word_list[0] if stripped_word_list else ""

# Updated 5/24/2021: only use lemmas as keywords
def _get_combined_keywords(_keywords, tokens, split_text):
    """
    :param _keywords:dict of keywords:scores
    :param split_text: list of strings
    :return: combined_keywords:list
    """
    result = []
    _keywords = _keywords.copy()
    # print('Keywords: ', _keywords)
    len_text = len(split_text)
    for i in range(len_text):
        word = _strip_word(split_text[i])
        # print('Word: ', word)
        if word in tokens:
            word = tokens[word].token
            # print('Word after retrieval: ', word)
        if word in _keywords:
            combined_word = [word]
            # print('Combined word (initial): ', combined_word)
            if i + 1 == len_text:
                result.append(word)   # appends last word if keyword and doesn't iterate
            for j in range(i + 1, len_text):
                other_word = _strip_word(split_text[j])
                # print('Other word (initial): ', other_word)
                if other_word in tokens:
                    other_word = tokens[other_word].token
                    # print('Other word (retrieval): ', other_word)
                if other_word in _keywords and other_word not in combined_word:
                    combined_word.append(other_word)
                    # print('Updated combined word: ', combined_word)
                else:
                    for keyword in combined_word:
                        _keywords.pop(keyword)
                        # print('Updated keywords: ', _keywords)
                    result.append(" ".join(combined_word))
                    # print('Updated result: ', result)
                    break
    return result


def _get_average_score(concept, _keywords):
    word_list = concept.split()
    word_counter = 0
    total = 0
    for word in word_list:
        total += _keywords[word]
        word_counter += 1
    return total / word_counter


def _format_results(_keywords, combined_keywords, split, scores):
    """
    :param _keywords:dict of keywords:scores
    :param combined_keywords:list of word/s
    """
    combined_keywords.sort(key=lambda w: _get_average_score(w, _keywords), reverse=True)
    if scores:
        return [(word, _get_average_score(word, _keywords)) for word in combined_keywords]
    if split:
        return combined_keywords
    return "\n".join(combined_keywords)


def keywords(text, ratio=1/3, words=None, language="english", split=False, scores=False, deaccent=False, additional_stopwords=None):
    if not isinstance(text, str):
        raise ValueError("Text parameter must be a Unicode object (str)!")

    # print('\n\n\nCluster of Text: ', text, '\n\n\n')
    # Gets a dict of word -> lemma
    tokens = _clean_text_by_word(text, language, deacc=deaccent, additional_stopwords=additional_stopwords)
    split_text = list(_tokenize_by_word(text))

    # Creates the graph and adds the edges
    graph = _build_graph(_get_words_for_graph(tokens))
    _set_graph_edges(graph, tokens, split_text)
    del split_text # It's no longer used

    _remove_unreachable_nodes(graph)

    # PageRank cannot be run in an empty graph.
    if len(graph.nodes()) == 0:
        return [] if split else ""

    # Ranks the tokens using the PageRank algorithm. Returns dict of lemma -> score
    pagerank_scores = _pagerank(graph)

    extracted_lemmas = _extract_tokens(graph.nodes(), pagerank_scores, ratio, words)

    keywords = _get_keywords_with_score(extracted_lemmas)

    # text.split() to keep numbers and punctuation marks, so separated concepts are not combined
    combined_keywords = _get_combined_keywords(keywords, tokens, text.split())

    return _format_results(keywords, combined_keywords, split, scores)


def get_graph(text, language="english", deaccent=False):
    tokens = _clean_text_by_word(text, language, deacc=deaccent)
    split_text = list(_tokenize_by_word(text, deacc=deaccent))

    graph = _build_graph(_get_words_for_graph(tokens))
    _set_graph_edges(graph, tokens, split_text)

    return graph
