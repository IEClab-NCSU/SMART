import shutil
import matplotlib.pyplot as plt
import nltk
import numpy as np
import math
from keyphrase_extractor_v2 import *
from nltk.stem import WordNetLemmatizer
from nltk.corpus import wordnet
from itertools import combinations
from scipy import stats


def valid_text(text, stop_words):
    new_text = "".join([char for char in text if char not in ['.', '!', '?']])
    new_text = [word for word in word_tokenize(new_text) if word not in stop_words]
    new_text = " ".join(new_text)
    valid = False
    for word in word_tokenize(new_text):
        if len(word) >= 2:
            valid = True
    return valid


def preprocess(text):
    # apply lowercasing
    new_text = text.lower()

    # remove punctuation and numeric characters
    new_text = re.sub(r"[^\s]+_[^\s]+", "", new_text)
    new_text = re.sub(r"[(][^)]*[)]", "", new_text)
    new_text = new_text.replace('_', '.')
    new_text = new_text.replace('-', ' ')
    new_text = re.sub(r"[^a-zA-Z\s!.?]", "", new_text)

    # remove extra whitespace
    new_text = " ".join(new_text.split())
    return new_text


def remove_stopwords(text):
    stop_words = stopwords.words('english') + ['correct', 'true', 'false', 'yes', 'following', 'mathrm']
    new_text = [word for word in word_tokenize(text) if word not in stop_words]
    new_text = " ".join(new_text)
    return new_text


def stem(text):
    ps = PorterStemmer()
    new_text = " ".join([ps.stem(word) for word in word_tokenize(text)])
    return new_text


def pos_tagger(nltk_tag):
    if nltk_tag.startswith('J'):
        return wordnet.ADJ
    elif nltk_tag.startswith('V'):
        return wordnet.VERB
    elif nltk_tag.startswith('N'):
        return wordnet.NOUN
    elif nltk_tag.startswith('R'):
        return wordnet.ADV
    else:
        return None


def lemmatize(text):
    lemmatizer = WordNetLemmatizer()
    lemmatized_sentence = []
    pos_tagged = nltk.pos_tag(word_tokenize(text))
    tagged_text = list(map(lambda x: (x[0], pos_tagger(x[1])), pos_tagged))
    for word, tag in tagged_text:
        if tag is None:
            lemmatized_sentence.append(word)
        else:
            lemmatized_sentence.append(lemmatizer.lemmatize(word, tag))
    lemmatized_text = " ".join(lemmatized_sentence)

    return lemmatized_text


def get_alt_precision_recall(pred_kp, target_kp):
    pred = lemmatize(pred_kp)
    # print(f'Original Prediction: {pred_kp}')
    # print(f'Lemmatized Prediction: {pred}')
    pred_kp_words = word_tokenize(pred)
    set_pred_kp_words = set(pred_kp_words)

    target = lemmatize(target_kp)
    # print(f'Original Target: {target_kp}')
    # print(f'Lemmatized Target: {target}')
    target_kp_words = word_tokenize(target)
    set_target_kp_words = set(target_kp_words)

    common_tokens = list(set_pred_kp_words.intersection(set_target_kp_words))
    num_common_tokens = len(common_tokens)

    num_predicted_words = len(list(set_pred_kp_words))
    num_target_words = len(list(set_target_kp_words))

    if num_predicted_words > 0:
        precision = num_common_tokens / num_predicted_words
    else:
        precision = 0.0

    if num_target_words > 0:
        recall = num_common_tokens / num_target_words
    else:
        recall = 0.0

    return precision, recall


def found_candidate(document, word_combo, stop_words):
    candidates = []
    for sentence in sent_tokenize(document):
        valid = valid_text(sentence, stop_words)
        # print(sentence)
        if valid:
            cv = CountVectorizer(ngram_range=(1, 3), stop_words=stop_words).fit([sentence])
            sent_ngrams = cv.get_feature_names()
            candidates += sent_ngrams

    combo_found = False
    for candidate in candidates:
        if word_combo[0] in candidate and word_combo[1] in candidate:
            combo_found = True

    return combo_found


def found_adjacent(document, frequent_words):
    stop_words = stopwords.words('english') + ['correct', 'true', 'false', 'yes', 'following', 'mathrm']
    # print(document, '\n')
    try:
        document_cv = CountVectorizer(ngram_range=(3, 3), stop_words=stop_words).fit([document])
        doc_ngrams = document_cv.get_feature_names()

    except:
        # print('exception')
        document_cv = CountVectorizer(ngram_range=(2, 2), stop_words=stop_words).fit([document])
        doc_ngrams = document_cv.get_feature_names()

    tgt_combos = list(combinations(frequent_words, 2))

    found = False
    for combo in tgt_combos:
        for ngram in doc_ngrams:
                if combo[0] in ngram and combo[1] in ngram:
                # if found_candidate(document, combo, stop_words=stop_words):
                    found = True
                    # print(sub_combo)
            # print('-----\n')

    return found


def get_good_precision_potential(document, target, verbose=False):
    # Inputs:
    #       document: document text (after preprocess, stopword removal, stemming)
    #       target: target keyphrase text (after preprocess, stopword removal, stemming)
    #       prediction: predicted keyphrase text (after preprocess, stopword removal, stemming)
    #
    #
    # Outputs:
    #       good_precision_potential:   Boolean on whether or not the document has potential for good precision using
    #                                   KeyBERT
    #       num_infrequent_words:       Number of words from the target keyphrase that are relatively infrequent
    #                                   in the document
    #       num_frequent_words:         Number of words from the target keyphrase that are relatively frequent
    #                                   in the document
    #       num_target_words:           Number of words in the target keyphrase

    stop_words = stopwords.words('english') + ['correct', 'true', 'false', 'yes', 'following', 'mathrm']
    pos_count_dict = {'Noun': 0, 'Adjective': 0, 'Verb': 0, 'Adverb': 0, 'Other': 0}

    document_cv = CountVectorizer(ngram_range=(1, 1), stop_words=stop_words)
    count_vector = document_cv.fit_transform([document])
    doc_term_index_dict = document_cv.vocabulary_
    doc_words = document_cv.get_feature_names()

    tgt_cv = CountVectorizer(ngram_range=(1, 1), stop_words=stop_words).fit([target])
    tgt_words = tgt_cv.get_feature_names()

    doc_term_count_dict = {}
    counts = []
    for term in doc_words:
        term_index = doc_term_index_dict[term]
        term_count = count_vector[0, term_index]
        doc_term_count_dict[term] = term_count
        counts.append(term_count)

    sorted_doc_dict = sorted(doc_term_count_dict.items(), key=lambda x: x[1], reverse=True)
    sorted_doc_words = [key for key, _ in sorted_doc_dict]

    num_doc_words = len(sorted_doc_words)
    if num_doc_words >= 10:
        threshold_index = math.ceil(num_doc_words * 0.25)
        threshold_count = sorted_doc_dict[threshold_index][1]
    else:
        threshold_index = num_doc_words - 1
        threshold_count = sorted_doc_dict[threshold_index][1]
        # threshold_index = 'Mean'
        # threshold_count = np.floor(np.mean(counts))
        # threshold_count = stats.mode(counts)[0][0]

    tgt_term_count_dict = {}
    num_infrequent_tgt_words = 0
    num_frequent_tgt_words = 0
    num_target_words = 0
    frequent_words = []
    for term in tgt_words:
        if stem(term) in doc_words:
            term_index = doc_term_index_dict[stem(term)]
            term_count = count_vector[0, term_index]
        else:
            tag = nltk.pos_tag([term])[0][1]
            # print(tag)
            if tag.startswith('J'):
                pos_count_dict['Adjective'] += 1
            elif tag.startswith('V'):
                pos_count_dict['Verb'] += 1
            elif tag.startswith('N'):
                pos_count_dict['Noun'] += 1
            elif tag.startswith('R'):
                pos_count_dict['Adverb'] += 1
            else:
                pos_count_dict['Other'] += 1
            term_count = 0

        tgt_term_count_dict[stem(term)] = term_count

        if term_count < threshold_count:
            num_infrequent_tgt_words += 1
        else:
            if stem(term) not in frequent_words:
                frequent_words.append(stem(term))
                num_frequent_tgt_words += 1
        num_target_words += 1

    good_precision_potential = False

    found_together = found_adjacent(document, frequent_words)

    if num_frequent_tgt_words >= 2 and found_together:
        good_precision_potential = True

    if verbose:
        print(f'A word is not frequent if it appears less than {threshold_count} (at {sorted_doc_dict[threshold_index][0]}) times in the document.')
        print(f'Number of words in the document: {num_doc_words}')
        print(f'Target words that are frequent in the document: {frequent_words}')
        print(f'Of the words in the document:')
        for item in sorted_doc_dict:
            print(f'\t{item[0]} : {item[1]}')

        sorted_dict = sorted(tgt_term_count_dict.items(), key=lambda x: x[1], reverse=True)
        print(f'Of the words in the target keyphrase ({target}):')
        for item in sorted_dict:
            print(f'\t{item[0]} : {item[1]}')

        print(f'The sample has {num_frequent_tgt_words} target words that are frequent in the document.')
        if num_doc_words >= 4:
            print(f'The sample has at least two target words that appear adjacent to one another in the document: {found_together}.')
        print(f'The sample has potential for good precision: {good_precision_potential}.')

    return good_precision_potential, num_infrequent_tgt_words, num_frequent_tgt_words, num_target_words, found_together, num_doc_words, pos_count_dict


def categorize_samples(orig_docs, targets, preds, course):

    print_number = 25

    count_both_good = 0
    count_good_prec_poor_recall = 0
    count_poor_prec_good_recall = 0
    count_both_poor = 0

    both_good_indices = []
    good_prec_poor_recall_indices = []
    poor_prec_good_recall_indices = []
    both_poor_indices = []

    prec_list = []
    recall_list = []

    num_poor_words_tgt_not_doc = 0
    num_poor_words_tgt = 0

    num_good_words_tgt_not_doc = 0
    num_good_words_tgt = 0

    num_unlikely_precision = 0
    num_good_lemmatized = 0

    num_lack_freq_tgt_words = 0
    num_not_found_together = 0

    good_percentages = []
    poor_percentages = []
    num_freq_list = []
    num_doc_words_list = []
    found_adj_list = []
    potential_list = []

    total_pos_count_dict = {'Noun': 0, 'Adjective': 0, 'Verb': 0, 'Adverb': 0, 'Other': 0}

    for index in range(0, len(targets)):
        # print(f'index: {index}')
        good_precision = True
        good_recall = True

        if len(preds[index]) == 1:
            pred_kp_orig = preprocess(preds[index][0])
            pred_kp = remove_stopwords(pred_kp_orig)
        else:
            pred_kp_orig = ''
            pred_kp = pred_kp_orig
        stem_pred_kp = stem(pred_kp_orig)
        pred_kp_words = word_tokenize(stem_pred_kp)
        set_pred_kp_words = set(pred_kp_words)

        target_kp_orig = preprocess(targets[index][0])
        target_kp_orig = re.sub(r"[!.?]", "", target_kp_orig)
        target_kp = remove_stopwords(target_kp_orig)
        stem_target_kp = stem(target_kp)
        target_kp_words = word_tokenize(stem_target_kp)
        set_target_kp_words = set(target_kp_words)
        list_target_kp_words = list(set_target_kp_words)

        orig_doc = preprocess(orig_docs[index])
        doc = remove_stopwords(orig_doc)
        doc = stem(doc)
        doc_words = word_tokenize(doc)
        set_orig_doc_words = set(doc_words)
        list_orig_doc_words = list(set_orig_doc_words)

        # print(index, '\n')
        if index == print_number:
            potential, _, num_freq, num_tgt_words, found_tog, num_doc_words, pos_count = get_good_precision_potential(doc, target_kp, verbose=True)
        else:
            potential, _, num_freq, num_tgt_words, found_tog, num_doc_words, pos_count = get_good_precision_potential(doc, target_kp)
        num_freq_list.append(num_freq)
        num_doc_words_list.append(num_doc_words)
        found_adj_list.append(found_tog)
        potential_list.append(potential)
        for key in ['Noun', 'Adjective', 'Verb', 'Adverb', 'Other']:
            total_pos_count_dict[key] += pos_count[key]

        if num_tgt_words >= 10 and num_freq < 2:
            num_lack_freq_tgt_words += 1
            # print(f'Less than 2 Frequent Target Words index: {index}')
        elif num_tgt_words < 10 and num_freq < 1:
            num_lack_freq_tgt_words += 1
            # print(f'Less than 1 Frequent Target Words index: {index}')
        elif not found_tog:
            num_not_found_together += 1
            # print(f'Not found together index: {index}')

        common_tokens = list(set_pred_kp_words.intersection(set_target_kp_words))
        num_common_tokens = len(common_tokens)

        num_predicted_words = len(list(set_pred_kp_words))
        num_target_words = len(list(set_target_kp_words))

        if num_predicted_words > 0:
            precision = num_common_tokens / num_predicted_words
        else:
            precision = 0.0

        if num_target_words > 0:
            recall = num_common_tokens / num_target_words
        else:
            recall = 0.0

        prec_list.append(precision)
        recall_list.append(recall)

        if precision < 0.66:
            good_precision = False
        if recall < 0.66:
            good_recall = False

        # print(f'Good Precision: {good_precision}, Good Recall: {good_recall}')

        if good_precision and good_recall:
            count_both_good += 1
            both_good_indices.append(index)

            # examine the percentage of words in target in orig. document
            num_good_words_tgt += num_target_words

            list_tgt_not_doc = [word for word in list_target_kp_words if word not in list_orig_doc_words]
            num_words_tgt_not_doc = len(list_tgt_not_doc)
            num_good_words_tgt_not_doc += num_words_tgt_not_doc

            good_percentages.append(num_words_tgt_not_doc / num_target_words)

        if good_precision and not good_recall:
            count_good_prec_poor_recall += 1
            good_prec_poor_recall_indices.append(index)

            # examine the percentage of words in target in orig. document
            num_good_words_tgt += num_target_words

            list_tgt_not_doc = [word for word in list_target_kp_words if word not in list_orig_doc_words]
            num_words_tgt_not_doc = len(list_tgt_not_doc)
            num_good_words_tgt_not_doc += num_words_tgt_not_doc

            good_percentages.append(num_words_tgt_not_doc / num_target_words)

        if not good_precision and good_recall:
            count_poor_prec_good_recall += 1
            poor_prec_good_recall_indices.append(index)
        if not good_precision and not good_recall:
            count_both_poor += 1
            both_poor_indices.append(index)

            # examine the percentage of words in target in orig. document
            num_poor_words_tgt += num_target_words

            list_tgt_not_doc = [word for word in list_target_kp_words if word not in list_orig_doc_words]
            num_words_tgt_not_doc = len(list_tgt_not_doc)
            num_poor_words_tgt_not_doc += num_words_tgt_not_doc

            if num_poor_words_tgt_not_doc / num_poor_words_tgt < 0.3:
                num_unlikely_precision += 1

            poor_percentages.append(num_words_tgt_not_doc / num_target_words)

            alt_prec, _ = get_alt_precision_recall(pred_kp_orig, target_kp_orig)

            if alt_prec >= 0.66:
                num_good_lemmatized += 1
        if index == print_number:
            print(f'Good Precision? {good_precision}, Good Recall? {good_recall}')
            print(f'Document:\n{orig_doc}')

    print(f'Percentage of Words in Target but Not Original Document (Good Performance): {num_good_words_tgt_not_doc / num_good_words_tgt}')
    print(f'Percentage of Words in Target but Not Original Document (Poor Performance): {num_poor_words_tgt_not_doc / num_poor_words_tgt}')
    print(f'Number with Unlikely Precision: {num_unlikely_precision}')
    print(f'Number with Good Precision after Lemmatization: {num_good_lemmatized}')
    print(f'Number of Samples Lacking Frequent Target Words: {num_lack_freq_tgt_words}')
    print(f'Number of Samples With Frequent Target Words that are Separated: {num_not_found_together}')

    print('\n-----')
    print(
        f'Total Number of Words in Target Keyphrases that do not appear in Original Document by Part of Speech ({course}).')
    for key, value in total_pos_count_dict.items():
        print(f'\t{key}: {value}')
    print('-----')

    percentages = good_percentages + poor_percentages

    # plt.hist(good_percentages, bins=np.linspace(0, 1, 20), alpha=0.5, label='Good Performance')
    plt.hist(percentages, bins=np.linspace(0, 1, 20), alpha=0.5)
    plt.title('Distribution for Percentage of Target Keywords that are Not in the Original Document')
    plt.xlabel('% Target Words Not Found in Original Document')
    plt.ylabel('Number of Samples')
    # plt.legend(loc='upper right')
    plt.savefig(f'detailed_performance/histogram_{course}.pdf')
    plt.close()

    return count_both_good, count_good_prec_poor_recall, count_poor_prec_good_recall, count_both_poor, both_good_indices, good_prec_poor_recall_indices, poor_prec_good_recall_indices, both_poor_indices, prec_list, recall_list, potential_list, num_freq_list, num_doc_words_list, found_adj_list, percentages


def main():

    output_path = 'detailed_performance/'
    if os.path.exists(output_path):
        shutil.rmtree(output_path)
    os.mkdir(output_path)

    course_percentages = []

    for course in ['oli-intro-bio', 'oli-gen-chem']:
        for algorithm in ['keybert']:  # , 'textrank', 'singlerank', 'topicrank', 'multipartiterank', 'rake', 'yake']:
            df_path = f'AKE_Result_Dataframes/{course}_{algorithm}_df.pkl'
            df = pd.read_pickle(df_path)

            orig_docs = df.Original.values
            # print(orig_docs)
            targets = df.Target.values
            preds = df.Predicted.values

            # print(targets)
            # print(preds)

            both_good, prec_not_recall, not_prec_recall, both_poor, both_good_indices, prec_not_recall_indices, not_prec_recall_indices, both_poor_indices, prec_list, recall_list, potential_list, num_freq_list, num_doc_words_list, found_adj_list, percentages = categorize_samples(orig_docs, targets, preds, course)

            performance_category = np.zeros(df.shape[0])
            for index in both_good_indices:
                performance_category[index] = 1
            for index in prec_not_recall_indices:
                performance_category[index] = 2
            for index in not_prec_recall_indices:
                performance_category[index] = 3
            for index in both_poor_indices:
                performance_category[index] = 4

            df['Performance_Category'] = performance_category
            df['Precision'] = prec_list
            df['Recall'] = recall_list
            df['Potential'] = potential_list
            df['# Frequent Target Tokens'] = num_freq_list
            df['# Tokens in Document'] = num_doc_words_list
            df['Target Tokens adjacent in Document'] = found_adj_list

            course_percentages.append(percentages)

            print(df.groupby(['Performance_Category', 'Potential']).size())

            outfile = f'detailed_performance/{course}_{algorithm}_performance_categorization.csv'
            df.to_csv(outfile)

            print('-----')
            print(f'Course: {course}')
            print(f'Algorithm: {algorithm}')
            print(f'Number of Samples with Good Precision and Good Recall: {both_good}')
            print(f'Number of Samples with Good Precision and Poor Recall: {prec_not_recall}')
            print(f'Number of Samples with Poor Precision and Good Recall: {not_prec_recall}')
            print(f'Number of Samples with Poor Precision and Poor Recall: {both_poor}')
            print('-----\n')

    titles = ['OLI Introduction to Biology', 'OLI General Chemistry 1']
    plt.hist(course_percentages[0], bins=np.linspace(0, 1, 20), fill=False, hatch='xxxx', label=titles[0])
    plt.hist(course_percentages[1], bins=np.linspace(0, 1, 20), fill=False, hatch='....', label=titles[1])
    plt.title('Target Keyphrase Terms that do not appear in the Original Document')
    plt.xlabel('# Target Skill Label Terms Not in Original Document / # Target Skill Label Terms')
    plt.ylabel('Target Skill Label Frequency')
    plt.legend(loc='upper right')
    plt.savefig(f'detailed_performance/target_keyphrase_histogram.pdf')
    plt.close()



if __name__ == '__main__':
    main()
