import re
import numpy as np
import scipy.sparse
from pymongo import MongoClient
from collections import defaultdict
from sklearn.feature_extraction.text import TfidfVectorizer
try:
    import cPickle as pickle
except:
    import pickle

from collections import Counter

# data preprocessing modules
from nltk import PorterStemmer
from nltk.corpus import stopwords
from sklearn.preprocessing import label_binarize
from sklearn.metrics import roc_curve, auc


def find_math_words(text):
    math_words = ''
    if '\\int' in text:
        math_words += ' integral'
    if '\\lim' in text:
        math_words += ' limit'
    if '\\sum' in text:
        math_words += ' sum'
    if '\\infty' in text:
        math_words += ' infinity'
    if '{matrix}' in text or '{pmatrix}' in text or '{bmatrix}' in text:
        math_words += ' matrix'
    if '{array}' in text:
        math_words += ' array'
    if '\\exp' in text or 'e^' in text:
        math_words += ' exponential'
    if 'ln(' in text or 'log(' in text:
        math_words += ' log'
    if '\\sqrt' in text:
        math_words += ' square_root'
    if '\\frac' in text:
        math_words += ' fraction'
    if '\\sin' in text:
        math_words += ' sine'
    if '\\cos' in text:
        math_words += ' cosine'
    if '\\tan' in text:
        math_words += ' tangent'
    if '\\arctan' in text:
        math_words += ' arctangent'
    if '\\pi' in text:
        math_words += ' pi'
    if '\\partial' in text:
        math_words += ' partial'
    if '\\Delta' in text:
        math_words += ' delta'
    if '\\geq' in text or '\\leq':
        math_words += ' greater_than'
    if '\\cdot' in text:
        math_words += ' cdot'
    if '\\subset' in text or '\\subseteq' in text:
        math_words += ' subset'
    if ('\\cup' in text or '\\cap' in text
            or '\\bigcup' in text or '\\bigcap' in text):
        math_words += ' cup'
    if '\\epsilon' in text or '\\varepsilon' in text:
        math_words += ' epsilon'
    if '\\inf' in text:
        math_words += ' infimum'
    if '\\sup' in text:
        math_words += ' supremum'
    if '\\min' in text:
        math_words += ' minimum'
    if '\\max' in text:
        math_words += ' maximum'
    if '\\det' in text:
        math_words += ' determinant'
    if '^T' in text:
        math_words += ' transpose'
    if '\\mod' in text:
        math_words += ' modulo'

    return math_words


def strip_text(text):
    ''' Remove html tags, latex tags, etc. '''

    math_words = find_math_words(text)

    text = text.replace('<span class=\"math\">', 'code_word_begin')
    text = text.replace('</span>', 'code_word_end')
    text = re.sub(r'(?<=code_word_begin)(.*?)(?=code_word_end)', ' ', text,
                  flags=re.DOTALL)

    text = text.replace('<em>', 'code_word_begin')
    text = text.replace('</em>', 'code_word_end')
    text = re.sub(r'(?<=code_word_begin)(.*?)(?=code_word_end)', ' ', text,
                  flags=re.DOTALL)

    text.strip()
    text = text.lower()
    text = text.replace('<p>', ' ')
    text = text.replace('</p>', ' ')
    text = text.replace('code_word_begin', ' ')
    text = text.replace('code_word_end', ' ')
    text = text.replace('.', ' ')
    text = text.replace(',', ' ')
    text = text.replace(';', ' ')
    text = text.replace('?', ' ')
    text = text.replace('!', ' ')
    text = text.replace('\n', '')
    text = re.sub(r'[^a-z ]', ' ', text)

    text = text + math_words

    list_voc = re.split(r'[ ]+', text)

    return list_voc


def get_all_MER_topics():
    '''Returns list of all topics on MER'''
    client = MongoClient()
    questions_collection = client['merdb'].questions
    return questions_collection.find().distinct("topics")


def get_questions_with_topics(topics):
    '''Returns list of questions with matching topics'''
    client = MongoClient()
    questions_collection = client['merdb'].questions
    if isinstance(topics, str):
        topics = [topics]
    qs = []
    for q in questions_collection.find({"topics": {"$in": topics}}):
        qs.append(q)
    return qs


def count_topics_in_questions(qs):
    count_dict = defaultdict(int)
    for q in qs:
        try:
            for topic in q['topics']:
                count_dict[topic] += 1
        except KeyError:
            pass
    return count_dict


def get_topic_to_parent_dict():
    '''returns dict topic -> parent_topic'''
    client = MongoClient()
    topics_collection = client['merdb'].topics
    topic_to_parent_dict = dict()
    for q in topics_collection.find():
        topic_to_parent_dict[q['topic']] = q['parent']
    return topic_to_parent_dict


def topic_to_parent(topic):
    '''returns parent for given topic'''
    try:
        return topic_to_parent_dict[topic]
    except NameError:
        topic_to_parent_dict = get_topic_to_parent_dict()
        return topic_to_parent_dict[topic]


def question_to_parents(q):
    '''returns sorted list of all unique parents of the questions,
    or [None] if question has no topics or topic is unknown.'''
    if not 'topics' in q.keys():
        return [None]
    parents = []
    for topic in q['topics']:
        try:
            parents.append(topic_to_parent(topic))
        except KeyError:
            pass
    return sorted(list(set(parents)))


def questions_to_parents(qs):
    '''returns list of sorted list of all unique parents for all questions.'''
    list_of_parents = []
    for q in qs:
        list_of_parents.append(question_to_parents(q))
    return list_of_parents


def unique_parents(qs):
    '''
    returns list of distinct parent topics in list of questions qs.
    Removes parents with only a single question!
    '''
    c = Counter(p for q in qs for p in question_to_parents(q))
    at_least_twice = [
        p for q in qs for p in question_to_parents(q) if c[p] > 1]
    return sorted(list(set(at_least_twice)))


def question_to_BOW(q, include_hint_and_sols=True):
    '''Transforms a question dictionary q to its bag of words'''
    def words_stemmed_no_stop(words):
        '''remove commonly used words and combine words with the same root'''
        stop = stopwords.words('english')
        res = []
        for word in words:
            stemmed = PorterStemmer().stem_word(word)
            # take words longer than 1 char
            if stemmed not in stop and len(stemmed) > 1:
                res.append(stemmed)
        return res

    all_text = q['statement_html']
    if include_hint_and_sols:
        for h in q['hints_html']:
            all_text += h
        for s in q['sols_html']:
            all_text += s

    all_words = strip_text(all_text)
    bow = words_stemmed_no_stop(all_words)
    return ' '.join([w for w in bow])


def questions_to_BOW(qs):
    '''Transforms list of questions to list of bag of words'''
    return [question_to_BOW(q) for q in qs]


def statement_to_BOW(statement):
    '''Transforms a statement to its bag of words'''
    def words_stemmed_no_stop(words):
        '''remove commonly used words and combine words with the same root'''
        stop = stopwords.words('english')
        res = []
        for word in words:
            stemmed = PorterStemmer().stem_word(word)
            # take words longer than 1 char
            if stemmed not in stop and len(stemmed) > 1:
                res.append(stemmed)
        return res

    all_words = strip_text(statement)
    bow = words_stemmed_no_stop(all_words)
    return ' '.join([w for w in bow])


def question_to_X(q, FILE_TO_LOAD="TfidfVectorizer.bin"):
    '''Transforms question to X vector. Uses vectorizer saved as 'vectorizer'
    or at FILE_TO_LOAD'''
    try:
        return vectorizer.transform([question_to_BOW(q)])
    except NameError:
        vectorizer = pickle.load(open(FILE_TO_LOAD, "r"))
        return vectorizer.transform([question_to_BOW(q)])


def statement_to_X(statement, FILE_TO_LOAD="TfidfVectorizer.bin"):
    try:
        return vectorizer.transform([statement_to_BOW(statement)])
    except NameError:
        vectorizer = pickle.load(open(FILE_TO_LOAD, "r"))
        return vectorizer.transform([statement_to_BOW(statement)])


def questions_to_X(qs):
    '''Transforms questions to X matrix. Uses vectorizer saved as 'vectorizer'
    or at FILE_TO_LOAD'''
    qs_X = [question_to_X(q) for q in qs]
    return scipy.sparse.vstack(qs_X)


def save_TfidfVectorizer(qs, WHERE_TO_SAVE='TfidfVectorizer.bin'):
    '''fits and saves TfidfVectorizer on input list of questions
    (training set!)'''
    vectorizer = TfidfVectorizer(min_df=2)
    vectorizer.fit(questions_to_BOW(qs))
    if WHERE_TO_SAVE:
        pickle.dump(vectorizer, open(WHERE_TO_SAVE, "wb"))
    return vectorizer


# !!! rewrote
def questions_to_topic_index(qs, topic_tags, parents=False):
    class_indices = range(0, len(topic_tags))
    topic_labels = []
    for q in qs:
            # go through topic_tags, if any of the topics is in the question's
            # topic list. Append its index to topic_labels
        for i in class_indices:
            if (((not parents) and (topic_tags[i] in q['topics']))
                    or
                    ((parents) and topic_tags[i] in question_to_parents(q))):
                topic_labels.append(i)
                # assumes there is only one topic for each question
                break

    return np.asarray(topic_labels)


def questions_to_y(qs, topic_tags, parents=False):
    if parents:
        class_indices = range(len(unique_parents(qs)))
    else:
        class_indices = range(len(topic_tags))

    return label_binarize(questions_to_topic_index(qs, topic_tags, parents),
                          class_indices)


def pred_to_topic(pred_array, topic_tags):
    '''returns topic with largest likelihood from vector of prediction
    probabilities for a single question'''
    return(topic_tags[np.argmax(pred_array)])


def preds_to_topic(pred_array, topic_tags, num_topics):
    '''returns topic with largest likelihood from vector of prediction
    probabilities for a single question'''
    pred_topics = []
    for t in range(num_topics):
        pred_topics.append(topic_tags[np.argmax(pred_array)])
        pred_array[0][np.argmax(pred_array)] = 0
    return(pred_topics)


def preds_to_topics(preds_array, topic_tags):
    '''returns topic with largest likelihood from vector of prediction
    probabilities for an array of questions'''
    result = []
    for p in preds_array:
        result.append(pred_to_topic(p, topic_tags))
    return result


def combined_roc_score(correct, predicted):
    '''returns micro roc for combined classifier
    and dict with roc for all classes'''
    fpr = dict()
    tpr = dict()
    roc_auc = dict()
    for i in range(predicted.shape[1]):
        fpr[i], tpr[i], _ = roc_curve(correct[:, i], predicted[:, i])
        roc_auc[i] = auc(fpr[i], tpr[i])
    # Compute micro-average ROC curve and ROC area
    fpr["micro"], tpr["micro"], _ = roc_curve(
        correct.ravel(), predicted.ravel())
    roc_auc["micro"] = auc(fpr["micro"], tpr["micro"])
    return roc_auc["micro"], roc_auc


def find_question(statement):
    '''Returns a question that matches the statement'''
    client = MongoClient()
    db = client.merdb
    questions = db['questions']
    for q in questions.find({"statement_html": statement}):
        return q


def predict_topic_for_question(q, classifier, topic_tags):
    vec = question_to_X(q)
    pred_prob = classifier.predict_proba(vec)
    pred_class = pred_to_topic(pred_prob, topic_tags)
    return pred_class


def predict_topics_for_question(q, classifier, topic_tags, num_topics):
    vec = question_to_X(q)
    pred_prob = classifier.predict_proba(vec)
    pred_topics = preds_to_topic(pred_prob, topic_tags, num_topics)
    return pred_topics


def predict_topics_for_statement(statement, classifier, topic_tags, num_topics):
    vec = statement_to_X(statement)
    pred_prob = classifier.predict_proba(vec)
    pred_topics = preds_to_topic(pred_prob, topic_tags, num_topics)
    return pred_topics


#!!!
# sort by probabilities


def predict_topics_for_questions(qs, classifier, topic_tags):
    return [predict_topic_for_question(q, classifier, topic_tags) for q in qs]


def determine_topic_for_question(q, classifier, topic_tags):
    # assumes only one topic
    if q is None:
        return None
    try:
        for t in topic_tags:
            if t in q['topics']:
                return t
    except KeyError:
        pass
    predicted = predict_topic_for_question(q, classifier, topic_tags)
    return predicted


def determine_topics_for_question(q, classifier, topic_tags, num_topics):
    # assumes only one topic
    if q is None:
        return None
    try:
        for t in topic_tags:
            if t in q['topics']:
                return t
    except KeyError:
        pass
    predicted = predict_topics_for_question(
        q, classifier, topic_tags, num_topics)
    return predicted


def determine_topics_for_questions(qs, classifier, topic_tags):
    return [determine_topic_for_question(q, classifier,
                                         topic_tags) for q in qs]


def determine_topics_for_statement(statement, classifier, topic_tags, num_topics):
    # assumes only one topic
    if statement is None:
        return None
    predicted = predict_topics_for_statement(
        statement, classifier, topic_tags, num_topics)
    return predicted


def beautify(topic):
    if isinstance(topic, str):
        if topic is None:
            return topic
        else:
            return topic.replace("_", " ")
    else:
        ntopic = []
        for to in topic:
            if to is not None:
                parent = topic_to_parent(to)
                ntopic.append(
                    ' ' + parent.replace("_", " ") + ': ' + to.replace("_", " "))
        return ntopic


def topic_result_with_parent(topic):
    if topic is None:
        return {'parent_topic': None, 'topic': None}
    parent = topic_to_parent(topic)
    return {'parent_topic': parent, 'topic': topic}


def topic_results_with_parents(topics):
    res = []
    for t in topics:
        res.append(topic_result_with_parent(t))
    return res
