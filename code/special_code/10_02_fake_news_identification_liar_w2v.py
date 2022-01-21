# -*- coding: utf-8 -*-
"""
Created on Sat Jun 12 18:02:36 2021

@author: luismiguells

Description: Classify if news is true or fake using the LIAR  dataset. 
This dataset is divided in train and test data. Creating Word2Vec vectors
to tranform the data. SVM, LR, SGDC, MNB, KNN and RF are the models used for 
the classification.
"""

from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from sklearn.neighbors import KNeighborsClassifier
from sklearn.linear_model import SGDClassifier
from sklearn.naive_bayes import MultinomialNB
from sklearn.preprocessing import normalize
from gensim.models import Word2Vec
from nltk.corpus import stopwords
from sklearn import metrics
from sklearn import svm
import numpy as np
import warnings
import time

# Remove to see warnings
warnings.filterwarnings('ignore')

class MySentences(object):
    def __init__(self, file_name):
        self.file_name = file_name
 
    def __iter__(self):
        for line in open(self.file_name):
            yield line.rstrip().split()

def my_tokenizer(s):
    """
    Parameters
    ----------
    s : A string.

    Returns
    -------
    A list with words splitted by space.

    """
    return s.split()

def read_labels(file, labels_names):
    """
    Parameters
    ----------
    file : Name of the file to read.
    labels_names : A list filled with the labels to use. 
                   For example, ['True', 'Fake'].

    Returns
    -------
    label : A list of labels using the index of labels_names variable.
    """
    label = []
    with open(file) as content_file:
        for line in content_file:
            category = line.strip()
            label.append(labels_names.index(category))
    return label

def clean_words(words, stop_words):
    """
    Parameters
    ----------
    words : A list of words.
    stop_words : A stopwords list from NLTK library.

    Returns
    -------
    text : A text that does not have short and long words, and without
           stopwords.
    """
    text = ' '.join([word for word in words if len(word)>2 and len(word)<35 and word not in stop_words])
    return text

def clean_en_words(text):
    """
    Parameters
    ----------
    text : A string.

    Returns
    -------
    clean_text : A string without English stopwords for Spanish text.
    """
    stop_words = stopwords.words('english')
    words = text.rstrip().split()
    clean_text = ' '.join([word for word in words if word not in stop_words])
    
    return clean_text

def read_text_data_with_emos(lang, text_file, emo_file):
    """
    Parameters
    ----------
    lang : The selected language for stopwords.
    text_file : A list that contains the text data.
    emo_file : A list that contains the emojis.

    Returns
    -------
    data : A clean list with text and emojis merged.
    """
    data = []
    if lang == 'spanish':
        stop_words = stopwords.words('spanish')
    elif lang == 'english':
        stop_words = stopwords.words('english')
    with open(text_file, encoding='utf-8') as text_content, open(emo_file, encoding='utf-8') as emo_content:
        for text_line, emo_line in zip(text_content, emo_content):
            words = text_line.rstrip().split()
            text = clean_words(words, stop_words)
            text += ' '+emo_line.rstrip()
            data.append(text)
    return data

def read_text_data(lang, file):
    """
    Parameters
    ----------
    lang : The selected language for stopwords.
    file : The file that contains the text.

    Returns
    -------
    data : A list that contains all the text data cleaned.
    """
    data = []
    if lang == 'spanish':
        stop_words = stopwords.words('spanish')
    elif lang == 'english':
        stop_words = stopwords.words('english')
    with open(file, encoding='utf-8') as content_file:
        for line in content_file:
            words = line.rstrip().split()
            text = clean_words(words, stop_words)
            
            # Clean possible English stopwords that could be in the Spanish corpus
            if lang == 'spanish':
                text = clean_en_words(text)
            data.append(text)
    return data

def remove_empty_text_data_w2v(corpus, labels):
    """
    Parameters
    ----------
    corpus : A list with text data.
    labels : A list that contains the labels.

    Returns
    -------
    corpus : A list without elements of length 0.
    labels : A list that contains labels, except those whose length 
             in the corpus is equal to 0.
    """
    count = 0
    l_index = []
    for line in corpus:
        if len(line) == 0:
            l_index.append(count)
        count += 1
    
    if len(l_index) > 0:
        l_index.reverse()
        for idx in l_index:
            corpus.pop(idx)
            labels.pop(idx)
        return corpus, labels
    else:
        return corpus, labels

def remove_nan_values(corpus, labels):
    """
    Parameters
    ----------
    corpus : A list with text data.
    labels : A list that contains the labels.

    Returns
    -------
    corpus : A list without NaN elements.
    labels : A list that contains labels, except those whose elemens
             in the corpus is equal to NaN.
    """
    count = 0
    l_index = []
    
    for i in corpus:
        if np.isnan(i).any():
            l_index.append(count)
        count += 1
    if len(l_index) > 0:
        labels = list(labels)
        l_index.reverse()
        for idx in l_index:
            corpus.pop(idx)
            labels.pop(idx)
        return corpus, np.asarray(labels)
    else:
        return corpus, labels

def val_score(clf, train_data, train_labels, valid_data, valid_labels):
    """
    Parameters
    ----------
    clf : A classifier (SVM, MNB, KNN, etc.).
    train_data : Data to train the model.
    train_labels : Labels that correspond to the train_data.
    valid_data : Validation data to test the model.
    valid_labels : Validation labels that correspond to the valid_data.

    Returns
    -------
    f1_macro : F1 metric on its macro version.
    """
    clf.fit(train_data, train_labels)
    predicted = clf.predict(valid_data)
    f1_macro = metrics.f1_score(valid_labels, predicted, average='macro')
    
    return f1_macro

def w2v_reader(w2v_file):
    """
    Parameters
    ----------
    w2v_file : fastText file that contains the vectors.

    Returns
    -------
    w2v_dict : A dictionary where the key is the word and the value of the word.
    """
    w2v_dict = {}
    with open(w2v_file, 'r', encoding='utf-8') as w2v_reader:
        for line in w2v_reader:
            tokens = line.strip().split()
            if len(tokens) > 301:
                pass
            else:
                vect = [float(token) for token in tokens[1:]]
                w2v_dict[tokens[0]] = vect

    return w2v_dict

def w2v_vectors(sentences):
    """
    Parameters
    ----------
    file : File that contains data (e.g., words, emojis, hashtags, etc.).

    Returns
    -------
    model : Dictionary where each feature is represented with a vector.
    """
    model = Word2Vec(sentences, min_count=1, vector_size=300, workers=4)
    
    return model


# Variables
dataset = 'liar_dataset'
lang = 'english'
feature = 'Word2Vec'

# OM = Own Model PM = Pre-trained Model PM+OM = Combination of both
model_type = 'PM' 

# Files
main_dir = 'C:/Users/luismiguel/Google Drive/MT/data/'+dataset+'/'
w2v_file = 'C:/Users/luismiguel/Google Drive/MT/data/pre_trained_models/word2vec_english_300.txt'
labels_names = ['True', 'Fake']

# Train file
labels_file_train = main_dir+'labels_train.txt'
words_file_train = main_dir+'split_train/words.txt'

# Validation file
labels_file_valid = main_dir+'labels_valid.txt'
words_file_valid = main_dir+'split_valid/words.txt'

# Test file
labels_file_test = main_dir+'labels_test.txt'
words_file_test = main_dir+'split_test/words.txt'

# Select the model of word representation to work with
if model_type == 'OM':
    sentences_train = MySentences(words_file_train)
    w2v_dict = w2v_vectors(sentences_train)
elif model_type == 'PM':
    w2v_dict = w2v_reader(w2v_file)
elif model_type == 'PM+OM':
    w2v_dict = w2v_reader(w2v_file)
    sentences_train = MySentences(words_file_train)
    w2v_dict_aux = w2v_vectors(sentences_train)
    
    vocab = w2v_dict_aux.wv.key_to_index
    
    for w in vocab.keys():
        if w not in w2v_dict:
            w2v_dict[w] = w2v_dict_aux.wv.get_vector(w)

# Reading train data
corpus_train = []
corpus_train = read_text_data(lang, words_file_train)
labels_list_train = read_labels(labels_file_train, labels_names)
labels_train = np.asarray(labels_list_train)

# Create the corpus train with fastText vectors
l_s = [line.split() for line in corpus_train]
s = np.zeros((300)) 
w2v_train = []
l_not_w2v = []

for l in l_s:
    i = 0
    for w in l:
        if model_type == 'OM':
            if w in w2v_dict.wv:
                s += w2v_dict.wv.get_vector(w)
                i += 1
            else:
                l_not_w2v.append(w)
        else:
            if w in w2v_dict:
                s += w2v_dict[w]
                i += 1
            else:
                l_not_w2v.append(w)
            
    s = s/i
    w2v_train.append(s)
    s = np.zeros((300))
    
# Remove possible elements value equal to NaN
w2v_train, labels_train = remove_nan_values(w2v_train, labels_train)
w2v_corpus_train = np.array(w2v_train)

# Normalize the data
n, m = w2v_corpus_train.shape
min_values = np.min(w2v_corpus_train, axis=1)
min_values = min_values.reshape((n, 1))
w2v_corpus_train = w2v_corpus_train+abs(min_values)
w2v_corpus_train = normalize(w2v_corpus_train, norm='l2')

# Reading validation data
corpus_valid = []
corpus_valid = read_text_data(lang, words_file_valid)
labels_list_valid = read_labels(labels_file_valid, labels_names)
labels_valid = np.asarray(labels_list_valid)

# Create the corpus validation with fastText vectors
l_s = [line.split() for line in corpus_valid]
s = np.zeros((300)) 
w2v_valid = []
l_not_w2v = []

for l in l_s:
    i = 0
    for w in l:
        if model_type == 'OM':
            if w in w2v_dict.wv:
                s += w2v_dict.wv.get_vector(w)
                i += 1
            else:
                l_not_w2v.append(w)
        else:
            if w in w2v_dict:
                s += w2v_dict[w]
                i += 1
            else:
                l_not_w2v.append(w)
            
    s = s/i
    w2v_valid.append(s)
    s = np.zeros((300))
    
# Remove possible elements value equal to NaN
w2v_valid, labels_valid = remove_nan_values(w2v_valid, labels_valid)
w2v_corpus_valid = np.array(w2v_valid)

# Normalize the data
n, m = w2v_corpus_valid.shape
min_values = np.min(w2v_corpus_valid, axis=1)
min_values = min_values.reshape((n, 1))
w2v_corpus_valid = w2v_corpus_valid+abs(min_values)
w2v_corpus_valid = normalize(w2v_corpus_valid, norm='l2')

# Reading test data
labels_list_test = read_labels(labels_file_test, labels_names)
corpus_test = []
corpus_test = read_text_data(lang, words_file_test)

labels_test = np.asarray(labels_list_test)
labels_set_test = set(labels_list_test)

# Create the corpus test with fastText vectors
l_s = [line.split() for line in corpus_test]
s = np.zeros((300)) 
w2v_test = []
l_not_w2v = []

for l in l_s:
    i = 0
    for w in l:
        if model_type == 'OM':
            if w in w2v_dict.wv:
                s += w2v_dict.wv.get_vector(w)
                i += 1
            else:
                l_not_w2v.append(w)
        else:
            if w in w2v_dict:
                s += w2v_dict[w]
                i += 1
            else:
                l_not_w2v.append(w)
            
    s = s/i
    w2v_test.append(s)
    s = np.zeros((300))
    
# Remove possible elements value equal to NaN
w2v_test, labels_test = remove_nan_values(w2v_test, labels_test)
w2v_corpus_test = np.array(w2v_test)
test_index = [i for i in range(len(labels_test))]
test_index = np.asarray(test_index)

# Normalize the data
n, m = w2v_corpus_test.shape
min_values = np.min(w2v_corpus_test, axis=1)
min_values = min_values.reshape((n, 1))
w2v_corpus_test = w2v_corpus_test+abs(min_values)
w2v_corpus_test = normalize(w2v_corpus_test, norm='l2')

# Classifiers to use
classifiers = ['SVM', 'LR', 'MNB', 'SGDC', 'KNN', 'RF']

for cl in classifiers:
    
    print('Training and testing with', cl)
    classifier = cl
    out_dir = 'C:/Users/luismiguel/Google Drive/MT/results/probabilities/'+classifier+'/'+feature+'/'+model_type+'/'+dataset+'/'
    
    start = time.time()

    if classifier == 'SVM':
               
        # Find the best hyper-parameter
        cs = [0.01, 0.1, 1, 10, 100]
        best_c = 0
        best_score = 0
    
        for c in cs:
            clf_inner = svm.SVC(C=c, kernel='linear')
            score = val_score(clf_inner, w2v_corpus_train, labels_train, w2v_corpus_valid, labels_valid)
            if score > best_score:
                best_score = score
                best_c = c
        
        # Create the model with the best hyper-parameter 
        clf = svm.SVC(C=best_c, kernel='linear', probability=True)
            
    elif classifier == 'LR':
        
        # Find the best hyper-parameter
        cs = [0.01, 0.1, 1, 10, 100]
        best_c = 0
        best_score = 0
    
        for c in cs:
            clf_inner = LogisticRegression(C=c, penalty='l2', solver='liblinear')
            score = val_score(clf_inner, w2v_corpus_train, labels_train, w2v_corpus_valid, labels_valid)
            if score > best_score:
                best_score = score
                best_c = c
                
        # Create the model with the best hyper-parameter        
        clf = LogisticRegression(C=best_c, penalty='l2', solver='liblinear')
    
    elif classifier == 'MNB':
        
        clf = MultinomialNB()
    
    elif classifier == 'SGDC':
        
        # Find the best hyper-parameter
        cs = [0.01, 0.1, 1, 10, 100]
        best_c = 0
        best_score = 0
    
        for c in cs:
            clf_inner = SGDClassifier(loss='log', alpha=1/c, max_iter=10000)
            score = val_score(clf_inner, w2v_corpus_train, labels_train, w2v_corpus_valid, labels_valid)
            if score > best_score:
                best_score = score
                best_c = c
                
        # Create the model with the best hyper-parameter        
        clf = SGDClassifier(loss='log', alpha=1/best_c, max_iter=10000)
    
    elif classifier == 'KNN':
        
        # Find the best hyper-parameter
        ks = [1, 2, 3, 5, 10]
        best_score = 0
        best_k = 0
    
        for k in ks:
            clf_inner = KNeighborsClassifier(n_neighbors=k)
            score = val_score(clf_inner, w2v_corpus_train, labels_train, w2v_corpus_valid, labels_valid)
            if score > best_score:
                best_score = score
                best_k = k
                
        # Create the model with the best hyper-parameter         
        clf = KNeighborsClassifier(n_neighbors=best_k)
        
    elif classifier == 'RF':
        
        # Find the best hyper-parameter
        rs = [10, 50, 100, 150, 200]
        best_score = 0
        best_r = 0
    
        for r in rs:
            clf_inner = RandomForestClassifier(n_estimators=r, random_state=0)
            score = val_score(clf_inner, w2v_corpus_train, labels_train, w2v_corpus_valid, labels_valid)
            if score > best_score:
                best_score = score
                best_r = r
                
        # Create the model with the best hyper-parameter        
        clf = RandomForestClassifier(n_estimators=best_r, random_state=0)
    
    # Concatenate the train and validation data
    corpus_train_grouped = np.concatenate((w2v_corpus_train, w2v_corpus_valid))
    labels_train_grouped = np.concatenate((labels_train, labels_valid))
    labels_train_grouped = np.asarray(labels_train_grouped)
    
    
    clf.fit(corpus_train_grouped, labels_train_grouped)
    predicted = clf.predict(w2v_corpus_test)
    predicted_proba = clf.predict_proba(w2v_corpus_test)
    accuracy = np.mean(predicted == labels_test)
    precission_macro = metrics.precision_score(labels_test, predicted, average='macro')
    recall_macro = metrics.recall_score(labels_test, predicted, average='macro')
    f1_macro = metrics.f1_score(labels_test, predicted, average='macro')
    kapha = metrics.cohen_kappa_score(labels_test, predicted) 
    roc = metrics.roc_auc_score(labels_test, predicted, average='macro')
    
    # Create an array to write it in a file
    predictions = np.concatenate((test_index[:, None], predicted_proba, predicted[:, None].astype(int), labels_test[:, None].astype(int)), axis=1)
    fmt = '%d', '%1.9f', '%1.9f', '%d', '%d'
    np.savetxt(out_dir+'probability.csv', predictions, fmt=fmt,delimiter=',', encoding='utf-8', header='test_index, probability_true, probability_fake, predicted_class, real_class', comments='')


    end = time.time()
    
        
    # Print the results
    print('Accuracy: %0.2f' % accuracy)
    print('Precision: %0.2f' % precission_macro)
    print('Recall: %0.2f' % recall_macro)
    print('F1: %0.2f' % f1_macro)
    print('Kapha: %0.2f' % kapha)
    print('ROC-AUC: %0.2f' % roc)
    print('Time of training + testing: %0.2f' % (end - start))
    print('\n')