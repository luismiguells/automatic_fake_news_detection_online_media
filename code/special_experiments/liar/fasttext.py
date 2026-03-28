from pathlib import Path
import sys
# -*- coding: utf-8 -*-
"""
Created on Sat Jun  5 15:49:55 2021

@author: luismiguells

Description: Classify if news is true or fake using the LIAR  dataset. 
This dataset is divided in train and test data. Creating fastText vectors
to tranform the data. SVM, LR, SGDC, MNB, KNN and RF are the models used for 
the classification.
"""

from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from sklearn.neighbors import KNeighborsClassifier
from sklearn.linear_model import SGDClassifier
from sklearn.naive_bayes import MultinomialNB
from sklearn.preprocessing import normalize
from nltk.corpus import stopwords
from sklearn import metrics
from sklearn import svm
import numpy as np
import fasttext
import warnings
import time

sys.path.append(str(Path(__file__).resolve().parent.parent.parent))
from utils import read_labels, read_text_data

# Remove to see warnings
warnings.filterwarnings('ignore')

# Variables
dataset = 'liar_dataset'
lang = 'english'
feature = 'fastText'

# OM = Own Model PM = Pre-trained Model PM+OM = Combination of both
model_type = 'PM' 

# Files
project_root = Path(__file__).resolve().parent.parent.parent.parent
main_dir = project_root / 'data' / dataset
fasttext_file = project_root / 'data' / \'pre_trained_models/fasttext_english_300.vec'
labels_names = ['True', 'Fake']

# Train file
labels_file_train = main_dir / 'labels_train.txt'
words_file_train = main_dir / 'split_train/words.txt'

# Validation file
labels_file_valid = main_dir / 'labels_valid.txt'
words_file_valid = main_dir / 'split_valid/words.txt'

# Test file
labels_file_test = main_dir / 'labels_test.txt'
words_file_test = main_dir / 'split_test/words.txt'

# Select the model of word representation to work with
if model_type == 'OM':
    fast_text_dict = fast_text_vectors(words_file_train)
elif model_type == 'PM':
    fast_text_dict = fasttext_reader(fasttext_file)
elif model_type == 'PM+OM':
    fast_text_dict = fasttext_reader(fasttext_file)
    fast_text_dict_aux = fast_text_vectors(words_file_train)
    
    vocab = fast_text_dict_aux.get_words()
    
    for w in vocab:
        if w not in fast_text_dict:
            fast_text_dict[w] = fast_text_dict_aux.get_word_vector(w)

# Reading train data
corpus_train = []
corpus_train = read_text_data(lang, words_file_train)
labels_list_train = read_labels(labels_file_train, labels_names)
labels_train = np.asarray(labels_list_train)

# Create the corpus train with fastText vectors
l_s = [line.split() for line in corpus_train]
s = np.zeros((300)) 
ft_train = []
l_not_fasttext = []

for l in l_s:
    i = 0
    for w in l:
        if w in fast_text_dict:
            s += fast_text_dict[w]
            i += 1
        else:
            l_not_fasttext.append(w)
            
    s = s/i
    ft_train.append(s)
    s = np.zeros((300))
    
# Remove possible elements value equal to NaN
ft_train, labels_train = remove_nan_values(ft_train, labels_train)
ft_corpus_train = np.array(ft_train)

# Normalize the data
n, m = ft_corpus_train.shape
min_values = np.min(ft_corpus_train, axis=1)
min_values = min_values.reshape((n, 1))
ft_corpus_train = ft_corpus_train+abs(min_values)
ft_corpus_train = normalize(ft_corpus_train, norm='l2')

# Reading validation data
corpus_valid = []
corpus_valid = read_text_data(lang, words_file_valid)
labels_list_valid = read_labels(labels_file_valid, labels_names)
labels_valid = np.asarray(labels_list_valid)

# Create the corpus validation with fastText vectors
l_s = [line.split() for line in corpus_valid]
s = np.zeros((300)) 
ft_valid = []
l_not_fasttext = []

for l in l_s:
    i = 0
    for w in l:
        if w in fast_text_dict:
            s += fast_text_dict[w]
            i += 1
        else:
            l_not_fasttext.append(w)
            
    s = s/i
    ft_valid.append(s)
    s = np.zeros((300))
    
# Remove possible elements value equal to NaN
ft_valid, labels_valid = remove_nan_values(ft_valid, labels_valid)
ft_corpus_valid = np.array(ft_valid)

# Normalize the data
n, m = ft_corpus_valid.shape
min_values = np.min(ft_corpus_valid, axis=1)
min_values = min_values.reshape((n, 1))
ft_corpus_valid = ft_corpus_valid+abs(min_values)
ft_corpus_valid = normalize(ft_corpus_valid, norm='l2')

# Reading test data
labels_list_test = read_labels(labels_file_test, labels_names)
corpus_test = []
corpus_test = read_text_data(lang, words_file_test)

labels_test = np.asarray(labels_list_test)
labels_set_test = set(labels_list_test)

# Create the corpus test with fastText vectors
l_s = [line.split() for line in corpus_test]
s = np.zeros((300)) 
ft_test = []
l_not_fasttext = []

for l in l_s:
    i = 0
    for w in l:
        if w in fast_text_dict:
            s += fast_text_dict[w]
            i += 1
        else:
            l_not_fasttext.append(w)
            
    s = s/i
    ft_test.append(s)
    s = np.zeros((300))
    
# Remove possible elements value equal to NaN
ft_test, labels_test = remove_nan_values(ft_test, labels_test)
ft_corpus_test = np.array(ft_test)
test_index = [i for i in range(len(labels_test))]
test_index = np.asarray(test_index)

# Normalize the data
n, m = ft_corpus_test.shape
min_values = np.min(ft_corpus_test, axis=1)
min_values = min_values.reshape((n, 1))
ft_corpus_test = ft_corpus_test+abs(min_values)
ft_corpus_test = normalize(ft_corpus_test, norm='l2')

# Classifiers to use
classifiers = ['SVM', 'LR', 'MNB', 'SGDC', 'KNN', 'RF']

for cl in classifiers:
    
    print('Training and testing with', cl)
    classifier = cl
    out_dir = project_root / 'results' / \'probabilities/'+classifier+'/'+feature+'/'+model_type+'/'+dataset+'/'
    
    start = time.time()

    if classifier == 'SVM':
               
        # Find the best hyper-parameter
        cs = [0.01, 0.1, 1, 10, 100]
        best_c = 0
        best_score = 0
    
        for c in cs:
            clf_inner = svm.SVC(C=c, kernel='linear')
            score = val_score(clf_inner, ft_corpus_train, labels_train, ft_corpus_valid, labels_valid)
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
            score = val_score(clf_inner, ft_corpus_train, labels_train, ft_corpus_valid, labels_valid)
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
            score = val_score(clf_inner, ft_corpus_train, labels_train, ft_corpus_valid, labels_valid)
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
            score = val_score(clf_inner, ft_corpus_train, labels_train, ft_corpus_valid, labels_valid)
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
            score = val_score(clf_inner, ft_corpus_train, labels_train, ft_corpus_valid, labels_valid)
            if score > best_score:
                best_score = score
                best_r = r
                
        # Create the model with the best hyper-parameter        
        clf = RandomForestClassifier(n_estimators=best_r, random_state=0)
    
    # Concatenate the train and validation data
    corpus_train_grouped = np.concatenate((ft_corpus_train, ft_corpus_valid))
    labels_train_grouped = np.concatenate((labels_train, labels_valid))
    labels_train_grouped = np.asarray(labels_train_grouped)
    
    
    clf.fit(corpus_train_grouped, labels_train_grouped)
    predicted = clf.predict(ft_corpus_test)
    predicted_proba = clf.predict_proba(ft_corpus_test)
    accuracy = np.mean(predicted == labels_test)
    precission_macro = metrics.precision_score(labels_test, predicted, average='macro')
    recall_macro = metrics.recall_score(labels_test, predicted, average='macro')
    f1_macro = metrics.f1_score(labels_test, predicted, average='macro')
    kapha = metrics.cohen_kappa_score(labels_test, predicted) 
    roc = metrics.roc_auc_score(labels_test, predicted, average='macro')
    
    # Create an array to write it in a file
    predictions = np.concatenate((test_index[:, None], predicted_proba, predicted[:, None].astype(int), labels_test[:, None].astype(int)), axis=1)
    fmt = '%d', '%1.9f', '%1.9f', '%d', '%d'
    np.savetxt(out_dir / 'probability.csv', predictions, fmt=fmt,delimiter=',', encoding='utf-8', header='test_index, probability_true, probability_fake, predicted_class, real_class', comments='')

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
