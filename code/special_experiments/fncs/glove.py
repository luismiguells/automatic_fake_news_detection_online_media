from pathlib import Path
import sys
# -*- coding: utf-8 -*-
"""
Created on Fri May 28 10:33:38 2021

@author: luismiguells

Description: Classify if news is true or fake using the Fake News Corpus Spanish 
dataset. This dataset is divided in train and test data. Using GloVe vectors
to tranform the data. SVM, LR, SGDC, MNB, KNN and RF are the models used for 
the classification.
"""

from sklearn.model_selection import cross_val_score
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import StratifiedKFold
from sklearn.ensemble import RandomForestClassifier
from sklearn.neighbors import KNeighborsClassifier
from sklearn.linear_model import SGDClassifier
from sklearn.naive_bayes import MultinomialNB
from sklearn.preprocessing import normalize
from nltk.corpus import stopwords
from sklearn import metrics
from sklearn import svm
import numpy as np
import warnings
import time

sys.path.append(str(Path(__file__).resolve().parent.parent.parent))
from utils import read_labels, read_text_data

# Remove to see warnings
warnings.filterwarnings('ignore')

# Variables
dataset = 'fncs_dataset'
lang = 'spanish'
feature = 'GloVe'

# Files
project_root = Path(__file__).resolve().parent.parent.parent.parent
main_dir = project_root / 'data' / dataset
glove_file = project_root / 'data' / \'glove-sbwc.i25.vec'
labels_names = ['True', 'Fake']

# Train file
labels_file_train = main_dir / 'labels_train.txt'
words_file_train = main_dir / 'split_train/words.txt'

# Test file
labels_file_test = main_dir / 'labels_test.txt'
words_file_test = main_dir / 'split_test/words.txt'

# Reading train data
labels_list_train = read_labels(labels_file_train, labels_names)
corpus_train = []
corpus_train = read_text_data(lang, words_file_train)

# Remove possible elements with length 0
corpus_train, labels_list_train = remove_empty_text_data_glove(corpus_train, labels_list_train)
labels_train = np.asarray(labels_list_train)
labels_set_train = set(labels_list_train)

# Read the GloVe file
glove_dict = glove_reader(glove_file)

# Create the corpus train with GloVe vectors
l_s = [line.split() for line in corpus_train]
s = np.zeros((300))
gv_train = []
l_not_glove = []

for l in l_s:
    i = 0
    for w in l:
        if w in glove_dict:
            s += glove_dict[w]
            i += 1
        else:
            l_not_glove.append(w)
        
    s = s/i
    gv_train.append(s)
    s = np.zeros((300))

# Remove possible elements value equal to NaN
gv_train, labels_train = remove_nan_values(gv_train, labels_train)
gv_corpus_train = np.array(gv_train)

# Normalize the data
n, m = gv_corpus_train.shape
min_values = np.min(gv_corpus_train, axis=1)
min_values = min_values.reshape((n, 1))
gv_corpus_train = gv_corpus_train+abs(min_values)
gv_corpus_train = normalize(gv_corpus_train, norm='l2')

# Reading test data
labels_list_test = read_labels(labels_file_test, labels_names)
corpus_test = []
corpus_test = read_text_data(lang, words_file_test)

# Remove possible elements with length 0
corpus_test, labels_list_test = remove_empty_text_data_glove(corpus_test, labels_list_test)
test_index = [i for i in range(len(labels_list_test))]
test_index = np.asarray(test_index)
labels_test = np.asarray(labels_list_test)
labels_set_test = set(labels_list_test)

# Create the corpus test with GloVe vectors
l_s = [line.split() for line in corpus_test]
s = np.zeros((300)) 
gv_test = []
l_not_glove = []

for l in l_s:
    i = 0
    for w in l:
        if w in glove_dict:
            s += glove_dict[w]
            i += 1
        else:
            l_not_glove.append(w)
        
    s = s/i
    gv_test.append(s)
    s = np.zeros((300))

# Remove possible elements value equal to NaN
gv_test, labels_test = remove_nan_values(gv_test, labels_test)
gv_corpus_test = np.array(gv_test)
n, m = gv_corpus_test.shape

# Normalize the data
min_values = np.min(gv_corpus_test, axis=1)
min_values = min_values.reshape((n, 1))
gv_corpus_test = gv_corpus_test+abs(min_values)
gv_corpus_test = normalize(gv_corpus_test, norm='l2')

# Classifiers to use
classifiers = ['SVM', 'LR', 'MNB', 'SGDC', 'KNN', 'RF']

for cl in classifiers:
    
    print('Training and testing with', cl)
    classifier = cl
    out_dir = project_root / 'results' / \'probabilities/'+classifier+'/'+feature+'/'+dataset+'/'
    
    
    start = time.time()
    
    if classifier == 'SVM':
               
        # Find the best hyper-parameter
        cs = [0.01, 0.1, 1, 10, 100]
        best_c = 0
        best_score = 0
    
        for c in cs:
            clf_inner = svm.SVC(C=c, kernel='linear')
            sub_skf = StratifiedKFold(n_splits=5, random_state=0, shuffle=True)
            scores_inner = cross_val_score(clf_inner, gv_corpus_train, labels_train, scoring='f1_macro', cv=sub_skf)
            score = np.mean(scores_inner)
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
            sub_skf = StratifiedKFold(n_splits=5, random_state=0, shuffle=True)
            scores_inner = cross_val_score(clf_inner, gv_corpus_train, labels_train, scoring='f1_macro', cv=sub_skf)
            score = np.mean(scores_inner)
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
            sub_skf = StratifiedKFold(n_splits=5, random_state=0, shuffle=True)
            scores_inner = cross_val_score(clf_inner, gv_corpus_train, labels_train, scoring='f1_macro', cv=sub_skf)
            score = np.mean(scores_inner)
            if score > best_score:
                best_score = score
                best_c = c
                
        # Create the model with the best hyper-parameter        
        clf = SGDClassifier(loss='log', alpha=1/best_c, max_iter=10000)
        
    elif classifier == 'MNB':
            
            clf = MultinomialNB() 
    
    elif classifier == 'KNN':
        
        # Find the best hyper-parameter
        ks = [1, 2, 3, 5, 10]
        best_score = 0
        best_k = 0
    
        for k in ks:
            clf_inner = KNeighborsClassifier(n_neighbors=k)
            sub_skf = StratifiedKFold(n_splits=5, random_state=0, shuffle=True)
            scores_inner = cross_val_score(clf_inner, gv_corpus_train, labels_train, scoring='f1_macro', cv=sub_skf)
            score = np.mean(scores_inner)
            if score > best_score:
                best_score = score
                best_k = k
                
        # Create the model with the best hyper-parameter         
        clf = KNeighborsClassifier(n_neighbors=best_k)
        
    elif classifier == 'RF':
        
        # Find the best hyper-parameter
        rs = [10, 50, 100, 200, 500]
        best_score = 0
        best_r = 0
    
        for r in rs:
            clf_inner = RandomForestClassifier(n_estimators=r, random_state=0)
            sub_skf = StratifiedKFold(n_splits=5, random_state=0, shuffle=True)
            scores_inner = cross_val_score(clf_inner, gv_corpus_train, labels_train, scoring='f1_macro', cv=sub_skf)
            score = np.mean(scores_inner)
            if score > best_score:
                best_score = score
                best_r = r
                
        # Create the model with the best hyper-parameter        
        clf = RandomForestClassifier(n_estimators=best_r, random_state=0)
    
    clf.fit(gv_corpus_train, labels_train)
    predicted = clf.predict(gv_corpus_test)
    predicted_proba = clf.predict_proba(gv_corpus_test)
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