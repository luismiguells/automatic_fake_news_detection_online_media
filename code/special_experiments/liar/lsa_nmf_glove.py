from pathlib import Path
import sys
# -*- coding: utf-8 -*-
"""
Created on Mon May 31 12:22:46 2021

@author: luismiguells

Description: Classify if news is true or fake using the LIAR dataset. 
This dataset is divided in train, validation and test data. Using GloVe vectors
to tranform the data. SVDR and NMFDR are the models
used for the classification.
"""

from sklearn.metrics.pairwise import cosine_similarity
from sklearn.decomposition import TruncatedSVD, NMF
from sklearn.preprocessing import normalize
from nltk.corpus import stopwords
from sklearn import metrics
import numpy as np
import warnings
import time

sys.path.append(str(Path(__file__).resolve().parent.parent.parent))
from utils import read_labels, read_text_data

# Remove to see warnings
warnings.filterwarnings('ignore')

# Variables
dataset = 'liar_dataset'
lang = 'english'
feature = 'GloVe'

# Files
project_root = Path(__file__).resolve().parent.parent.parent.parent
main_dir = project_root / 'data' / dataset
glove_file = project_root / 'data' / \'glove.twitter.27B.200d.txt'
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

# Reading train data
labels_list_train = read_labels(labels_file_train, labels_names)
corpus_train = []
corpus_train = read_text_data(lang, words_file_train)
labels_train = np.asarray(labels_list_train)
labels_set_train = set(labels_list_train)

# Read the GloVe file
glove_dict = glove_reader(glove_file)

# Create the corpus train with GloVe vectors
l_s = [line.split() for line in corpus_train]
s = np.zeros((200)) 
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
    s = np.zeros((200))

# Remove possible elements value equal to NaN
gv_train, labels_train = remove_nan_values(gv_train, labels_train)
gv_corpus_train = np.array(gv_train)

# Normalize the data
n, m = gv_corpus_train.shape
min_values = np.min(gv_corpus_train, axis=1)
min_values = min_values.reshape((n, 1))
gv_corpus_train = gv_corpus_train+abs(min_values)
gv_corpus_train = normalize(gv_corpus_train, norm='l2')

# Reading valid data
labels_list_valid = read_labels(labels_file_valid, labels_names)
corpus_valid = []
corpus_valid = read_text_data(lang, words_file_valid)
labels_valid = np.asarray(labels_list_valid)
labels_set_valid = set(labels_list_valid)

# Create the corpus test with GloVe vectors
l_s = [line.split() for line in corpus_valid]
s = np.zeros((200)) 
gv_valid = []
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
    gv_valid.append(s)
    s = np.zeros((200))

# Remove possible elements value equal to NaN
gv_valid, labels_test = remove_nan_values(gv_valid, labels_valid)
gv_corpus_valid = np.array(gv_valid)

# Normalize the data
n, m = gv_corpus_valid.shape
min_values = np.min(gv_corpus_valid, axis=1)
min_values = min_values.reshape((n, 1))
gv_corpus_valid = gv_corpus_valid+abs(min_values)
gv_corpus_valid = normalize(gv_corpus_valid, norm='l2')

# Reading test data
labels_list_test = read_labels(labels_file_test, labels_names)
corpus_test = []
corpus_test = read_text_data(lang, words_file_test)
test_index = [i for i in range(len(labels_list_test))]
test_index = np.asarray(test_index)
labels_test = np.asarray(labels_list_test)
labels_set_test = set(labels_list_test)

# Create the corpus test with GloVe vectors
l_s = [line.split() for line in corpus_test]
s = np.zeros((200)) 
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
    s = np.zeros((200))

# Remove possible elements value equal to NaN
gv_test, labels_test = remove_nan_values(gv_test, labels_test)
gv_corpus_test = np.array(gv_test)

# Normalize the data
n, m = gv_corpus_test.shape
min_values = np.min(gv_corpus_test, axis=1)
min_values = min_values.reshape((n, 1))
gv_corpus_test = gv_corpus_test+abs(min_values)
gv_corpus_test = normalize(gv_corpus_test, norm='l2')

# Classifiers to use
classifiers = ['SVDR', 'NMFDR']

for cl in classifiers:
    
    print('Training and testing with', cl)
    classifier = cl
    out_dir = project_root / 'results' / \'probabilities/'+classifier+'/'+feature+'/'+dataset+'/'
    
    
    start = time.time()
    
    # Training and testing 
    
    if classifier == 'SVDR':
                    
        # Find the best hyper-parameter
        ns = [1, 2, 4, 8, 16, 32]
        best_n = 0
        best_score = 0
        
        for n in ns:
            sub_svdr = train_pca(gv_corpus_train, labels_train, labels_set_train, n)
            sub_predicted, temp_scores = predict_pca(gv_corpus_valid, sub_svdr)
            score = metrics.f1_score(labels_valid, sub_predicted, average='macro')
            if score > best_score:
                best_score = score
                best_n = n
                
        # Concatenate the train and validation data
        corpus_train_grouped = np.concatenate((gv_corpus_train, gv_corpus_valid))
        labels_train_grouped = labels_list_train+labels_list_valid
        labels_train_grouped = np.asarray(labels_train_grouped) 
        
        # Traing with the best hyper-parameter
        pcadr = train_pca(corpus_train_grouped, labels_train_grouped, labels_set_train, best_n)
        
        # Testing with data 
        predicted, predicted_proba = predict_pca(gv_corpus_test, pcadr)
    
    elif classifier == 'NMFDR':
            
        # Find the best hyper-parameter
        ns = [1, 2, 4, 8, 16, 32]
        best_n = 0
        best_score = 0
        
        for n in ns:
            sub_nmfdr = train_nmf(gv_corpus_train, labels_train, labels_set_train, n)
            sub_predicted, temp_scores = predict_nmf(gv_corpus_valid, sub_svdr)
            score = metrics.f1_score(labels_valid, sub_predicted, average='macro')
            if score > best_score:
                best_score = score
                best_n = n
        
        # Concatenate the train and validation data
        corpus_train_grouped = np.concatenate((gv_corpus_train, gv_corpus_valid))
        labels_train_grouped = labels_list_train+labels_list_valid
        labels_train_grouped = np.asarray(labels_train_grouped) 
        
        
        # Traing with the best hyper-parameter
        nmfdr = train_nmf(corpus_train_grouped, labels_train_grouped, labels_set_train, best_n)
        
        # Testing with data
        predicted, predicted_proba = predict_nmf(gv_corpus_test, nmfdr)
    
    
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