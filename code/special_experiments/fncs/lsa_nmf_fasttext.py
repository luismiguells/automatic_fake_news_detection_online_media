from pathlib import Path
import sys
# -*- coding: utf-8 -*-
"""
Created on Sun Jun  6 16:55:41 2021

@author: luismiguells

Description: Classify if news is true or fake using the Fake News Corpus Spanish 
dataset. This dataset is divided in train and test data. Creating fastText vectors
to tranform the data. SVDR and NMFDR are the models used for 
the classification.
"""

from sklearn.metrics.pairwise import cosine_similarity
from sklearn.model_selection import StratifiedKFold
from sklearn.decomposition import TruncatedSVD, NMF
from sklearn.preprocessing import normalize
from nltk.corpus import stopwords
from sklearn import metrics
import numpy as np
import fasttext
import warnings
import time

sys.path.append(str(Path(__file__).resolve().parent.parent.parent))
from utils import read_labels, read_text_data

# Remove to see warnings
warnings.filterwarnings('ignore')

# Variables
dataset = 'fncs_dataset'
lang = 'spanish'
feature = 'fastText'

# OM = Own Model PM = Pre-trained Model PM+OM = Combination of both
model_type = 'PM+OM' 

# Files
project_root = Path(__file__).resolve().parent.parent.parent.parent
main_dir = project_root / 'data' / dataset
fasttext_file = project_root / 'data' / \'pre_trained_models/fasttext_spanish_300.vec'
labels_names = ['True', 'Fake']

# Train file
labels_file_train = main_dir / 'labels_train.txt'
words_file_train = main_dir / 'split_train/words.txt'

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
labels_set_train = set(labels_list_train)

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

# Reading test data
labels_list_test = read_labels(labels_file_test, labels_names)
corpus_test = []
corpus_test = read_text_data(lang, words_file_test)
test_index = [i for i in range(len(labels_list_test))]
test_index = np.asarray(test_index)
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

# Normalize the data
n, m = ft_corpus_test.shape
min_values = np.min(ft_corpus_test, axis=1)
min_values = min_values.reshape((n, 1))
ft_corpus_test = ft_corpus_test+abs(min_values)
ft_corpus_test = normalize(ft_corpus_test, norm='l2')

# Classfiers to use
classifiers = ['SVDR', 'NMFDR']

for cl in classifiers:
    
    print('Training and testing with', cl)
    classifier = cl
    out_dir = project_root / 'results' / \'probabilities/'+classifier+'/'+feature+'/'+model_type+'/'+dataset+'/'
    
    start = time.time()
    
    # Training and testing 
    
    if classifier == 'SVDR':
        
        # Find the best hyper-parameter
        ns = [1, 2, 4, 8, 16, 32]
        best_n = 0
        best_score = 0
        
        for n in ns:
            
            sub_skf = StratifiedKFold(n_splits=5, random_state=0, shuffle=True)
            scores_inner = []
            for sub_train_index, sub_test_index in sub_skf.split(ft_corpus_train, labels_train):
                sub_data_train, sub_data_test = ft_corpus_train[sub_train_index], ft_corpus_train[sub_test_index]
                sub_labels_train, sub_labels_test = labels_train[sub_train_index], labels_train[sub_test_index]
                sub_svdr = train_pca(sub_data_train, sub_labels_train, labels_set_train, n)
                sub_predicted, temp_scores = predict_pca(sub_data_test, sub_svdr)
                sub_f1_macro = metrics.f1_score(sub_labels_test, sub_predicted, average='macro')
                scores_inner.append(sub_f1_macro)
            score = np.mean(scores_inner)
            if score > best_score:
                best_score = score
                best_n = n
        
        # Traing with the best hyper-parameter
        pcadr = train_pca(ft_corpus_train, labels_train, labels_set_train, best_n)
        
        # Testing with data 
        predicted, predicted_proba = predict_pca(ft_corpus_test, pcadr)
                
    elif classifier == 'NMFDR':
        
        # Find the best hyper-parameter
        ns = [1, 2, 4, 8, 16, 32]
        best_n = 0
        best_score = 0
        
        for n in ns:
            
            sub_skf = StratifiedKFold(n_splits=5, random_state=0, shuffle=True)
            scores_inner = []
            for sub_train_index, sub_test_index in sub_skf.split(ft_corpus_train, labels_train):
                sub_data_train, sub_data_test = ft_corpus_train[sub_train_index], ft_corpus_train[sub_test_index]
                sub_labels_train, sub_labels_test = labels_train[sub_train_index], labels_train[sub_test_index]
                sub_nmfdr = train_nmf(sub_data_train, sub_labels_train, labels_set_train, n)
                sub_predicted, temp_scores = predict_nmf(sub_data_test, sub_nmfdr)
                sub_f1_macro = metrics.f1_score(sub_labels_test, sub_predicted, average='macro')
                scores_inner.append(sub_f1_macro)
            score = np.mean(scores_inner)
            if score > best_score:
                best_score = score
                best_n = n
                
        # Traing with the best hyper-parameter
        nmfdr = train_nmf(ft_corpus_train, labels_train, labels_set_train, best_n)
        
        # Testing with data
        predicted, predicted_proba = predict_nmf(ft_corpus_test, nmfdr)
    
    
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
