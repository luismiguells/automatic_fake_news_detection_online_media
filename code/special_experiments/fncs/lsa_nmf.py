from pathlib import Path
import sys
# -*- coding: utf-8 -*-
"""
Created on Sun May 30 12:26:06 2021

@author: luismiguells

Description: Classify if news is true or fake using the Fake News Corpus Spanish 
dataset. This dataset is divided in train and test data. Using the method TF-IDF
to tranform the data. SVDR and NMFDR are the models 
used for the classification.
"""

from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity
from sklearn.model_selection import StratifiedKFold
from sklearn.decomposition import TruncatedSVD, NMF
from nltk.corpus import stopwords
from sklearn import metrics
import numpy as np
import warnings
import time

sys.path.append(str(Path(__file__).resolve().parent.parent.parent))
from utils import read_all_features, read_labels, read_text_data

# Remove to see warnings
warnings.filterwarnings('ignore') 

# Variables
dataset = 'fncs_dataset'
lang = 'spanish'
feature = 'Raw'

# Files
project_root = Path(__file__).resolve().parent.parent.parent.parent
main_dir = project_root / 'data' / dataset
labels_names = ['True', 'Fake']

# Train file
labels_file_train = main_dir / 'labels_train.txt'
words_file_train = main_dir / 'split_train/words.txt'
hashs_file_train = main_dir / 'split_train/hashtags.txt'
ats_file_train = main_dir / 'split_train/ats.txt'
emo_file_train = main_dir / 'split_train/emoticons.txt'
links_file_train = main_dir / 'split_train/links.txt'
abvs_file_train = main_dir / 'split_train/abvs.txt'
file_train = main_dir / 'corpus_train.txt'

# Test file
labels_file_test = main_dir / 'labels_test.txt'
words_file_test = main_dir / 'split_test/words.txt'
hashs_file_test = main_dir / 'split_test/hashtags.txt'
ats_file_test = main_dir / 'split_test/ats.txt'
emo_file_test = main_dir / 'split_test/emoticons.txt'
links_file_test = main_dir / 'split_test/links.txt'
abvs_file_test = main_dir / 'split_test/abvs.txt'
file_test = main_dir / 'corpus_test.txt'

# Reading train data
labels_list_train = read_labels(labels_file_train, labels_names)
corpus_train = []

if feature == 'Words':
    corpus_train = read_text_data(lang, words_file_train)
elif feature == 'AF':
    corpus_train = read_all_features(lang, words_file_train, emo_file_train, hashs_file_train, ats_file_train, links_file_train, abvs_file_train)    
elif feature == 'Raw':
    corpus_train = read_text_data(lang, file_train)
    
labels_train = np.asarray(labels_list_train)
labels_set_train = list(set(labels_list_train))

# Reading test data
labels_list_test = read_labels(labels_file_test, labels_names)
corpus_test = []

if feature == 'Words':
    corpus_test = read_text_data(lang, words_file_test)
elif feature == 'AF':
    corpus_test = read_all_features(lang, words_file_test, emo_file_test, hashs_file_test, ats_file_test, links_file_test, abvs_file_test)    
elif feature == 'Raw':
    corpus_test = read_text_data(lang, file_test)

test_index = [i for i in range(len(labels_list_test))]
test_index = np.asarray(test_index)
labels_test = np.asarray(labels_list_test)
labels_set_test = set(labels_list_test)

# Classifiers to use
classifiers = ['SVDR', 'NMFDR']

for cl in classifiers:
    
    print('Training and testing with', cl)
    classifier = cl
    out_dir = project_root / 'results' / \'probabilities/'+classifier+'/'+feature+'/'+dataset+'/'
    
    start = time.time()
    
    # Training and testing 
    vec = TfidfVectorizer(min_df=1, norm='l2', analyzer = 'word', tokenizer=my_tokenizer)
    train_tfidf = vec.fit_transform(corpus_train)
    
    if classifier == 'SVDR':
        
        # Find the best hyper-parameter
        ns = [1, 2, 4, 8, 16, 32]
        best_n = 0
        best_score = 0
        
        for n in ns:
            
            sub_skf = StratifiedKFold(n_splits=5, random_state=0, shuffle=True)
            scores_inner = []
            for sub_train_index, sub_test_index in sub_skf.split(train_tfidf, labels_train):
                sub_data_train, sub_data_test = train_tfidf[sub_train_index], train_tfidf[sub_test_index]
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
        pcadr = train_pca(train_tfidf, labels_train, labels_set_train, best_n)
        
        # Testing with data 
        test_tfidf = vec.transform(corpus_test)
        predicted, predicted_proba = predict_pca(test_tfidf, pcadr)
                
    elif classifier == 'NMFDR':
        
        # Find the best hyper-parameter
        ns = [1, 2, 4, 8, 16, 32]
        best_n = 0
        best_score = 0
        
        for n in ns:
            
            sub_skf = StratifiedKFold(n_splits=5, random_state=0, shuffle=True)
            scores_inner = []
            for sub_train_index, sub_test_index in sub_skf.split(train_tfidf, labels_train):
                sub_data_train, sub_data_test = train_tfidf[sub_train_index], train_tfidf[sub_test_index]
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
        nmfdr = train_nmf(train_tfidf, labels_train, labels_set_train, best_n)
        
        # Testing with data
        test_tfidf = vec.transform(corpus_test)
        predicted, predicted_proba = predict_nmf(test_tfidf, nmfdr)
    
    
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