# -*- coding: utf-8 -*-
"""
Classify fake news using TF-IDF and various classifiers.

This module uses the Fake News Corpus Spanish dataset and transforms
the data using TF-IDF. SVM, LR, SGDC, MNB, KNN, and RF models are used
for classification.
"""

import sys
import time
import warnings
from pathlib import Path

import numpy as np
from sklearn import metrics, svm
from sklearn.ensemble import RandomForestClassifier
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression, SGDClassifier
from sklearn.model_selection import StratifiedKFold, cross_val_score
from sklearn.neighbors import KNeighborsClassifier
from sklearn.naive_bayes import MultinomialNB

sys.path.append(str(Path(__file__).resolve().parent.parent.parent))
from utils import read_all_features, read_labels, read_text_data

# Add parent directory to path to import utils

    my_tokenizer, read_labels, read_text_data, 
    read_text_data_with_emos, read_all_features
)

# Remove to see warnings
warnings.filterwarnings('ignore')

def train_and_test(cl_name, train_tfidf, labels_train, corpus_test, test_index, labels_test, vec):
    """
    Trains and tests a classifier.

    Args:
        cl_name (str): Name of the classifier.
        train_tfidf: TF-IDF transformed training data.
        labels_train: Training labels.
        corpus_test: Test corpus.
        test_index: Indices for testing.
        labels_test: Test labels.
        vec: TfidfVectorizer instance.

    Returns:
        dict: A dictionary containing performance metrics.
    """
    print(f'Training and testing with {cl_name}')
    start = time.time()
    
    if cl_name == 'SVM':
        cs = [0.01, 0.1, 1, 10, 100]
        best_c = 0
        best_score = 0
        for c in cs:
            clf_inner = svm.SVC(C=c, kernel='linear')
            sub_skf = StratifiedKFold(n_splits=5, random_state=0, shuffle=True)
            scores_inner = cross_val_score(clf_inner, train_tfidf, labels_train, scoring='f1_macro', cv=sub_skf)
            score = np.mean(scores_inner)
            if score > best_score:
                best_score = score
                best_c = c
        clf = svm.SVC(C=best_c, kernel='linear', probability=True)
            
    elif cl_name == 'LR':
        cs = [0.01, 0.1, 1, 10, 100]
        best_c = 0
        best_score = 0
        for c in cs:
            clf_inner = LogisticRegression(C=c, penalty='l2', solver='liblinear')
            sub_skf = StratifiedKFold(n_splits=5, random_state=0, shuffle=True)
            scores_inner = cross_val_score(clf_inner, train_tfidf, labels_train, scoring='f1_macro', cv=sub_skf)
            score = np.mean(scores_inner)
            if score > best_score:
                best_score = score
                best_c = c
        clf = LogisticRegression(C=best_c, penalty='l2', solver='liblinear')
    
    elif cl_name == 'MNB':
        clf = MultinomialNB()
    
    elif cl_name == 'SGDC':
        cs = [0.01, 0.1, 1, 10, 100]
        best_c = 0
        best_score = 0
        for c in cs:
            clf_inner = SGDClassifier(loss='log', alpha=1/c, max_iter=10000)
            sub_skf = StratifiedKFold(n_splits=5, random_state=0, shuffle=True)
            scores_inner = cross_val_score(clf_inner, train_tfidf, labels_train, scoring='f1_macro', cv=sub_skf)
            score = np.mean(scores_inner)
            if score > best_score:
                best_score = score
                best_c = c
        clf = SGDClassifier(loss='log', alpha=1/best_c, max_iter=10000)
    
    elif cl_name == 'KNN':
        ks = [1, 2, 3, 5, 10]
        best_score = 0
        best_k = 0
        for k in ks:
            clf_inner = KNeighborsClassifier(n_neighbors=k)
            sub_skf = StratifiedKFold(n_splits=5, random_state=0, shuffle=True)
            scores_inner = cross_val_score(clf_inner, train_tfidf, labels_train, scoring='f1_macro', cv=sub_skf)
            score = np.mean(scores_inner)
            if score > best_score:
                best_score = score
                best_k = k
        clf = KNeighborsClassifier(n_neighbors=best_k)
        
    elif cl_name == 'RF':
        rs = [10, 50, 100, 200, 500]
        best_score = 0
        best_r = 0
        for r in rs:
            clf_inner = RandomForestClassifier(n_estimators=r, random_state=0)
            sub_skf = StratifiedKFold(n_splits=5, random_state=0, shuffle=True)
            scores_inner = cross_val_score(clf_inner, train_tfidf, labels_train, scoring='f1_macro', cv=sub_skf)
            score = np.mean(scores_inner)
            if score > best_score:
                best_score = score
                best_r = r
        clf = RandomForestClassifier(n_estimators=best_r, random_state=0)
    
    clf.fit(train_tfidf, labels_train)
    test_tfidf = vec.transform(corpus_test)
    predicted = clf.predict(test_tfidf)
    predicted_proba = clf.predict_proba(test_tfidf)
    
    accuracy = np.mean(predicted == labels_test)
    precision_macro = metrics.precision_score(labels_test, predicted, average='macro')
    recall_macro = metrics.recall_score(labels_test, predicted, average='macro')
    f1_macro = metrics.f1_score(labels_test, predicted, average='macro')
    kapha = metrics.cohen_kappa_score(labels_test, predicted) 
    roc = metrics.roc_auc_score(labels_test, predicted, average='macro')
    
    end = time.time()
    
    results = {
        'accuracy': accuracy,
        'precision': precision_macro,
        'recall': recall_macro,
        'f1': f1_macro,
        'kappa': kapha,
        'roc': roc,
        'time': end - start,
        'predicted_proba': predicted_proba,
        'predicted': predicted
    }
    
    print(f'Accuracy: {accuracy:0.2f}')
    print(f'Precision: {precision_macro:0.2f}')
    print(f'Recall: {recall_macro:0.2f}')
    print(f'F1: {f1_macro:0.2f}')
    print(f'Kappa: {kapha:0.2f}')
    print(f'ROC-AUC: {roc:0.2f}')
    print(f'Time of training + testing: {end - start:0.2f}\n')
    
    return results

def main():
    project_root = Path(__file__).resolve().parent.parent.parent
    project_root = Path(__file__).resolve().parent.parent.parent.parent
        """Main execution function."""
    # Variables
    dataset = 'fncs_dataset'
    lang = 'spanish'
    feature = 'Raw'
    labels_names = ['True', 'Fake']

    # Files
    main_dir = project_root / 'data' / dataset
    
    # Train files
    labels_file_train = main_dir / 'labels_train.txt'
    words_file_train = main_dir / 'split_train/words.txt'
    hashs_file_train = main_dir / 'split_train/hashtags.txt'
    ats_file_train = main_dir / 'split_train/ats.txt'
    emo_file_train = main_dir / 'split_train/emoticons.txt'
    links_file_train = main_dir / 'split_train/links.txt'
    abvs_file_train = main_dir / 'split_train/abvs.txt'
    file_train = main_dir / 'corpus_train.txt'

    # Test files
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
    
    if feature == 'Words':
        corpus_train = read_text_data(lang, words_file_train)
    elif feature == 'AF':
        corpus_train = read_all_features(lang, words_file_train, emo_file_train, hashs_file_train, ats_file_train, links_file_train, abvs_file_train)    
    elif feature == 'Raw':
        corpus_train = read_text_data(lang, file_train)
        
    labels_train = np.asarray(labels_list_train)

    # Reading test data
    labels_list_test = read_labels(labels_file_test, labels_names)
    
    if feature == 'Words':
        corpus_test = read_text_data(lang, words_file_test)
    elif feature == 'AF':
        corpus_test = read_all_features(lang, words_file_test, emo_file_test, hashs_file_test, ats_file_test, links_file_test, abvs_file_test)    
    elif feature == 'Raw':
        corpus_test = read_text_data(lang, file_test)

    test_index = np.arange(len(labels_list_test))
    labels_test = np.asarray(labels_list_test)

    # Classifiers to use    
    classifiers = ['SVM', 'LR', 'MNB', 'SGDC', 'KNN', 'RF']

    vec = TfidfVectorizer(min_df=1, norm='l2', analyzer='word', tokenizer=my_tokenizer)
    train_tfidf = vec.fit_transform(corpus_train)

    for cl in classifiers:
        out_dir = project_root / 'results' / cl / feature / dataset
        out_dir.mkdir(parents=True, exist_ok=True)
        
        res = train_and_test(cl, train_tfidf, labels_train, corpus_test, test_index, labels_test, vec)
        
        # Save results
        predictions = np.concatenate((
            test_index[:, None], 
            res['predicted_proba'], 
            res['predicted'][:, None].astype(int), 
            labels_test[:, None].astype(int)
        ), axis=1)
        fmt = '%d', '%1.9f', '%1.9f', '%d', '%d'
        header = 'test_index, probability_true, probability_fake, predicted_class, real_class'
        np.savetxt(out_dir / 'probability.csv', predictions, fmt=fmt, delimiter=',', encoding='utf-8', header=header, comments='')

if __name__ == "__main__":
    main()
