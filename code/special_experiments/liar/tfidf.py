# -*- coding: utf-8 -*-
"""
Classify fake news using TF-IDF and various classifiers on the LIAR dataset.

This module uses the LIAR dataset (train, validation, and test) and
transforms the data using TF-IDF. SVM, LR, SGDC, MNB, KNN, and RF models
are used for classification.
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

def val_score(clf, train_data, train_labels, valid_data, valid_labels):
    """
    Evaluates a classifier using F1 macro score on validation data.

    Args:
        clf: A classifier instance.
        train_data: Data to train the model.
        train_labels: Labels for training data.
        valid_data: Validation data to test the model.
        valid_labels: Validation labels.

    Returns:
        float: F1 macro score.
    """
    clf.fit(train_data, train_labels)
    predicted = clf.predict(valid_data)
    return metrics.f1_score(valid_labels, predicted, average='macro')

def train_and_test(cl_name, train_tfidf, labels_train, valid_tfidf, labels_valid, corpus_test, test_index, labels_test, vec):
    """
    Finds best hyper-parameters on validation set, then trains on train+valid and tests.

    Args:
        cl_name (str): Name of the classifier.
        train_tfidf: TF-IDF training data.
        labels_train: Training labels.
        valid_tfidf: TF-IDF validation data.
        labels_valid: Validation labels.
        corpus_test: Test corpus.
        test_index: Indices for testing.
        labels_test: Test labels.
        vec: TfidfVectorizer instance.

    Returns:
        dict: Performance metrics and predictions.
    """
    print(f'Training and testing with {cl_name}')
    start = time.time()
    
    if cl_name == 'SVM':
        cs = [0.01, 0.1, 1, 10, 100]
        best_c, best_score = 0, 0
        for c in cs:
            clf_inner = svm.SVC(C=c, kernel='linear')
            score = val_score(clf_inner, train_tfidf, labels_train, valid_tfidf, labels_valid)
            if score > best_score:
                best_score, best_c = score, c
        clf = svm.SVC(C=best_c, kernel='linear', probability=True)
            
    elif cl_name == 'LR':
        cs = [0.01, 0.1, 1, 10, 100]
        best_c, best_score = 0, 0
        for c in cs:
            clf_inner = LogisticRegression(C=c, penalty='l2', solver='liblinear')
            score = val_score(clf_inner, train_tfidf, labels_train, valid_tfidf, labels_valid)
            if score > best_score:
                best_score, best_c = score, c
        clf = LogisticRegression(C=best_c, penalty='l2', solver='liblinear')
    
    elif cl_name == 'MNB':
        clf = MultinomialNB()
    
    elif cl_name == 'SGDC':
        cs = [0.01, 0.1, 1, 10, 100]
        best_c, best_score = 0, 0
        for c in cs:
            clf_inner = SGDClassifier(loss='log', alpha=1/c, max_iter=10000)
            score = val_score(clf_inner, train_tfidf, labels_train, valid_tfidf, labels_valid)
            if score > best_score:
                best_score, best_c = score, c
        clf = SGDClassifier(loss='log', alpha=1/best_c, max_iter=10000)
    
    elif cl_name == 'KNN':
        ks = [1, 2, 3, 5, 10]
        best_k, best_score = 0, 0
        for k in ks:
            clf_inner = KNeighborsClassifier(n_neighbors=k)
            score = val_score(clf_inner, train_tfidf, labels_train, valid_tfidf, labels_valid)
            if score > best_score:
                best_score, best_k = score, k
        clf = KNeighborsClassifier(n_neighbors=best_k)
        
    elif cl_name == 'RF':
        rs = [10, 50, 100, 200, 500]
        best_r, best_score = 0, 0
        for r in rs:
            clf_inner = RandomForestClassifier(n_estimators=r, random_state=0)
            score = val_score(clf_inner, train_tfidf, labels_train, valid_tfidf, labels_valid)
            if score > best_score:
                best_score, best_r = score, r
        clf = RandomForestClassifier(n_estimators=best_r, random_state=0)
    
    return clf, start

def main():
    project_root = Path(__file__).resolve().parent.parent.parent
    project_root = Path(__file__).resolve().parent.parent.parent.parent
        """Main execution function."""
    # Variables
    dataset = 'liar_dataset'
    lang = 'english'
    feature = 'Raw'
    labels_names = ['True', 'Fake']

    # Path handling
    main_dir = project_root / 'data' / dataset
    
    # Define file paths
    files = {
        'train': {
            'labels': main_dir / 'labels_train.txt',
            'words': main_dir / 'split_train/words.txt',
            'hashs': main_dir / 'split_train/hashtags.txt',
            'ats': main_dir / 'split_train/ats.txt',
            'emo': main_dir / 'split_train/emoticons.txt',
            'links': main_dir / 'split_train/links.txt',
            'abvs': main_dir / 'split_train/abvs.txt',
            'raw': main_dir / 'corpus_train.txt'
        },
        'valid': {
            'labels': main_dir / 'labels_valid.txt',
            'words': main_dir / 'split_valid/words.txt',
            'hashs': main_dir / 'split_valid/hashtags.txt',
            'ats': main_dir / 'split_valid/ats.txt',
            'emo': main_dir / 'split_valid/emoticons.txt',
            'links': main_dir / 'split_valid/links.txt',
            'abvs': main_dir / 'split_valid/abvs.txt',
            'raw': main_dir / 'corpus_valid.txt'
        },
        'test': {
            'labels': main_dir / 'labels_test.txt',
            'words': main_dir / 'split_test/words.txt',
            'hashs': main_dir / 'split_test/hashtags.txt',
            'ats': main_dir / 'split_test/ats.txt',
            'emo': main_dir / 'split_test/emoticons.txt',
            'links': main_dir / 'split_test/links.txt',
            'abvs': main_dir / 'split_test/abvs.txt',
            'raw': main_dir / 'corpus_test.txt'
        }
    }

    def load_corpus(split_name):
        f = files[split_name]
        if feature == 'Words':
            return read_text_data(lang, f['words'])
        elif feature == 'AF':
            return read_all_features(lang, f['words'], f['emo'], f['hashs'], f['ats'], f['links'], f['abvs'])
        return read_text_data(lang, f['raw'])

    # Loading data
    labels_list_train = read_labels(files['train']['labels'], labels_names)
    corpus_train = load_corpus('train')
    labels_train = np.asarray(labels_list_train)

    labels_list_valid = read_labels(files['valid']['labels'], labels_names)
    corpus_valid = load_corpus('valid')
    labels_valid = np.asarray(labels_list_valid)

    labels_list_test = read_labels(files['test']['labels'], labels_names)
    corpus_test = load_corpus('test')
    labels_test = np.asarray(labels_list_test)
    test_index = np.arange(len(labels_list_test))

    # Classifiers to use
    classifiers = ['SVM', 'LR', 'MNB', 'SGDC', 'KNN', 'RF']

    # Pre-transform for hyper-parameter search
    vec_init = TfidfVectorizer(min_df=1, norm='l2', analyzer='word', tokenizer=my_tokenizer)
    train_tfidf = vec_init.fit_transform(corpus_train)
    valid_tfidf = vec_init.transform(corpus_valid)

    for cl in classifiers:
        out_dir = project_root / 'results' / cl / feature / dataset
        out_dir.mkdir(parents=True, exist_ok=True)
        
        clf, start_time = train_and_test(cl, train_tfidf, labels_train, valid_tfidf, labels_valid, corpus_test, test_index, labels_test, vec_init)
        
        # Concatenate train and validation data for final training
        corpus_combined = corpus_train + corpus_valid
        labels_combined = np.concatenate([labels_train, labels_valid])
        
        vec = TfidfVectorizer(min_df=1, norm='l2', analyzer='word', tokenizer=my_tokenizer)
        train_combined_tfidf = vec.fit_transform(corpus_combined)
        
        clf.fit(train_combined_tfidf, labels_combined)
        test_tfidf = vec.transform(corpus_test)
        
        predicted = clf.predict(test_tfidf)
        predicted_proba = clf.predict_proba(test_tfidf)
        
        accuracy = np.mean(predicted == labels_test)
        precision = metrics.precision_score(labels_test, predicted, average='macro')
        recall = metrics.recall_score(labels_test, predicted, average='macro')
        f1 = metrics.f1_score(labels_test, predicted, average='macro')
        kappa = metrics.cohen_kappa_score(labels_test, predicted)
        roc = metrics.roc_auc_score(labels_test, predicted, average='macro')
        
        end_time = time.time()
        
        print(f'Accuracy: {accuracy:0.2f}')
        print(f'Precision: {precision:0.2f}')
        print(f'Recall: {recall:0.2f}')
        print(f'F1: {f1:0.2f}')
        print(f'Kappa: {kappa:0.2f}')
        print(f'ROC-AUC: {roc:0.2f}')
        print(f'Time of training + testing: {end_time - start_time:0.2f}\n')

        # Save results
        predictions = np.concatenate((
            test_index[:, None], 
            predicted_proba, 
            predicted[:, None].astype(int), 
            labels_test[:, None].astype(int)
        ), axis=1)
        fmt = '%d', '%1.9f', '%1.9f', '%d', '%d'
        header = 'test_index, probability_true, probability_fake, predicted_class, real_class'
        np.savetxt(out_dir / 'probability.csv', predictions, fmt=fmt, delimiter=',', encoding='utf-8', header=header, comments='')

if __name__ == "__main__":
    main()
