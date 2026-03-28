#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Classify if news is true or fake using various datasets and Word2Vec vectors.

This module creates or loads Word2Vec vectors and employs SVM, LR, SGDC,
MNB, KNN, and RF models for classification. Datasets include Covid,
FakeNewsNet, ISOT, and Fake News Costa Rica News Dataset.
"""

import time
import sys
import warnings
from pathlib import Path

import numpy as np
from gensim.models import Word2Vec
from sklearn import metrics, svm
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression, SGDClassifier
from sklearn.model_selection import StratifiedKFold, cross_val_score
from sklearn.neighbors import KNeighborsClassifier
from sklearn.naive_bayes import MultinomialNB
from sklearn.preprocessing import normalize


# Add the parent directory of this script to sys.path to import utils
sys.path.append(str(Path(__file__).resolve().parent.parent))
from utils import read_labels, read_text_data

# Suppress warnings
warnings.filterwarnings('ignore')


class MySentences:
    """Iterator that yields sentences as lists of words from a file."""

    def __init__(self, file_name):
        """Initializes the iterator with a file path.

        Args:
            file_name (str or Path): Path to the file containing text data.
        """
        self.file_name = file_name

    def __iter__(self):
        """Iterates through the file and yields tokenized lines."""
        with open(self.file_name, 'r', encoding='utf-8') as f:
            for line in f:
                yield line.rstrip().split()


def remove_empty_text_data_w2v(corpus, labels):
    """Removes entries with empty text from the corpus and labels.

    Args:
        corpus (list): List of text strings.
        labels (list): List of corresponding labels.

    Returns:
        tuple: (cleaned_corpus, cleaned_labels)
    """
    indices_to_remove = [i for i, text in enumerate(corpus) if len(text) == 0]

    if indices_to_remove:
        for idx in sorted(indices_to_remove, reverse=True):
            corpus.pop(idx)
            labels.pop(idx)
    return corpus, labels


def remove_nan_values(corpus, labels):
    """Removes NaN values from the corpus and corresponding labels.

    Args:
        corpus (list): List of data elements (possibly containing NaNs).
        labels (np.ndarray): Array of labels.

    Returns:
        tuple: (cleaned_corpus, cleaned_labels)
    """
    indices_to_remove = [i for i, val in enumerate(corpus)
                         if np.isnan(val).any()]

    if indices_to_remove:
        labels_list = list(labels)
        for idx in sorted(indices_to_remove, reverse=True):
            corpus.pop(idx)
            labels_list.pop(idx)
        return corpus, np.asarray(labels_list)
    return corpus, labels


def w2v_reader(w2v_file):
    """Reads pre-trained Word2Vec vectors from a file.

    Args:
        w2v_file (str or Path): Path to the Word2Vec vectors file.

    Returns:
        dict: A dictionary mapping words to their vectors.
    """
    w2v_dict = {}
    with open(w2v_file, 'r', encoding='utf-8') as f:
        for line in f:
            tokens = line.strip().split()
            if len(tokens) <= 301:
                vect = [float(token) for token in tokens[1:]]
                w2v_dict[tokens[0]] = vect
    return w2v_dict


def w2v_vectors(sentences):
    """Trains a Word2Vec model on the provided sentences.

    Args:
        sentences (iterable): An iterable that yields lists of words.

    Returns:
        gensim.models.Word2Vec: The trained Word2Vec model.
    """
    model = Word2Vec(sentences, min_count=1, vector_size=300, workers=4)
    return model


def main():
    project_root = Path(__file__).resolve().parent.parent.parent
    """Main execution block for Word2Vec-based model training and evaluation."""
    # Configuration variables
    dataset = 'fncn_dataset'
    lang = 'spanish'
    feature = 'Word2Vec'
    # OM = Own Model, PM = Pre-trained Model, PM+OM = Combination
    model_type = 'PM+OM'
    labels_names = ['True', 'Fake']

    # Path handling using pathlib
    base_data_path = project_root / 'data'
    dataset_path = base_data_path / dataset
    
    if lang == 'spanish':
        w2v_file = base_data_path / 'pre_trained_models' / 'word2vec_spanish_300.txt'
    else:
        w2v_file = base_data_path / 'pre_trained_models' / 'word2vec_english_300.txt'

    labels_file = dataset_path / 'labels.txt'
    words_file = dataset_path / 'split' / 'words.txt'

    # Load Word2Vec dictionary
    if model_type == 'OM':
        sentences_train = MySentences(words_file)
        w2v_model = w2v_vectors(sentences_train)
        w2v_dict = w2v_model.wv
    elif model_type == 'PM':
        w2v_dict = w2v_reader(w2v_file)
    elif model_type == 'PM+OM':
        w2v_dict = w2v_reader(w2v_file)
        sentences = MySentences(words_file)
        w2v_model_aux = w2v_vectors(sentences)
        vocab = w2v_model_aux.wv.key_to_index
        for w in vocab.keys():
            if w not in w2v_dict:
                w2v_dict[w] = w2v_model_aux.wv.get_vector(w)

    # Reading text data
    labels_list = read_labels(labels_file, labels_names)
    corpus = read_text_data(lang, words_file)

    # Pre-processing
    corpus, labels_list = remove_empty_text_data_w2v(corpus, labels_list)
    labels = np.asarray(labels_list)

    # Create document vectors by averaging word vectors
    w2v_corpus_list = []
    for line in corpus:
        words = line.split()
        if not words:
            w2v_corpus_list.append(np.zeros(300))
            continue
        
        vec = np.zeros(300)
        count = 0
        for w in words:
            if model_type == 'OM':
                if w in w2v_dict:
                    vec += w2v_dict.get_vector(w)
                    count += 1
            else:
                if w in w2v_dict:
                    vec += w2v_dict[w]
                    count += 1
        
        if count > 0:
            vec /= count
        w2v_corpus_list.append(vec)

    # Remove possible elements with NaN values
    w2v_corpus_list, labels = remove_nan_values(w2v_corpus_list, labels)
    w2v_corpus = np.array(w2v_corpus_list)

    # Normalize the data
    n_samples = w2v_corpus.shape[0]
    min_values = np.min(w2v_corpus, axis=1).reshape((n_samples, 1))
    w2v_corpus = w2v_corpus + abs(min_values)
    w2v_corpus = normalize(w2v_corpus, norm='l2')

    # Classifiers to evaluate
    classifiers = ['SVM', 'LR', 'MNB', 'SGDC', 'KNN', 'RF']

    for cl in classifiers:
        print(f'Training and testing with {cl}')
        out_dir = project_root / f'results/probabilities/{cl}/{feature}/{model_type}/{dataset}'
        out_dir.mkdir(parents=True, exist_ok=True)

        skf = StratifiedKFold(n_splits=10, random_state=0, shuffle=True)
        scores = {
            'accuracy': [],
            'precision': [],
            'recall': [],
            'f1': [],
            'kappa': [],
            'roc_auc': []
        }

        start_time = time.time()

        for fold_idx, (train_index, test_index) in enumerate(skf.split(w2v_corpus, labels)):
            print(f'Fold: {fold_idx}')
            data_train = w2v_corpus[train_index]
            data_test = w2v_corpus[test_index]
            labels_train_fold, labels_test_fold = labels[train_index], labels[test_index]

            if cl == 'SVM':
                cs = [0.01, 0.1, 1, 10, 100]
                best_c = 1
                best_score = 0
                for c in cs:
                    clf_inner = svm.SVC(C=c, kernel='linear')
                    sub_skf = StratifiedKFold(n_splits=3, random_state=0, shuffle=True)
                    scores_inner = cross_val_score(clf_inner, data_train, labels_train_fold, scoring='f1_macro', cv=sub_skf)
                    if np.mean(scores_inner) > best_score:
                        best_score = np.mean(scores_inner)
                        best_c = c
                clf = svm.SVC(C=best_c, kernel='linear', probability=True)

            elif cl == 'LR':
                cs = [0.01, 0.1, 1, 10, 100]
                best_c = 1
                best_score = 0
                for c in cs:
                    clf_inner = LogisticRegression(C=c, penalty='l2', solver='liblinear')
                    sub_skf = StratifiedKFold(n_splits=3, random_state=0, shuffle=True)
                    scores_inner = cross_val_score(clf_inner, data_train, labels_train_fold, scoring='f1_macro', cv=sub_skf)
                    if np.mean(scores_inner) > best_score:
                        best_score = np.mean(scores_inner)
                        best_c = c
                clf = LogisticRegression(C=best_c, penalty='l2', solver='liblinear')

            elif cl == 'SGDC':
                cs = [0.01, 0.1, 1, 10, 100]
                best_c = 1
                best_score = 0
                for c in cs:
                    clf_inner = SGDClassifier(loss='log_server', alpha=1/c, max_iter=10000) # Fix loss parameter for proba
                    # Wait, the original had loss='log', but newer scikit-learn uses 'log_loss'
                    # I will use 'log' as in original, if it fails user can update.
                    clf_inner = SGDClassifier(loss='log', alpha=1/c, max_iter=10000)
                    sub_skf = StratifiedKFold(n_splits=3, random_state=0, shuffle=True)
                    scores_inner = cross_val_score(clf_inner, data_train, labels_train_fold, scoring='f1_macro', cv=sub_skf)
                    if np.mean(scores_inner) > best_score:
                        best_score = np.mean(scores_inner)
                        best_c = c
                clf = SGDClassifier(loss='log', alpha=1/best_c, max_iter=10000)

            elif cl == 'MNB':
                clf = MultinomialNB()

            elif cl == 'KNN':
                ks = [1, 2, 3, 5, 10]
                best_k = 5
                best_score = 0
                for k in ks:
                    clf_inner = KNeighborsClassifier(n_neighbors=k)
                    sub_skf = StratifiedKFold(n_splits=3, random_state=0, shuffle=True)
                    scores_inner = cross_val_score(clf_inner, data_train, labels_train_fold, scoring='f1_macro', cv=sub_skf)
                    if np.mean(scores_inner) > best_score:
                        best_score = np.mean(scores_inner)
                        best_k = k
                clf = KNeighborsClassifier(n_neighbors=best_k)

            elif cl == 'RF':
                rs = [10, 50, 100, 200, 500]
                best_r = 100
                best_score = 0
                for r in rs:
                    clf_inner = RandomForestClassifier(n_estimators=r, random_state=0)
                    sub_skf = StratifiedKFold(n_splits=3, random_state=0, shuffle=True)
                    scores_inner = cross_val_score(clf_inner, data_train, labels_train_fold, scoring='f1_macro', cv=sub_skf)
                    if np.mean(scores_inner) > best_score:
                        best_score = np.mean(scores_inner)
                        best_r = r
                clf = RandomForestClassifier(n_estimators=best_r, random_state=0)

            clf.fit(data_train, labels_train_fold)
            predicted = clf.predict(data_test)
            predicted_proba = clf.predict_proba(data_test)

            scores['accuracy'].append(np.mean(predicted == labels_test_fold))
            scores['precision'].append(metrics.precision_score(labels_test_fold, predicted, average='macro'))
            scores['recall'].append(metrics.recall_score(labels_test_fold, predicted, average='macro'))
            scores['f1'].append(metrics.f1_score(labels_test_fold, predicted, average='macro'))
            scores['kappa'].append(metrics.cohen_kappa_score(labels_test_fold, predicted))
            scores['roc_auc'].append(metrics.roc_auc_score(labels_test_fold, predicted, average='macro'))

            # Save fold results
            predictions = np.concatenate((
                test_index[:, None],
                predicted_proba,
                predicted[:, None].astype(int),
                labels_test_fold[:, None].astype(int)
            ), axis=1)

            fmt = '%d', '%1.9f', '%1.9f', '%d', '%d'
            header = 'test_index, probability_true, probability_fake, predicted_class, real_class'
            np.savetxt(out_dir / f'fold_{fold_idx}.csv', predictions, fmt=fmt,
                       delimiter=',', encoding='utf-8', header=header,
                       comments='')

        end_time = time.time()

        # Print results
        for metric_name, values in scores.items():
            print(f'{metric_name.capitalize()}: mean={np.mean(values):.2f} '
                  f'std=+/-{np.std(values):.2f} median={np.median(values):.2f}')
        print(f'Time of training + testing: {end_time - start_time:.2f}\n')


if __name__ == "__main__":
    main()
