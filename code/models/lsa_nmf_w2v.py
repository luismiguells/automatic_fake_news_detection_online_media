#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Classify if news is true or fake using various datasets and Word2Vec vectors.

This module employs SVD-based and NMF-based dimensionality reduction for
classification. Datasets include Covid, FakeNewsNet, ISOT, and Fake News
Costa Rica News Dataset.
"""

import time
import sys
import warnings
from pathlib import Path

import numpy as np
from gensim.models import Word2Vec
from sklearn import metrics
from sklearn.decomposition import NMF, TruncatedSVD
from sklearn.metrics.pairwise import cosine_similarity
from sklearn.model_selection import StratifiedKFold
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


def train_pca(data_train, labels_train, labels_set, n_components):
    """Trains a TruncatedSVD model for each class.

    Args:
        data_train (np.ndarray): Training data.
        labels_train (np.ndarray): Training labels.
        labels_set (set): Set of unique labels.
        n_components (int): Number of SVD components.

    Returns:
        list: A list of TruncatedSVD models, one per class.
    """
    pcadr = []
    for label in labels_set:
        cat_train = data_train[labels_train == label]
        svd = TruncatedSVD(n_components=n_components, n_iter=15,
                           random_state=0)
        svd.fit(cat_train)
        pcadr.append(svd)
    return pcadr


def predict_pca(data_test, pcadr):
    """Predicts class labels using class-specific SVD models.

    Args:
        data_test (np.ndarray): Test data.
        pcadr (list): List of TruncatedSVD models.

    Returns:
        tuple: (predicted_labels, probabilities)
    """
    predicted = []
    probas = []
    for example in data_test:
        example = example.reshape(1, -1)
        sims = []
        for svdpca in pcadr:
            example_proj = svdpca.transform(example)
            example_rec = svdpca.inverse_transform(example_proj)
            norm = np.linalg.norm(example_rec)
            if norm > 0 and not np.isnan(norm):
                example_rec = example_rec / norm
            
            sim = cosine_similarity(example_rec, example)
            sims.append(sim[0, 0])
        
        predicted.append(np.argmax(sims))
        tot = sum(sims)
        probas.append([p / tot if tot > 0 else 0 for p in sims])
    
    return np.array(predicted), np.array(probas)


def train_nmf(data_train, labels_train, labels_set, n_components):
    """Trains an NMF model for each class.

    Args:
        data_train (np.ndarray): Training data.
        labels_train (np.ndarray): Training labels.
        labels_set (set): Set of unique labels.
        n_components (int): Number of NMF components.

    Returns:
        list: A list of NMF models, one per class.
    """
    nmfdr = []
    for label in labels_set:
        cat_train = data_train[labels_train == label]
        model = NMF(n_components=n_components, init='random', random_state=0)
        model.fit(cat_train)
        nmfdr.append(model)
    return nmfdr


def predict_nmf(data_test, nmfdr):
    """Predicts class labels using class-specific NMF models.

    Args:
        data_test (np.ndarray): Test data.
        nmfdr (list): List of NMF models.

    Returns:
        tuple: (predicted_labels, probabilities)
    """
    predicted = []
    probas = []
    for example in data_test:
        example = example.reshape(1, -1)
        sims = []
        for nmf_model in nmfdr:
            example_proj = nmf_model.transform(example)
            example_rec = nmf_model.inverse_transform(example_proj)
            norm = np.linalg.norm(example_rec)
            if norm > 0 and not np.isnan(norm):
                example_rec = example_rec / norm
            
            sim = cosine_similarity(example_rec, example)
            sims.append(sim[0, 0])
            
        predicted.append(np.argmax(sims))
        tot = sum(sims)
        probas.append([p / tot if tot > 0 else 0 for p in sims])

    return np.array(predicted), np.array(probas)


def main():
    project_root = Path(__file__).resolve().parent.parent.parent
    """Main execution block for SVD/NMF-based model training and evaluation."""
    # Configuration variables
    dataset = 'fncn_dataset'
    lang = 'spanish'
    feature = 'Word2Vec'
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

    # Reading data
    labels_list = read_labels(labels_file, labels_names)
    corpus = read_text_data(lang, words_file)

    # Pre-processing
    corpus, labels_list = remove_empty_text_data_w2v(corpus, labels_list)
    labels = np.asarray(labels_list)
    labels_set = set(labels_list)

    # Create document vectors
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

    w2v_corpus_list, labels = remove_nan_values(w2v_corpus_list, labels)
    w2v_corpus = np.array(w2v_corpus_list)

    # Normalize the data
    n_samples = w2v_corpus.shape[0]
    min_values = np.min(w2v_corpus, axis=1).reshape((n_samples, 1))
    w2v_corpus = w2v_corpus + abs(min_values)
    w2v_corpus = normalize(w2v_corpus, norm='l2')

    # Classifiers to evaluate
    classifiers = ['SVDR', 'NMFDR']

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

            best_n = 1
            best_score = 0
            ns = [1, 2, 4, 8, 16, 32]

            for n in ns:
                sub_skf = StratifiedKFold(n_splits=3, random_state=0, shuffle=True)
                inner_f1s = []
                for sub_train_idx, sub_test_idx in sub_skf.split(data_train, labels_train_fold):
                    sub_d_train, sub_d_test = data_train[sub_train_idx], data_train[sub_test_idx]
                    sub_l_train, sub_l_test = labels_train_fold[sub_train_idx], labels_train_fold[sub_test_idx]
                    
                    if cl == 'SVDR':
                        models = train_pca(sub_d_train, sub_l_train, labels_set, n)
                        pred, _ = predict_pca(sub_d_test, models)
                    else:
                        models = train_nmf(sub_d_train, sub_l_train, labels_set, n)
                        pred, _ = predict_nmf(sub_d_test, models)
                    
                    inner_f1s.append(metrics.f1_score(sub_l_test, pred, average='macro'))
                
                avg_f1 = np.mean(inner_f1s)
                if avg_f1 > best_score:
                    best_score = avg_f1
                    best_n = n

            # Final training for fold with best n
            if cl == 'SVDR':
                final_models = train_pca(data_train, labels_train_fold, labels_set, best_n)
                predicted, predicted_proba = predict_pca(data_test, final_models)
            else:
                final_models = train_nmf(data_train, labels_train_fold, labels_set, best_n)
                predicted, predicted_proba = predict_nmf(data_test, final_models)

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
