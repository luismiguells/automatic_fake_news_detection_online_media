# -*- coding: utf-8 -*-
"""
LSA (SVD) and NMF model with GloVe vectors for fake news classification.
Combines GloVe word embeddings with dimensionality reduction techniques 
(TruncatedSVD and NMF) for classification based on reconstruction similarity.
"""

import time
import sys
import warnings
import numpy as np
from pathlib import Path
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

def glove_reader(glove_file):
    """
    Reads a GloVe file and returns a dictionary of word vectors.

    Args:
        glove_file (Path): Path to the GloVe file.

    Returns:
        dict: A dictionary where keys are words and values are vectors.
    """
    glove_dict = {}
    with open(glove_file, 'r', encoding='utf-8') as f:
        for line in f:
            tokens = line.strip().split()
            if len(tokens) > 1:
                word = tokens[0]
                vector = [float(token) for token in tokens[1:]]
                glove_dict[word] = vector
    return glove_dict

def remove_empty_text_data_glove(corpus, labels):
    """
    Removes elements from corpus and labels where the corpus entry is empty.

    Args:
        corpus (list): List of text strings.
        labels (list): List of labels.

    Returns:
        tuple: (cleaned_corpus, cleaned_labels)
    """
    indices_to_remove = [i for i, line in enumerate(corpus) if len(line) == 0]
    for idx in sorted(indices_to_remove, reverse=True):
        corpus.pop(idx)
        labels.pop(idx)
    return corpus, labels

def remove_nan_values(corpus, labels):
    """
    Removes elements from corpus and labels where the corpus entry contains NaN values.

    Args:
        corpus (list/np.array): List or array of vectors.
        labels (np.array): Array of labels.

    Returns:
        tuple: (cleaned_corpus, cleaned_labels)
    """
    indices_to_remove = [i for i, vec in enumerate(corpus) if np.isnan(vec).any()]
    if indices_to_remove:
        corpus = list(corpus)
        labels = list(labels)
        for idx in sorted(indices_to_remove, reverse=True):
            corpus.pop(idx)
            labels.pop(idx)
        return corpus, np.asarray(labels)
    return corpus, labels

def train_svdr(data_train, labels_train, labels_set, n_components):
    """
    Trains a TruncatedSVD model for each class.

    Args:
        data_train (np.array): Training data.
        labels_train (np.array): Training labels.
        labels_set (list): Set of unique labels.
        n_components (int): Number of components for SVD.

    Returns:
        list: List of trained TruncatedSVD models.
    """
    models = []
    for label in labels_set:
        cat_train = data_train[labels_train == label]
        svd = TruncatedSVD(n_components=n_components, n_iter=15, random_state=0)
        svd.fit(cat_train)
        models.append(svd)
    return models

def predict_svdr(data_test, models):
    """
    Predicts labels using reconstruction similarity with SVD models.

    Args:
        data_test (np.array): Test data.
        models (list): Trained SVD models.

    Returns:
        tuple: (predicted_labels, probabilities)
    """
    predicted = []
    probas = []
    for example in data_test:
        example_reshaped = example.reshape(1, -1)
        sims = []
        for model in models:
            example_proj = model.transform(example_reshaped)
            example_rec = model.inverse_transform(example_proj)
            norm = np.linalg.norm(example_rec)
            if norm > 0:
                example_rec = example_rec / norm
            
            sim = cosine_similarity(example_rec, example_reshaped)
            sims.append(np.sum(sim))
            
        predicted.append(np.argmax(sims))
        tot = sum(sims)
        probas.append([p / tot if tot > 0 else 0 for p in sims])
        
    return np.array(predicted), np.array(probas)

def train_nmfdr(data_train, labels_train, labels_set, n_components):
    """
    Trains an NMF model for each class.

    Args:
        data_train (np.array): Training data.
        labels_train (np.array): Training labels.
        labels_set (list): Set of unique labels.
        n_components (int): Number of components for NMF.

    Returns:
        list: List of trained NMF models.
    """
    models = []
    for label in labels_set:
        cat_train = data_train[labels_train == label]
        model = NMF(n_components=n_components, init='random', random_state=0)
        model.fit(cat_train)
        models.append(model)
    return models

def predict_nmfdr(data_test, models):
    """
    Predicts labels using reconstruction similarity with NMF models.

    Args:
        data_test (np.array): Test data.
        models (list): Trained NMF models.

    Returns:
        tuple: (predicted_labels, probabilities)
    """
    predicted = []
    probas = []
    for example in data_test:
        example_reshaped = example.reshape(1, -1)
        sims = []
        for model in models:
            example_proj = model.transform(example_reshaped)
            example_rec = model.inverse_transform(example_proj)
            norm = np.linalg.norm(example_rec)
            if norm > 0:
                example_rec = example_rec / norm
            
            sim = cosine_similarity(example_rec, example_reshaped)
            sims.append(np.sum(sim))
            
        predicted.append(np.argmax(sims))
        tot = sum(sims)
        probas.append([p / tot if tot > 0 else 0 for p in sims])
        
    return np.array(predicted), np.array(probas)

def run_lsa_nmf_glove_classification(dataset, lang, main_dir, glove_file, results_base_dir):
    """
    Runs the LSA/NMF with GloVe classification pipeline.

    Args:
        dataset (str): Name of the dataset.
        lang (str): Language ('spanish' or 'english').
        main_dir (Path): Directory containing dataset files.
        glove_file (Path): Path to GloVe vectors.
        results_base_dir (Path): Base directory for results.
    """
    labels_file = main_dir / 'labels.txt'
    words_file = main_dir / 'split/words.txt'
    labels_names = ['True', 'Fake']

    if not labels_file.exists():
        print(f"Labels file not found: {labels_file}")
        return

    # Reading data
    labels_list = read_labels(labels_file, labels_names)
    corpus = read_text_data(lang, words_file)
    corpus, labels_list = remove_empty_text_data_glove(corpus, labels_list)
    labels = np.asarray(labels_list)
    labels_set = list(sorted(set(labels_list)))

    # Load GloVe vectors
    print(f"Loading GloVe vectors from {glove_file}...")
    glove_dict = glove_reader(glove_file)
    vector_dim = 300 if lang == 'spanish' else 200
    
    gv = []
    for line in corpus:
        words = line.split()
        s = np.zeros(vector_dim)
        count = 0
        for w in words:
            if w in glove_dict:
                s += glove_dict[w]
                count += 1
        if count > 0:
            s = s / count
        gv.append(s)

    gv, labels = remove_nan_values(gv, labels)
    gv_corpus = np.array(gv)

    # Normalize the data
    n_samples = gv_corpus.shape[0]
    min_values = np.min(gv_corpus, axis=1).reshape((n_samples, 1))
    gv_corpus = gv_corpus + abs(min_values)
    gv_corpus = normalize(gv_corpus, norm='l2')

    classifiers = ['SVDR', 'NMFDR']

    for cl_name in classifiers:
        print(f'Training and testing with {cl_name}')
        out_dir = results_base_dir / cl_name / 'GloVe' / dataset
        out_dir.mkdir(parents=True, exist_ok=True)
        
        skf = StratifiedKFold(n_splits=10, random_state=0, shuffle=True)
        scores = {
            'accuracy': [], 'precision': [], 'recall': [], 
            'f1': [], 'kappa': [], 'roc_auc': []
        }
        
        start_time = time.time()
        
        for i, (train_index, test_index) in enumerate(skf.split(gv_corpus, labels)):
            print(f'Fold: {i}')
            data_train, data_test = gv_corpus[train_index], gv_corpus[test_index]
            labels_train, labels_test = labels[train_index], labels[test_index]
            
            # Hyperparameter tuning for n_components
            ns = [1, 2, 4, 8, 16, 32]
            best_n = 0
            best_score = 0
            
            for n in ns:
                sub_skf = StratifiedKFold(n_splits=3, random_state=0, shuffle=True)
                inner_scores = []
                for sub_train_idx, sub_test_idx in sub_skf.split(data_train, labels_train):
                    s_train, s_test = data_train[sub_train_idx], data_train[sub_test_idx]
                    sl_train, sl_test = labels_train[sub_train_idx], labels_train[sub_test_idx]
                    
                    if cl_name == 'SVDR':
                        sub_models = train_svdr(s_train, sl_train, labels_set, n)
                        sub_pred, _ = predict_svdr(s_test, sub_models)
                    else:
                        sub_models = train_nmfdr(s_train, sl_train, labels_set, n)
                        sub_pred, _ = predict_nmfdr(s_test, sub_models)
                    
                    inner_scores.append(metrics.f1_score(sl_test, sub_pred, average='macro'))
                
                avg_score = np.mean(inner_scores)
                if avg_score > best_score:
                    best_score = avg_score
                    best_n = n
            
            # Train and predict with best n
            if cl_name == 'SVDR':
                models = train_svdr(data_train, labels_train, labels_set, best_n)
                predicted, predicted_proba = predict_svdr(data_test, models)
            else:
                models = train_nmfdr(data_train, labels_train, labels_set, best_n)
                predicted, predicted_proba = predict_nmfdr(data_test, models)
            
            # Metrics
            scores['accuracy'].append(np.mean(predicted == labels_test))
            scores['precision'].append(metrics.precision_score(labels_test, predicted, average='macro'))
            scores['recall'].append(metrics.recall_score(labels_test, predicted, average='macro'))
            scores['f1'].append(metrics.f1_score(labels_test, predicted, average='macro'))
            scores['kappa'].append(metrics.cohen_kappa_score(labels_test, predicted))
            scores['roc_auc'].append(metrics.roc_auc_score(labels_test, predicted, average='macro'))
            
            # Save predictions
            predictions = np.concatenate((
                test_index[:, None], 
                predicted_proba, 
                predicted[:, None].astype(int), 
                labels_test[:, None].astype(int)
            ), axis=1)
            header = 'test_index, probability_true, probability_fake, predicted_class, real_class'
            np.savetxt(out_dir / f'fold_{i}.csv', predictions, fmt=['%d', '%1.9f', '%1.9f', '%d', '%d'],
                       delimiter=',', encoding='utf-8', header=header, comments='')
        
        end_time = time.time()
        for metric, values in scores.items():
            print(f'{metric.capitalize()}: mean={np.mean(values):.2f} std={np.std(values):.2f} median={np.median(values):.2f}')
        print(f'Total time: {end_time - start_time:.2f}s\n')

if __name__ == "__main__":
    # Configuration
    DATASET = 'fncn_dataset'
    LANG = 'spanish'
    
        project_root = Path(__file__).resolve().parent.parent.parent
    BASE_DIR = project_root / 'data'
    if LANG == 'spanish':
        GLOVE_PATH = BASE_DIR / 'glove-sbwc.i25.vec'
    else:
        GLOVE_PATH = BASE_DIR / 'glove.twitter.27B.200d.txt'
        
        RESULTS_DIR = project_root / 'results' / 'probabilities'
    run_lsa_nmf_glove_classification(DATASET, LANG, BASE_DIR / DATASET, GLOVE_PATH, RESULTS_DIR)
