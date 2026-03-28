# -*- coding: utf-8 -*-
"""
LSA (SVD) and NMF model with fastText vectors for fake news classification.
Combines fastText word embeddings with dimensionality reduction techniques 
(TruncatedSVD and NMF) for classification based on reconstruction similarity.
"""

import time
import sys
import warnings
import numpy as np
import fasttext
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

def fasttext_reader(fasttext_file):
    """
    Reads a fastText .vec file and returns a dictionary of word vectors.

    Args:
        fasttext_file (Path): Path to the fastText file.

    Returns:
        dict: A dictionary where keys are words and values are vectors.
    """
    ft_dict = {}
    with open(fasttext_file, 'r', encoding='utf-8') as f:
        for line in f:
            tokens = line.strip().split()
            if len(tokens) > 1:
                word = tokens[0]
                vector = [float(token) for token in tokens[1:]]
                ft_dict[word] = vector
    return ft_dict

def train_fasttext_model(corpus_file):
    """
    Trains an unsupervised fastText model on the given corpus.

    Args:
        corpus_file (Path): Path to the text file for training.

    Returns:
        fasttext.FastText._FastText: The trained fastText model.
    """
    return fasttext.train_unsupervised(str(corpus_file), 'skipgram', dim=300, epoch=20, verbose=0)

def remove_empty_text_data_fasttext(corpus, labels):
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

def run_lsa_nmf_fasttext_classification(dataset, lang, model_type, main_dir, fasttext_pre_trained_file, results_base_dir):
    """
    Runs the LSA/NMF with fastText classification pipeline.
    """
    labels_file = main_dir / 'labels.txt'
    words_file = main_dir / 'split/words.txt'
    labels_names = ['True', 'Fake']

    if not labels_file.exists():
        print(f"Labels file not found: {labels_file}")
        return

    # Prepare fastText dictionary
    print(f"Preparing fastText dictionary (Type: {model_type})...")
    if model_type == 'OM':
        ft_model = train_fasttext_model(words_file)
        ft_dict = ft_model # Keep model to use get_word_vector
    elif model_type == 'PM':
        ft_dict = fasttext_reader(fasttext_pre_trained_file)
    elif model_type == 'PM+OM':
        ft_dict = fasttext_reader(fasttext_pre_trained_file)
        ft_model_aux = train_fasttext_model(words_file)
        for w in ft_model_aux.get_words():
            if w not in ft_dict:
                ft_dict[w] = ft_model_aux.get_word_vector(w)
    else:
        print(f"Unknown model type: {model_type}")
        return

    # Reading data
    labels_list = read_labels(labels_file, labels_names)
    corpus = read_text_data(lang, words_file)
    corpus, labels_list = remove_empty_text_data_fasttext(corpus, labels_list)
    labels = np.asarray(labels_list)
    labels_set = list(sorted(set(labels_list)))

    # Vectorize corpus
    ft_vectors = []
    for line in corpus:
        words = line.split()
        s = np.zeros(300)
        count = 0
        for w in words:
            if model_type == 'OM':
                if w in ft_dict.get_words():
                    s += ft_dict.get_word_vector(w)
                    count += 1
            else:
                if w in ft_dict:
                    s += ft_dict[w]
                    count += 1
        if count > 0:
            s = s / count
        ft_vectors.append(s)

    ft_vectors, labels = remove_nan_values(ft_vectors, labels)
    ft_corpus = np.array(ft_vectors)

    # Normalize
    n_samples = ft_corpus.shape[0]
    min_values = np.min(ft_corpus, axis=1).reshape((n_samples, 1))
    ft_corpus = ft_corpus + abs(min_values)
    ft_corpus = normalize(ft_corpus, norm='l2')

    classifiers = ['SVDR', 'NMFDR']

    for cl_name in classifiers:
        print(f'Training and testing with {cl_name}')
        out_dir = results_base_dir / cl_name / 'fastText' / model_type / dataset
        out_dir.mkdir(parents=True, exist_ok=True)
        
        skf = StratifiedKFold(n_splits=10, random_state=0, shuffle=True)
        scores = {
            'accuracy': [], 'precision': [], 'recall': [], 
            'f1': [], 'kappa': [], 'roc_auc': []
        }
        
        start_time = time.time()
        for i, (train_index, test_index) in enumerate(skf.split(ft_corpus, labels)):
            print(f'Fold: {i}')
            data_train, data_test = ft_corpus[train_index], ft_corpus[test_index]
            labels_train, labels_test = labels[train_index], labels[test_index]
            
            # Tuning
            ns = [1, 2, 4, 8, 16, 32]
            best_n, best_score = 0, 0
            for n in ns:
                sub_skf = StratifiedKFold(n_splits=3, random_state=0, shuffle=True)
                inner_scores = []
                for sub_tr_idx, sub_ts_idx in sub_skf.split(data_train, labels_train):
                    s_tr, s_ts = data_train[sub_tr_idx], data_train[sub_ts_idx]
                    sl_tr, sl_ts = labels_train[sub_tr_idx], labels_train[sub_ts_idx]
                    if cl_name == 'SVDR':
                        sub_models = train_svdr(s_tr, sl_tr, labels_set, n)
                        sub_pred, _ = predict_svdr(s_ts, sub_models)
                    else:
                        sub_models = train_nmfdr(s_tr, sl_tr, labels_set, n)
                        sub_pred, _ = predict_nmfdr(s_ts, sub_models)
                    inner_scores.append(metrics.f1_score(sl_ts, sub_pred, average='macro'))
                avg_score = np.mean(inner_scores)
                if avg_score > best_score:
                    best_score, best_n = avg_score, n
            
            # Final models
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
            
            predictions = np.concatenate((test_index[:, None], predicted_proba, predicted[:, None].astype(int), labels_test[:, None].astype(int)), axis=1)
            np.savetxt(out_dir / f'fold_{i}.csv', predictions, fmt=['%d', '%1.9f', '%1.9f', '%d', '%d'],
                       delimiter=',', encoding='utf-8', header='test_index, probability_true, probability_fake, predicted_class, real_class', comments='')
        
        end_time = time.time()
        for m, v in scores.items():
            print(f'{m.capitalize()}: mean={np.mean(v):.2f} std={np.std(v):.2f} median={np.median(v):.2f}')
        print(f'Total time: {end_time - start_time:.2f}s\n')

if __name__ == "__main__":
    DATASET = 'fnn_dataset'
    LANG = 'english'
    MODEL_TYPE = 'PM+OM'
        project_root = Path(__file__).resolve().parent.parent.parent
    BASE_DIR = project_root / 'data'
    FASTTEXT_PATH = BASE_DIR / ('pre_trained_models/fasttext_spanish_300.vec' if LANG == 'spanish' else 'pre_trained_models/fasttext_english_300.vec')
        RESULTS_DIR = project_root / 'results' / 'probabilities'
    run_lsa_nmf_fasttext_classification(DATASET, LANG, MODEL_TYPE, BASE_DIR / DATASET, FASTTEXT_PATH, RESULTS_DIR)
