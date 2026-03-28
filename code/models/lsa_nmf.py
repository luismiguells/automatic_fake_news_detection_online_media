# -*- coding: utf-8 -*-
"""
LSA (SVD) and NMF model for fake news classification.
Uses dimensionality reduction techniques (TruncatedSVD and NMF) 
to classify news based on reconstruction similarity.
"""

import time
import sys
import warnings
import numpy as np
from pathlib import Path
from sklearn import metrics
from sklearn.decomposition import NMF, TruncatedSVD
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity
from sklearn.model_selection import StratifiedKFold
from nltk.corpus import stopwords


# Add the parent directory of this script to sys.path to import utils
sys.path.append(str(Path(__file__).resolve().parent.parent))
from utils import my_tokenizer, read_labels, read_text_data, clean_words, clean_en_words

# Suppress warnings
warnings.filterwarnings('ignore')

def read_all_features(lang, text_file, emo_file, hashs_file, ats_file, links_file, abvs_file):
    """
    Reads and merges all features into a single corpus.

    Args:
        lang (str): Language for stopwords.
        text_file (Path): Path to text file.
        emo_file (Path): Path to emojis file.
        hashs_file (Path): Path to hashtags file.
        ats_file (Path): Path to mentions file.
        links_file (Path): Path to links file.
        abvs_file (Path): Path to abbreviations file.

    Returns:
        list: Merged corpus.
    """
    data = []
    stop_words = set(stopwords.words(lang))
    
    with open(text_file, encoding='utf-8') as text_f, \
         open(emo_file, encoding='utf-8') as emo_f, \
         open(hashs_file, encoding='utf-8') as hash_f, \
         open(ats_file, encoding='utf-8') as at_f, \
         open(links_file, encoding='utf-8') as link_f, \
         open(abvs_file, encoding='utf-8') as abv_f:
        
        for t_l, e_l, h_l, a_l, l_l, ab_l in zip(text_f, emo_f, hash_f, at_f, link_f, abv_f):
            words = t_l.rstrip().split()
            text = clean_words(words, stop_words)
            if lang == 'spanish':
                text = clean_en_words(text)
            
            merged = f"{text} {e_l.rstrip()} {h_l.rstrip()} {a_l.rstrip()} {l_l.rstrip()} {ab_l.rstrip()}"
            data.append(merged)
    return data

def train_pca(data_train, labels_train, labels_set, n_components):
    """
    Trains a TruncatedSVD model for each class.

    Args:
        data_train (sparse matrix): Training data.
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

def predict_pca(data_test, models):
    """
    Predicts labels using reconstruction similarity with SVD models.

    Args:
        data_test (sparse matrix): Test data.
        models (list): Trained SVD models.

    Returns:
        tuple: (predicted_labels, probabilities)
    """
    predicted = []
    probas = []
    for example in data_test:
        sims = []
        for model in models:
            example_proj = model.transform(example)
            example_rec = model.inverse_transform(example_proj)
            norm = np.linalg.norm(example_rec)
            if norm > 0:
                example_rec = example_rec / norm
            
            sim = cosine_similarity(example_rec, example)
            sims.append(np.sum(sim))
            
        predicted.append(np.argmax(sims))
        tot = sum(sims)
        probas.append([p / tot if tot > 0 else 0 for p in sims])
        
    return np.array(predicted), np.array(probas)

def train_nmf(data_train, labels_train, labels_set, n_components):
    """
    Trains an NMF model for each class.

    Args:
        data_train (sparse matrix): Training data.
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

def predict_nmf(data_test, models):
    """
    Predicts labels using reconstruction similarity with NMF models.

    Args:
        data_test (sparse matrix): Test data.
        models (list): Trained NMF models.

    Returns:
        tuple: (predicted_labels, probabilities)
    """
    predicted = []
    probas = []
    for example in data_test:
        sims = []
        for model in models:
            example_proj = model.transform(example)
            example_rec = model.inverse_transform(example_proj)
            norm = np.linalg.norm(example_rec)
            if norm > 0:
                example_rec = example_rec / norm
            
            sim = cosine_similarity(example_rec, example)
            sims.append(np.sum(sim))
            
        predicted.append(np.argmax(sims))
        tot = sum(sims)
        probas.append([p / tot if tot > 0 else 0 for p in sims])
        
    return np.array(predicted), np.array(probas)

def run_lsa_nmf_classification(dataset, lang, feature, main_dir, results_base_dir):
    """
    Runs the LSA/NMF classification pipeline.

    Args:
        dataset (str): Name of the dataset.
        lang (str): Language of the dataset.
        feature (str): Feature type ('Words', 'AF', 'Raw').
        main_dir (Path): Directory containing dataset files.
        results_base_dir (Path): Base directory to save results.
    """
    labels_file = main_dir / 'labels.txt'
    words_file = main_dir / 'split/words.txt'
    hashs_file = main_dir / 'split/hashtags.txt'
    ats_file = main_dir / 'split/ats.txt'
    emo_file = main_dir / 'split/emoticons.txt'
    links_file = main_dir / 'split/links.txt'
    abvs_file = main_dir / 'split/abvs.txt'
    file_raw = main_dir / 'corpus.txt'
    labels_names = ['True', 'Fake']

    if not labels_file.exists():
        print(f"Labels file not found: {labels_file}")
        return

    # Reading data
    labels_list = read_labels(labels_file, labels_names)
    if feature == 'Words':
        corpus = read_text_data(lang, words_file)
    elif feature == 'AF':
        corpus = read_all_features(lang, words_file, emo_file, hashs_file, ats_file, links_file, abvs_file)
    elif feature == 'Raw':
        corpus = read_text_data(lang, file_raw)
    else:
        print(f"Unknown feature type: {feature}")
        return
        
    labels = np.asarray(labels_list)
    labels_set = list(sorted(set(labels_list)))
    classifiers = ['SVDR', 'NMFDR']

    for cl_name in classifiers:
        print(f'Training and testing with {cl_name}')
        out_dir = results_base_dir / cl_name / feature / dataset
        out_dir.mkdir(parents=True, exist_ok=True)
        
        skf = StratifiedKFold(n_splits=10, random_state=0, shuffle=True)
        scores = {
            'accuracy': [], 'precision': [], 'recall': [], 
            'f1': [], 'kappa': [], 'roc_auc': []
        }
        
        start_time = time.time()
        
        for i, (train_index, test_index) in enumerate(skf.split(corpus, labels)):
            print(f'Fold: {i}')
            data_train = [corpus[x] for x in train_index]
            data_test = [corpus[x] for x in test_index]
            labels_train, labels_test = labels[train_index], labels[test_index]
            
            vec = TfidfVectorizer(min_df=1, norm='l2', analyzer='word', tokenizer=my_tokenizer)
            train_tfidf = vec.fit_transform(data_train)
            test_tfidf = vec.transform(data_test)
            
            # Hyperparameter tuning for n_components
            ns = [1, 2, 4, 8, 16, 32]
            best_n = 0
            best_score = 0
            
            for n in ns:
                sub_skf = StratifiedKFold(n_splits=3, random_state=0, shuffle=True)
                inner_scores = []
                for sub_train_idx, sub_test_idx in sub_skf.split(train_tfidf, labels_train):
                    s_train, s_test = train_tfidf[sub_train_idx], train_tfidf[sub_test_idx]
                    sl_train, sl_test = labels_train[sub_train_idx], labels_train[sub_test_idx]
                    
                    if cl_name == 'SVDR':
                        sub_models = train_pca(s_train, sl_train, labels_set, n)
                        sub_pred, _ = predict_pca(s_test, sub_models)
                    else:
                        sub_models = train_nmf(s_train, sl_train, labels_set, n)
                        sub_pred, _ = predict_nmf(s_test, sub_models)
                    
                    inner_scores.append(metrics.f1_score(sl_test, sub_pred, average='macro'))
                
                avg_score = np.mean(inner_scores)
                if avg_score > best_score:
                    best_score = avg_score
                    best_n = n
            
            # Train and predict with best n
            if cl_name == 'SVDR':
                models = train_pca(train_tfidf, labels_train, labels_set, best_n)
                predicted, predicted_proba = predict_pca(test_tfidf, models)
            else:
                models = train_nmf(train_tfidf, labels_train, labels_set, best_n)
                predicted, predicted_proba = predict_nmf(test_tfidf, models)
            
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
        
        # Print results
        for metric, values in scores.items():
            print(f'{metric.capitalize()}: mean={np.mean(values):.2f} std={np.std(values):.2f} median={np.median(values):.2f}')
        print(f'Total time: {end_time - start_time:.2f}s\n')

if __name__ == "__main__":
    # Configuration
    DATASET = 'fncn_dataset'
    LANG = 'spanish'
    FEATURE = 'Raw'
    
    BASE_DATA_DIR = project_root / 'data'
        RESULTS_DIR = project_root / 'results' / 'probabilities'
    
    run_lsa_nmf_classification(DATASET, LANG, FEATURE, BASE_DATA_DIR / DATASET, RESULTS_DIR)
