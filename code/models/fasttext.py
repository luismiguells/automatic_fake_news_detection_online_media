# -*- coding: utf-8 -*-
"""
fastText model for fake news classification.
Transforms text data using fastText embeddings (own, pre-trained, or combined)
and applies several classifiers (SVM, LR, SGDC, MNB, KNN, RF) with cross-validation.
"""

import time
import sys
import warnings
import numpy as np
import fasttext
from pathlib import Path
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
    # Note: fasttext expects string path
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

def run_fasttext_classification(dataset, lang, model_type, main_dir, fasttext_pre_trained_file, results_base_dir):
    """
    Runs the fastText classification pipeline.

    Args:
        dataset (str): Name of the dataset.
        lang (str): Language ('spanish' or 'english').
        model_type (str): Type of model ('OM', 'PM', 'PM+OM').
        main_dir (Path): Directory containing dataset files.
        fasttext_pre_trained_file (Path): Path to pre-trained fastText vectors.
        results_base_dir (Path): Base directory for results.
    """
    labels_file = main_dir / 'labels.txt'
    words_file = main_dir / 'split/words.txt'
    labels_names = ['True', 'Fake']

    if not labels_file.exists():
        print(f"Labels file not found: {labels_file}")
        return

    # Prepare fastText dictionary based on model_type
    print(f"Preparing fastText dictionary (Type: {model_type})...")
    if model_type == 'OM':
        ft_model = train_fasttext_model(words_file)
        ft_dict = {w: ft_model.get_word_vector(w) for w in ft_model.get_words()}
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

    # Vectorize corpus
    ft_vectors = []
    for line in corpus:
        words = line.split()
        s = np.zeros(300)
        count = 0
        for w in words:
            if w in ft_dict:
                s += ft_dict[w]
                count += 1
        if count > 0:
            s = s / count
        ft_vectors.append(s)

    ft_vectors, labels = remove_nan_values(ft_vectors, labels)
    ft_corpus = np.array(ft_vectors)

    # Normalize the data
    n_samples = ft_corpus.shape[0]
    min_values = np.min(ft_corpus, axis=1).reshape((n_samples, 1))
    ft_corpus = ft_corpus + abs(min_values)
    ft_corpus = normalize(ft_corpus, norm='l2')

    classifiers = ['SVM', 'LR', 'MNB', 'SGDC', 'KNN', 'RF']

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
            
            # Model selection with hyperparameter tuning
            if cl_name == 'SVM':
                cs = [0.01, 0.1, 1, 10, 100]
                best_c = 0
                best_score = 0
                for c in cs:
                    clf_inner = svm.SVC(C=c, kernel='linear')
                    inner_cv = StratifiedKFold(n_splits=3, random_state=0, shuffle=True)
                    score = np.mean(cross_val_score(clf_inner, data_train, labels_train, scoring='f1_macro', cv=inner_cv))
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
                    inner_cv = StratifiedKFold(n_splits=3, random_state=0, shuffle=True)
                    score = np.mean(cross_val_score(clf_inner, data_train, labels_train, scoring='f1_macro', cv=inner_cv))
                    if score > best_score:
                        best_score = score
                        best_c = c
                clf = LogisticRegression(C=best_c, penalty='l2', solver='liblinear')
            
            elif cl_name == 'SGDC':
                cs = [0.01, 0.1, 1, 10, 100]
                best_c = 0
                best_score = 0
                for c in cs:
                    clf_inner = SGDClassifier(loss='log', alpha=1/c, max_iter=10000)
                    inner_cv = StratifiedKFold(n_splits=3, random_state=0, shuffle=True)
                    score = np.mean(cross_val_score(clf_inner, data_train, labels_train, scoring='f1_macro', cv=inner_cv))
                    if score > best_score:
                        best_score = score
                        best_c = c
                clf = SGDClassifier(loss='log', alpha=1/best_c, max_iter=10000)
            
            elif cl_name == 'MNB':
                clf = MultinomialNB()
            
            elif cl_name == 'KNN':
                ks = [1, 2, 3, 5, 10]
                best_k = 0
                best_score = 0
                for k in ks:
                    clf_inner = KNeighborsClassifier(n_neighbors=k)
                    inner_cv = StratifiedKFold(n_splits=3, random_state=0, shuffle=True)
                    score = np.mean(cross_val_score(clf_inner, data_train, labels_train, scoring='f1_macro', cv=inner_cv))
                    if score > best_score:
                        best_score = score
                        best_k = k
                clf = KNeighborsClassifier(n_neighbors=best_k)
                
            elif cl_name == 'RF':
                rs = [10, 50, 100, 200, 500]
                best_r = 0
                best_score = 0
                for r in rs:
                    clf_inner = RandomForestClassifier(n_estimators=r, random_state=0)
                    inner_cv = StratifiedKFold(n_splits=3, random_state=0, shuffle=True)
                    score = np.mean(cross_val_score(clf_inner, data_train, labels_train, scoring='f1_macro', cv=inner_cv))
                    if score > best_score:
                        best_score = score
                        best_r = r
                clf = RandomForestClassifier(n_estimators=best_r, random_state=0)
            
            clf.fit(data_train, labels_train)
            predicted = clf.predict(data_test)
            predicted_proba = clf.predict_proba(data_test)
            
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
    MODEL_TYPE = 'PM+OM'
    
        project_root = Path(__file__).resolve().parent.parent.parent
    BASE_DIR = project_root / 'data'
    if LANG == 'spanish':
        FASTTEXT_PATH = BASE_DIR / 'pre_trained_models/fasttext_spanish_300.vec'
    else:
        FASTTEXT_PATH = BASE_DIR / 'pre_trained_models/fasttext_english_300.vec'
        
        RESULTS_DIR = project_root / 'results' / 'probabilities'
    run_fasttext_classification(DATASET, LANG, MODEL_TYPE, BASE_DIR / DATASET, FASTTEXT_PATH, RESULTS_DIR)
