# -*- coding: utf-8 -*-
"""
TF-IDF model for fake news classification.
Transforms text data using TF-IDF and applies several classifiers
(SVM, LR, SGDC, MNB, KNN, RF) with cross-validation.
"""

import time
import sys
import warnings
import numpy as np
from pathlib import Path
from sklearn import metrics, svm

# Add the parent directory of this script to sys.path to import utils
sys.path.append(str(Path(__file__).resolve().parent.parent))
from utils import my_tokenizer, read_labels, read_text_data, clean_words, clean_en_words

from sklearn.ensemble import RandomForestClassifier
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression, SGDClassifier
from sklearn.model_selection import StratifiedKFold, cross_val_score
from sklearn.neighbors import KNeighborsClassifier
from sklearn.naive_bayes import MultinomialNB
from nltk.corpus import stopwords

# Suppress warnings
warnings.filterwarnings('ignore')

def read_text_data_with_emos(lang, text_file, emo_file):
    """
    Reads text data and emojis, merging them into a single corpus.

    Args:
        lang (str): Language for stopwords ('spanish' or 'english').
        text_file (Path): Path to the text data file.
        emo_file (Path): Path to the emojis file.

    Returns:
        list: A cleaned list with text and emojis merged.
    """
    data = []
    stop_words = set(stopwords.words(lang))
    
    with open(text_file, encoding='utf-8') as text_content, \
         open(emo_file, encoding='utf-8') as emo_content:
        for text_line, emo_line in zip(text_content, emo_content):
            words = text_line.rstrip().split()
            text = clean_words(words, stop_words)
            if lang == 'spanish':
                text = clean_en_words(text)
            text += ' ' + emo_line.rstrip()
            data.append(text)
    return data

def read_all_features(lang, text_file, emo_file, hashs_file, ats_file, links_file, abvs_file):
    """
    Reads and merges all features (text, emojis, hashtags, mentions, links, abbreviations).

    Args:
        lang (str): Language for stopwords.
        text_file (Path): Path to the text data.
        emo_file (Path): Path to the emojis.
        hashs_file (Path): Path to the hashtags.
        ats_file (Path): Path to the mentions.
        links_file (Path): Path to the links.
        abvs_file (Path): Path to the abbreviations.

    Returns:
        list: A merged corpus of all features.
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

def run_tfidf_classification(dataset, lang, feature, main_dir, results_base_dir):
    """
    Runs the TF-IDF classification pipeline.

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
    classifiers = ['SVM', 'LR', 'MNB', 'SGDC', 'KNN', 'RF']

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
            
            # Model selection with hyperparameter tuning
            if cl_name == 'SVM':
                cs = [0.01, 0.1, 1, 10, 100]
                best_c = 0
                best_score = 0
                for c in cs:
                    clf_inner = svm.SVC(C=c, kernel='linear')
                    inner_cv = StratifiedKFold(n_splits=3, random_state=0, shuffle=True)
                    score = np.mean(cross_val_score(clf_inner, train_tfidf, labels_train, scoring='f1_macro', cv=inner_cv))
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
                    score = np.mean(cross_val_score(clf_inner, train_tfidf, labels_train, scoring='f1_macro', cv=inner_cv))
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
                    inner_cv = StratifiedKFold(n_splits=3, random_state=0, shuffle=True)
                    score = np.mean(cross_val_score(clf_inner, train_tfidf, labels_train, scoring='f1_macro', cv=inner_cv))
                    if score > best_score:
                        best_score = score
                        best_c = c
                clf = SGDClassifier(loss='log', alpha=1/best_c, max_iter=10000)
            
            elif cl_name == 'KNN':
                ks = [1, 2, 3, 5, 10]
                best_k = 0
                best_score = 0
                for k in ks:
                    clf_inner = KNeighborsClassifier(n_neighbors=k)
                    inner_cv = StratifiedKFold(n_splits=3, random_state=0, shuffle=True)
                    score = np.mean(cross_val_score(clf_inner, train_tfidf, labels_train, scoring='f1_macro', cv=inner_cv))
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
                    score = np.mean(cross_val_score(clf_inner, train_tfidf, labels_train, scoring='f1_macro', cv=inner_cv))
                    if score > best_score:
                        best_score = score
                        best_r = r
                clf = RandomForestClassifier(n_estimators=best_r, random_state=0)
            
            clf.fit(train_tfidf, labels_train)
            predicted = clf.predict(test_tfidf)
            predicted_proba = clf.predict_proba(test_tfidf)
            
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
    DATASET = 'isot_dataset'
    LANG = 'english'
    FEATURE = 'Raw'
    
    project_root = Path(__file__).resolve().parent.parent.parent
    BASE_DATA_DIR = project_root / 'data'
    RESULTS_DIR = project_root / 'results' / 'probabilities'
    
    run_tfidf_classification(DATASET, LANG, FEATURE, BASE_DATA_DIR / DATASET, RESULTS_DIR)
