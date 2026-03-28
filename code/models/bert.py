#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Classify if news is true or fake using various datasets and BERT models.

This module utilizes the ktrain library to employ BERT-based models
(including Spanish-specific BERT) for news classification.
Datasets include Covid, FakeNewsNet, ISOT, and Fake News Costa Rica News Dataset.
"""

import os
import sys
import random
import time
import warnings
from pathlib import Path

# Add the parent directory of this script to sys.path to import utils
sys.path.append(str(Path(__file__).resolve().parent.parent))

import ktrain
import numpy as np
import tensorflow as tf
from keras.callbacks import EarlyStopping
from ktrain import text
from nltk.corpus import stopwords
from sklearn import metrics
from sklearn.model_selection import StratifiedKFold, train_test_split

from utils import clean_en_words, clean_words, read_labels, read_text_data

# Suppress warnings
warnings.filterwarnings('ignore')

# Set seed for reproducibility
os.environ['PYTHONHASHSEED'] = str(0)
tf.random.set_seed(0)
np.random.seed(0)
random.seed(0)


def read_text_data_with_emos(lang, text_file, emo_file):
    """Reads and cleans text data, merging it with emoji data.

    Args:
        lang (str): The language for stopwords ('spanish' or 'english').
        text_file (str or Path): Path to the text data file.
        emo_file (str or Path): Path to the emojis file.

    Returns:
        list: A list of strings where each string is cleaned text merged with emojis.
    """
    data = []
    stop_words = set(stopwords.words(lang))

    with open(text_file, 'r', encoding='utf-8') as text_content, \
         open(emo_file, 'r', encoding='utf-8') as emo_content:
        for text_line, emo_line in zip(text_content, emo_content):
            words = text_line.rstrip().split()
            txt = clean_words(words, stop_words)
            txt += ' ' + emo_line.rstrip()
            data.append(txt)
    return data


def read_all_features(lang, text_file, emo_file, hashs_file, ats_file,
                      links_file, abvs_file):
    """Reads and cleans text data, merging it with multiple additional features.

    Features include emojis, hashtags, mentions (@), links, and abbreviations.

    Args:
        lang (str): The language for stopwords ('spanish' or 'english').
        text_file (str or Path): Path to the main text data file.
        emo_file (str or Path): Path to the emojis file.
        hashs_file (str or Path): Path to the hashtags file.
        ats_file (str or Path): Path to the mentions (@) file.
        links_file (str or Path): Path to the links file.
        abvs_file (str or Path): Path to the abbreviations file.

    Returns:
        list: A list of strings with all features merged and cleaned.
    """
    data = []
    stop_words = set(stopwords.words(lang))

    with open(text_file, 'r', encoding='utf-8') as text_content, \
         open(emo_file, 'r', encoding='utf-8') as emo_content, \
         open(hashs_file, 'r', encoding='utf-8') as hashs_content, \
         open(ats_file, 'r', encoding='utf-8') as ats_content, \
         open(links_file, 'r', encoding='utf-8') as links_content, \
         open(abvs_file, 'r', encoding='utf-8') as abvs_content:

        zipped_content = zip(text_content, emo_content, hashs_content,
                             ats_content, links_content, abvs_content)

        for t_l, e_l, h_l, a_l, l_l, ab_l in zipped_content:
            words = t_l.rstrip().split()
            txt = clean_words(words, stop_words)

            if lang == 'spanish':
                txt = clean_en_words(txt)

            combined_text = (
                f"{txt} {e_l.rstrip()} {h_l.rstrip()} "
                f"{a_l.rstrip()} {l_l.rstrip()} {ab_l.rstrip()}"
            )
            data.append(combined_text)
    return data


def main():
    project_root = Path(__file__).resolve().parent.parent.parent
    """Main execution block for BERT-based model training and evaluation."""
    # Configuration variables
    dataset = 'fncn_dataset'
    lang = 'spanish'
    feature = 'AF'
    labels_names = ['True', 'Fake']

    if lang == 'spanish':
        model_name = 'dccuchile/bert-base-spanish-wwm-cased'
    else:
        model_name = 'bert-base-cased'

    # Path handling using pathlib
    project_root = Path(__file__).resolve().parent.parent.parent
    base_data_path = project_root / 'data' / dataset
    labels_file = base_data_path / 'labels.txt'
    words_file = base_data_path / 'split' / 'words.txt'
    hashs_file = base_data_path / 'split' / 'hashtags.txt'
    ats_file = base_data_path / 'split' / 'ats.txt'
    emo_file = base_data_path / 'split' / 'emoticons.txt'
    links_file = base_data_path / 'split' / 'links.txt'
    abvs_file = base_data_path / 'split' / 'abvs.txt'
    file_train = base_data_path / 'corpus.txt'

    # Reading data
    labels_list = read_labels(labels_file, labels_names)
    corpus = []

    if feature == 'Words':
        corpus = read_text_data(lang, words_file)
    elif feature == 'AF':
        corpus = read_all_features(lang, words_file, emo_file, hashs_file,
                                   ats_file, links_file, abvs_file)
    elif feature == 'Raw':
        corpus = read_text_data(lang, file_train)

    labels = list(set(labels_list))
    labels_list_arr = np.array(labels_list)

    skf = StratifiedKFold(n_splits=10, random_state=0, shuffle=True)
    scores = {
        'accuracy': [],
        'precision': [],
        'recall': [],
        'f1': [],
        'kappa': [],
        'roc_auc': []
    }

    classifier = 'BERT'
    results_base_dir = project_root / 'results' / 'probabilities'
    out_dir = results_base_dir / classifier / feature / dataset
    out_dir.mkdir(parents=True, exist_ok=True)

    start_time = time.time()

    for fold_idx, (train_index, test_index) in enumerate(skf.split(corpus, labels_list_arr)):
        print(f'Fold: {fold_idx}')
        data_train = [corpus[x] for x in train_index]
        data_test = [corpus[x] for x in test_index]
        labels_train_fold = labels_list_arr[train_index]
        labels_test_fold = labels_list_arr[test_index]

        # Train/Validation split
        t_data, v_data, t_labels, v_labels = train_test_split(
            data_train, labels_train_fold, test_size=0.1,
            random_state=0, stratify=labels_train_fold
        )

        # Preprocess with ktrain Transformer
        preproc = text.Transformer(model_name, maxlen=512, class_names=labels)
        train_preproc = preproc.preprocess_train(t_data, t_labels)
        val_preproc = preproc.preprocess_test(v_data, v_labels)

        tf.keras.backend.clear_session()

        # Model creation and training
        model = preproc.get_classifier()
        learner = ktrain.get_learner(model, train_data=train_preproc,
                                     val_data=val_preproc, batch_size=6)
        
        learner.fit_onecycle(lr=5e-5, epochs=7, verbose=0,
                             callbacks=[EarlyStopping(monitor='val_auc',
                                                      patience=2, mode='max',
                                                      restore_best_weights=True)])

        # Prediction
        predictor = ktrain.get_predictor(model, preproc)
        predicted = predictor.predict(data_test)
        predicted_proba = predictor.predict_proba(data_test)

        scores['accuracy'].append(np.mean(predicted == labels_test_fold))
        scores['precision'].append(metrics.precision_score(labels_test_fold,
                                                          predicted,
                                                          average='macro'))
        scores['recall'].append(metrics.recall_score(labels_test_fold,
                                                    predicted,
                                                    average='macro'))
        scores['f1'].append(metrics.f1_score(labels_test_fold,
                                             predicted,
                                             average='macro'))
        scores['kappa'].append(metrics.cohen_kappa_score(labels_test_fold,
                                                        predicted))
        scores['roc_auc'].append(metrics.roc_auc_score(labels_test_fold,
                                                      predicted,
                                                      average='macro'))

        # Save fold results
        predicted_arr = np.array(predicted)
        predictions = np.concatenate((
            test_index[:, None],
            predicted_proba,
            predicted_arr[:, None],
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
