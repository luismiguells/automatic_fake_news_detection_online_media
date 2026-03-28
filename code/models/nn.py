#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Classify if news is true or fake using various datasets.

This module uses tokenization to transform data and employs LSTM, GRU,
and BiLSTM models for classification. Datasets include Covid, FakeNewsNet,
ISOT, and Fake News Costa Rica News Dataset.
"""

import time
import sys
import warnings
from pathlib import Path

import numpy as np
import pandas as pd
import tensorflow as tf
from keras.callbacks import EarlyStopping
from keras.layers import (LSTM, Bidirectional, Dense, Dropout, Embedding,
                          GRU)
from keras.metrics import AUC, Accuracy
from keras.models import Sequential
from keras.preprocessing.sequence import pad_sequences
from keras.preprocessing.text import Tokenizer
from nltk.corpus import stopwords
from sklearn import metrics
from sklearn.model_selection import StratifiedKFold


# Add the parent directory of this script to sys.path to import utils
sys.path.append(str(Path(__file__).resolve().parent.parent))
from utils import clean_en_words, clean_words, read_labels, read_text_data

# Suppress warnings
warnings.filterwarnings('ignore')

# Set seed for reproducibility
tf.random.set_seed(0)
np.random.seed(0)


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
            text = clean_words(words, stop_words)
            text += ' ' + emo_line.rstrip()
            data.append(text)
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
            text = clean_words(words, stop_words)

            if lang == 'spanish':
                text = clean_en_words(text)

            combined_text = (
                f"{text} {e_l.rstrip()} {h_l.rstrip()} "
                f"{a_l.rstrip()} {l_l.rstrip()} {ab_l.rstrip()}"
            )
            data.append(combined_text)
    return data


def main():
    project_root = Path(__file__).resolve().parent.parent.parent
    """Main execution block for model training and evaluation."""
    # Configuration variables
    dataset = 'fncn_dataset'
    lang = 'spanish'
    feature = 'Raw'
    embedding_dimension = 300
    epochs = 20
    labels_names = ['True', 'Fake']

    # Path handling using pathlib
    base_data_path = project_root / 'data'/ dataset
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

    labels = np.asarray(labels_list)

    len_news = [len(line.split()) for line in corpus]
    max_len = int(np.mean(len_news))

    # Classifiers to evaluate
    classifiers = ['LSTM', 'GRU', 'BiLSTM']

    for cl in classifiers:
        print(f'Training and testing with {cl}')
        out_dir = project_root / f'results/probabilities/{cl}/{feature}/{dataset}'
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

        for fold_idx, (train_index, test_index) in enumerate(skf.split(corpus, labels)):
            print(f'Fold: {fold_idx}')
            data_train = [corpus[x] for x in train_index]
            data_test = [corpus[x] for x in test_index]
            labels_train_fold = labels[train_index]
            labels_test_fold = labels[test_index]

            # Tokenization
            tokenizer = Tokenizer()
            tokenizer.fit_on_texts(data_train)

            data_train = tokenizer.texts_to_sequences(data_train)
            data_test = tokenizer.texts_to_sequences(data_test)

            data_train = pad_sequences(data_train, maxlen=max_len,
                                       padding='post', truncating='post')
            data_test = pad_sequences(data_test, maxlen=max_len,
                                      padding='post', truncating='post')

            # One-hot encoding for training labels
            labels_train_encoded = pd.get_dummies(labels_train_fold).values

            tf.keras.backend.clear_session()
            vocab_size = len(tokenizer.word_index) + 1

            # Model definition
            model = Sequential()
            model.add(Embedding(input_dim=vocab_size,
                                output_dim=embedding_dimension,
                                input_length=max_len))

            if cl == 'LSTM':
                model.add(LSTM(128))
            elif cl == 'GRU':
                model.add(GRU(128))
            elif cl == 'BiLSTM':
                for _ in range(3):
                    model.add(Bidirectional(LSTM(128, return_sequences=True,
                                                 recurrent_dropout=0.2)))
                    model.add(Dropout(0.5))
                model.add(Bidirectional(LSTM(128, recurrent_dropout=0.2)))
                model.add(Dropout(0.5))
                model.add(Dense(256, activation='relu'))

            model.add(Dropout(rate=0.5))
            model.add(Dense(units=len(labels_names), activation='softmax'))
            model.compile(optimizer='adamax', loss='categorical_crossentropy',
                          metrics=[AUC(), Accuracy()])

            # Training
            model.fit(data_train, labels_train_encoded, epochs=epochs,
                      batch_size=64, verbose=0, validation_split=0.1,
                      callbacks=[EarlyStopping(monitor='val_auc', patience=5,
                                               mode='max',
                                               restore_best_weights=True)])

            # Evaluation
            predicted_proba = model.predict(data_test)
            predicted = np.argmax(predicted_proba, axis=-1)

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
