#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Classify if news is true or fake using various datasets and GloVe vectors.

This module uses GloVe embeddings to transform data and employs LSTM, GRU,
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
from sklearn import metrics
from sklearn.model_selection import StratifiedKFold


# Add the parent directory of this script to sys.path to import utils
sys.path.append(str(Path(__file__).resolve().parent.parent))
from utils import read_labels, read_text_data

# Suppress warnings
warnings.filterwarnings('ignore')

# Set seed for reproducibility
tf.random.set_seed(0)
np.random.seed(0)


def glove_reader(glove_file):
    """Reads GloVe vectors from a file into a dictionary.

    Args:
        glove_file (str or Path): Path to the GloVe vectors file.

    Returns:
        dict: A dictionary where keys are words and values are their vectors.
    """
    glove_dict = {}
    with open(glove_file, 'r', encoding='utf-8') as f:
        for line in f:
            tokens = line.strip().split()
            if not tokens:
                continue
            word = tokens[0]
            vector = [float(token) for token in tokens[1:]]
            glove_dict[word] = vector
    return glove_dict


def remove_empty_text_data_glove(corpus, labels):
    """Removes entries with empty text from the corpus and labels.

    Args:
        corpus (list): List of text strings.
        labels (list): List of corresponding labels.

    Returns:
        tuple: (cleaned_corpus, cleaned_labels)
    """
    indices_to_remove = [i for i, text in enumerate(corpus) if len(text) == 0]

    if indices_to_remove:
        # Remove from end to start to maintain index validity
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


def main():
    project_root = Path(__file__).resolve().parent.parent.parent
    """Main execution block for GloVe-based model training and evaluation."""
    # Configuration variables
    dataset = 'fncn_dataset'
    lang = 'spanish'
    feature = 'GloVe'
    epochs = 20
    labels_names = ['True', 'Fake']

    # Embedding dimension based on language
    embedding_dimension = 300 if lang == 'spanish' else 200

    # Path handling using pathlib
    base_data_path = project_root / 'data'
    if lang == 'spanish':
        glove_file = base_data_path / 'glove-sbwc.i25.vec'
    else:
        glove_file = base_data_path / 'glove.twitter.27B.200d.txt'

    dataset_path = base_data_path / dataset
    labels_file = dataset_path / 'labels.txt'
    words_file = dataset_path / 'split' / 'words.txt'

    # Reading data
    labels_list = read_labels(labels_file, labels_names)
    corpus = read_text_data(lang, words_file)

    # Pre-processing
    corpus, labels_list = remove_empty_text_data_glove(corpus, labels_list)
    labels = np.asarray(labels_list)

    # Load GloVe embeddings
    embeddings_index = glove_reader(glove_file)

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
            word_index = tokenizer.word_index

            data_train = tokenizer.texts_to_sequences(data_train)
            data_test = tokenizer.texts_to_sequences(data_test)

            data_train = pad_sequences(data_train, maxlen=max_len,
                                       padding='post', truncating='post')
            data_test = pad_sequences(data_test, maxlen=max_len,
                                      padding='post', truncating='post')

            # One-hot encoding for training labels
            labels_train_encoded = pd.get_dummies(labels_train_fold).values

            # Create embedding matrix
            vocab_size = len(word_index) + 1
            embedding_matrix = np.zeros((vocab_size, embedding_dimension))
            for word, idx in word_index.items():
                if idx < vocab_size:
                    embedding_vector = embeddings_index.get(word)
                    if embedding_vector is not None:
                        embedding_matrix[idx] = embedding_vector

            tf.keras.backend.clear_session()

            # Model definition
            model = Sequential()
            model.add(Embedding(input_dim=vocab_size,
                                output_dim=embedding_dimension,
                                input_length=max_len,
                                trainable=False,
                                weights=[embedding_matrix]))

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
