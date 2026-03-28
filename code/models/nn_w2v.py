#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Classify if news is true or fake using various datasets and Word2Vec vectors.

This module uses Word2Vec embeddings to transform data and employs LSTM, GRU,
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
from gensim.models import Word2Vec
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
    epochs = 20
    embedding_dimension = 300
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
        words = w2v_model.wv.index_to_key
        vectors = w2v_model.wv.vectors
        w2v_dict = dict(zip(words, vectors))
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

    len_news = [len(line.split()) for line in corpus]
    max_len = int(np.mean(len_news))

    # Classifiers to evaluate
    classifiers = ['LSTM', 'GRU', 'BiLSTM']

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
                    embedding_vector = w2v_dict.get(word)
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
