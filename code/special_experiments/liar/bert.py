#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Classify if news is true or fake using the LIAR dataset.

This dataset is divided into train, validation, and test data.
Uses BERT models for transformation and classification.
"""

import os
import random
import sys
import time
import warnings
from pathlib import Path

import ktrain
import numpy as np
import tensorflow as tf
from keras.callbacks import EarlyStopping
from ktrain import text
from sklearn import metrics

sys.path.append(str(Path(__file__).resolve().parent.parent.parent))
from utils import read_all_features, read_labels, read_text_data

# Add parent directory to path to import utils

# Remove to see warnings
warnings.filterwarnings('ignore')

# Set seed for reproducibility
os.environ['PYTHONHASHSEED'] = str(0)
tf.random.set_seed(0)
np.random.seed(0)
random.seed(0)

def main():
    project_root = Path(__file__).resolve().parent.parent.parent
    project_root = Path(__file__).resolve().parent.parent.parent.parent
    """Main execution block for BERT classification on LIAR dataset."""
    # Variables
    dataset = 'liar_dataset'
    lang = 'english'
    model_name = 'bert-base-cased'
    classifier = 'BERT'
    feature = 'Words'

    # Path handling using pathlib
        data_dir = project_root / 'data' / dataset
    results_dir = project_root / 'results' / 'probabilities' / classifier / feature / dataset
    results_dir.mkdir(parents=True, exist_ok=True)

    labels_names = ['True', 'Fake']

    # Train files
    labels_file_train = data_dir / 'labels_train.txt'
    words_file_train = data_dir / 'split_train' / 'words.txt'
    hashs_file_train = data_dir / 'split_train' / 'hashtags.txt'
    ats_file_train = data_dir / 'split_train' / 'ats.txt'
    emo_file_train = data_dir / 'split_train' / 'emoticons.txt'
    links_file_train = data_dir / 'split_train' / 'links.txt'
    abvs_file_train = data_dir / 'split_train' / 'abvs.txt'
    file_train = data_dir / 'corpus_train.txt'

    # Validation files
    labels_file_valid = data_dir / 'labels_valid.txt'
    words_file_valid = data_dir / 'split_valid' / 'words.txt'
    hashs_file_valid = data_dir / 'split_valid' / 'hashtags.txt'
    ats_file_valid = data_dir / 'split_valid' / 'ats.txt'
    emo_file_valid = data_dir / 'split_valid' / 'emoticons.txt'
    links_file_valid = data_dir / 'split_valid' / 'links.txt'
    abvs_file_valid = data_dir / 'split_valid' / 'abvs.txt'
    file_valid = data_dir / 'corpus_valid.txt'

    # Test files
    labels_file_test = data_dir / 'labels_test.txt'
    words_file_test = data_dir / 'split_test' / 'words.txt'
    hashs_file_test = data_dir / 'split_test' / 'hashtags.txt'
    ats_file_test = data_dir / 'split_test' / 'ats.txt'
    emo_file_test = data_dir / 'split_test' / 'emoticons.txt'
    links_file_test = data_dir / 'split_test' / 'links.txt'
    abvs_file_test = data_dir / 'split_test' / 'abvs.txt'
    file_test = data_dir / 'corpus_test.txt'

    # Reading train data
    print(f"Reading {feature} features for training...")
    labels_list_train = read_labels(labels_file_train, labels_names)
    corpus_train = []

    if feature == 'Words':
        corpus_train = read_text_data(lang, words_file_train)
    elif feature == 'AF':
        corpus_train = read_all_features(
            lang, words_file_train, emo_file_train, hashs_file_train,
            ats_file_train, links_file_train, abvs_file_train
        )
    elif feature == 'Raw':
        corpus_train = read_text_data(lang, file_train)

    # Reading validation data
    print(f"Reading {feature} features for validation...")
    labels_list_valid = read_labels(labels_file_valid, labels_names)
    corpus_valid = []

    if feature == 'Words':
        corpus_valid = read_text_data(lang, words_file_valid)
    elif feature == 'AF':
        corpus_valid = read_all_features(
            lang, words_file_valid, emo_file_valid, hashs_file_valid,
            ats_file_valid, links_file_valid, abvs_file_valid
        )
    elif feature == 'Raw':
        corpus_valid = read_text_data(lang, file_valid)

    # Reading test data
    print(f"Reading {feature} features for testing...")
    labels_list_test = read_labels(labels_file_test, labels_names)
    corpus_test = []

    if feature == 'Words':
        corpus_test = read_text_data(lang, words_file_test)
    elif feature == 'AF':
        corpus_test = read_all_features(
            lang, words_file_test, emo_file_test, hashs_file_test,
            ats_file_test, links_file_test, abvs_file_test
        )
    elif feature == 'Raw':
        corpus_test = read_text_data(lang, file_test)

    test_index = np.arange(len(labels_list_test))
    labels_test = np.asarray(labels_list_test)

    # Create a list with the labels
    labels = list(set(labels_list_train))

    # Preprocess data with the BERT model of selection
    preproc = text.Transformer(model_name, maxlen=512, class_names=labels)
    train_data_preproc = preproc.preprocess_train(corpus_train, labels_list_train)
    val_data_preproc = preproc.preprocess_test(corpus_valid, labels_list_valid)

    # Clear session
    tf.keras.backend.clear_session()

    start = time.time()

    # Model creation and training
    model = preproc.get_classifier()
    learner = ktrain.get_learner(
        model, train_data=train_data_preproc,
        val_data=val_data_preproc, batch_size=6
    )
    learner.fit_onecycle(
        lr=5e-5, epochs=7, verbose=0,
        callbacks=[EarlyStopping(
            monitor='val_auc', patience=2, mode='max', restore_best_weights=True
        )]
    )

    # Prediction
    predictor = ktrain.get_predictor(model, preproc)
    predicted = predictor.predict(corpus_test)
    predicted_proba = predictor.predict_proba(corpus_test)

    # Testing
    accuracy = np.mean(predicted == labels_test)
    precision_macro = metrics.precision_score(labels_test, predicted, average='macro')
    recall_macro = metrics.recall_score(labels_test, predicted, average='macro')
    f1_macro = metrics.f1_score(labels_test, predicted, average='macro')
    kappa = metrics.cohen_kappa_score(labels_test, predicted)
    roc = metrics.roc_auc_score(labels_test, predicted, average='macro')

    # Create an array to write it in a file
    predictions = np.concatenate((
        test_index[:, None],
        predicted_proba,
        np.asarray(predicted)[:, None].astype(int),
        labels_test[:, None].astype(int)
    ), axis=1)

    fmt = '%d', '%1.9f', '%1.9f', '%d', '%d'
    header = 'test_index, probability_true, probability_fake, predicted_class, real_class'
    np.savetxt(
        results_dir / 'probability.csv', predictions, fmt=fmt,
        delimiter=',', encoding='utf-8', header=header, comments=''
    )

    end = time.time()

    print(f'Accuracy: {accuracy:0.2f}')
    print(f'Precision: {precision_macro:0.2f}')
    print(f'Recall: {recall_macro:0.2f}')
    print(f'F1: {f1_macro:0.2f}')
    print(f'Kappa: {kappa:0.2f}')
    print(f'ROC-AUC: {roc:0.2f}')
    print(f'Time of training + testing: {(end - start):0.2f}')
    print('\n')

if __name__ == "__main__":
    main()
