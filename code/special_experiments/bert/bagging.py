#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Classify if news is true or fake using the Fake News Corpus Spanish dataset.

Uses Bagging with BERT models for transformation and classification.
"""

import os
import random
import sys
import warnings
from pathlib import Path

import ktrain
import numpy as np
import tensorflow as tf
from keras.callbacks import EarlyStopping
from ktrain import text
from sklearn import metrics
from sklearn.model_selection import train_test_split

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

def train_bert(train_data, val_data, corpus_test, preproc):
    """
    Train a BERT model and return predictions.

    Args:
        train_data: Preprocessed training data.
        val_data: Preprocessed validation data.
        corpus_test: Test corpus strings.
        preproc: ktrain Transformer instance.

    Returns:
        tuple: (predicted classes, predicted probabilities)
    """
    # Clear session
    tf.keras.backend.clear_session()

    # Model creation and training
    model = preproc.get_classifier()
    learner = ktrain.get_learner(
        model, train_data=train_data,
        val_data=val_data, batch_size=6
    )
    learner.fit_onecycle(
        lr=5e-5, epochs=20, verbose=0,
        callbacks=[EarlyStopping(
            monitor='val_auc', patience=2, mode='max', restore_best_weights=True
        )]
    )

    # Prediction
    predictor = ktrain.get_predictor(model, preproc)
    predicted = predictor.predict(corpus_test)
    predicted_proba = predictor.predict_proba(corpus_test)

    return predicted, predicted_proba

def main():
    project_root = Path(__file__).resolve().parent.parent.parent
    project_root = Path(__file__).resolve().parent.parent.parent.parent
    """Main execution block for Bagging with BERT on FNCS dataset."""
    # Variables
    dataset = 'fncs_dataset'
    lang = 'spanish'
    model_name = 'dccuchile/bert-base-spanish-wwm-cased'
    feature = 'Words'

    # Path handling using pathlib
        data_dir = project_root / 'data' / dataset

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
        corpus_train = read_text_data(lang, file_train)
    else:
        corpus_train = read_all_features(
            lang, words_file_train, emo_file_train, hashs_file_train,
            ats_file_train, links_file_train, abvs_file_train
        )

    # Reading test data
    print(f"Reading {feature} features for testing...")
    labels_list_test = read_labels(labels_file_test, labels_names)
    corpus_test = []

    if feature == 'Words':
        corpus_test = read_text_data(lang, file_test)
    else:
        corpus_test = read_all_features(
            lang, words_file_test, emo_file_test, hashs_file_test,
            ats_file_test, links_file_test, abvs_file_test
        )

    labels_test = np.asarray(labels_list_test)

    # Create a list with the labels
    labels = list(set(labels_list_train))

    # NumPy array to store the predictions
    predictions_bagging = np.zeros((len(corpus_test), 5))

    # Create a list with all the indexes in the corpus train
    list_index = list(range(len(corpus_train)))

    predictions_file_bert = data_dir / 'predictions_bert_bagging_5_.txt'

    with open(predictions_file_bert, 'w', encoding='utf-8') as w:
        w.write('Accuracy,Precision,Recall,F1,Kapha,ROC\n')

    for iteration in range(50):
        print(f'Iteration: {iteration}')

        for i in range(5):
            # Select random indexes
            indexes = random.sample(list_index, int(len(corpus_train) * 0.7))

            # Create a sub-corpus and sub-labels
            sub_corpus = [corpus_train[x] for x in indexes]
            sub_labels = [labels_list_train[x] for x in indexes]

            # Split train data into train data and val data
            train_data, val_data, train_labels, val_labels = train_test_split(
                sub_corpus, sub_labels, test_size=0.1,
                random_state=0, stratify=sub_labels
            )

            # Preprocess data with the BERT model of selection
            preproc = text.Transformer(model_name, maxlen=512, class_names=labels)
            train_data_preproc = preproc.preprocess_train(train_data, train_labels)
            val_data_preproc = preproc.preprocess_test(val_data, val_labels)

            predicted, _ = train_bert(
                train_data_preproc, val_data_preproc, corpus_test, preproc
            )
            predicted = np.array(predicted)

            predictions_bagging[:, i] = predicted

        # Majority class
        majority_predictions = [
            1 if (np.count_nonzero(row == 1)) >= 3 else 0
            for row in predictions_bagging
        ]

        # Testing
        accuracy = np.mean(majority_predictions == labels_test)
        precision_macro = metrics.precision_score(labels_test, majority_predictions, average='macro')
        recall_macro = metrics.recall_score(labels_test, majority_predictions, average='macro')
        f1_macro = metrics.f1_score(labels_test, majority_predictions, average='macro')
        kappa = metrics.cohen_kappa_score(labels_test, majority_predictions)
        roc = metrics.roc_auc_score(labels_test, majority_predictions, average='macro')

        print(f'Accuracy: {accuracy:0.4f}')
        print(f'Precision: {precision_macro:0.4f}')
        print(f'Recall: {recall_macro:0.4f}')
        print(f'F1: {f1_macro:0.4f}')
        print(f'Kappa: {kappa:0.4f}')
        print(f'ROC-AUC: {roc:0.4f}')
        print('\n')

        with open(predictions_file_bert, 'a', encoding='utf-8') as w:
            p_bert = (
                f'{accuracy:1.4f},{precision_macro:1.4f},{recall_macro:1.4f},'
                f'{f1_macro:1.4f},{kappa:1.4f},{roc:1.4f}'
            )
            w.write(p_bert + '\n')

if __name__ == "__main__":
    main()
