#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Classify if news is true or fake using the Fake News Corpus Spanish dataset.

Performs hyperparameter experimentation (epochs and learning rates) with BERT.
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

def main():
    project_root = Path(__file__).resolve().parent.parent.parent
    project_root = Path(__file__).resolve().parent.parent.parent.parent
    """Main execution block for BERT hyperparameter experimentation on FNCS dataset."""
    # Variables
    dataset = 'fncs_dataset'
    lang = 'spanish'
    model_name = 'dccuchile/bert-base-spanish-wwm-cased'
    feature = 'Words'

    # Path handling using pathlib
        data_dir = project_root / 'data' / dataset
    out_dir = data_dir / 'experimentation_bert'
    out_dir.mkdir(parents=True, exist_ok=True)

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

    # Split train data into train data and val data
    train_data, val_data, train_labels, val_labels = train_test_split(
        corpus_train, labels_list_train, test_size=0.1,
        random_state=0, stratify=labels_list_train
    )

    # Preprocess data with the BERT model of selection
    preproc = text.Transformer(model_name, maxlen=512, class_names=labels)
    train_data_preproc = preproc.preprocess_train(train_data, train_labels)
    val_data_preproc = preproc.preprocess_test(val_data, val_labels)

    epochs = [5, 10, 20, 30, 50]
    learning_rates = [3e-4, 1e-4, 5e-5, 3e-5]

    for e in epochs:
        for l in learning_rates:
            print(f'Epochs: {e} Learning Rate: {l}')

            predictions_file_bert = out_dir / f'predictions_bert_{e}_{l}.txt'

            with open(predictions_file_bert, 'w', encoding='utf-8') as w:
                w.write('Accuracy,Precision,Recall,F1,Kapha,ROC\n')

            for i in range(5):
                print(f'Iteration: {i}')

                # Clear session
                tf.keras.backend.clear_session()

                # Model creation and training
                model = preproc.get_classifier()
                learner = ktrain.get_learner(
                    model, train_data=train_data_preproc,
                    val_data=val_data_preproc, batch_size=6
                )
                learner.fit_onecycle(
                    lr=l, epochs=e, verbose=0,
                    callbacks=[EarlyStopping(
                        monitor='val_auc', patience=2, mode='max', restore_best_weights=True
                    )]
                )

                # Prediction
                predictor = ktrain.get_predictor(model, preproc)
                predicted = predictor.predict(corpus_test)

                # Testing
                accuracy = np.mean(predicted == labels_test)
                precision_macro = metrics.precision_score(labels_test, predicted, average='macro')
                recall_macro = metrics.recall_score(labels_test, predicted, average='macro')
                f1_macro = metrics.f1_score(labels_test, predicted, average='macro')
                kappa = metrics.cohen_kappa_score(labels_test, predicted)
                roc = metrics.roc_auc_score(labels_test, predicted, average='macro')

                with open(predictions_file_bert, 'a', encoding='utf-8') as w:
                    p_bert = (
                        f'{accuracy:1.4f},{precision_macro:1.4f},{recall_macro:1.4f},'
                        f'{f1_macro:1.4f},{kappa:1.4f},{roc:1.4f}'
                    )
                    w.write(p_bert + '\n')

                print(f'Accuracy: {accuracy:0.4f}')
                print(f'Precision: {precision_macro:0.4f}')
                print(f'Recall: {recall_macro:0.4f}')
                print(f'F1: {f1_macro:0.4f}')
                print(f'Kappa: {kappa:0.4f}')
                print(f'ROC-AUC: {roc:0.4f}')
                print('\n')

if __name__ == "__main__":
    main()
