#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Extract statistics from the Fake News Corpus Spanish dataset.

This module calculates the number of true and fake news, average length
per feature per class, and vocabulary size per feature per class.
"""

import sys
from pathlib import Path

import numpy as np

sys.path.append(str(Path(__file__).resolve().parent.parent.parent))
from utils import read_labels, read_raw_data, read_text_data

# Add parent directory to path to import utils

def main():
    project_root = Path(__file__).resolve().parent.parent.parent
    project_root = Path(__file__).resolve().parent.parent.parent.parent
        """Main execution block for extracting dataset statistics."""
    # Variables
    dataset = 'fncs_dataset'
    lang = 'spanish'
    labels_names = ['True', 'Fake']
    features_list = ['words', 'emojis', 'hashtags', 'ats', 'links', 'abvs']

    # Path handling using pathlib
    main_dir = project_root / 'data' / dataset
    feat_files = [
        'split_train/words.txt', 'split_train/emoticons.txt',
        'split_train/hashtags.txt', 'split_train/ats.txt',
        'split_train/links.txt', 'split_train/abvs.txt'
    ]
    labels_file = main_dir / 'labels_train.txt'

    # Reading labels
    labels_list = read_labels(labels_file, labels_names)
    labels = np.asarray(labels_list)

    # Count true and fake news
    unique, counts = np.unique(labels, return_counts=True)
    dict_news = dict(zip(unique, counts))

    print(f'The number of true news in {dataset} dataset is: {dict_news.get(0, 0)}')
    print(f'The number of fake news in {dataset} dataset is: {dict_news.get(1, 0)}')

    # Calculate statistics per feature
    for i, feat_path_suffix in enumerate(feat_files):
        feat_path = main_dir / feat_path_suffix
        feature_name = features_list[i]

        print(f'Loading feature data: {feature_name}...')
        if i == 0:
            corpus = read_text_data(lang, feat_path)
        else:
            corpus = read_raw_data(feat_path)
        print('Feature data loaded!')

        true_news_len = [len(line.split()) for line, label in zip(corpus, labels) if label == 0]
        fake_news_len = [len(line.split()) for line, label in zip(corpus, labels) if label == 1]

        # Print length statistics
        for label_name, lengths in [('true', true_news_len), ('fake', fake_news_len)]:
            print(f'Mean, median and std of {label_name} news with {feature_name} in {dataset}:')
            if lengths:
                print(f'Mean: {np.mean(lengths):0.2f}')
                print(f'Median: {np.median(lengths):0.2f}')
                print(f'STD: {np.std(lengths):0.2f}')
                min_val = np.min(lengths)
                print(f'Max/Min: {np.max(lengths) / min_val if min_val != 0 else "inf":0.2f}')
            else:
                print('No data for this class.')
            print('\n')

        # Vocabulary size
        l_voc = set()
        for line in corpus:
            l_voc.update(line.split())
        print(f'The vocabulary size of {feature_name} in {dataset}:')
        print(f'Size: {len(l_voc)}\n')

        # Vocabulary size per class
        for label_idx, label_name in [(0, 'TRUE NEWS'), (1, 'FAKE NEWS')]:
            class_voc = set()
            class_corpus = [line for line, lbl in zip(corpus, labels) if lbl == label_idx]
            for line in class_corpus:
                class_voc.update(line.split())
            print(f'The vocabulary size of {label_name} with {feature_name} in {dataset}:')
            print(f'Size: {len(class_voc)}\n')

if __name__ == "__main__":
    main()
