# -*- coding: utf-8 -*-
"""
Dataset statistics module for extracting different metrics from fake news datasets.
Calculates class distribution, average length per feature, and vocabulary size.
"""

import sys
import numpy as np
from pathlib import Path

# Add the parent directory of this script to sys.path to import utils
sys.path.append(str(Path(__file__).resolve().parent.parent))
from utils import read_labels, read_text_data, read_raw_data

def calculate_statistics(dataset_name, lang, main_dir, labels_names):
    """
    Calculates and prints statistics for a given dataset.

    Args:
        dataset_name (str): Name of the dataset.
        lang (str): Language of the text ('spanish' or 'english').
        main_dir (Path): Base directory for the dataset files.
        labels_names (list): List of label names (e.g., ['True', 'Fake']).
    """
    features = ['words', 'emojis', 'hashtags', 'ats', 'links', 'abvs']
    feat_files = [
        'split/words.txt', 'split/emoticons.txt', 'split/hashtags.txt',
        'split/ats.txt', 'split/links.txt', 'split/abvs.txt'
    ]
    labels_file = main_dir / 'labels.txt'

    if not labels_file.exists():
        print(f"Labels file not found at {labels_file}")
        return

    # Reading labels
    labels_list = read_labels(labels_file, labels_names)
    labels = np.asarray(labels_list)

    # Count the number of true and fake news
    unique, counts = np.unique(labels, return_counts=True)
    dict_news = dict(zip(unique, counts))

    print(f"Statistics for {dataset_name} dataset:")
    print(f"  Number of True news: {dict_news.get(0, 0)}")
    print(f"  Number of Fake news: {dict_news.get(1, 0)}\n")

    # Calculate statistics per feature
    for i, feat_name in enumerate(features):
        feat_path = main_dir / feat_files[i]
        if not feat_path.exists():
            print(f"Feature file {feat_path} not found. Skipping {feat_name}.")
            continue

        print(f"Processing feature: {feat_name}")
        
        # Load data
        if i == 0:
            corpus = read_text_data(lang, feat_path)
        else:
            corpus = read_raw_data(feat_path)

        # Calculate lengths per class
        # Assuming label 0 is True and 1 is Fake
        true_news_len = [len(line.split()) for line, label in zip(corpus, labels) if label == 0]
        fake_news_len = [len(line.split()) for line, label in zip(corpus, labels) if label == 1]

        def print_stats(name, data):
            if not data:
                print(f"  No data for {name}")
                return
            print(f"  {name} Stats:")
            print(f"    Mean: {np.mean(data):.2f}")
            print(f"    Median: {np.median(data):.2f}")
            print(f"    STD: {np.std(data):.2f}")
            min_val = np.min(data)
            max_val = np.max(data)
            ratio = max_val / min_val if min_val != 0 else float('inf')
            print(f"    Max/Min Ratio: {ratio:.2f}")

        print_stats("TRUE NEWS", true_news_len)
        print_stats("FAKE NEWS", fake_news_len)

        # Vocabulary size
        all_tokens = [token for line in corpus for token in line.split()]
        voc_s = len(set(all_tokens))
        print(f"  Total Vocabulary Size: {voc_s}")

        true_tokens = [token for line, label in zip(corpus, labels) if label == 0 for token in line.split()]
        fake_tokens = [token for line, label in zip(corpus, labels) if label == 1 for token in line.split()]
        
        print(f"  True News Vocabulary Size: {len(set(true_tokens))}")
        print(f"  Fake News Vocabulary Size: {len(set(fake_tokens))}\n")

if __name__ == "__main__":
    # Example configuration
    DATASET = 'fncn_dataset'
    LANGUAGE = 'spanish'
    BASE_PATH = Path(__file__).resolve().parent.parent.parent / 'data' / DATASET
    LABELS = ['True', 'Fake']

    if BASE_PATH.exists():
        calculate_statistics(DATASET, LANGUAGE, BASE_PATH, LABELS)
    else:
        print(f"Base path not found: {BASE_PATH}")
