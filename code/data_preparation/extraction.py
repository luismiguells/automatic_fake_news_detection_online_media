# -*- coding: utf-8 -*-
"""
Data extraction module for various fake news datasets.
This module provides functions to read raw data from different sources 
and save them in a consistent format (corpus.txt and labels.txt).
"""

import sys
import pandas as pd
from pathlib import Path

# Add the parent directory of this script to sys.path to import utils
sys.path.append(str(Path(__file__).resolve().parent.parent))
from utils import read_raw_data

def save_formatted_data(corpus, labels, corpus_file, labels_file):
    """
    Saves corpus and labels to text files, ensuring no newlines or tabs in text.

    Args:
        corpus (list): List of text strings.
        labels (list): List of label strings.
        corpus_file (Path): Path to save the corpus.
        labels_file (Path): Path to save the labels.
    """
    with open(corpus_file, 'w', encoding='utf-8') as c_f, \
         open(labels_file, 'w', encoding='utf-8') as l_f:
        for line, label in zip(corpus, labels):
            # Clean text by replacing newlines, tabs, and carriage returns
            clean_line = str(line).replace('\n', ' ').replace('\t', ' ').replace('\r', ' ')
            c_f.write(clean_line + '\n')
            l_f.write(str(label) + '\n')

def extract_fncn(data_path, out_dir):
    """
    Extracts data from the FNCN dataset (Excel format).

    Args:
        data_path (Path): Path to the source Excel file.
        out_dir (Path): Directory to save the output files.
    """
    data = pd.read_excel(data_path)
    data_text = list(data['Texto'])
    data_labels = list(data['Clasificacion'])

    labels = ['True' if i == 'V' else 'Fake' for i in data_labels]
    
    save_formatted_data(data_text, labels, out_dir / 'corpus.txt', out_dir / 'labels.txt')

def extract_fnn(gossip_fake_dir, gossip_real_dir, poli_fake_dir, poli_real_dir, out_dir):
    """
    Extracts data from the FNN dataset (Gossipcop and Politifact).

    Args:
        gossip_fake_dir (Path): Path to gossipcop_fake.txt.
        gossip_real_dir (Path): Path to gossipcop_real.txt.
        poli_fake_dir (Path): Path to politifact_fake.txt.
        poli_real_dir (Path): Path to politifact_real.txt.
        out_dir (Path): Directory to save the output files.
    """
    gossip_real_text = read_raw_data(gossip_real_dir)
    gossip_fake_text = read_raw_data(gossip_fake_dir)
    poli_real_text = read_raw_data(poli_real_dir)
    poli_fake_text = read_raw_data(poli_fake_dir)

    gossip_real_labels = ['True'] * len(gossip_real_text)
    gossip_fake_labels = ['Fake'] * len(gossip_fake_text)
    poli_real_labels = ['True'] * len(poli_real_text)
    poli_fake_labels = ['Fake'] * len(poli_fake_text)

    corpus = gossip_real_text + poli_real_text + gossip_fake_text + poli_fake_text
    labels = gossip_real_labels + poli_real_labels + gossip_fake_labels + poli_fake_labels

    save_formatted_data(corpus, labels, out_dir / 'corpus.txt', out_dir / 'labels.txt')

def extract_covid(data_path, out_dir):
    """
    Extracts data from the COVID dataset (CSV format).

    Args:
        data_path (Path): Path to the source CSV file.
        out_dir (Path): Directory to save the output files.
    """
    data = pd.read_csv(data_path)
    corpus = list(data['headlines'])
    data_labels = list(data['outcome'])

    labels = ['Fake' if label == 0 else 'True' for label in data_labels]

    save_formatted_data(corpus, labels, out_dir / 'corpus.txt', out_dir / 'labels.txt')

def extract_liar(train_dir, test_dir, val_dir, out_dir):
    """
    Extracts data from the LIAR dataset (TSV format).

    Args:
        train_dir (Path): Path to train.tsv.
        test_dir (Path): Path to test.tsv.
        val_dir (Path): Path to valid.tsv.
        out_dir (Path): Directory to save the output files.
    """
    train_data = pd.read_csv(train_dir, delimiter='\t')
    test_data = pd.read_csv(test_dir, delimiter='\t')
    val_data = pd.read_csv(val_dir, delimiter='\t')

    def map_liar_labels(labels):
        mapped = []
        for label in labels:
            if label in ['true', 'mostly-true']:
                mapped.append('True')
            else:
                mapped.append('Fake')
        return mapped

    save_formatted_data(
        train_data['statement'], map_liar_labels(train_data['label']),
        out_dir / 'corpus_train.txt', out_dir / 'labels_train.txt'
    )
    save_formatted_data(
        test_data['statement'], map_liar_labels(test_data['label']),
        out_dir / 'corpus_test.txt', out_dir / 'labels_test.txt'
    )
    save_formatted_data(
        val_data['statement'], map_liar_labels(val_data['label']),
        out_dir / 'corpus_valid.txt', out_dir / 'labels_valid.txt'
    )

def extract_isot(true_path, fake_path, out_dir):
    """
    Extracts data from the ISOT dataset (CSV format).

    Args:
        true_path (Path): Path to True.csv.
        fake_path (Path): Path to Fake.csv.
        out_dir (Path): Directory to save the output files.
    """
    true_data = pd.read_csv(true_path)
    fake_data = pd.read_csv(fake_path)

    true_corpus = [line for line in true_data['text'] if not str(line).isspace()]
    fake_corpus = [line for line in fake_data['text'] if not str(line).isspace()]

    labels = (['True'] * len(true_corpus)) + (['Fake'] * len(fake_corpus))
    corpus = true_corpus + fake_corpus

    save_formatted_data(corpus, labels, out_dir / 'corpus.txt', out_dir / 'labels.txt')

def extract_fncs(train_path, test_path, out_dir):
    """
    Extracts data from the FNCS dataset (Excel format).

    Args:
        train_path (Path): Path to train.xlsx.
        test_path (Path): Path to test.xlsx.
        out_dir (Path): Directory to save the output files.
    """
    train_data = pd.read_excel(train_path)
    test_data = pd.read_excel(test_path)

    save_formatted_data(
        train_data['Text'], train_data['Category'],
        out_dir / 'corpus_train.txt', out_dir / 'labels_train.txt'
    )
    save_formatted_data(
        test_data['Text'], test_data['Category'],
        out_dir / 'corpus_test.txt', out_dir / 'labels_test.txt'
    )

if __name__ == "__main__":
    # Example execution for LIAR dataset (was active in original script)
    # Update paths as necessary for your environment
    liar_base = Path(__file__).resolve().parent.parent.parent / 'data' / 'liar_dataset'
    liar_data = liar_base / 'data'
    
    # Check if directory exists before running to avoid errors in different environments
    if liar_data.exists():
        extract_liar(
            liar_data / 'train.tsv',
            liar_data / 'test.tsv',
            liar_data / 'valid.tsv',
            liar_base
        )
    else:
        print(f"LIAR data directory not found at {liar_data}")
