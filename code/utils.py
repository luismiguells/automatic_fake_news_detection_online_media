# -*- coding: utf-8 -*-
"""
Common utility functions for the fake news detection project.
"""

from nltk.corpus import stopwords
import pathlib

def my_tokenizer(s):
    """
    Standard tokenizer that splits a string by whitespace.

    Args:
        s (str): The string to tokenize.

    Returns:
        list: A list of words.
    """
    return s.split()

def read_labels(file_path, labels_names):
    """
    Reads labels from a file and maps them to their index in labels_names.

    Args:
        file_path (str or pathlib.Path): Path to the labels file.
        labels_names (list): List of possible label strings (e.g., ['True', 'Fake']).

    Returns:
        list: A list of label indices.
    """
    label_indices = []
    with open(file_path, 'r', encoding='utf-8') as f:
        for line in f:
            category = line.strip()
            if category in labels_names:
                label_indices.append(labels_names.index(category))
    return label_indices

def clean_words(words, stop_words):
    """
    Cleans a list of words by removing stopwords and filtering by length.

    Args:
        words (list): List of words to clean.
        stop_words (set or list): Stopwords to remove.

    Returns:
        str: A space-separated string of cleaned words.
    """
    return ' '.join([
        word for word in words 
        if 2 < len(word) < 35 and word not in stop_words
    ])

def clean_en_words(text):
    """
    Removes English stopwords from a given text.

    Args:
        text (str): The text to clean.

    Returns:
        str: Text without English stopwords.
    """
    en_stop_words = set(stopwords.words('english'))
    words = text.rstrip().split()
    return ' '.join([word for word in words if word not in en_stop_words])

def read_text_data(lang, file_path, remove_en_stopwords=True):
    """
    Reads and cleans text data from a file.

    Args:
        lang (str): The language for stopwords ('spanish' or 'english').
        file_path (str or pathlib.Path): Path to the text file.
        remove_en_stopwords (bool): Whether to remove English stopwords from Spanish text.

    Returns:
        list: A list of cleaned text strings.
    """
    data = []
    stop_words = set(stopwords.words(lang))
    
    with open(file_path, 'r', encoding='utf-8') as f:
        for line in f:
            words = line.rstrip().split()
            text = clean_words(words, stop_words)
            
            # Clean possible English stopwords that could be in the Spanish corpus
            if lang == 'spanish' and remove_en_stopwords:
                text = clean_en_words(text)
            data.append(text)
    return data

def read_raw_data(file_path):
    """
    Reads data from a file line by line and returns it as a list.

    Args:
        file_path (str or pathlib.Path): Path to the file.

    Returns:
        list: A list of lines, stripped of trailing whitespace.
    """
    data = []
    with open(file_path, 'r', encoding='utf-8') as f:
        for line in f:
            data.append(line.strip())
    return data

def read_text_data_with_emos(lang, text_file, emo_file):
    """
    Reads and cleans text data from a file and merges it with emojis.

    Args:
        lang (str): The language for stopwords ('spanish' or 'english').
        text_file (str or pathlib.Path): Path to the text file.
        emo_file (str or pathlib.Path): Path to the emojis file.

    Returns:
        list: A list of merged and cleaned text strings.
    """
    data = []
    stop_words = set(stopwords.words(lang))
    with open(text_file, encoding='utf-8') as text_content, \
         open(emo_file, encoding='utf-8') as emo_content:
        for text_line, emo_line in zip(text_content, emo_content):
            words = text_line.rstrip().split()
            text = clean_words(words, stop_words)
            text += ' ' + emo_line.rstrip()
            data.append(text)
    return data

def read_all_features(lang, text_file, emo_file, hashs_file, ats_file, links_file, abvs_file, remove_en_stopwords=True):
    """
    Reads and cleans text data and merges it with all extra features.

    Args:
        lang (str): The language for stopwords.
        text_file (str or pathlib.Path): Path to the text file.
        emo_file (str or pathlib.Path): Path to the emojis file.
        hashs_file (str or pathlib.Path): Path to the hashtags file.
        ats_file (str or pathlib.Path): Path to the ats file.
        links_file (str or pathlib.Path): Path to the links file.
        abvs_file (str or pathlib.Path): Path to the abbreviations file.
        remove_en_stopwords (bool): Whether to remove English stopwords from Spanish text.

    Returns:
        list: A list of merged and cleaned text strings.
    """
    data = []
    stop_words = set(stopwords.words(lang))

    with open(text_file, encoding='utf-8') as text_f, \
         open(emo_file, encoding='utf-8') as emo_f, \
         open(hashs_file, encoding='utf-8') as hashs_f, \
         open(ats_f, encoding='utf-8') as ats_f, \
         open(links_f, encoding='utf-8') as links_f, \
         open(abvs_f, encoding='utf-8') as abvs_f:

        for text_l, emo_l, hash_l, ats_l, links_l, abvs_l in zip(
            text_f, emo_f, hashs_f, ats_f, links_f, abvs_f
        ):
            words = text_l.rstrip().split()
            text = clean_words(words, stop_words)

            if lang == 'spanish' and remove_en_stopwords:
                text = clean_en_words(text)

            features = [emo_l, hash_l, ats_l, links_l, abvs_l]
            text += ' ' + ' '.join(f.rstrip() for f in features)
            data.append(text)
    return data
