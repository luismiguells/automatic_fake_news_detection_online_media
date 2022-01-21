# -*- coding: utf-8 -*-
"""
Created on Fri Oct 15 15:09:56 2021

@author: luismiguells

Description: Extract different statistics from the datasets: LIAR dataset
"""

from nltk.corpus import stopwords
import numpy as np


def my_tokenizer(s):
    """
    Parameters
    ----------
    s : A string.

    Returns
    -------
    A list with words splitted by space.

    """
    return s.split()

def read_labels(file, labels_names):
    """
    Parameters
    ----------
    file : Name of the file to read.
    labels_names : Labels to use. For example, ['True', 'Fake'].

    Returns
    -------
    label : A list of labels using numbers. For example, [0, 1].
    """
    label = []
    with open(file) as content_file:
        for line in content_file:
            category = line.strip()
            label.append(labels_names.index(category))
    return label

def clean_words(words, stop_words):
    """
    Parameters
    ----------
    words : A list of words.
    stop_words : A stopwords list from NLTK library.

    Returns
    -------
    text : A text that does not have short and long words, and without
           stopwords.
    """
    text = ' '.join([word for word in words if len(word)>2 and len(word)<35 and word not in stop_words])
    return text

def clean_en_words(text):
    """
    Parameters
    ----------
    text : A string.

    Returns
    -------
    clean_text : A string without English stopwords for Spanish text.
    """
    stop_words = stopwords.words('english')
    words = text.rstrip().split()
    clean_text = ' '.join([word for word in words if word not in stop_words])
    
    return clean_text

def read_text_data(lang, file):
    """
    Parameters
    ----------
    lang : The selected language for stopwords.
    file : The file that contains the text.

    Returns
    -------
    data : A list that contains all the text data cleaned.
    """
    data = []
    if lang == 'spanish':
        stop_words = stopwords.words('spanish')
    elif lang == 'english':
        stop_words = stopwords.words('english')
    with open(file, encoding='utf-8') as content_file:
        for line in content_file:
            words = line.rstrip().split()
            text = clean_words(words, stop_words)
            
            # Clean possible English stopwords that could be in the Spanish corpus
            if lang == 'spanish':
                text = clean_en_words(text)
            data.append(text)
    return data

def read_data(file):
    """
    Read data from a file line by line, removing the new line chars at the
    end of the line, and store the lines in a list.

    Parameters:
    file : A string with the file name to read.

    Returns
    -------
    A list with the lines of the file as strings.
    """
    data = []
    with open(file, 'r', encoding='utf-8') as reader:
        for line in reader:
            line = line.strip()
            data.append(line)
    return data

    



# Variables
dataset = 'liar_dataset'
lang = 'english'

# Files
main_dir = 'C:/Users/luismiguel/Google Drive/MT/data/'+dataset+'/'
labels_names = ['True', 'Fake']

features = ['words', 'emojis', 'hashtags', 'ats', 'links', 'abvs']

# Features train file names
feat_files_train = ['split_train/words.txt', 'split_train/emoticons.txt', 'split_train/hashtags.txt', 'split_train/ats.txt', 'split_train/links.txt', 'split_train/abvs.txt']
labels_file_train = main_dir+'labels_train.txt'

feat_files_valid = ['split_valid/words.txt', 'split_valid/emoticons.txt', 'split_valid/hashtags.txt', 'split_valid/ats.txt', 'split_valid/links.txt', 'split_valid/abvs.txt']
labels_file_valid = main_dir+'labels_train.txt'

# Reading data
labels_list_train = read_labels(labels_file_train, labels_names)
labels_list_valid = read_labels(labels_file_valid, labels_names)
labels_list = labels_list_train+labels_list_valid
labels = np.asarray(labels_list)
labels_set = set(labels_list)

# Count the number of true and fake news in each dataset

unique, counts = np.unique(labels, return_counts=True)
dict_news = dict(zip(unique, counts))

print('The number of true news in', dataset, 'dataset is:', dict_news[0])
print('The number of fake news in', dataset, 'dataset is:',  dict_news[1])

# Calculate the average length of news per feature

for feat in range(6):
    
    if feat == 0:
        print('Loading feature data...')
        corpus_train = read_text_data(lang, main_dir+feat_files_train[feat])
        corpus_valid = read_text_data(lang, main_dir+feat_files_valid[feat])
        print('Feature data loaded!')
    else:
        print('Loading feature data...')
        corpus_train = read_data(main_dir+feat_files_train[feat])
        corpus_valid = read_data(main_dir+feat_files_valid[feat])
        print('Feature data loaded!')


    feature_word = features[feat]
    
    corpus = corpus_train+corpus_valid
    
    true_news_len = [len(line.split()) for line, label in zip(corpus, labels) if label == 0]
    fake_news_len = [len(line.split()) for line, label in zip(corpus, labels) if label == 1]
    
    # Print the average length of feature per class
    print('Mean, median and std of true news with', feature_word, 'in the dataset', dataset, ':')
    print('Mean:', np.mean(true_news_len))
    print('Median:', np.median(true_news_len))
    print('STD:', np.std(true_news_len))
    print('Max/Min:', np.max(true_news_len)/np.min(true_news_len))
    print('\n')
    
    print('Mean, median and std of fake news with', feature_word, 'in the dataset', dataset, ':')
    print('Mean:', np.mean(fake_news_len))
    print('Median:', np.median(fake_news_len))
    print('STD:', np.std(fake_news_len))
    print('Max/Min:', np.max(fake_news_len)/np.min(fake_news_len))
    print('\n')
    
    # Calculate the vocabulary size in whole dataset
    
    l_voc = []
    
    for feature in corpus:
        tokens = feature.split()
        l_voc.extend(tokens)
    l_voc= set(l_voc)
    voc_s = len(l_voc)
    
    print('The vocabulary size of', feature_word, 'in the dataset', dataset, ':')
    print('Size:', voc_s)
    print('\n')
    
    # Calculate the vocabulary size in each class
    
    l_true_voc = []
    l_fake_voc = []
    
    corpus_true = [line for line, label in zip(corpus, labels) if label == 0]
    corpus_fake = [line for line, label in zip(corpus, labels) if label == 1]
    
    for feature in corpus_true:
        tokens = feature.split()
        l_true_voc.extend(tokens)
    l_true_voc = set(l_true_voc)
    voc_true_s = len(l_true_voc)
    
    for feature in corpus_fake:
        tokens = feature.split()
        l_fake_voc.extend(tokens)
    l_fake_voc = set(l_fake_voc)
    voc_fake_s = len(l_fake_voc)
    
    print('The vocabulary size of TRUE NEWS with', feature_word, 'in the dataset', dataset, ':')
    print('Size:', voc_true_s)
    print('\n')
    
    print('The vocabulary size of FAKE NEWS with', feature_word, 'in the dataset', dataset, ':')
    print('Size:', voc_fake_s)
    print('\n')

