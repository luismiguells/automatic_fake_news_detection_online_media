#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Wed May 19 21:36:13 2021

@author: luismiguells

Description: Classify if news is true or fake using the datasets Covid Dataset, 
FakeNewsNet Dataset, ISOT Dataset and Fake News Costa Rica News Dataset. 
Using a tokenization to tranform the data. LSTM, GRU and BiLSTM are the models 
used for the classification.
"""

from keras.layers import Dense, Embedding, LSTM, GRU, Dropout, Bidirectional
from keras.preprocessing.sequence import pad_sequences
from sklearn.model_selection import StratifiedKFold
from keras.preprocessing.text import Tokenizer
from keras.callbacks import EarlyStopping
from keras.metrics import AUC, Accuracy
from keras.models import Sequential
from nltk.corpus import stopwords
from sklearn import metrics
import tensorflow as tf
import pandas as pd
import numpy as np
import warnings
import time

# Remove to see warnings
warnings.filterwarnings('ignore') 

# Set seed for reproducibility
tf.random.set_seed(0)
np.random.seed(0)
 
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
    labels_names : A list filled with the labels to use. 
                   For example, ['True', 'Fake'].

    Returns
    -------
    label : A list of labels using the index of labels_names variable.
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

def read_text_data_with_emos(lang, text_file, emo_file):
    """
    Parameters
    ----------
    lang : The selected language for stopwords.
    text_file : A list that contains the text data.
    emo_file : A list that contains the emojis.

    Returns
    -------
    data : A clean list with text and emojis merged.
    """
    data = []
    if lang == 'spanish':
        stop_words = stopwords.words('spanish')
    elif lang == 'english':
        stop_words = stopwords.words('english')
    with open(text_file, encoding='utf-8') as text_content, open(emo_file, encoding='utf-8') as emo_content:
        for text_line, emo_line in zip(text_content, emo_content):
            words = text_line.rstrip().split()
            text = clean_words(words, stop_words)
            text += ' '+emo_line.rstrip()
            data.append(text)
    return data

def read_all_features(lang, text_file, emo_file, hashs_file, ats_file, links_file, abvs_file):
    """
    Parameters
    ----------
    lang : The selected language for stopwords.
    text_file : A list that contains the text data of the corpus.
    emo_file : A list that contains the emojis of the corpus.
    hash_file : A list that contains the hashtags (#) of the corpus.
    ats_file : A list thtat contains the ats (@) of the corpus.
    links_file : A list that contains the links (https::\\www...) of the corpus.
    abvs_file : A list that contains the abbrevations (ggg) of the corpus.

    Returns
    -------
    data : A corpus with all the features aforementioned merged.
    """
    data = []
    if lang == 'spanish':
        stop_words = stopwords.words('spanish')
    elif lang == 'english':
        stop_words = stopwords.words('english')
    with open(text_file, encoding='utf-8') as text_content, open(emo_file, encoding='utf-8') as emo_content, open(hashs_file, encoding='utf-8') as hashs_content, open(ats_file, encoding='utf-8') as ats_content, open(links_file, encoding='utf-8') as links_content, open(abvs_file, encoding='utf-8') as abvs_content:
        for text_line, emo_line, hashs_line, ats_line, links_line, abvs_line in zip(text_content, emo_content, hashs_content, ats_content, links_content, abvs_content):
            words = text_line.rstrip().split()
            text = clean_words(words, stop_words)
            
            # Clean possible English stopwords that could be in the Spanish corpus
            if lang == 'spanish':
                text = clean_en_words(text)
            text += ' '+emo_line.rstrip()+' '+hashs_line.rstrip()+' '+ats_line.rstrip()+' '+links_line.rstrip()+' '+abvs_line.rstrip()
            data.append(text)
    return data

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

"""
Datasets
----------------------------------------------
covid_dataset: COVID Dataset
fnn_dataset: Fake NewsNet Dataset
isot_dataset: ISOT Dataset
fncn_dataset: Fake News Costa Rica News Dataset
"""


# Variables
dataset = 'fncn_dataset'
lang = 'spanish'
feature = 'Raw'

# Neural network parameters
embedding_dimension = 300
# max_len = 2000
epochs = 20


# Files
main_dir = 'C:/Users/Administrador/Google Drive/MT/data/'+dataset+'/'
labels_file = main_dir+'labels.txt'
words_file = main_dir+'split/words.txt'
hashs_file = main_dir+'split/hashtags.txt'
ats_file = main_dir+'split/ats.txt'
emo_file = main_dir+'split/emoticons.txt'
links_file = main_dir+'split/links.txt'
abvs_file = main_dir+'split/abvs.txt'
file_train = main_dir+'corpus.txt'
labels_names = ['True', 'Fake']

# Reading data
labels_list = read_labels(labels_file, labels_names)
corpus = []

if feature == 'Words':
    corpus = read_text_data(lang, words_file)
elif feature == 'AF':
    corpus = read_all_features(lang, words_file, emo_file, hashs_file, ats_file, links_file, abvs_file)
elif feature == 'Raw':
    corpus = read_text_data(lang, file_train)
    
labels = np.asarray(labels_list)
labels_set = set(labels_list)

len_news = [len(line.split()) for line in corpus]
max_len = int(np.mean(len_news))


# Classifiers to use
classifiers = ['LSTM', 'GRU', 'BiLSTM']

for cl in classifiers:
    print('Training and testing with', cl)
    classifier = cl
    out_dir = 'C:/Users/Administrador/Google Drive/MT/results/probabilities/'+classifier+'/'+feature+'/'+dataset+'/'
    
    skf = StratifiedKFold(n_splits=10, random_state=0, shuffle=True) #10 
    scores_accuracy = []
    scores_precission_macro = []
    scores_recall_macro = []
    scores_f1_macro = []
    scores_kapha = [] 
    scores_roc = []
    
    start = time.time()
    i = 0
    
    for train_index, test_index in skf.split(corpus, labels):
        print('Fold:', i)
        data_train = [corpus[x] for x in train_index]
        data_test = [corpus[x] for x in test_index]
        labels_train = labels[train_index]
        labels_test = labels[test_index]
        
        # Tokenizer
        tokenizer = Tokenizer()
        tokenizer.fit_on_texts(data_train)
        
        # Convert text into a sequence of ints
        data_train = tokenizer.texts_to_sequences(data_train)
        data_test = tokenizer.texts_to_sequences(data_test)
        
        # pad the sequences
        data_train = pad_sequences(data_train, maxlen=max_len, padding='post', truncating='post')        
        data_test = pad_sequences(data_test, maxlen=max_len, padding='post', truncating='post') 
        
        # Convert the train labels into a array of n=classes dimenssion
        labels_train = pd.get_dummies(labels_train).values
        
        # Clear session
        tf.keras.backend.clear_session()
        
        if cl == 'LSTM':
    
            # Create the neural network
            clf = Sequential()
            clf.add(Embedding(input_dim=len(tokenizer.word_index)+1, output_dim=embedding_dimension, input_length=max_len))
            clf.add(LSTM(128))
            clf.add(Dropout(rate=0.5))
            clf.add(Dense(units=len(labels_names), activation='softmax'))
            clf.compile(optimizer='adamax', loss='categorical_crossentropy', metrics=[AUC(), Accuracy()])
            
            # Training
            clf.fit(data_train, labels_train, epochs=epochs, batch_size=64, verbose=0, validation_split=0.1, callbacks=[EarlyStopping(monitor='val_auc', patience=5, mode='max', restore_best_weights=True)])
            
        elif cl == 'GRU':
            
            # Create the neural network
            clf = Sequential()
            clf.add(Embedding(input_dim=len(tokenizer.word_index)+1, output_dim=embedding_dimension, input_length=max_len))
            clf.add(GRU(128))
            clf.add(Dropout(rate=0.5))
            clf.add(Dense(units=len(labels_names), activation='softmax'))
            clf.compile(optimizer='adamax', loss='categorical_crossentropy', metrics=[AUC(), Accuracy()])
            
            # Training
            clf.fit(data_train, labels_train, epochs=epochs, batch_size=64, verbose=0, validation_split=0.1, callbacks=[EarlyStopping(monitor='val_auc', patience=5, mode='max', restore_best_weights=True)])
        
        elif cl == 'BiLSTM':
            
            # Create the neural network
            clf = Sequential()
            clf.add(Embedding(input_dim=len(tokenizer.word_index)+1, output_dim=embedding_dimension, input_length=max_len))
            for j in range(0, 3):
                
                # Add a bidirectional LSTM layer
                clf.add(Bidirectional(LSTM(128, return_sequences=True, recurrent_dropout=0.2)))
                
                # Add a dropout layer after each LSTM layer
                clf.add(Dropout(0.5))
                
            clf.add(Bidirectional(LSTM(128, recurrent_dropout=0.2)))
            clf.add(Dropout(0.5))
                
            # Add the fully connected layer with 256 neurons and relu activation
            clf.add(Dense(256, activation='relu'))
            clf.add(Dense(units=len(labels_names), activation='softmax'))
            clf.compile(optimizer='adamax', loss='categorical_crossentropy', metrics=[AUC(), Accuracy()])
            
            # Training
            clf.fit(data_train, labels_train, epochs=epochs, batch_size=64, verbose=0, validation_split=0.1, callbacks=[EarlyStopping(monitor='val_auc', patience=5,    mode='max', restore_best_weights=True)])
            
        # Testing
        predicted = np.argmax(clf.predict(data_test), axis=-1)
        predicted_proba = clf.predict(data_test)
        accuracy = np.mean(predicted == labels_test)
        precission_macro = metrics.precision_score(labels_test, predicted, average='macro')
        recall_macro = metrics.recall_score(labels_test, predicted, average='macro')
        f1_macro = metrics.f1_score(labels_test, predicted, average='macro')
        kapha = metrics.cohen_kappa_score(labels_test, predicted)
        roc = metrics.roc_auc_score(labels_test, predicted, average='macro')
        
        # Create an array to write it in a file
        predictions = np.concatenate((test_index[:, None], predicted_proba, predicted[:, None].astype(int), labels_test[:, None].astype(int)), axis=1)
        fmt = '%d', '%1.9f', '%1.9f', '%d', '%d'
        np.savetxt(out_dir+'fold_'+str(i)+'.csv', predictions, fmt=fmt,delimiter=',', encoding='utf-8', header='test_index, probability_true, probability_fake, predicted_class, real_class', comments='')
        
        scores_accuracy.append(accuracy)
        scores_precission_macro.append(precission_macro)
        scores_recall_macro.append(recall_macro)
        scores_f1_macro.append(f1_macro)
        scores_kapha.append(kapha)
        scores_roc.append(roc)
        i += 1
        
        
    end = time.time()
    
    
    # Print the results
    print('Accuracy: mean=%0.2f std=+/-%0.2f median=%0.2f' % (np.mean(scores_accuracy), np.std(scores_accuracy), np.median(scores_accuracy)))
    print('Precision: mean=%0.2f std=+/-%0.2f median=%0.2f' % (np.mean(scores_precission_macro), np.std(scores_precission_macro), np.median(scores_precission_macro)))
    print('Recall: mean=%0.2f std=+/-%0.2f median=%0.2f' % (np.mean(scores_recall_macro), np.std(scores_recall_macro), np.median(scores_recall_macro)))
    print('F1: mean=%0.2f std=+/-%0.2f median=%0.2f' % (np.mean(scores_f1_macro), np.std(scores_f1_macro), np.median(scores_f1_macro)))
    print('Kapha: mean=%0.2f std=+/-%0.2f median=%0.2f' % (np.mean(scores_kapha), np.std(scores_kapha), np.median(scores_kapha)))
    print('ROC-AUC: mean=%0.2f std=+/-%0.2f median=%0.2f' % (np.mean(scores_roc), np.std(scores_roc), np.median(scores_roc)))
    print('Time of training + testing: %0.2f' % (end - start))
    print('\n')