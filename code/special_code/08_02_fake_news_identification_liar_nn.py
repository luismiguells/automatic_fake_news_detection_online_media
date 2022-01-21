# -*- coding: utf-8 -*-
"""
Created on Mon Jun  7 16:13:19 2021

@author: luismiguells

Description: Classify if news is true or fake using the LIAR dataset. 
This dataset is divided in train, validation and test data. Using a tokenization
to tranform the data. LSTM, GRU and BiLSTM are the models 
used for the classification.
"""

from keras.layers import Dense, Embedding, LSTM, GRU, Dropout, Bidirectional
from keras.preprocessing.sequence import pad_sequences
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

# Variables
dataset = 'liar_dataset'
lang = 'english'
feature = 'Raw'

# Neural network parameters
embedding_dimension = 300
# max_len = 2000
epochs = 20

# Files
main_dir = 'C:/Users/Administrador/Google Drive/MT/data/'+dataset+'/'
labels_names = ['True', 'Fake']

# Train file
labels_file_train = main_dir+'labels_train.txt'
words_file_train = main_dir+'split_train/words.txt'
hashs_file_train = main_dir+'split_train/hashtags.txt'
ats_file_train = main_dir+'split_train/ats.txt'
emo_file_train = main_dir+'split_train/emoticons.txt'
links_file_train = main_dir+'split_train/links.txt'
abvs_file_train = main_dir+'split_train/abvs.txt'
file_train = main_dir+'corpus_train.txt'

# Validation file
labels_file_valid = main_dir+'labels_valid.txt'
words_file_valid = main_dir+'split_valid/words.txt'
hashs_file_valid = main_dir+'split_valid/hashtags.txt'
ats_file_valid = main_dir+'split_valid/ats.txt'
emo_file_valid = main_dir+'split_valid/emoticons.txt'
links_file_valid = main_dir+'split_valid/links.txt'
abvs_file_valid = main_dir+'split_valid/abvs.txt'
file_valid = main_dir+'corpus_valid.txt'

# Test file
labels_file_test = main_dir+'labels_test.txt'
words_file_test = main_dir+'split_test/words.txt'
hashs_file_test = main_dir+'split_test/hashtags.txt'
ats_file_test = main_dir+'split_test/ats.txt'
emo_file_test = main_dir+'split_test/emoticons.txt'
links_file_test = main_dir+'split_test/links.txt'
abvs_file_test = main_dir+'split_test/abvs.txt'
file_test = main_dir+'corpus_test.txt'



# Reading train data
labels_list_train = read_labels(labels_file_train, labels_names)
corpus_train = []

if feature == 'Words':
    corpus_train = read_text_data(lang, words_file_train)
elif feature == 'AF':
    corpus_train = read_all_features(lang, words_file_train, emo_file_train, hashs_file_train, ats_file_train, links_file_train, abvs_file_train)    
elif feature == 'Raw':
    corpus_train = read_text_data(lang, file_train)
    
labels_train = np.asarray(labels_list_train)
labels_set_train = set(labels_list_train)

# Reading valid data
labels_list_valid = read_labels(labels_file_valid, labels_names)
corpus_valid = []

if feature == 'Words':
    corpus_valid = read_text_data(lang, words_file_valid)
elif feature == 'AF':
    corpus_valid = read_all_features(lang, words_file_valid, emo_file_valid, hashs_file_valid, ats_file_valid, links_file_valid, abvs_file_valid)        
elif feature == 'Raw':
    corpus_valid = read_text_data(lang, file_valid)
    
labels_valid = np.asarray(labels_list_valid)
labels_set_valid = set(labels_list_valid)


# Reading test data
labels_list_test = read_labels(labels_file_test, labels_names)
corpus_test = []

if feature == 'Words':
    corpus_test = read_text_data(lang, words_file_test)
elif feature == 'AF':
    corpus_test = read_all_features(lang, words_file_test, emo_file_test, hashs_file_test, ats_file_test, links_file_test, abvs_file_test)    
elif feature == 'Raw':
    corpus_test = read_text_data(lang, file_test)
    
test_index = [i for i in range(len(labels_list_test))]
test_index = np.asarray(test_index)
labels_test = np.asarray(labels_list_test)
labels_set_test = set(labels_list_test)

len_news = [len(line.split()) for line in corpus_train]
max_len = int(np.mean(len_news))

# Convert the train and validation labels into a array of n=classes dimenssion
labels_train = pd.get_dummies(labels_train).values
labels_valid = pd.get_dummies(labels_valid).values

# Classifiers to use
classifiers = ['LSTM', 'GRU', 'BiLSTM']

for cl in classifiers:
    print('Training and testing with', cl)
    classifier = cl
    out_dir = 'C:/Users/Administrador/Google Drive/MT/results/probabilities/'+classifier+'/'+feature+'/'+dataset+'/'
    
    start = time.time()
    
    # Tokenizer
    tokenizer = Tokenizer()
    tokenizer.fit_on_texts(corpus_train)
    
    # Convert text into a sequence of ints
    data_train = tokenizer.texts_to_sequences(corpus_train)
    data_valid = tokenizer.texts_to_sequences(corpus_valid)
    data_test = tokenizer.texts_to_sequences(corpus_test)
    
    # Pad the sequences
    data_train = pad_sequences(data_train, maxlen=max_len, padding='post', truncating='post')        
    data_valid = pad_sequences(data_valid, maxlen=max_len, padding='post', truncating='post')
    data_test = pad_sequences(data_test, maxlen=max_len, padding='post', truncating='post')
    
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
        clf.fit(data_train, labels_train, epochs=epochs, batch_size=64, verbose=0, validation_data=(data_valid, labels_valid), callbacks=[EarlyStopping(monitor='val_auc', patience=5, mode='max', restore_best_weights=True)])
            
    elif cl == 'GRU':
        
        # Create the neural network
        clf = Sequential()
        clf.add(Embedding(input_dim=len(tokenizer.word_index)+1, output_dim=embedding_dimension, input_length=max_len))
        clf.add(GRU(128))
        clf.add(Dropout(rate=0.5))
        clf.add(Dense(units=len(labels_names), activation='softmax'))
        clf.compile(optimizer='adamax', loss='categorical_crossentropy', metrics=[AUC(), Accuracy()])
        
        # Training
        clf.fit(data_train, labels_train, epochs=epochs, batch_size=64, verbose=0, validation_data=(data_valid, labels_valid), callbacks=[EarlyStopping(monitor='val_auc', patience=5, mode='max', restore_best_weights=True)])
    
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
        clf.fit(data_train, labels_train, epochs=epochs, batch_size=64, verbose=0, validation_data=(data_valid, labels_valid), callbacks=[EarlyStopping(monitor='val_auc', patience=5,    mode='max', restore_best_weights=True)])

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
    np.savetxt(out_dir+'probability.csv', predictions, fmt=fmt,delimiter=',', encoding='utf-8', header='test_index, probability_true, probability_fake, predicted_class, real_class', comments='')


    end = time.time()
    
        
    # Print the results
    print('Accuracy: %0.2f' % accuracy)
    print('Precision: %0.2f' % precission_macro)
    print('Recall: %0.2f' % recall_macro)
    print('F1: %0.2f' % f1_macro)
    print('Kapha: %0.2f' % kapha)
    print('ROC-AUC: %0.2f' % roc)
    print('Time of training + testing: %0.2f' % (end - start))
    print('\n')