from pathlib import Path
import sys
# -*- coding: utf-8 -*-
"""
Created on Mon Jun 28 15:49:55 2021

@author: luismiguells

Description: Classify if news is true or fake using the LIAR  dataset. 
This dataset is divided in train and test data. Creating fastText vectors
to tranform the data. LSTM, GRU and BiLSTM are the models used for the 
classification.
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
import fasttext
import warnings
import time

sys.path.append(str(Path(__file__).resolve().parent.parent.parent))
from utils import read_labels, read_text_data

# Remove to see warnings
warnings.filterwarnings('ignore')

# Set seed for reproducibility
tf.random.set_seed(0)
np.random.seed(0)

# Variables
dataset = 'liar_dataset'
lang = 'english'
feature = 'fastText'

# Neural network parameters
embedding_dimension = 300
# max_len = 2000
epochs = 20

# OM = Own Model PM = Pre-trained Model PM+OM = Combination of both
model_type = 'PM+OM' 

# Files
project_root = Path(__file__).resolve().parent.parent.parent.parent
main_dir = project_root / 'data' / dataset
fasttext_file = project_root / 'data' / \'pre_trained_models/fasttext_english_300.vec'
labels_names = ['True', 'Fake']

# Train file
labels_file_train = main_dir / 'labels_train.txt'
words_file_train = main_dir / 'split_train/words.txt'

# Validation file
labels_file_valid = main_dir / 'labels_valid.txt'
words_file_valid = main_dir / 'split_valid/words.txt'

# Test file
labels_file_test = main_dir / 'labels_test.txt'
words_file_test = main_dir / 'split_test/words.txt'

# Select the model of word representation to work with
if model_type == 'OM':
    fast_text_dict = fast_text_vectors(words_file_train)
elif model_type == 'PM':
    fast_text_dict = fasttext_reader(fasttext_file)
elif model_type == 'PM+OM':
    fast_text_dict = fasttext_reader(fasttext_file)
    fast_text_dict_aux = fast_text_vectors(words_file_train)
    
    vocab = fast_text_dict_aux.get_words()
    
    for w in vocab:
        if w not in fast_text_dict:
            fast_text_dict[w] = fast_text_dict_aux.get_word_vector(w)

# Reading train data
labels_list_train = read_labels(labels_file_train, labels_names)
corpus_train = []
corpus_train = read_text_data(lang, words_file_train)

# Remove possible elements with length 0
corpus_train, labels_list_train = remove_empty_text_data_fasttext(corpus_train, labels_list_train)
labels_train = np.asarray(labels_list_train)
labels_set_train = set(labels_list_train)

# Reading valid data
labels_list_valid = read_labels(labels_file_valid, labels_names)
corpus_valid = []
corpus_valid = read_text_data(lang, words_file_valid)

# Remove possible elements with length 0
corpus_valid, labels_list_test = remove_empty_text_data_fasttext(corpus_valid, labels_list_valid)
labels_valid = np.asarray(labels_list_test)
labels_set_valid = set(labels_list_test)

# Reading test data
labels_list_test = read_labels(labels_file_test, labels_names)
corpus_test = []
corpus_test = read_text_data(lang, words_file_test)

# Remove possible elements with length 0
corpus_test, labels_list_test = remove_empty_text_data_fasttext(corpus_test, labels_list_test)
test_index = [i for i in range(len(labels_list_test))]
test_index = np.asarray(test_index)
labels_test = np.asarray(labels_list_test)
labels_set_test = set(labels_list_test)

# Create the embedding matrix
embeddings_index = fast_text_dict

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
    out_dir = project_root / 'results' / \'probabilities/'+classifier+'/'+feature+'/'+model_type+'/'+dataset+'/'
    
    start = time.time()

    # Tokenizer
    tokenizer = Tokenizer()
    tokenizer.fit_on_texts(corpus_train)
    word_index = tokenizer.word_index
    
    # Convert text into a sequence of ints
    data_train = tokenizer.texts_to_sequences(corpus_train)
    data_valid = tokenizer.texts_to_sequences(corpus_valid)
    data_test = tokenizer.texts_to_sequences(corpus_test)
    
    # Pad the sequences
    data_train = pad_sequences(data_train, maxlen=max_len, padding='post', truncating='post')        
    data_valid = pad_sequences(data_valid, maxlen=max_len, padding='post', truncating='post')
    data_test = pad_sequences(data_test, maxlen=max_len, padding='post', truncating='post')
    
    # Create the embedding train matrix
    embedding_matrix = np.zeros((len(word_index)+1, embedding_dimension))
    for word, idx in word_index.items():
        if idx < len(word_index)+1:
            embedding_vector = embeddings_index.get(word)
            if embedding_vector is not None:
                embedding_matrix[idx] = embedding_vector
    
    # Clear session
    tf.keras.backend.clear_session()
    
    if cl == 'LSTM':
    
        # Create the neural network
        clf = Sequential()
        clf.add(Embedding(input_dim=len(tokenizer.word_index)+1, output_dim=embedding_dimension, input_length=max_len, trainable=False, weights=[embedding_matrix]))
        clf.add(LSTM(128))
        clf.add(Dropout(rate=0.5))
        clf.add(Dense(units=len(labels_names), activation='softmax'))
        clf.compile(optimizer='adamax', loss='categorical_crossentropy', metrics=[AUC(), Accuracy()])
        
        # Training
        clf.fit(data_train, labels_train, epochs=epochs, batch_size=64, verbose=0, validation_data=(data_valid, labels_valid), callbacks=[EarlyStopping(monitor='val_auc', patience=5, mode='max', restore_best_weights=True)])
            
    elif cl == 'GRU':
        
        # Create the neural network
        clf = Sequential()
        clf.add(Embedding(input_dim=len(tokenizer.word_index)+1, output_dim=embedding_dimension, input_length=max_len, trainable=False, weights=[embedding_matrix]))
        clf.add(GRU(128))
        clf.add(Dropout(rate=0.5))
        clf.add(Dense(units=len(labels_names), activation='softmax'))
        clf.compile(optimizer='adamax', loss='categorical_crossentropy', metrics=[AUC(), Accuracy()])
        
        # Training
        clf.fit(data_train, labels_train, epochs=epochs, batch_size=64, verbose=0, validation_data=(data_valid, labels_valid), callbacks=[EarlyStopping(monitor='val_auc', patience=5, mode='max', restore_best_weights=True)])
    
    elif cl == 'BiLSTM':
        
        # Create the neural network
        clf = Sequential()
        clf.add(Embedding(input_dim=len(tokenizer.word_index)+1, output_dim=embedding_dimension, input_length=max_len, trainable=False, weights=[embedding_matrix]))
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
    np.savetxt(out_dir / 'probability.csv', predictions, fmt=fmt,delimiter=',', encoding='utf-8', header='test_index, probability_true, probability_fake, predicted_class, real_class', comments='')

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
