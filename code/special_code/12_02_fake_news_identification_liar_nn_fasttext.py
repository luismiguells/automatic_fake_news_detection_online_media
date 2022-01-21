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

def remove_empty_text_data_fasttext(corpus, labels):
    """
    Parameters
    ----------
    corpus : A list with text data.
    labels : A list that contains the labels.

    Returns
    -------
    corpus : A list without elements of length 0.
    labels : A list that contains labels, except those whose length 
             in the corpus is equal to 0.
    """
    count = 0
    l_index = []
    for line in corpus:
        if len(line) == 0:
            l_index.append(count)
        count += 1
    
    if len(l_index) > 0:
        l_index.reverse()
        for idx in l_index:
            corpus.pop(idx)
            labels.pop(idx)
        return corpus, labels
    else:
        return corpus, labels

def remove_nan_values(corpus, labels):
    """
    Parameters
    ----------
    corpus : A list with text data.
    labels : A list that contains the labels.

    Returns
    -------
    corpus : A list without NaN elements.
    labels : A list that contains labels, except those whose elemens
             in the corpus is equal to NaN.
    """
    count = 0
    l_index = []
    
    for i in corpus:
        if np.isnan(i).any():
            l_index.append(count)
        count += 1
    if len(l_index) > 0:
        labels = list(labels)
        l_index.reverse()
        for idx in l_index:
            corpus.pop(idx)
            labels.pop(idx)
        return corpus, np.asarray(labels)
    else:
        return corpus, labels

def val_score(clf, train_data, train_labels, valid_data, valid_labels):
    """
    Parameters
    ----------
    clf : A classifier (SVM, MNB, KNN, etc.).
    train_data : Data to train the model.
    train_labels : Labels that correspond to the train_data.
    valid_data : Validation data to test the model.
    valid_labels : Validation labels that correspond to the valid_data.

    Returns
    -------
    f1_macro : F1 metric on its macro version.
    """
    clf.fit(train_data, train_labels)
    predicted = clf.predict(valid_data)
    f1_macro = metrics.f1_score(valid_labels, predicted, average='macro')
    
    return f1_macro

def fasttext_reader(fasttext_file):
    """
    Parameters
    ----------
    fasttext_file : fastText file that contains the vectors.

    Returns
    -------
    fasttext_dict : A dictionary where the key is the word and the value of the word.
    """
    fasttext_dict = {}
    with open(fasttext_file, 'r', encoding='utf-8') as fasttext_reader:
        for line in fasttext_reader:
            tokens = line.strip().split()
            vect = [float(token) for token in tokens[1:]]
            fasttext_dict[tokens[0]] = vect

    return fasttext_dict

def fast_text_vectors(file):
    """
    Parameters
    ----------
    file : File that contains data (e.g., words, emojis, hashtags, etc.).

    Returns
    -------
    model : Dictionary where each feature is represented with a vector.
    """
    model = fasttext.train_unsupervised(file, 'skipgram', dim=300, epoch=20, verbose=0)
    
    return model

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
main_dir = 'C:/Users/Administrador/Google Drive/MT/data/'+dataset+'/'
fasttext_file = 'C:/Users/Administrador/Google Drive/MT/data/pre_trained_models/fasttext_english_300.vec'
labels_names = ['True', 'Fake']


# Train file
labels_file_train = main_dir+'labels_train.txt'
words_file_train = main_dir+'split_train/words.txt'

# Validation file
labels_file_valid = main_dir+'labels_valid.txt'
words_file_valid = main_dir+'split_valid/words.txt'

# Test file
labels_file_test = main_dir+'labels_test.txt'
words_file_test = main_dir+'split_test/words.txt'

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
    out_dir = 'C:/Users/Administrador/Google Drive/MT/results/probabilities/'+classifier+'/'+feature+'/'+model_type+'/'+dataset+'/'
    
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
