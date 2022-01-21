# -*- coding: utf-8 -*-
"""
Created on Fri Jul  2 10:18:44 2021

@author: luismiguells

Description: Classify if news is true or fake using the datasets Covid Dataset, 
FakeNewsNet Dataset, ISOT Dataset and Fake News Costa Rica News Dataset. 
Creating Word2Vec vectors to tranform the data. LSTM, GRU and BiLSTM are the 
models used for the classification.
"""

from keras.layers import Dense, Embedding, LSTM, GRU, Dropout, Bidirectional
from keras.preprocessing.sequence import pad_sequences
from sklearn.model_selection import StratifiedKFold
from keras.preprocessing.text import Tokenizer
from keras.callbacks import EarlyStopping
from keras.metrics import AUC, Accuracy
from keras.models import Sequential
from gensim.models import Word2Vec
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

class MySentences(object):
    def __init__(self, file_name):
        self.file_name = file_name
 
    def __iter__(self):
        for line in open(self.file_name):
            yield line.rstrip().split()

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

def remove_empty_text_data_w2v(corpus, labels):
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

def w2v_reader(w2v_file):
    """
    Parameters
    ----------
    w2v_file : fastText file that contains the vectors.

    Returns
    -------
    w2v_file : A dictionary where the key is the word and the value of the word.
    """
    w2v_dict = {}
    with open(w2v_file, 'r', encoding='utf-8') as w2v_reader:
        for line in w2v_reader:
            tokens = line.strip().split()
            if len(tokens) > 301:
                pass
            else:
                vect = [float(token) for token in tokens[1:]]
                w2v_dict[tokens[0]] = vect
        

    return w2v_dict

def w2v_vectors(sentences):
    """
    Parameters
    ----------
    file : File that contains data (e.g., words, emojis, hashtags, etc.).

    Returns
    -------
    model : Dictionary where each feature is represented with a vector.
    """
    model = Word2Vec(sentences, min_count=1, vector_size=300, workers=4)
    
    return model

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
feature = 'Word2Vec'

# Neural network parameters
embedding_dimension = 300
# max_len = 2000
epochs = 20

# OM = Own Model PM = Pre-trained Model PM+OM = Combination of both
model_type = 'PM+OM' 


main_dir = 'C:/Users/Administrador/Google Drive/MT/data/'+dataset+'/'

if lang == 'spanish':
    w2v_file = 'C:/Users/Administrador/Google Drive/MT/data/pre_trained_models/word2vec_spanish_300.txt'
else:
    w2v_file = 'C:/Users/Administrador/Google Drive/MT/data/pre_trained_models/word2vec_english_300.txt'
    
labels_file = main_dir+'labels.txt'
words_file = main_dir+'split/words.txt'
labels_names = ['True', 'Fake']

# Select the model of word representation to work with
if model_type == 'OM':
    sentences_train = MySentences(words_file)
    w2v_dict = w2v_vectors(sentences_train)
elif model_type == 'PM':
    w2v_dict = w2v_reader(w2v_file)
elif model_type == 'PM+OM':
    w2v_dict = w2v_reader(w2v_file)
    sentences = MySentences(words_file)
    w2v_dict_aux = w2v_vectors(sentences)
    
    vocab = w2v_dict_aux.wv.key_to_index
    
    for w in vocab.keys():
        if w not in w2v_dict:
            w2v_dict[w] = w2v_dict_aux.wv.get_vector(w)


# Reading data
labels_list = read_labels(labels_file, labels_names)
corpus = []
corpus = read_text_data(lang, words_file)

if model_type == 'OM':
    words = w2v_dict.wv.index_to_key
    vectors = w2v_dict.wv.vectors
    
    w2v_dict = dict(zip(words, vectors))
    
# Remove possible elements with length 0
corpus, labels_list = remove_empty_text_data_w2v(corpus, labels_list)
labels = np.asarray(labels_list)
labels_set = set(labels_list)

embeddings_index = w2v_dict

len_news = [len(line.split()) for line in corpus]
max_len = int(np.mean(len_news))

# Classifiers to use
classifiers = ['LSTM', 'GRU', 'BiLSTM']

for cl in classifiers:
    print('Training and testing with', cl)
    classifier = cl
    out_dir = 'C:/Users/Administrador/Google Drive/MT/results/probabilities/'+classifier+'/'+feature+'/'+model_type+'/'+dataset+'/'
    
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
        word_index = tokenizer.word_index
        
        # Convert text into a sequence of ints
        data_train = tokenizer.texts_to_sequences(data_train)
        data_test = tokenizer.texts_to_sequences(data_test)
        
        # pad the sequences
        data_train = pad_sequences(data_train, maxlen=max_len, padding='post', truncating='post')        
        data_test = pad_sequences(data_test, maxlen=max_len, padding='post', truncating='post') 
        
        # Convert the train labels into a array of n=classes dimenssion
        labels_train = pd.get_dummies(labels_train).values
        
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
            clf.fit(data_train, labels_train, epochs=epochs, batch_size=64, verbose=0, validation_split=0.1, callbacks=[EarlyStopping(monitor='val_auc', patience=5, mode='max', restore_best_weights=True)])
            
        elif cl == 'GRU':
            
            # Create the neural network
            clf = Sequential()
            clf.add(Embedding(input_dim=len(tokenizer.word_index)+1, output_dim=embedding_dimension, input_length=max_len, trainable=False, weights=[embedding_matrix]))
            clf.add(GRU(128))
            clf.add(Dropout(rate=0.5))
            clf.add(Dense(units=len(labels_names), activation='softmax'))
            clf.compile(optimizer='adamax', loss='categorical_crossentropy', metrics=[AUC(), Accuracy()])
            
            # Training
            clf.fit(data_train, labels_train, epochs=epochs, batch_size=64, verbose=0, validation_split=0.1, callbacks=[EarlyStopping(monitor='val_auc', patience=5, mode='max', restore_best_weights=True)])
        
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