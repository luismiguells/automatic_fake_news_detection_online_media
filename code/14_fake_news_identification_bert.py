# -*- coding: utf-8 -*-
"""
Created on Mon Sep  6 10:01:22 2021

@author: luismiguells

Description: Classify if news is true or fake using the datasets Covid Dataset, 
FakeNewsNet Dataset, ISOT Dataset and Fake News Costa Rica News Dataset. 
Creating Word2Vec vectors to tranform the data. LSTM, GRU and BiLSTM are the 
models used for the classification.
"""

from sklearn.model_selection import train_test_split
from sklearn.model_selection import StratifiedKFold
from keras.callbacks import EarlyStopping
from keras.metrics import AUC, Accuracy
from nltk.corpus import stopwords
from sklearn import metrics
import tensorflow as tf
from ktrain import text
import pandas as pd
import numpy as np
import warnings
import ktrain
import random
import time
import os


# Remove to see warnings
warnings.filterwarnings('ignore') 

# Set seed for reproducibility
os.environ['PYTHONHASHSEED']=str(0)
tf.random.set_seed(0)
np.random.seed(0)
random.seed(0)


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
    txt : A text that does not have short and long words, and without
           stopwords.
    """
    txt = ' '.join([word for word in words if len(word)>2 and len(word)<35 and word not in stop_words])
    return txt

def clean_en_words(txt):
    """
    Parameters
    ----------
    text : A string.

    Returns
    -------
    clean_text : A string without English stopwords for Spanish text.
    """
    stop_words = stopwords.words('english')
    words = txt.rstrip().split()
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
            txt = clean_words(words, stop_words)
            txt += ' '+emo_line.rstrip()
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

if lang == 'spanish':
    model_name = 'dccuchile/bert-base-spanish-wwm-cased'
else:
    model_name = 'bert-base-cased'
feature = 'AF'

main_dir = 'C:/Users/Administrator/Google Drive/MT/data/'+dataset+'/'
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

# Create a list with the labels
labels = list(set(labels_list))
labels_list = np.array(labels_list)

skf = StratifiedKFold(n_splits=10, random_state=0, shuffle=True) #10 
scores_accuracy = []
scores_precission_macro = []
scores_recall_macro = []
scores_f1_macro = []
scores_kapha = [] 
scores_roc = []

classifier = 'BERT'
out_dir = 'C:/Users/Administrator/Google Drive/MT/results/probabilities/'+classifier+'/'+feature+'/'+dataset+'/'

i = 0
start = time.time()

for train_index, test_index in skf.split(corpus, labels_list):
    
    print('Fold :',i)
    data_train = [corpus[x] for x in train_index]
    data_test = [corpus[x] for x in test_index]
    labels_train, labels_test = labels_list[train_index], labels_list[test_index]
    
    # Split train data into train data and val data
    train_data, val_data, train_labels, val_labels = train_test_split(data_train, labels_train, test_size=0.1, random_state=0, stratify=labels_train)
    
    # Preprocess data with the BERT model of selection
    preproc = text.Transformer(model_name, maxlen=512, class_names=labels)
    train_data_preproc = preproc.preprocess_train(train_data, train_labels)
    val_data_preproc = preproc.preprocess_test(val_data, val_labels)
    
    # Clear session
    tf.keras.backend.clear_session()
    
    # Model creation and training
    model = preproc.get_classifier()
    learner = ktrain.get_learner(model, train_data=train_data_preproc, val_data=val_data_preproc, batch_size=6)
    learner.fit_onecycle(lr=5e-5, epochs=7, verbose=0, callbacks=[EarlyStopping(monitor='val_auc', patience=2, mode='max', restore_best_weights=True)])
    
    # Prediction
    predictor = ktrain.get_predictor(model, preproc)
    predicted = predictor.predict(data_test)
    predicted_proba = predictor.predict_proba(data_test)
    
    accuracy = np.mean(predicted == labels_test)
    precission_macro = metrics.precision_score(labels_test, predicted, average='macro')
    recall_macro = metrics.recall_score(labels_test, predicted, average='macro')
    f1_macro = metrics.f1_score(labels_test, predicted, average='macro')
    kapha = metrics.cohen_kappa_score(labels_test, predicted) 
    roc = metrics.roc_auc_score(labels_test, predicted, average='macro')
    
    # Create an array to write it in a file
    predicted = np.array(predicted)
    predictions = np.concatenate((test_index[:, None], predicted_proba, predicted[:, None], labels_test[:, None].astype(int)), axis=1)
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

