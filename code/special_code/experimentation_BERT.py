# -*- coding: utf-8 -*-
"""
Created on Wed Aug  4 18:03:41 2021

@author: luismiguells

Description: Classify if news is true or fake using the Fake News Corpus Spanish 
dataset. This dataset is divided in train and test data. Using BERT models for
trainsformation and classification
"""

from sklearn.model_selection import train_test_split
from keras.callbacks import EarlyStopping
from nltk.corpus import stopwords
from sklearn import metrics
from ktrain import text
import tensorflow as tf
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

def read_text_data_without_remove_en_stopwords(lang, file):
    """
    Parameters
    ----------
    file : The file that contains the text.

    Returns
    -------
    data : A list that contains all the text data.
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
            data.append(text)
    return data

# Variables
dataset = 'fncs_dataset'
lang = 'spanish'
model_name = 'dccuchile/bert-base-spanish-wwm-cased'
feature = 'Words'

# Files
main_dir = 'C:/Users/Administrator/Google Drive/MT/data/'+dataset+'/'
out_dir = 'C:/Users/Administrator/Google Drive/MT/data/fncs_dataset/experimentation_bert/'
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
    corpus_train = read_text_data(lang, file_train)
    # corpus_train = read_text_data_without_remove_en_stopwords(lang, words_file_train)
else:
    corpus_train = read_all_features(lang, words_file_train, emo_file_train, hashs_file_train, ats_file_train, links_file_train, abvs_file_train)

# Reading test data
labels_list_test = read_labels(labels_file_test, labels_names)
corpus_test = []

if feature == 'Words':
    corpus_test = read_text_data(lang, file_test)
    # corpus_test = read_text_data_without_remove_en_stopwords(lang, words_file_test)
else:
    corpus_test = read_all_features(lang, words_file_test, emo_file_test, hashs_file_test, ats_file_test, links_file_test, abvs_file_test)

labels_test = np.asarray(labels_list_test)

# Create a list with the labels
labels = list(set(labels_list_train))


# Split train data into train data and val data
train_data, val_data, train_labels, val_labels = train_test_split(corpus_train, labels_list_train, test_size=0.1, random_state=0, stratify=labels_list_train)

# Preprocess data with the BERT model of selection
preproc = text.Transformer(model_name, maxlen=512, class_names=labels)
train_data_preproc = preproc.preprocess_train(train_data, train_labels)
val_data_preproc = preproc.preprocess_test(val_data, val_labels)


epochs = [5, 10, 20, 30, 50]
learning_rates = [3e-4, 1e-4, 5e-5, 3e-5]

for e in epochs:
    for l in learning_rates: 
        
        print('Epochs: %d Learning Rate: %f'% (e, l))
        
        predictions_file_bert = out_dir+'predictions_bert_'+str(e)+'_'+str(l)+'.txt'
        
        with open(predictions_file_bert, 'w', encoding='utf-8') as w:
            w.write('Accuracy,Precision,Recall,F1,Kapha,ROC\n')

        for i in range(5):
            
            print('Iteration:', i)
        
            # Clear session
            tf.keras.backend.clear_session()
            
            # Model creation and training
            model = preproc.get_classifier()
            learner = ktrain.get_learner(model, train_data=train_data_preproc, val_data=val_data_preproc, batch_size=6)
            learner.fit_onecycle(lr=l, epochs=e, verbose=0, callbacks=[EarlyStopping(monitor='val_auc', patience=2, mode='max', restore_best_weights=True)])
            
            # Prediction
            predictor = ktrain.get_predictor(model, preproc)
            predicted = predictor.predict(corpus_test)
            predicted_proba = predictor.predict_proba(corpus_test)
            
            
            # Testing
            accuracy = np.mean(predicted == labels_test)
            precission_macro = metrics.precision_score(labels_test, predicted, average='macro')
            recall_macro = metrics.recall_score(labels_test, predicted, average='macro')
            f1_macro = metrics.f1_score(labels_test, predicted, average='macro')
            kapha = metrics.cohen_kappa_score(labels_test, predicted)
            roc = metrics.roc_auc_score(labels_test, predicted, average='macro')
            
            with open(predictions_file_bert, 'a', encoding='utf-8') as w:
                p_bert = '{0:1.4f},{1:1.4f},{2:1.4f},{3:1.4f},{4:1.4f},{5:1.4f}'.format(accuracy, precission_macro, recall_macro, f1_macro, kapha, roc)
                w.write(p_bert+'\n')
            
            
            print('Accuracy: %0.4f' % accuracy)
            print('Precision: %0.4f' % precission_macro)
            print('Recall: %0.4f' % recall_macro)
            print('F1: %0.4f' % f1_macro)
            print('Kapha: %0.4f' % kapha)
            print('ROC-AUC: %0.4f' % roc)
            print('\n')
            
            
            # # Recalculate with probabilities of other classifier
            # predictions_proba = predicted_proba
            # RF_probabilities = 'C:/Users/Administrator/Google Drive/MT/results/probabilities/RF/AF/fncs_dataset/probability.csv'
            # SVM_probablities = 'C:/Users/Administrator/Google Drive/MT/results/probabilities/SVM/AF/fncs_dataset/probability.csv'
            
            
            # # Extract the probabilities
            # data_RF = pd.read_csv(RF_probabilities, encoding='utf-8')
            # data_SVM = pd.read_csv(SVM_probablities, encoding='utf-8')
            
            # data_no_fake_RF = np.array(data_RF[' probability_true'])
            # data_fake_RF = np.array(data_RF[' probability_fake'])
            # data_no_fake_SVM = np.array(data_SVM[' probability_true'])
            # data_fake_SVM = np.array(data_SVM[' probability_fake'])
            
            # data_no_fake_RF = data_no_fake_RF[:, None]
            # data_fake_RF = data_fake_RF[:, None]
            # data_no_fake_SVM = data_no_fake_SVM[:, None]
            # data_fake_SVM = data_fake_SVM[:, None]
            
            
            # proba_RF = np.concatenate((data_no_fake_RF, data_fake_RF), axis=1)
            # proba_SVM = np.concatenate((data_no_fake_SVM, data_fake_SVM), axis=1)
            
            # # Multuply the probabilities of BERT model with RF model
            # proba_final = predicted_proba*proba_RF
            # predictions = np.argmax(proba_final, axis=1)
            
            # accuracy = np.mean(predictions == labels_test)
            # precission_macro = metrics.precision_score(labels_test, predictions, average='macro')
            # recall_macro = metrics.recall_score(labels_test, predictions, average='macro')
            # f1_macro = metrics.f1_score(labels_test, predictions, average='macro')
            # kapha = metrics.cohen_kappa_score(labels_test, predictions)
            # roc = metrics.roc_auc_score(labels_test, predictions, average='macro')
            
            # with open(predictions_file_rf_bert, 'a', encoding='utf-8') as w:
            #     p_bert_rf = '{0:1.4f},{1:1.4f},{2:1.4f},{3:1.4f},{4:1.4f},{5:1.4f}'.format(accuracy, precission_macro, recall_macro, f1_macro, kapha, roc)
            #     w.write(p_bert_rf+'\n')
            
            # print('Accuracy: %0.4f' % accuracy)
            # print('Precision: %0.4f' % precission_macro)
            # print('Recall: %0.4f' % recall_macro)
            # print('F1: %0.4f' % f1_macro)
            # print('Kapha: %0.4f' % kapha)
            # print('ROC-AUC: %0.4f' % roc)
            # print('\n')		


