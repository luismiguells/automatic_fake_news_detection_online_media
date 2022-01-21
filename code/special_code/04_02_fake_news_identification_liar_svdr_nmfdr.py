# -*- coding: utf-8 -*-
"""
Created on Sun May 30 13:51:48 2021

@author: luismiguells

Description: Classify if news is true or fake using the LIAR dataset. 
This dataset is divided in train, validation and test data. Using the method 
TF-IDF to tranform the data. SVDR and NMFDR are the models
used for the classification.
"""

from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity
from sklearn.decomposition import TruncatedSVD, NMF
from nltk.corpus import stopwords
from sklearn import metrics
import numpy as np
import warnings
import time


# Remove to see warnings
warnings.filterwarnings('ignore')

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

def train_pca(data_train, labels_train, labels_set, n):
    pcadr = []
    for label in labels_set:
       cat_train = data_train[labels_train==label]
       svd = TruncatedSVD(n_components=n, n_iter=15, random_state=0)
       svd.fit(cat_train)
       pcadr.append(svd)
    return pcadr

def predict_pca(data_test, pcadr):
    predicted = []
    probas = []
    for example in data_test:
        sims = []
        for svdpca in pcadr:
            example_proj = svdpca.transform(example)
            example_rec = svdpca.inverse_transform(example_proj)
            aux = example_rec/np.linalg.norm(example_rec)
            if not np.isnan(np.sum(aux)):
                example_rec = example_rec/np.linalg.norm(example_rec)
            # error = np.linalg.norm(example_rec-example)
            sim = cosine_similarity(example_rec, example)
            sim = sum(sum(sim))
            sims.append(sim)
        predicted.append(np.argmax(sims))
        tot = sum(sims)
        proba = [p/tot for p in sims]
        probas.append(proba)
    predicted = np.array(predicted)
    probas = np.array(probas)
    
    return predicted, probas

def train_nmf(data_train, labels_train, labels_set, n):
    nmfdr = []
    for label in labels_set:
        cat_train = data_train[labels_train==label]
        model = NMF(n_components=n, init='random', random_state=0)
        model.fit(cat_train)
        nmfdr.append(model)
    return nmfdr

def predict_nmf(data_test, nmfdr):
    predicted = []
    probas = []
    for example in data_test:
        sims = []
        for svdnmf in nmfdr:
            example_proj = svdnmf.transform(example)
            example_rec = svdnmf.inverse_transform(example_proj)
            aux = example_rec/np.linalg.norm(example_rec)
            if not np.isnan(np.sum(aux)):
                example_rec = example_rec/np.linalg.norm(example_rec)
            # error = np.linalg.norm(example_rec-example)
            sim = cosine_similarity(example_rec, example)
            sim = sum(sum(sim))
            sims.append(sim)
        predicted.append(np.argmax(sims))
        tot = sum(sims)
        proba = [p/tot for p in sims]
        probas.append(proba)

    predicted = np.array(predicted)
    probas = np.array(probas)
    
    return predicted, probas

# Variables
dataset = 'liar_dataset'
lang = 'english'
feature = 'Raw'


# Files
main_dir = 'C:/Users/luismiguel/Google Drive/MT/data/'+dataset+'/'
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
labels_set_train = list(set(labels_list_train))

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
labels_set_valid = list(set(labels_list_valid))


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
labels_set_test = list(set(labels_list_test))

# Classifiers to use
classifiers = ['SVDR', 'NMFDR']

for cl in classifiers:
    
    print('Training and testing with', cl)
    classifier = cl
    out_dir = 'C:/Users/luismiguel/Google Drive/MT/results/probabilities/'+classifier+'/'+feature+'/'+dataset+'/'
    
    
    start = time.time()
    
    # Training and testing 
    vec = TfidfVectorizer(min_df=1, norm='l2', analyzer = 'word', tokenizer=my_tokenizer)
    train_tfidf = vec.fit_transform(corpus_train)
    valid_tfidf = vec.transform(corpus_valid)
    
    
    if classifier == 'SVDR':
                    
        # Find the best hyper-parameter
        ns = [1, 2, 4, 8, 16, 32]
        best_n = 0
        best_score = 0
        
        for n in ns:
            sub_svdr = train_pca(train_tfidf, labels_train, labels_set_train, n)
            sub_predicted, temp_scores = predict_pca(valid_tfidf, sub_svdr)
            score = metrics.f1_score(labels_valid, sub_predicted, average='macro')
            if score > best_score:
                best_score = score
                best_n = n
                
        # Concatenate the train and validation data
        corpus_train_grouped = corpus_train+corpus_valid
        labels_train_grouped = labels_list_train+labels_list_valid
        labels_train_grouped = np.asarray(labels_train_grouped) 
        
        vec = TfidfVectorizer(min_df=1, norm='l2', analyzer = 'word', tokenizer=my_tokenizer)
        train_grouped_tfidf = vec.fit_transform(corpus_train_grouped)
        
        # Traing with the best hyper-parameter
        pcadr = train_pca(train_grouped_tfidf, labels_train_grouped, labels_set_train, best_n)
        
        # Testing with data 
        test_tfidf = vec.transform(corpus_test)
        predicted, predicted_proba = predict_pca(test_tfidf, pcadr)
    
    elif classifier == 'NMFDR':
            
        # Find the best hyper-parameter
        ns = [1, 2, 4, 8, 16, 32]
        best_n = 0
        best_score = 0
        
        for n in ns:
            sub_nmfdr = train_nmf(train_tfidf, labels_train, labels_set_train, n)
            sub_predicted, temp_scores = predict_nmf(valid_tfidf, sub_svdr)
            score = metrics.f1_score(labels_valid, sub_predicted, average='macro')
            if score > best_score:
                best_score = score
                best_n = n
        
        # Concatenate the train and validation data
        corpus_train_grouped = corpus_train+corpus_valid
        labels_train_grouped = labels_list_train+labels_list_valid
        labels_train_grouped = np.asarray(labels_train_grouped) 
        
        
        vec = TfidfVectorizer(min_df=1, norm='l2', analyzer = 'word', tokenizer=my_tokenizer)
        train_grouped_tfidf = vec.fit_transform(corpus_train_grouped)
        
        # Traing with the best hyper-parameter
        nmfdr = train_nmf(train_grouped_tfidf, labels_train_grouped, labels_set_train, best_n)
        
        # Testing with data
        test_tfidf = vec.transform(corpus_test)
        predicted, predicted_proba = predict_nmf(test_tfidf, nmfdr)
    
    
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