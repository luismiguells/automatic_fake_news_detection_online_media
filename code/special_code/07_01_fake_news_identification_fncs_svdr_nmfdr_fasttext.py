# -*- coding: utf-8 -*-
"""
Created on Sun Jun  6 16:55:41 2021

@author: luismiguells

Description: Classify if news is true or fake using the Fake News Corpus Spanish 
dataset. This dataset is divided in train and test data. Creating fastText vectors
to tranform the data. SVDR and NMFDR are the models used for 
the classification.
"""

from sklearn.metrics.pairwise import cosine_similarity
from sklearn.model_selection import StratifiedKFold
from sklearn.decomposition import TruncatedSVD, NMF
from sklearn.preprocessing import normalize
from nltk.corpus import stopwords
from sklearn import metrics
import numpy as np
import fasttext
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
        example = example.reshape(1, -1)
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
        example = example.reshape(1, -1)
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
dataset = 'fncs_dataset'
lang = 'spanish'
feature = 'fastText'

# OM = Own Model PM = Pre-trained Model PM+OM = Combination of both
model_type = 'PM+OM' 

# Files
main_dir = 'C:/Users/luismiguel/Google Drive/MT/data/'+dataset+'/'
fasttext_file = 'C:/Users/luismiguel/Google Drive/MT/data/pre_trained_models/fasttext_spanish_300.vec'
labels_names = ['True', 'Fake']

# Train file
labels_file_train = main_dir+'labels_train.txt'
words_file_train = main_dir+'split_train/words.txt'

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
corpus_train = []
corpus_train = read_text_data(lang, words_file_train)
labels_list_train = read_labels(labels_file_train, labels_names)
labels_train = np.asarray(labels_list_train)
labels_set_train = set(labels_list_train)

# Create the corpus train with fastText vectors
l_s = [line.split() for line in corpus_train]
s = np.zeros((300)) 
ft_train = []
l_not_fasttext = []

for l in l_s:
    i = 0
    for w in l:
        if w in fast_text_dict:
            s += fast_text_dict[w]
            i += 1
        else:
            l_not_fasttext.append(w)
            
    s = s/i
    ft_train.append(s)
    s = np.zeros((300))
    
# Remove possible elements value equal to NaN
ft_train, labels_train = remove_nan_values(ft_train, labels_train)
ft_corpus_train = np.array(ft_train)

# Normalize the data
n, m = ft_corpus_train.shape
min_values = np.min(ft_corpus_train, axis=1)
min_values = min_values.reshape((n, 1))
ft_corpus_train = ft_corpus_train+abs(min_values)
ft_corpus_train = normalize(ft_corpus_train, norm='l2')

# Reading test data
labels_list_test = read_labels(labels_file_test, labels_names)
corpus_test = []
corpus_test = read_text_data(lang, words_file_test)
test_index = [i for i in range(len(labels_list_test))]
test_index = np.asarray(test_index)
labels_test = np.asarray(labels_list_test)
labels_set_test = set(labels_list_test)

# Create the corpus test with fastText vectors
l_s = [line.split() for line in corpus_test]
s = np.zeros((300)) 
ft_test = []
l_not_fasttext = []

for l in l_s:
    i = 0
    for w in l:
        if w in fast_text_dict:
            s += fast_text_dict[w]
            i += 1
        else:
            l_not_fasttext.append(w)
            
    s = s/i
    ft_test.append(s)
    s = np.zeros((300))
    
# Remove possible elements value equal to NaN
ft_test, labels_test = remove_nan_values(ft_test, labels_test)
ft_corpus_test = np.array(ft_test)

# Normalize the data
n, m = ft_corpus_test.shape
min_values = np.min(ft_corpus_test, axis=1)
min_values = min_values.reshape((n, 1))
ft_corpus_test = ft_corpus_test+abs(min_values)
ft_corpus_test = normalize(ft_corpus_test, norm='l2')

# Classfiers to use
classifiers = ['SVDR', 'NMFDR']

for cl in classifiers:
    
    print('Training and testing with', cl)
    classifier = cl
    out_dir = 'C:/Users/luismiguel/Google Drive/MT/results/probabilities/'+classifier+'/'+feature+'/'+model_type+'/'+dataset+'/'
    
    start = time.time()
    
    # Training and testing 
    
    if classifier == 'SVDR':
        
        # Find the best hyper-parameter
        ns = [1, 2, 4, 8, 16, 32]
        best_n = 0
        best_score = 0
        
        for n in ns:
            
            sub_skf = StratifiedKFold(n_splits=5, random_state=0, shuffle=True)
            scores_inner = []
            for sub_train_index, sub_test_index in sub_skf.split(ft_corpus_train, labels_train):
                sub_data_train, sub_data_test = ft_corpus_train[sub_train_index], ft_corpus_train[sub_test_index]
                sub_labels_train, sub_labels_test = labels_train[sub_train_index], labels_train[sub_test_index]
                sub_svdr = train_pca(sub_data_train, sub_labels_train, labels_set_train, n)
                sub_predicted, temp_scores = predict_pca(sub_data_test, sub_svdr)
                sub_f1_macro = metrics.f1_score(sub_labels_test, sub_predicted, average='macro')
                scores_inner.append(sub_f1_macro)
            score = np.mean(scores_inner)
            if score > best_score:
                best_score = score
                best_n = n
        
        # Traing with the best hyper-parameter
        pcadr = train_pca(ft_corpus_train, labels_train, labels_set_train, best_n)
        
        # Testing with data 
        predicted, predicted_proba = predict_pca(ft_corpus_test, pcadr)
                
    elif classifier == 'NMFDR':
        
        # Find the best hyper-parameter
        ns = [1, 2, 4, 8, 16, 32]
        best_n = 0
        best_score = 0
        
        for n in ns:
            
            sub_skf = StratifiedKFold(n_splits=5, random_state=0, shuffle=True)
            scores_inner = []
            for sub_train_index, sub_test_index in sub_skf.split(ft_corpus_train, labels_train):
                sub_data_train, sub_data_test = ft_corpus_train[sub_train_index], ft_corpus_train[sub_test_index]
                sub_labels_train, sub_labels_test = labels_train[sub_train_index], labels_train[sub_test_index]
                sub_nmfdr = train_nmf(sub_data_train, sub_labels_train, labels_set_train, n)
                sub_predicted, temp_scores = predict_nmf(sub_data_test, sub_nmfdr)
                sub_f1_macro = metrics.f1_score(sub_labels_test, sub_predicted, average='macro')
                scores_inner.append(sub_f1_macro)
            score = np.mean(scores_inner)
            if score > best_score:
                best_score = score
                best_n = n
                
        # Traing with the best hyper-parameter
        nmfdr = train_nmf(ft_corpus_train, labels_train, labels_set_train, best_n)
        
        # Testing with data
        predicted, predicted_proba = predict_nmf(ft_corpus_test, nmfdr)
    
    
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
