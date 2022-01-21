# -*- coding: utf-8 -*-
"""
Created on Wed May  5 13:43:24 2021

@author: luismiguells

Description: Classify if news is true or fake using the datasets Covid Dataset, 
FakeNewsNet Dataset, ISOT Dataset and Fake News Costa Rica News Dataset. 
Using the method TF-IDF to tranform the data. SVDR and NMFDR are the models 
used for the classification.
"""


from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity
from sklearn.model_selection import StratifiedKFold
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

# Files
main_dir = 'C:/Users/luismiguel/Google Drive/MT/data/'+dataset+'/'
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
labels_set = list(set(labels_list))

# Classifiers to use
classifiers = ['SVDR', 'NMFDR']

for cl in classifiers:
    
    print('Training and testing with', cl)
    classifier = cl
    out_dir = 'C:/Users/luismiguel/Google Drive/MT/results/probabilities/'+classifier+'/'+feature+'/'+dataset+'/'
    
    skf = StratifiedKFold(n_splits=10, random_state=0, shuffle=True) #10 
    scores_accuracy = []
    scores_precission_macro = []
    scores_recall_macro = []
    scores_f1_macro = []
    scores_kapha = [] 
    scores_roc = []
    
    
    i = 0
    start = time.time()
    
    # Training and testing 
    for train_index, test_index in skf.split(corpus, labels):
        print('Fold :',i)
        data_train = [corpus[x] for x in train_index]
        data_test = [corpus[x] for x in test_index]
        labels_train, labels_test = labels[train_index], labels[test_index]
        vec = TfidfVectorizer(min_df=1, norm='l2', analyzer='word', tokenizer=my_tokenizer)
        train_tfidf = vec.fit_transform(data_train)
        
        
        
        if classifier == 'SVDR':
                   
            # Find the best hyper-parameter
            ns = [1, 2, 4, 8, 16, 32]
            best_n = 0
            best_score = 0
            
            for n in ns:
                
                sub_skf = StratifiedKFold(n_splits=3, random_state=0, shuffle=True)
                scores_inner = []
                for sub_train_index, sub_test_index in sub_skf.split(train_tfidf, labels_train):
                    sub_data_train, sub_data_test = train_tfidf[sub_train_index], train_tfidf[sub_test_index]
                    sub_labels_train, sub_labels_test = labels_train[sub_train_index], labels_train[sub_test_index]
                    sub_svdr = train_pca(sub_data_train, sub_labels_train, labels_set, n)
                    sub_predicted, temp_scores = predict_pca(sub_data_test, sub_svdr)
                    sub_f1_macro = metrics.f1_score(sub_labels_test, sub_predicted, average='macro')
                    scores_inner.append(sub_f1_macro)
                score = np.mean(scores_inner)
                if score > best_score:
                    best_score = score
                    best_n = n
            
            # Traing with the best hyper-parameter
            pcadr = train_pca(train_tfidf, labels_train, labels_set, best_n)
            
            # Testing with data 
            test_tfidf = vec.transform(data_test)
            predicted, predicted_proba = predict_pca(test_tfidf, pcadr)
                
        elif classifier == 'NMFDR':
            
            # Find the best hyper-parameter
            ns = [1, 2, 4, 8, 16, 32]
            best_n = 0
            best_score = 0
            
            for n in ns:
                
                sub_skf = StratifiedKFold(n_splits=3, random_state=0, shuffle=True)
                scores_inner = []
                for sub_train_index, sub_test_index in sub_skf.split(train_tfidf, labels_train):
                    sub_data_train, sub_data_test = train_tfidf[sub_train_index], train_tfidf[sub_test_index]
                    sub_labels_train, sub_labels_test = labels_train[sub_train_index], labels_train[sub_test_index]
                    sub_nmfdr = train_nmf(sub_data_train, sub_labels_train, labels_set, n)
                    sub_predicted, temp_scores = predict_nmf(sub_data_test, sub_nmfdr)
                    sub_f1_macro = metrics.f1_score(sub_labels_test, sub_predicted, average='macro')
                    scores_inner.append(sub_f1_macro)
                score = np.mean(scores_inner)
                if score > best_score:
                    best_score = score
                    best_n = n
                    
            # Traing with the best hyper-parameter
            nmfdr = train_nmf(train_tfidf, labels_train, labels_set, best_n)
            
            # Testing with data
            test_tfidf = vec.transform(data_test)
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