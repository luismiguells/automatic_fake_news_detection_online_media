# -*- coding: utf-8 -*-
"""
Created on Sun Nov 22 19:55:31 2020

@author: luismiguells
"""

import pandas as pd

def read_data(file):
    data = []
    with open(file, 'r', encoding='utf-8') as file_reader:
        for line in file_reader:
            data.append(line)
    return data

#------------------------- FNCN dataset ---------------------------------------
# data_dir = 'C:/Users/luismiguel/Documents/Tesis/data/fncn_dataset/data/datasource_clasificado_webhose.xls'
# data = pd.read_excel(data_dir)

# data_text = list(data['Texto'])
# data_labels = list(data['Clasificacion'])

# labels = []

# for i in data_labels:
#     if i == 'V':
#         labels.append('True')
#     else:
#         labels.append('Fake')

# corpus = data_text

# out_dir = 'C:/Users/luismiguel/Documents/Tesis/data/fncn_dataset/'
# corpus_file = out_dir+'corpus.txt'
# labels_file = out_dir+'labels.txt'

# with open(corpus_file, 'w', encoding='utf-8') as c_f, open(labels_file, 'w', encoding='utf-8') as l_f:
#     for line, label in zip(corpus, labels):
#         line = line.replace('\n', ' ')
#         line = line.replace('\t', ' ')
#         line = line.replace('\r', ' ')
#         c_f.write(line+'\n')
#         l_f.write(label+'\n')

#------------------------- FNN dataset ----------------------------------------
# gossip_fake_dir = 'C:/Users/luismiguel/Google Drive/MT/data/fnn_dataset/data/gossipcop_fake.txt'
# gossip_real_dir = 'C:/Users/luismiguel/Google Drive/MT/data/fnn_dataset/data/gossipcop_real.txt'
# poli_fake_dir = 'C:/Users/luismiguel/Google Drive/MT/data/fnn_dataset/data/politifact_fake.txt'
# poli_real_dir = 'C:/Users/luismiguel/Google Drive/MT/data/fnn_dataset/data/politifact_real.txt'


# gossip_real_text = read_data(gossip_real_dir)
# gossip_fake_text = read_data(gossip_fake_dir)
# poli_real_text = read_data(poli_real_dir)
# poli_fake_text = read_data(poli_fake_dir)

# gossip_real_labels = ['True' for i in range(len(gossip_real_text))]
# gossip_fake_labels = ['Fake' for i in range(len(gossip_fake_text))]
# poli_real_labels = ['True' for i in range(len(poli_real_text))]
# poli_fake_labels = ['Fake' for i in range(len(poli_fake_text))]

# corpus = gossip_real_text+poli_real_text+gossip_fake_text+poli_fake_text
# labels = gossip_real_labels+poli_real_labels+gossip_fake_labels+poli_fake_labels

# out_dir = 'C:/Users/luismiguel/Google Drive/MT/data/fnn_dataset/'
# corpus_file = out_dir+'corpus.txt'
# labels_file = out_dir+'labels.txt'

# with open(corpus_file, 'w', encoding='utf-8') as c_f, open(labels_file, 'w', encoding='utf-8') as l_f:
#     for line, label in zip(corpus, labels):
#         line = line.replace('\n', ' ')
#         line = line.replace('\t', ' ')
#         line = line.replace('\r', ' ')
#         c_f.write(line+'\n')
#         l_f.write(label+'\n')


#------------------------- Covid dataset --------------------------------------
# data_dir = '/Users/luismiguells/Dropbox/Tesis/Tesis maestría/data/covid_dataset/COVIDFakeNewsData.csv'

# data = pd.read_csv(data_dir)

# corpus, data_labels = list(data['headlines']), list(data['outcome'])

# labels = []
# for label in data_labels:
#     if label == 0:
#         labels.append('Fake')
#     else:
#         labels.append('True')


# out_dir = '/Users/luismiguells/Dropbox/Tesis/Tesis maestría/data/covid_dataset/'
# corpus_file = out_dir+'corpus.txt'
# labels_file = out_dir+'labels.txt'

# with open(corpus_file, 'w', encoding='utf-8') as c_f, open(labels_file, 'w', encoding='utf-8') as l_f:
#     for line, label in zip(corpus, labels):
#         line = line.replace('\n', ' ')
#         line = line.replace('\t', ' ')
#         line = line.replace('\r', ' ')
#         c_f.write(line+'\n')
#         l_f.write(label+'\n')


#------------------------- LIAR dataset ---------------------------------------
train_dir = 'C:/Users/luismiguel/Google Drive/MT/data/liar_dataset/data/train.tsv'
test_dir = 'C:/Users/luismiguel/Google Drive/MT/data/liar_dataset/data/test.tsv'
val_dir = 'C:/Users/luismiguel/Google Drive/MT/data/liar_dataset/data/valid.tsv'


train_data = pd.read_csv(train_dir, delimiter='\t')
test_data = pd.read_csv(test_dir, delimiter='\t')
val_data = pd.read_csv(val_dir, delimiter='\t')



train_text, test_text, val_text = list(train_data['statement']), list(test_data['statement']), list(val_data['statement'])
train_labels, test_labels, val_labels = list(train_data['label']), list(test_data['label']), list(val_data['label'])


labels_train = []
labels_test = []
labels_val = []

for line in train_labels:
    if line == 'true':
        labels_train.append('True')
    elif line == 'mostly-true':
        labels_train.append('True')
    elif line == 'barely-true':
        labels_train.append('Fake')
    elif line == 'half-true':
        labels_train.append('Fake')
    elif line == 'false':
        labels_train.append('Fake')
    elif line == 'pants-fire':
        labels_train.append('Fake')

for line in test_labels:
    if line == 'true':
        labels_test.append('True')
    elif line == 'mostly-true':
        labels_test.append('True')
    elif line == 'barely-true':
        labels_test.append('Fake')
    elif line == 'half-true':
        labels_test.append('Fake')
    elif line == 'false':
        labels_test.append('Fake')
    elif line == 'pants-fire':
        labels_test.append('Fake')

for line in val_labels:
    if line == 'true':
        labels_val.append('True')
    elif line == 'mostly-true':
        labels_val.append('True')
    elif line == 'barely-true':
        labels_val.append('Fake')
    elif line == 'half-true':
        labels_val.append('Fake')
    elif line == 'false':
        labels_val.append('Fake')
    elif line == 'pants-fire':
        labels_val.append('Fake')



out_dir = 'C:/Users/luismiguel/Google Drive/MT/data/liar_dataset/'
corpus_train_file = out_dir+'corpus_train.txt'
labels_train_file = out_dir+'labels_train.txt'
corpus_test_file = out_dir+'corpus_test.txt'
labels_test_file = out_dir+'labels_test.txt'
corpus_valid_file = out_dir+'corpus_valid.txt'
labels_valid_file = out_dir+'labels_valid.txt'

with open(corpus_train_file, 'w', encoding='utf-8') as c_f, open(labels_train_file, 'w', encoding='utf-8') as l_f:
    for line, label in zip(train_text, labels_train):
        line = line.replace('\n', ' ')
        line = line.replace('\t', ' ')
        line = line.replace('\r', ' ')
        c_f.write(line+'\n')
        l_f.write(label+'\n')
        
with open(corpus_test_file, 'w', encoding='utf-8') as c_f, open(labels_test_file, 'w', encoding='utf-8') as l_f:
    for line, label in zip(test_text, labels_test):
        line = line.replace('\n', ' ')
        line = line.replace('\t', ' ')
        line = line.replace('\r', ' ')
        c_f.write(line+'\n')
        l_f.write(label+'\n')

with open(corpus_valid_file, 'w', encoding='utf-8') as c_f, open(labels_valid_file, 'w', encoding='utf-8') as l_f:
    for line, label in zip(val_text, labels_val):
        line = line.replace('\n', ' ')
        line = line.replace('\t', ' ')
        line = line.replace('\r', ' ')
        c_f.write(line+'\n')
        l_f.write(label+'\n')

#------------------------- ISOT dataset ---------------------------------------
# true_dir = '/Users/luismiguells/Dropbox/Tesis/Tesis maestría/data/isot_dataset/True.csv'
# fake_dir = '/Users/luismiguells/Dropbox/Tesis/Tesis maestría/data/isot_dataset/Fake.csv'


# true_data = pd.read_csv(true_dir)
# fake_data = pd.read_csv(fake_dir)


# true_text, fake_text = list(true_data['text']), list(fake_data['text'])


# true_corpus = [line for line in true_text if line.isspace() != True]
# fake_corpus = [line for line in fake_text if line.isspace() != True]


# labels_true = ['True' for i in range(len(true_corpus))]
# labels_fake = ['Fake' for i in range(len(fake_corpus))]

# corpus = true_corpus+fake_corpus
# labels = labels_true+labels_fake


# out_dir = '/Users/luismiguells/Dropbox/Tesis/Tesis maestría/data/isot_dataset/'
# corpus_file = out_dir+'corpus.txt'
# labels_file = out_dir+'labels.txt'

# with open(corpus_file, 'w', encoding='utf-8') as c_f, open(labels_file, 'w', encoding='utf-8') as l_f:
#     for line, label in zip(corpus, labels):
#         line = line.replace('\n', ' ')
#         line = line.replace('\t', ' ')
#         line = line.replace('\r', ' ')
#         c_f.write(line+'\n')
#         l_f.write(label+'\n')


#------------------------- FNCS dataset ---------------------------------------
# train_dir = 'C:/Users/luismiguel/Google Drive/MT/data/fncs_dataset/data/train.xlsx'
# test_dir = 'C:/Users/luismiguel/Google Drive/MT/data/fncs_dataset/data/test.xlsx'

# train_data = pd.read_excel(train_dir)
# test_data = pd.read_excel(test_dir)

# labels_train, text_train_data = list(train_data['Category']), list(train_data['Text'])
# labels_test, text_test_data = list(test_data['Category']), list(test_data['Text'])


# out_dir = 'C:/Users/luismiguel/Google Drive/MT/data/fncs_dataset/'
# corpus_train_file = out_dir+'corpus_train.txt'
# labels_train_file = out_dir+'labels_train.txt'
# corpus_test_file = out_dir+'corpus_test.txt'
# labels_test_file = out_dir+'labels_test.txt'

# with open(corpus_train_file, 'w', encoding='utf-8') as c_f, open(labels_train_file, 'w', encoding='utf-8') as l_f:
#     for line, label in zip(text_train_data, labels_train):
#         line = line.replace('\n', ' ')
#         line = line.replace('\t', ' ')
#         line = line.replace('\r', ' ')
#         c_f.write(line+'\n')
#         l_f.write(label+'\n')

# with open(corpus_test_file, 'w', encoding='utf-8') as c_f, open(labels_test_file, 'w', encoding='utf-8') as l_f:
#     for line, label in zip(text_test_data, labels_test):
#         line = line.replace('\n', ' ')
#         line = line.replace('\t', ' ')
#         line = line.replace('\r', ' ')
#         c_f.write(line+'\n')
#         l_f.write(label+'\n')