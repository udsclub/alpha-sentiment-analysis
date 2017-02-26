#!/usr/bin/python
# -*- coding: utf-8 -*-

"""
Using pre-trained GloVe word embeddings with Keras
"""
import os
import json
import numpy as np
from keras.models import model_from_json
from keras.preprocessing.text import Tokenizer
from keras.preprocessing.sequence import pad_sequences
from keras.utils.np_utils import to_categorical
from keras.models import Sequential
from keras.layers import Dense, Dropout, Activation, Input
from keras.layers import Embedding
from keras.layers import LSTM
from keras.callbacks import ModelCheckpoint, TensorBoard, EarlyStopping
import matplotlib.pyplot as plt

BASE_DIR = ''
GLOVE_DIR = BASE_DIR + 'glove.6B/' # http://nlp.stanford.edu/projects/glove/ pretrained vectors
TEXT_DATA_DIR = 'data/'
HEADER = True

def loading_data(name, TEXT_DATA_DIR='data/', HEADER=True):
    '''
    Функция возвращает массив ревью и меток
    name - название файла
    '''
    X = []
    y = []
    with open(os.path.join(TEXT_DATA_DIR, str(name)), "r") as f:
        if HEADER:
            header = next(f)
        for line in f:
            try:
                temp_y, temp_x = line.rstrip("\n").split("|")
            except ValueError:
                print(line)
            X.append(temp_x)
            y.append(temp_y)
    return (X, y)

MAX_NB_WORDS = 20000
EMBEDDING_DIM = 100
MAX_SEQUENCE_LENGTH = 50
VALIDATION_SPLIT = 0.2

def index_word_vector(name='glove.6B.100d.txt'):
    """
    Function return Indexing word vectors
    name - название файла с предобученными словами
    """
    print(name)
    embeddings_index = {}
    f = open(os.path.join(GLOVE_DIR, name), encoding='utf-8')
    for line in f:
        values = line.split()
        word = values[0]
        coefs = np.asarray(values[1:], dtype='float32')
        embeddings_index[word] = coefs
    f.close()
    print(len(embeddings_index))
    return embeddings_index

def preprocessing_text(name_data, name_glove, EMBEDDING_DIM, MAX_SEQUENCE_LENGTH=50):
    """
    Функция делает предобработку текста, включая
    Tokenize text
    Glove word-embeddings repressentation
    Preparing embedding matrix
    Split the data into a training set and a validation set
    """
    tokenizer = Tokenizer(nb_words=MAX_NB_WORDS) # create dictionary of MAX_NB_WORDS, other words will not be used
    X, y = loading_data(name_data)
    tokenizer.fit_on_texts(X)
    sequences = tokenizer.texts_to_sequences(X) # transform words to its indexes
    word_index = tokenizer.word_index # dictionary of word:index
    print('Found %s unique tokens.' % len(word_index))
    data = pad_sequences(sequences, maxlen=MAX_SEQUENCE_LENGTH) # transform a list to numpy array with shape (nb_samples, MAX_SEQUENCE_LENGTH)
                                                                # be careful because it takes only last MAX_SEQUENCE_LENGTH words
    labels = to_categorical(np.asarray(y))
    print('Shape of data tensor:', data.shape)
    print('Shape of label tensor:', labels.shape)

    nb_words = min(MAX_NB_WORDS, len(word_index))
    embedding_matrix = np.zeros((nb_words, EMBEDDING_DIM))
    embeddings_index = index_word_vector(name_glove)
    for word, i in word_index.items():
        if i >= MAX_NB_WORDS:
            continue
        embedding_vector = embeddings_index.get(word)
        if embedding_vector is not None:
            # words not found in embedding index will be all-zeros.
            embedding_matrix[i] = embedding_vector
    
    indices = np.arange(data.shape[0])
    np.random.shuffle(indices)
    data = data[indices]
    labels = labels[indices]
    nb_validation_samples = int(VALIDATION_SPLIT * data.shape[0])

    x_train = data[:-nb_validation_samples]
    y_train = labels[:-nb_validation_samples]
    x_val = data[-nb_validation_samples:]
    y_val = labels[-nb_validation_samples:]

    return (x_train, y_train, x_val, y_val, nb_words, embedding_matrix)

def model(name_data, name_glove, EMBEDDING_DIM):

    x_train, y_train, x_val, y_val, nb_words, embedding_matrix = preprocessing_text(name_data, name_glove, EMBEDDING_DIM)

    embedding_layer = Embedding(nb_words,
                                EMBEDDING_DIM,
                                weights=[embedding_matrix],
                                input_length=MAX_SEQUENCE_LENGTH,
                                trainable=False)

    callback_2 = TensorBoard(log_dir='./logs', histogram_freq=0, write_graph=True, write_images=False)
    callback_3 = EarlyStopping(monitor='val_acc', min_delta=0, patience=3, verbose=0, mode='auto')

    model = Sequential()
    model.add(embedding_layer)
    model.add(LSTM(64, dropout_U=0.2, dropout_W=0.2))
    model.add(Dropout(0.4))
    model.add(Dense(2))
    model.add(Activation('sigmoid'))

    model.compile(loss='binary_crossentropy',
                optimizer='adam',
                metrics=['accuracy'])

    model.fit(x_train, y_train, validation_data = [x_val, y_val], batch_size=128, nb_epoch=15, callbacks=[callback_2, callback_3])
    print(model.summary())

    # evaluate loaded model on test data
    score = model.evaluate(x_val, y_val, verbose=0)
    print("%s    %s: %.2f%%  \n" % (name_glove, model.metrics_names[1], score[1]*100))

    with open('result.txt', 'a') as file:
        file.write("%s    %s: %.2f%%  \n" % (name_glove, model.metrics_names[1], score[1]*100))


model("reviews_rt_all_ascii.csv", 'glove.6B.50d.txt', 50)
model("reviews_rt_all_ascii.csv", 'glove.6B.100d.txt', 100)
model("reviews_rt_all_ascii.csv", 'glove.6B.200d.txt', 200)
model("reviews_rt_all_ascii.csv", 'glove.6B.300d.txt', 300)