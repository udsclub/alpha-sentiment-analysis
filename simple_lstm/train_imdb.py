import os
import numpy as np
from gensim import models
import six.moves.cPickle
from datetime import datetime
from keras.preprocessing.text import Tokenizer
from keras.preprocessing.sequence import pad_sequences
from keras.utils.np_utils import to_categorical
from keras.models import Sequential
from keras.layers import Dense, Dropout, Activation
from keras.layers import Embedding
from keras.layers import LSTM
from keras.callbacks import ModelCheckpoint, TensorBoard, EarlyStopping
from keras.optimizers import Adam
from sklearn.model_selection import train_test_split


BASE_DIR = '../'
EMBEDDING_DIR = BASE_DIR + 'embeddings/'  # http://nlp.stanford.edu/projects/glove/ pretrained vectors
EMBEDDING_FILE = "GoogleNews-vectors-negative300.bin"
TEXT_DATA_DIR = BASE_DIR + '../data/'
TEXT_DATA_FILE = "imdb_small.csv"
NAME = "imdb"
HEADER = True
raw = True
suffix = str(datetime.now().strftime("%Y-%m-%d-%H-%M"))
print(suffix)


def load_data():
    reviews = []
    ranks = []
    with open(os.path.join(TEXT_DATA_DIR, TEXT_DATA_FILE), "r") as f:
        if HEADER:
            _ = next(f)
        for line in f:
            temp_y, temp_x = line.rstrip("\n").split("|", 1)
            reviews.append(temp_x)
            ranks.append(temp_y)

    return reviews, ranks


def transform(tokenizer_object, train, test):
    sequences_train = tokenizer_object.texts_to_sequences(train)  # transform words to its indexes
    sequences_test = tokenizer_object.texts_to_sequences(test)

    word_indexes = tokenizer_object.word_index  # dictionary of word:index

    # transform a list to numpy array with shape (nb_samples, MAX_SEQUENCE_LENGTH)
    # be careful because it takes only last MAX_SEQUENCE_LENGTH words
    train = pad_sequences(sequences_train, maxlen=MAX_SEQUENCE_LENGTH)
    test = pad_sequences(sequences_test, maxlen=MAX_SEQUENCE_LENGTH)

    return train, test, word_indexes


def prepare_embeddings(raw=True):

    def load_w2v():
        _fname = EMBEDDING_DIR + EMBEDDING_FILE
        w2v_model = models.KeyedVectors.load_word2vec_format(_fname, binary=True)
        return w2v_model

    if raw:
        embeddings = load_w2v()
        # prepare embedding matrix
        nb_words = min(MAX_NB_WORDS, len(word_index))
        prepared_embedding_matrix = np.zeros((nb_words, EMBEDDING_DIM))
        for word, i in word_index.items():
            if i >= MAX_NB_WORDS:
                continue
            try:
                embedding_vector = embeddings.word_vec(word)
            except:
                embedding_vector = None
            if embedding_vector is not None:
                # words not found in embedding index will be all-zeros.
                prepared_embedding_matrix[i] = embedding_vector
        del embeddings
    else:
        prepared_embedding_matrix = np.load(EMBEDDING_DIR + EMBEDDING_FILE)

    return prepared_embedding_matrix


def model_initialization(embeddings):

    model = Sequential()
    model.add(embeddings)
    model.add(LSTM(128, dropout_U=0.2, dropout_W=0.2))
    model.add(Dropout(0.4))
    model.add(Dense(2))
    model.add(Activation('sigmoid'))

    return model


def train_model(optimizer, embeddings):

    callback_1 = TensorBoard(log_dir='./logs/logs_{}'.format(NAME), histogram_freq=0,
                             write_graph=False, write_images=False)
    callback_2 = EarlyStopping(monitor='val_acc', min_delta=0, patience=4, verbose=0, mode='auto')
    callback_3 = ModelCheckpoint("models/model_{}.hdf5".format(NAME), monitor='val_acc',
                                 save_best_only=True, verbose=0)

    embedding_layer = Embedding(embeddings.shape[0],
                                embeddings.shape[1],
                                weights=[embeddings],
                                input_length=MAX_SEQUENCE_LENGTH,
                                trainable=False)

    model = model_initialization(embedding_layer)
    model.compile(loss='binary_crossentropy',
                  optimizer=optimizer,
                  metrics=['accuracy'])

    model.fit(X_train, y_train, validation_data=[X_test, y_test], batch_size=256, nb_epoch=1000,
              callbacks=[callback_1, callback_2, callback_3])


data, labels = load_data()
labels = np.asarray(labels, dtype='int8')

MAX_NB_WORDS = 20000
EMBEDDING_DIM = 300
MAX_SEQUENCE_LENGTH = 50
VALIDATION_SPLIT = 0.2
RANDOM_SEED = 42

data_train, data_test, labels_train, labels_test = \
    train_test_split(data, np.asarray(labels, dtype='int8'),
                     test_size=VALIDATION_SPLIT, random_state=RANDOM_SEED, stratify=labels)

tokenizer = Tokenizer(nb_words=MAX_NB_WORDS)  # create dictionary of MAX_NB_WORDS, other words will not be used
tokenizer.fit_on_texts(data_train)
six.moves.cPickle.dump(tokenizer, open("../tokenizers/{}_tokenizer".format(NAME), "wb"))

X_train, X_test, word_index = transform(tokenizer, data_train, data_test)
y_train, y_test = to_categorical(np.asarray(labels_train)), to_categorical(np.asarray(labels_test))

embedding_matrix = prepare_embeddings(raw)

train_model(Adam(lr=0.001), embedding_matrix)











