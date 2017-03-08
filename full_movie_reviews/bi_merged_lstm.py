import os
import numpy as np
from gensim import models
import six.moves.cPickle
from datetime import datetime
from keras.preprocessing.text import Tokenizer
from keras.preprocessing.sequence import pad_sequences
from keras.utils.np_utils import to_categorical
from keras.models import Sequential
from keras.layers import Dense, Dropout, Activation, Merge
from keras.layers import Embedding
from keras.layers import Bidirectional, LSTM
from keras.callbacks import ModelCheckpoint, TensorBoard, EarlyStopping
from keras.optimizers import Adam
from sklearn.model_selection import train_test_split


BASE_DIR = '../'
EMBEDDING_DIR = BASE_DIR + 'embeddings/'  # http://nlp.stanford.edu/projects/glove/ pretrained vectors
EMBEDDING_FILE = "GoogleNews-vectors-negative300.bin"
TEXT_DATA_DIR = BASE_DIR + 'data/'
TEXT_DATA_FILE = "movie_reviews.csv"
LR = 0.001
HEADER = True
raw = True
suffix = str(datetime.now().strftime("%Y-%m-%d-%H-%M"))
NAME = "bidirectional_lstm" + suffix
BOTH_SIDES = False
print(suffix)


MAX_NB_WORDS = 20000
EMBEDDING_DIM = 300
MAX_SEQUENCE_LENGTH = 50
VALIDATION_SPLIT = 0.06552
RANDOM_SEED = 42


def load_data():
    x = []
    y = []
    with open(os.path.join(TEXT_DATA_DIR, TEXT_DATA_FILE), "r") as f:
        if HEADER:
            _ = next(f)
        for line in f:
            temp_y, temp_x = line.rstrip("\n").split(",", 1)
            x.append(temp_x)
            y.append(temp_y)

    return x, y


def transform(tokenizer_object, train, test):
    sequences_train = tokenizer_object.texts_to_sequences(train)  # transform words to its indexes
    sequences_test = tokenizer_object.texts_to_sequences(test)
    word_indexes = tokenizer_object.word_index  # dictionary of word:index

    # transform a list to numpy array with shape (nb_samples, MAX_SEQUENCE_LENGTH)
    # be careful because it takes only last MAX_SEQUENCE_LENGTH words
    train_r = pad_sequences(sequences_train, maxlen=MAX_SEQUENCE_LENGTH)
    test_r = pad_sequences(sequences_test, maxlen=MAX_SEQUENCE_LENGTH)

    train_l = pad_sequences(sequences_train, maxlen=MAX_SEQUENCE_LENGTH, truncating='post')
    test_l = pad_sequences(sequences_test, maxlen=MAX_SEQUENCE_LENGTH, truncating='post')
    return train_r, test_r, word_indexes, train_l, test_l


def prepare_embeddings(word_indexes):
    def load_w2v():
        _fname = EMBEDDING_DIR + EMBEDDING_FILE
        w2v_model = models.KeyedVectors.load_word2vec_format(_fname, binary=True)
        return w2v_model

    embeddings = load_w2v()
    # prepare embedding matrix
    nb_words = min(MAX_NB_WORDS, len(word_indexes))
    prepared_embedding_matrix = np.zeros((nb_words, EMBEDDING_DIM))
    for word, n in word_indexes.items():
        if n >= MAX_NB_WORDS:
            continue
        try:
            embedding_vector = embeddings.word_vec(word)
            prepared_embedding_matrix[n] = embedding_vector
        except:
            continue

    return prepared_embedding_matrix


def train_model(optimizer, embeddings, add_to_name):
    callback_1 = TensorBoard(log_dir='./logs/logs_{}'.format(NAME + str(add_to_name)), histogram_freq=0,
                             write_graph=False, write_images=False)
    callback_2 = EarlyStopping(monitor='val_acc', min_delta=0, patience=7, verbose=0, mode='auto')
    callback_3 = ModelCheckpoint("models/model_{}.hdf5".format(NAME + str(add_to_name)), monitor='val_acc',
                                 save_best_only=True, verbose=0)

    embedding_layer_1 = Embedding(embeddings.shape[0],
                                  embeddings.shape[1],
                                  weights=[embeddings],
                                  input_length=MAX_SEQUENCE_LENGTH,
                                  trainable=False)
    embedding_layer_2 = Embedding(embeddings.shape[0],
                                  embeddings.shape[1],
                                  weights=[embeddings],
                                  input_length=MAX_SEQUENCE_LENGTH,
                                  trainable=False)
    model_r = Sequential()
    model_r.add(embedding_layer_1)
    model_r.add(Dropout(0.2))
    model_r.add(Bidirectional(LSTM(128, dropout_U=0.1, dropout_W=0.1)))
    model_r.add(Dropout(0.2))

    model_l = Sequential()
    model_l.add(embedding_layer_2)
    model_l.add(Dropout(0.2))
    model_l.add(Bidirectional(LSTM(128, dropout_U=0.1, dropout_W=0.1)))
    model_l.add(Dropout(0.2))

    model = Sequential()
    model.add(Merge([model_l, model_r], mode='concat'))
    model.add(Dropout(0.2))
    model.add(Dense(64, activation='relu'))

    model.add(Dense(2))
    model.add(Activation('sigmoid'))
    # model = model_initialization(embedding_layer)
    model.compile(loss='binary_crossentropy',
                  optimizer=optimizer,
                  metrics=['accuracy'])
    model.fit([X_train_l, X_train_r], y_train, validation_data=([X_test_l, X_test_r], y_test),
              batch_size=2048, nb_epoch=1000, callbacks=[callback_1, callback_2, callback_3])

data, labels = load_data()
labels = np.asarray(labels, dtype='int8')

data_train, data_test, labels_train, labels_test = \
    train_test_split(data, np.asarray(labels, dtype='int8'),
                     test_size=VALIDATION_SPLIT, random_state=RANDOM_SEED, stratify=labels)

tokenizer = Tokenizer(nb_words=MAX_NB_WORDS)  # create dictionary of MAX_NB_WORDS, other words will not be used
tokenizer.fit_on_texts(data_train)
six.moves.cPickle.dump(tokenizer, open("../tokenizers/{}_tokenizer".format(NAME), "wb"))

for MAX_SEQUENCE_LENGTH in [100, 50, 25]:
    X_train_r, X_test_r, word_index, X_train_l, X_test_l = transform(tokenizer, data_train, data_test)
    y_train, y_test = to_categorical(np.asarray(labels_train)), to_categorical(np.asarray(labels_test))

    embedding_matrix = prepare_embeddings(word_index)
    train_model(Adam(lr=LR), embedding_matrix, 'merged_{}'.format(MAX_SEQUENCE_LENGTH))
