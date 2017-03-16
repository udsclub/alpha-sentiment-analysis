import os
import numpy as np
import pandas as pd
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

os.environ["CUDA_DEVICE_ORDER"] = "PCI_BUS_ID"  # see issue #152
os.environ["CUDA_VISIBLE_DEVICES"] = "1"
BASE_DIR = '../'
EMBEDDING_DIR = BASE_DIR + 'embeddings/'  # http://nlp.stanford.edu/projects/glove/ pretrained vectors
EMBEDDING_FILE = "GoogleNews-vectors-negative300.bin"
TEXT_DATA_DIR = BASE_DIR + 'data/'
TEXT_DATA_FILE = "train_movies.csv"
LR = 0.001
HEADER = True
raw = True
suffix = str(datetime.now().strftime("%Y-%m-%d-%H-%M"))
NAME = "amazon_" + suffix
BOTH_SIDES = False
print(suffix)

MAX_NB_WORDS = 50000
EMBEDDING_DIM = 300
MAX_SEQUENCE_LENGTH = 110
VALIDATION_SPLIT = 0.1
RANDOM_SEED = 42

negatives = {
    "didn't": "didn_`_t",
    "couldn't": "couldn_`_t",
    "don't": "don_`_t",
    "wouldn't": "wouldn_`_t",
    "doesn't": "doesn_`_t",
    "wasn't": "wasn_`_t",
    "weren't": "weren_`_t",
    "shouldn't": "shouldn_`_t",
    "isn't": "isn_`_t",
    "aren't": "aren_`_t",
}


def preprocess(text):
    text = text.lower()
    text = text.replace('<br />', ' ')
    for k, v in negatives.items():
        text = text.replace(k, v)
    return text


def load_data(n):
    df_train = pd.read_csv(os.path.join(TEXT_DATA_DIR, TEXT_DATA_FILE))
    df_train, _ = train_test_split(df_train, train_size=n, random_state=42)
    train, test = train_test_split(df_train.asin.unique(), test_size=VALIDATION_SPLIT, random_state=42)
    train_reviews, labels_train_reviews = df_train.loc[(df_train.asin.isin(train)) & (~pd.isnull(df_train.reviewText)),
                                                       "reviewText"].values, \
                                          df_train.loc[(df_train.asin.isin(train)) & (~pd.isnull(df_train.reviewText)),
                                                       "overall"].values
    test_reviews, labels_test_reviews = df_train.loc[(df_train.asin.isin(test)) & (~pd.isnull(df_train.reviewText)),
                                                     "reviewText"].values, df_train.loc[(df_train.asin.isin(test)) &
                                                                                        (~pd.isnull(
                                                                                            df_train.reviewText)),
                                                                                        "overall"].values
    for x in range(len(train_reviews)):
        train_reviews[x] = preprocess(train_reviews[x])
    for x in range(len(test_reviews)):
        test_reviews[x] = preprocess(test_reviews[x])

    return train_reviews, labels_train_reviews, test_reviews, labels_test_reviews


def transform(tokenizer_object, train, test, l_train, l_test, both_sides=True):
    sequences_train = tokenizer_object.texts_to_sequences(train)  # transform words to its indexes
    sequences_test = tokenizer_object.texts_to_sequences(test)

    word_indexes = tokenizer_object.word_index  # dictionary of word:index

    # transform a list to numpy array with shape (nb_samples, MAX_SEQUENCE_LENGTH)
    # be careful because it takes only last MAX_SEQUENCE_LENGTH words
    train = pad_sequences(sequences_train, maxlen=MAX_SEQUENCE_LENGTH)
    test = pad_sequences(sequences_test, maxlen=MAX_SEQUENCE_LENGTH)

    return train, test, word_indexes, l_train, l_test


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


def model_initialization(embeddings):
    model = Sequential()
    model.add(embeddings)
    model.add(Dropout(0.3))
    model.add(LSTM(128, dropout_U=0.1, dropout_W=0.1))
    model.add(Dropout(0.3))
    model.add(Dense(2))
    model.add(Activation('sigmoid'))

    return model


def train_model(optimizer, embeddings, X_train, y_train, X_test, y_test, add_to_name):
    callback_1 = TensorBoard(log_dir='./logs/logs_{}'.format(NAME + "_" + str(add_to_name)), histogram_freq=0,
                             write_graph=False, write_images=False)
    callback_2 = EarlyStopping(monitor='val_acc', min_delta=0, patience=10, verbose=0, mode='auto')
    callback_3 = ModelCheckpoint("models/model_{}.hdf5".format(NAME + "_" + str(add_to_name)), monitor='val_acc',
                                 save_best_only=True, verbose=0)

    embedding_layer = Embedding(embeddings.shape[0],
                                embeddings.shape[1],
                                weights=[embeddings],
                                input_length=MAX_SEQUENCE_LENGTH,
                                trainable=False,
                                mask_zero=True)

    model = model_initialization(embedding_layer)
    model.compile(loss='binary_crossentropy',
                  optimizer=optimizer,
                  metrics=['accuracy', 'fmeasure', 'precision', 'recall'])

    model.fit(X_train, y_train, validation_data=[X_test, y_test], batch_size=1024, nb_epoch=1000,
              callbacks=[callback_1, callback_2, callback_3], verbose=2)


def run(n):
    data_train, labels_train, data_test, labels_test = load_data(n)
    print(len(data_train), len(data_test))
    # tokenizer = Tokenizer(nb_words=MAX_NB_WORDS)  # create dictionary of MAX_NB_WORDS, other words will not be used
    tokenizer = Tokenizer(nb_words=MAX_NB_WORDS, filters='"#$%&()*+-/:;<=>@[\\]^{|}~\t\n,.')
    tokenizer.fit_on_texts(data_train)
    six.moves.cPickle.dump(tokenizer, open("../tokenizers/{}_tokenizer_{}".format(NAME, n), "wb"))

    X_train, X_test, word_index, labels_train, labels_test = transform(tokenizer, data_train, data_test,
                                                                       labels_train, labels_test, BOTH_SIDES)
    y_train, y_test = to_categorical(np.asarray(labels_train)), to_categorical(np.asarray(labels_test))
    embedding_matrix = prepare_embeddings(word_index)
    train_model(Adam(lr=LR), embedding_matrix, X_train, y_train, X_test, y_test, n)

for i in [100000, 200000, 500000, 1000000]:
    run(i)

