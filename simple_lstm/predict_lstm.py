import os
import numpy as np
from keras.models import load_model
from keras.preprocessing.sequence import pad_sequences
import six.moves.cPickle
from keras.utils.np_utils import to_categorical

MODEL_DIR = "../scripts/models/"
MODEL_NAME = "model_imdb.hdf5"
TEXT_DATA_DIR = "../../data/"
TEXT_DATA_FILE = "reviews_rt_all.csv"  # "imdb_small.csv"
HEADER = True

data = []
labels = []
with open(os.path.join(TEXT_DATA_DIR, TEXT_DATA_FILE), "r") as f:
    if HEADER:
        header = next(f)
    for line in f:
        temp_y, temp_x = line.rstrip("\n").split("|", 1)
        data.append(temp_x)
        labels.append(temp_y)
labels = np.asarray(labels, dtype='int8')

MAX_NB_WORDS = 20000
EMBEDDING_DIM = 300
MAX_SEQUENCE_LENGTH = 50

tokenizer = six.moves.cPickle.load(open("../tokenizers/imdb_tokenizer", "rb"))
sequences = tokenizer.texts_to_sequences(data)  # transform words to its indexes

# transform a list to numpy array with shape (nb_samples, MAX_SEQUENCE_LENGTH)
# be careful because it takes only last MAX_SEQUENCE_LENGTH words
X = pad_sequences(sequences, maxlen=MAX_SEQUENCE_LENGTH)
y = to_categorical(np.asarray(labels))

model = load_model(MODEL_DIR + MODEL_NAME)
score = model.evaluate(X, y, verbose=1)
print("%s: %.2f%%" % (model.metrics_names[1], score[1]*100))
