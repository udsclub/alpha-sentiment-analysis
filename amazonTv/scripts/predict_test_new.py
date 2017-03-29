import os
import numpy as np
import pandas as pd
from keras.models import load_model
from keras.preprocessing.sequence import pad_sequences
import six
from sklearn.metrics import classification_report, accuracy_score, precision_score, recall_score, f1_score


os.environ["CUDA_DEVICE_ORDER"] = "PCI_BUS_ID"
os.environ["CUDA_VISIBLE_DEVICES"] = "1"
BASE_DIR = ''
MODELs_DIR = 'models/'
TOKENIZERs_DIR = '../tokenizers/'
DATA_DIR = '../data/'
TEXT_DATA_DIR = BASE_DIR + DATA_DIR + 'test'
TEXT_DATA_FILE_1 = "rt-polarity_neg.txt"
TEXT_DATA_FILE_2 = "rt-polarity_pos.txt"
TEXT_DATA_FILE_3 = "test_movies.csv"
TEXT_DATA_FILEs_amazon = ["Digital_Music_5.csv", "Office_Products_5.csv", "Video_Games_5.csv"]
HEADER = True
MAX_SEQUENCE_LENGTH = 110

MODEL_NAMEs = ["model_amazon_2017-03-19-19-45_0.25.hdf5",
               "model_amazon_2017-03-19-19-45_0.5.hdf5",
               "model_amazon_2017-03-18-20-10_0.75.hdf5",
               "model_amazon_2017-03-18-20-10_0.9.hdf5",
               "model_amazon_2017-03-17-16-41_1.hdf5",
               "model_amazon_2017-03-18-20-10_1.1.hdf5",
               "model_amazon_2017-03-18-20-10_1.25.hdf5",
               "model_amazon_2017-03-18-00-34_1.5.hdf5",
               "model_amazon_2017-03-18-00-34_2.5.hdf5",
               "model_amazon_2017-03-18-00-34_5.hdf5",
               "model_amazon_2017-03-16-00-16base.hdf5"]
TOKENIZERs = ["amazon_2017-03-19-19-45_tokenizer_0.25",
              "amazon_2017-03-19-19-45_tokenizer_0.5",
              "amazon_2017-03-18-20-10_tokenizer_0.75",
              "amazon_2017-03-18-20-10_tokenizer_0.9",
              "amazon_2017-03-18-00-34_tokenizer_1",
              "amazon_2017-03-18-20-10_tokenizer_1.1",
              "amazon_2017-03-18-20-10_tokenizer_1.25",
              "amazon_2017-03-18-00-34_tokenizer_1.5",
              "amazon_2017-03-18-00-34_tokenizer_2.5",
              "amazon_2017-03-18-00-34_tokenizer_5",
              "amazon_2017-03-16-00-16_tokenizer"]
NAMEs = ['0.25', '0.5', '0.75', '0.9', '1', '1.1', '1.25', '1.5', '2.5', '5', 'all']

negatives = {
    "didn't": "didn_`_t",
    "couldn't": "couldn_`_t",
    "don't": "don_`_t",
    "wouldn't": "wouldn_`_t",
    "doesn't": "doesn_`_t",
    "wasn't": "wasn_`_t",
    "weren't": "weren_`_t",
    "shouldn't":"shouldn_`_t",
    "isn't": "isn_`_t",
    "aren't": "aren_`_t",
}


def preprocess(text):
    text = text.lower()
    text = text.replace('<br />', ' ')
    for k, v in negatives.items():
        text = text.replace(k, v)
    return text


def load_data():
    x = []
    y = []
    for i in [TEXT_DATA_FILE_1, TEXT_DATA_FILE_2]:
        with open(os.path.join(TEXT_DATA_DIR, i), "r", encoding='utf-8', errors='ignore') as f:
            if HEADER:
                _ = next(f)
            if i[-7:-4] == "pos":
                temp_y = 1
            else:
                temp_y = 0
            for line in f:
                x.append(line.rstrip("\n"))
                y.append(temp_y)
    y = np.asarray(y, dtype='int8')

    return x, y

HEADER = True
def load_data_amazon_test():
    x = []
    y = []
    with open(os.path.join(BASE_DIR, DATA_DIR, TEXT_DATA_FILE_amazon), "r", errors='ignoring') as f:
        if HEADER:
            _ = next(f)
        for line in f:
            temp_y, temp_x = line.rstrip("\n").split("|", 1)
            x.append(temp_x)
            y.append(temp_y)

    y = np.asarray(y, dtype='int8')
    return x, y


def load_data_amazon():
    x = pd.read_csv(os.path.join(BASE_DIR, DATA_DIR, TEXT_DATA_FILE_3))
    y = x.loc[~x['reviewText'].isnull(), 'overall'].values
    y = np.asarray(y, dtype='int8')
    x = x.loc[~x['reviewText'].isnull(), 'reviewText'].apply(preprocess).values
    return x, y


def get_score(data, y, tokenizer, model, name):
    tokenizer = six.moves.cPickle.load(open(os.path.join(TOKENIZERs_DIR, tokenizer), "rb"))
    model = load_model(os.path.join(MODELs_DIR, model))
    sequences = tokenizer.texts_to_sequences(data)
    x = pad_sequences(sequences, maxlen=MAX_SEQUENCE_LENGTH)
    y_pred = model.predict_classes(x, verbose=0)
    print(name)
    print("%s: %.2f%%" % ("accuracy", accuracy_score(y, y_pred)*100))
    print("%s: %.2f%%" % ("precision", precision_score(y, y_pred)*100))
    print("%s: %.2f%%" % ("recall", recall_score(y, y_pred)*100))
    print("%s: %.2f%%" % ("f1score", f1_score(y, y_pred)*100))
    print(classification_report(y, y_pred))
    print("=======================================================")
    print("=======================================================")
    print("=======================================================")

#print("Test on RT")
#data_test, labels = load_data()
#labels = np.asarray(labels, dtype='int8')
#for t, m, n in zip(TOKENIZERs, MODEL_NAMEs, NAMEs):
#   get_score(data_test, labels, t, m, n)

#print("Test on amazon")
#data_amazon, labels = load_data_amazon()
#for t, m, n in zip(TOKENIZERs, MODEL_NAMEs, NAMEs):
#    get_score(data_amazon, labels, t, m, n)

for TEXT_DATA_FILE_amazon in TEXT_DATA_FILEs_amazon:
    print(TEXT_DATA_FILE_amazon)
    data_amazon, labels = load_data_amazon_test()
    for t, m, n in zip(TOKENIZERs, MODEL_NAMEs, NAMEs):
        get_score(data_amazon, labels, t, m, n)


