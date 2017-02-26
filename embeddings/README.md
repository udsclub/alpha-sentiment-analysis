# Experiments with word embeddings

1. Pretrained word2vec embeddings( https://drive.google.com/file/d/0B7XkCwpI5KDYNlNUTTlSS21pQmM/ ) – google_word2vec.ipynb 

Optimizer: adam

Batch: 128

Model params: LSTM(128), dropout_U=0.2, dropout_W=0.2, dropout(0.4)

Train accuracy: 85.15%

Train loss: 0.3286

Validation accuracy: 82.68%

Validation loss: 0.3801

2. Pretrained Glove embeddings(http://nlp.stanford.edu/data/glove.6B.zip) – glove.py

Optimizer: adam

Batch: 128

Model params: LSTM(64), dropout_U=0.2, dropout_W=0.2, dropout(0.4)

- glove.6B.50d.txt      82s - loss: 0.4734 - acc: 0.7657 - val_loss: 0.4561 - val_acc: 0.7750
- glove.6B.100d.txt     93s - loss: 0.4206 - acc: 0.7991 - val_loss: 0.4194 - val_acc: 0.8022
- glove.6B.200d.txt     126s - loss: 0.3745 - acc: 0.8260 - val_loss: 0.4032 - val_acc: 0.8124
- glove.6B.300d.txt     150s - loss: 0.3417 - acc: 0.8453 - val_loss: 0.4020 - val_acc: 0.8159



