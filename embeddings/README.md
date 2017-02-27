# Experiments with word embeddings

### 1. Pretrained word2vec embeddings( https://drive.google.com/file/d/0B7XkCwpI5KDYNlNUTTlSS21pQmM/ ) – google_word2vec.ipynb 

Optimizer: adam

Batch: 128

Model params: LSTM(128), dropout_U=0.2, dropout_W=0.2, dropout(0.4)

Train accuracy: 85.15%

Train loss: 0.3286

Validation accuracy: 82.68%

Validation loss: 0.3801

### 2. Pretrained Glove embeddings(http://nlp.stanford.edu/data/glove.6B.zip) – glove.py

Optimizer: adam

Batch: 128

Model params: LSTM(64), dropout_U=0.2, dropout_W=0.2, dropout(0.4)

- glove.6B.50d.txt      82s - loss: 0.4734 - acc: 0.7657 - val_loss: 0.4561 - val_acc: 0.7750
- glove.6B.100d.txt     93s - loss: 0.4206 - acc: 0.7991 - val_loss: 0.4194 - val_acc: 0.8022
- glove.6B.200d.txt     126s - loss: 0.3745 - acc: 0.8260 - val_loss: 0.4032 - val_acc: 0.8124
- glove.6B.300d.txt     150s - loss: 0.3417 - acc: 0.8453 - val_loss: 0.4020 - val_acc: 0.8159

### 3. Pretrained Glove embeddings(http://nlp.stanford.edu/data/glove.6B.zip) – glove.py

Optimizer: adam

Model params: LSTM(128), dropout_U=0.2, dropout_W=0.2, dropout(0.4)

- glove.6B.100d.txt, batch_size=256     155s - loss: 0.3857 - acc: 0.8189 - val_loss: 0.4136 - val_acc: 0.8062     
- glove.6B.300d.txt, batch_size=256     217s - loss: 0.3204 - acc: 0.8565 - val_loss: 0.4025 - val_acc: 0.8186

- glove.6B.100d.txt, batch_size=128     171s - loss: 0.3734 - acc: 0.8272 - val_loss: 0.4086 - val_acc: 0.8095
- glove.6B.300d.txt, batch_size=128     254s - loss: 0.3187 - acc: 0.8587 - val_loss: 0.4141 - val_acc: 0.8137

- glove.6B.100d.txt, batch_size=64      283s - loss: 0.3658 - acc: 0.8329 - val_loss: 0.4237 - val_acc: 0.8073
- glove.6B.300d.txt, batch_size=64      364s - loss: 0.2966 - acc: 0.8694 - val_loss: 0.4223 - val_acc: 0.8183

