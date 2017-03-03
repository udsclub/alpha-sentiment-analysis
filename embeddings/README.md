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
=======
n_epochs: 13

Optimizer: sgd with momentum (lr=1, momentum=0.6)

Batch: 128

Model params: LSTM(128), dropout_U=0.2, dropout_W=0.2, dropout(0.4)

Train accuracy: 85.54%

Train loss: 0.3230

Validation accuracy: 82.43%

Validation loss: 0.3801

n_epochs: 29


2. Pretrained Glove embeddings(http://nlp.stanford.edu/data/glove.6B.zip) – glove.py

Optimizer: adam
Model params: LSTM(64), dropout_U=0.2, dropout_W=0.2, dropout(0.4)

Name of Glove pretrained file| Batch | Train accuracy | Validation accuracy  |
| ------------- |:-------------:| -----:| -----:|
| glove.6B.50d.txt| 128    |  76.57% |  77.50% |       
| glove.6B.100d.txt| 128    |  79.91% |  80.22% |
| glove.6B.200d.txt| 128    |  82.60% |  81.24% |
| glove.6B.300d.txt| 128    |  **84.53%** |  **81.59%** |

### 3. Pretrained Glove embeddings(http://nlp.stanford.edu/data/glove.6B.zip) – glove.py

Optimizer: adam

Model params: LSTM(128), dropout_U=0.2, dropout_W=0.2, dropout(0.4), MAX_SEQUENCE_LENGTH=50

Name of Glove pretrained file| Batch | Train accuracy | Validation accuracy  |
| ------------- |:-------------:| -----:| -----:|
| glove.6B.100d.txt| 256    |  81.89% |  80.62% |       
| glove.6B.300d.txt| 256    |  85.65% |  **81.86%** |
| glove.6B.100d.txt| 128    |  82.72% |  80.95% |
| glove.6B.300d.txt| 128    |  85.87% |  81.37% |
| glove.6B.100d.txt| 64     |  83.29% |  80.73% |
| glove.6B.300d.txt| 64     |  **86.94%** |  81.83% |

### 4. Trained Glove embeddings on rotten tomatoes database (https://yadi.sk/d/UlT88tKF3Em92X) 

Trained with embedding_size=300, context_size=10, min_occurrences=1, learning_rate=0.05, batch_size=512
and num_epochs=100, log_dir="log/example", summary_batch_interval=1000

Optimizer: adam

Model params: LSTM(128), dropout_U=0.2, dropout_W=0.2, dropout(0.4), batch_size=64

Name of Glove pretrained file| Batch | Train accuracy | Validation accuracy  |
| ------------- |:-------------:| -----:| -----:|
| my_glove.txt| 64    |  84.14% |  79.14% |

### 5. Pretrained word2vec embeddings( https://drive.google.com/file/d/0B7XkCwpI5KDYNlNUTTlSS21pQmM/ ) 

Optimizer: adam

Model params: LSTM(128), dropout_U=0.2, dropout_W=0.2

MAX_SEQUENCE_LENGTH| Batch |     Dropout      | Train accuracy           | Validation accuracy  | Test accuracy  |Epochs |
| ------------- |:-------------:| -----:| -----:| -----:| -----:| -----:|
| 50     | 128 | 0.4 | 90.44%|  85.01% | 76.17% | **10** |
| 50     | 2000 | 0.4 | 85.76%|  83.40% | 74.18% | 35 |
| 50     | 128 | 0.2 | 86.56%|  84.32% | **76.37%** | 27 |
| 100     | 128 | 0.4 | **91.16%**|  **88.81%** | 76.29% | 27 |
| 100     | 2000 | 0.4 | 89.85%|  88.09% | 76.32% | 68 |


### 6. Optimizers comparisons on rotten tomatoes database (https://yadi.sk/d/UlT88tKF3Em92X) 

| Optimizer        | Train accuracy           | Validation accuracy  | Epochs |
| ------------- |:-------------:| -----:| -----:| -----:|
| SGD nesterov momentum     | 85.79% | 82.33% | 46| 
| RMSprop     | 87.55% | 82.90% | 23|
| Adam | **89.03%** | 82.80% | 26|
| Adamax | 87.40% | **82.99%** | 30|
| Nadam | 88.12% | 82.65% | **14**|
| Adadelta | 83.70% | 82.50% | 58|
| Adadelta | 83.59% | 81.89% | 68|

Validation accuracy plot 

![alt text](https://github.com/udsclub/alpha-sentiment-analysis/blob/master/plots/val_acc_optimizers_comp.png)

Validation loss plot 

![alt text](https://github.com/udsclub/alpha-sentiment-analysis/blob/master/plots/val_loss_optimizers_comp.png)


