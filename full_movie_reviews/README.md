# Experiments with full rt + imdb datasets

### 1. Results

| Approach        | Train Accuracy           | Validation Accuracy  | Epochs |MAX_NB_WORDS | Droupout before LSTM cell |
| --------------- |:------------------------:| :-------------------:| :-----:|:-----------:|:-----------:|
| LSTM            | 89.32%                   | 84.89%               | 33     |50000        |0            |
| LSTM            | 89.27%                   | 84.79%               | 36     |20000        |0            |
| LSTM            | 86.89%                   | 84.91%               | 35     |20000        |0.2          |


LSTM – whole dataset; split VALIDATION_SPLIT = 0.06552, RANDOM_SEED = 42; MAX_SEQUENCE_LENGTH = 50; LSTM(128, dropout_U=0.2, dropout_W=0.2), Dropout(0.3).

