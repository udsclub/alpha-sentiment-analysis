# Experiments with full rt + imdb datasets

### 1. Results

| Approach| Train Accuracy| Validation Accuracy|Epochs|MAX_NB_WORDS|Droupout before LSTM cell|Droupout after LSTM|MAX_SEQUENCE_LENGTH|
|--------|:------:| :-----:|:----:|:-----:|:---:|:---:|:----:|
| LSTM   | 89.32% | 84.89% | 33   |50000  |0    |0.3  | 50   |
| LSTM   | 89.27% | 84.79% | 36   |20000  |0    |0.3  | 50   |
| LSTM   | 86.89% | 84.91% | 35   |20000  |0.2  |0.3  | 50   |
| LSTM   | 89.15% | 84.59% | 34   |20000  |0    |0.4  | 50   |
| LSTM   | 88.17% | 86.20% | 32   |20000  |0.2  |0.2  | 100  |
| biLSTM | 86.63% | 84.64% | 29   |20000  |0.2  |0.2  | 50  |
| biLSTM | 87.04% | 84.94% | 45   |20000  |0.2  |0.2  | 50  |


LSTM – whole dataset; split VALIDATION_SPLIT = 0.06552, RANDOM_SEED = 42; LSTM(128, dropout_U=0.2, dropout_W=0.2).
biLSTM_1 – Bidirectional LSTM, whole dataset; split VALIDATION_SPLIT = 0.06552, RANDOM_SEED = 42; LSTM(128, dropout_U=0.1, dropout_W=0.1)
biLSTM_2 – Bidirectional LSTM, whole dataset; split VALIDATION_SPLIT = 0.06552, RANDOM_SEED = 42; LSTM(128, dropout_U=0.2, dropout_W=0.2).

| Approach| Train Accuracy|Validation Accuracy|Epochs|MAX_NB_WORDS|MAX_SEQUENCE_LENGTH|
| ------- |:-------------:| :----------------:|:----:|:----------:|:-----------------:|
| LSTM    | 86.26%        | 83.86%            | 55   |20000       | 50                |
| LSTM    | 85.69%        | 84.23%            | 41   |20000       | 100               |



LSTM – whole dataset; split VALIDATION_SPLIT = 0.2, RANDOM_SEED = 42; LSTM(128, dropout_U=0.2, dropout_W=0.2), Dropout(0.4), , batch_size=2000.

