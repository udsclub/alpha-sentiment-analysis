# Experiments with full rt + imdb datasets

### 1. Results

| Approach| Train Accuracy| Validation Accuracy|Epochs|MAX WORDS|Droupout before LSTM cell|Droupout after LSTM|MAX SEQUENCE LENGTH|dropout_U|dropout_W|LSTM output|Dense|Droupout before Dense|
|--------|:------:|:------:|:----:|:-----:|:---:|:---:|:----:|:---:|:----:|:----:|:----:|:----:|
| LSTM   |**89.32%**|84.89%| 33   |50000  |0    |0.3  | 50   |0.2  |0.2   |128   |-     |-     |
| LSTM   | 89.27% | 84.79% | 36   |20000  |0    |0.3  | 50   |0.2  |0.2   |128   |-     |-     |
| LSTM   | 86.89% | 84.91% | 35   |20000  |0.2  |0.3  | 50   |0.2  |0.2   |128   |-     |-     |
| LSTM   | 89.15% | 84.59% | 34   |20000  |0    |0.4  | 50   |0.2  |0.2   |128   |-     |-     |
| LSTM   | 88.17% |**86.20%**| 32 |20000  |0.2  |0.2  | 100  |0.2  |0.2   |128   |-     |-     |
| biLSTM | 86.63% | 84.64% | 29   |20000  |0.2  |0.2  | 50   |0.1  |0.1   |128   |-     |-     |
| biLSTM | 87.04% | 84.94% | 45   |20000  |0.2  |0.2  | 50   |0.2  |0.2   |128   |-     |-     |
| mLSTM  | 86.67% | 84.43% |**18**|20000  |0    |0    | 50   |0.1  |0.1   |64    |32    |0     |
| mLSTM  | 88.43% | 85.59% |32    |20000  |0.2  |0.2  | 50   |0.1  |0.1   |128   |64    |0.2   |
| mBiLSTM| 87.95% | 83.85% |24    |20000  |0.2  |0.2  | 25   |0.1  |0.1   |128   |64    |0.2   |
| mBiLSTM| 89.56% | 85.61% |34    |20000  |0.2  |0.2  | 50   |0.1  |0.1   |128   |64    |0.2   |
| mBiLSTM| 88.43% | 86.18% |28    |20000  |0.2  |0.2  | 100  |0.1  |0.1   |128   |64    |0.2   |



LSTM – whole dataset; split VALIDATION_SPLIT = 0.06552, RANDOM_SEED = 42

biLSTM – Bidirectional LSTM, whole dataset; split VALIDATION_SPLIT = 0.06552, RANDOM_SEED = 42

mLSTM – Merged LSTMs with Dense layer (LSTM_1 - left MAX SEQUENCE LENGTH words,  LSTM_2 - right MAX SEQUENCE LENGTH words => merged(concatenation) => Dense)

mBiLSTM – Merged Bidirectional LSTMs with Dense layer (biLSTM_1 - left MAX SEQUENCE LENGTH words,  biLSTM_2 - right MAX SEQUENCE LENGTH words => merged(concatenation) => Dense)

| Approach| Train Accuracy|Validation Accuracy|Epochs|MAX_NB_WORDS|MAX_SEQUENCE_LENGTH|
| ------- |:-------------:| :----------------:|:----:|:----------:|:-----------------:|
| LSTM    | **86.26%**    | 83.86%            | 55   |20000       | 50                |
| LSTM    | 85.69%        | **84.23%**        | 41   |20000       | 100               |



LSTM – whole dataset; split VALIDATION_SPLIT = 0.2, RANDOM_SEED = 42; LSTM(128, dropout_U=0.2, dropout_W=0.2), Dropout(0.4), , batch_size=2000.

