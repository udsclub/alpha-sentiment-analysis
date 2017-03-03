# Experiments with full rt + imdb datasets

### 1. Results

| Approach        | Train Accuracy           | Validation Accuracy  | Epochs |MAX_NB_WORDS | Droupout before LSTM cell | Droupout|
| --------------- |:------------------------:| :-------------------:| :-----:|:-----------:|:-----------:|:-----------:|
| LSTM            | 89.32%                   | 84.89%               | 33     |50000        |0            |0.3          |
| LSTM            | 89.27%                   | 84.79%               | 36     |20000        |0            |0.3          |
| LSTM            | 86.89%                   | 84.91%               | 35     |20000        |0.2          |0.3          |
| LSTM            | 89.15%                   | 84.59%               | 34     |20000        |0            |0.4          |




LSTM – whole dataset; split VALIDATION_SPLIT = 0.06552, RANDOM_SEED = 42; MAX_SEQUENCE_LENGTH = 50; LSTM(128, dropout_U=0.2, dropout_W=0.2), Dropout(0.3).

| Approach        | Train Accuracy           | Validation Accuracy  | Epochs |MAX_NB_WORDS | MAX_SEQUENCE_LENGTH |
| --------------- |:------------------------:| :-------------------:| :-----:|:-----------:|:-----------:|
| LSTM            | 86.26%                   | 83.86%               | 55     |20000        | 50          |
| LSTM            | 85.69%                   | 84.23%               | 41     |20000        | 100         |



LSTM – whole dataset; split VALIDATION_SPLIT = 0.2, RANDOM_SEED = 42; LSTM(128, dropout_U=0.2, dropout_W=0.2), Dropout(0.4), , batch_size=2000.


