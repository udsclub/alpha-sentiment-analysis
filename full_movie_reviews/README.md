# Experiments with full rt + imdb datasets

### 1. Results

| Approach        | Train Accuracy           | Validation Accuracy  | Epochs |
| --------------- |:------------------------:| --------------------:| ------:|
| Simple          | 89.32%                   | 84.89%               | 33     |

Simple – whole dataset; split VALIDATION_SPLIT = 0.06552, RANDOM_SEED = 42; MAX_NB_WORDS = 50000; MAX_SEQUENCE_LENGTH = 50; LSTM(128, dropout_U=0.2, dropout_W=0.2), Dropout(0.3). 
