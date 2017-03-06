# Experiments with full rt + imdb datasets

If review length is more than 100 words, divide one into reviews as first 50 words and last 50 words.

### 1. Results

| Dropout before LSTM|Dropout_U|Dropout_W|Dropout before LSTM|Train Accuracy|Validation Accuracy|Epochs| 
|:------------------:|:-------:|:-------:|:-----------------:|-------------:|------------------:|-----:|
| 0.2                | 0.1     |0.3      |0.3                | 85.98%       | **84.75%**        |98    |
| 0.1                | 0.2     |0.3      |0.3                | 86.31%       | 84.68%            |129   |
| 0.2                | 0       |0.3      |0.1                | 86.95%       | 84.65%            |79    |
| 0.1                | 0.1     |0.3      |0.3                | 86.79%       | 84.62%            |103   |
| 0.1                | 0.2     |0.3      |0.1                | 85.82%       | 84.60%            |90    |
| 0.2                | 0.1     |0.1      |0                  | 87.75%       | 84.55%            |**65**|
| 0.2                | 0.2     |0        |0                  | **88.85%**   | 84.51%            |87    |

Full results: https://docs.google.com/spreadsheets/d/1UWdwxw11Kek5mEKz8h2W45ZmbOGMCwVAIZtwPLGf8CY/edit?usp=sharing

LSTM, split VALIDATION_SPLIT = 0.06552, RANDOM_SEED = 42; MAX_SEQUENCE_LENGTH = 50
