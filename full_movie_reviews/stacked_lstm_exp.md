# Experiments with full rt + imdb datasets

Experiments with stacked LSTMs

### 1. Results

**n\_layers**|**output\_1**|**output\_2**|**output\_3**|**train**|**val**|**step**
:-----:|:-----:|:-----:|:-----:|:-----:|:-----:|:-----:
3|128|64|64|0.87064|0.84905|38
2|128|128|-|0.872888|0.8484|38
2|64|32|-|0.858425|0.8475|60
2|64|64|-|0.856749|0.84725|51
2|256|128|-|0.895751|0.847|33
2|128|64|-|0.870847|0.84655|40
3|128|64|32|0.865125|0.84595|33

Full results: https://docs.google.com/spreadsheets/d/1ceBksWNRCp3e2ZGGwyxMddb3uqbkVEg9ZKZak8G3pJU/edit?usp=sharing

LSTM, split VALIDATION_SPLIT = 0.06552, RANDOM_SEED = 42; MAX_SEQUENCE_LENGTH = 50
Word2vec google embeddings -> dropout(0.2) -> LSTMs(dropout_u=0.1, dropout_w=0.1) -> dropout(0.2)
