# Experiments with full rt + imdb datasets

Experiments with stacked LSTMs

### 1. Results

**n\_layers**|**output\_1**|**output\_2**|**output\_3**|**train**|**val**|**step**
:-----:|:-----:|:-----:|:-----:|:-----:|:-----:|:-----:
3|128|64|64|0.87064|**0.84905**|38
2|128|128|-|0.872888|0.8484|38
2|64|32|-|0.858425|0.8475|60
2|64|64|-|0.856749|0.84725|51
2|256|128|-|**0.895751**|0.847|**33**
2|128|64|-|0.870847|0.84655|40
3|128|64|32|0.865125|0.84595|**33**

Full results: https://docs.google.com/spreadsheets/d/1ceBksWNRCp3e2ZGGwyxMddb3uqbkVEg9ZKZak8G3pJU/edit?usp=sharing

LSTM, split VALIDATION_SPLIT = 0.06552, RANDOM_SEED = 42; MAX_SEQUENCE_LENGTH = 50
Word2vec google embeddings -> dropout(0.2) -> LSTMs(dropout_u=0.1, dropout_w=0.1) -> dropout(0.2)

### 2. Results with MAX_SEQUENCE_LENGTH = 100

**n\_layers**|**output\_1**|**output\_2**|**output\_3**|**train**|**val**|**step**
:-----:|:-----:|:-----:|:-----:|:-----:|:-----:|:-----:
3|256|256|128|0.903215|**0.8624**|**14**
2|128|64|-|0.885601|0.86165|34
3|128|64|64|0.888265|0.8615|38
3|64|64|64|0.881172|0.8614|64
3|64|64|32|0.877502|0.8612|57
3|128|64|32|0.884819|0.8608|33
2|256|256|-|**0.910213**|0.85985|27

Full results: https://docs.google.com/spreadsheets/d/19Vone7dXCnoIRlYECgiUXi3APvhwI1QZ_JVMRjm0xCc/edit?usp=sharing

LSTM, split VALIDATION_SPLIT = 0.06552, RANDOM_SEED = 42; MAX_SEQUENCE_LENGTH = 100
Word2vec google embeddings -> dropout(0.2) -> LSTMs(dropout_u=0.1, dropout_w=0.1) dropout(0.1) -> dropout(0.2)
