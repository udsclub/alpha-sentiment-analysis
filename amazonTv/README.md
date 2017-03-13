# Experiments on TV and Movie amazon dataset

| Approach| Train Accuracy| Validation Accuracy|Test Accuracy|Epochs|MAX WORDS|Droupout before LSTM cell|Droupout after LSTM|MAX SEQUENCE LENGTH|dropout_U|dropout_W|LSTM_1 output|LSTM_2 output|Dense|Droupout before Dense|
|--------|:------:|:------:|:------:|:----:|:-----:|:---:|:---:|:----:|:---:|:----:|:----:|:----:|:----:|:----:|
| LSTM&m |**96.09%**|**95.96%**|**95.96%**|52|50000|0.2  |0.2  | 110  |0.2  |0.2   |128   |-     |-     |-     |

LSTM&m – whole dataset; split VALIDATION_SPLIT = 0.06552, RANDOM_SEED = 42; masking(0)

Test – train_test_split(data, labels, test_size=0.1, random_state=42, stratify=labels)
