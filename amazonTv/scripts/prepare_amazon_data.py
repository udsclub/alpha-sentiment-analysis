import pandas as pd
import gzip
from sklearn.model_selection import train_test_split

PATH = '../data/Movies_and_TV_5.json.gz' # http://jmcauley.ucsd.edu/data/amazon/links.html


def parse():
    g = gzip.open(PATH, 'rb')
    for l in g:
        yield eval(l)


def get_df():
    i = 0
    df = {}
    for d in parse(PATH):
        df[i] = d
        i += 1
    return pd.DataFrame.from_dict(df, orient='index')

df = get_df()
train, test = train_test_split(df.asin.unique(), test_size=0.1, random_state=42)
df_train = df[df.asin.isin(train) & (df.overall != 3)].copy()
df_test = df[df.asin.isin(test) & (df.overall != 3)].copy()
df_train['overall'] = df_train['overall'].apply(lambda x: 1 if x > 3 else 0)
df_test['overall'] = df_test['overall'].apply(lambda x: 1 if x > 3 else 0)
df_train[['overall', 'reviewText', 'asin']].to_csv('../data/train_movies.csv', index=False)
df_test[['overall', 'reviewText']].to_csv('../data/test_movies.csv', index=False)
