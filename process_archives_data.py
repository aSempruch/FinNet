import pandas as pd
from gensim.models import Word2Vec, KeyedVectors

from util import preprocess_text

#%% Const
VECTOR_SIZE = 200
DF_FILE = 'Datasets/archive/2018_05_112b52537b67659ad3609a234388c50a/loaded-1000.csv'

#%% Load Data
df = pd.read_csv(DF_FILE)

#%% Preprocess Steps
df['text'] = preprocess_text(df['text'])

#%% Word2Vec model
vec_model = Word2Vec(df['text'], vector_size=VECTOR_SIZE)

#%%