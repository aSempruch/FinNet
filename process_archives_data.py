import pandas as pd

from util import preprocess_text

#%% Const
DF_FILE = 'Datasets/archive/2018_05_112b52537b67659ad3609a234388c50a/loaded-1000.csv'

#%% Load Data
df = pd.read_csv(DF_FILE)

#%% Preprocess Steps
df['text'] = preprocess_text(df['text'])
