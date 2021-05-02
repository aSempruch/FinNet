import numpy as np
import pandas as pd

from tensorflow.keras import preprocessing as kprocessing
from sklearn.preprocessing import normalize

from models.baseline_rnn import BaselineRNN
from models.rnn1 import RNN1
from util import load_merged_data, score_model, score_sequence_model
from processing.news_processor import construct_doc2vec

# %% Load Data
df = load_merged_data(test=False)

# %% Construct doc2vec embeddings
df['doc2vec'] = construct_doc2vec(df['body'], {
    'vector_size': 10,
    'window': 2,
    'min_count': 1,
    'workers': 8
})


def generate_sequence_list(df, sequence_length, x_selector, y_selector):
    sequences = list()
    df_sorted = df.sort_values(by='datetime')

    for ticker in df_sorted.ticker.unique():
        df_single_ticker = df_sorted[df_sorted.ticker == ticker]
        for year in df_single_ticker.year.unique():
            df_single_year = df_single_ticker[df_single_ticker.year == year]

            # Skip batch of articles if doesn't meet batch size
            if len(df_single_year) < sequence_length:
                continue

            # print(len(df_single_year))
            x = x_selector(df_single_year)
            y = y_selector(df_single_year)
            x_sequence = kprocessing.timeseries_dataset_from_array(x, targets=y, sequence_length=sequence_length)
            sequences.append(x_sequence)

    return sequences


# %% Prepare data
train_data = df[(df.year == 2016) | (df.year == 2017)]
train_list = generate_sequence_list(train_data, sequence_length=5,
                                    x_selector=lambda df: normalize(np.array(df['doc2vec'].tolist())),
                                    y_selector=lambda df: df['is_dps_cut'].astype('float32').to_numpy().reshape((-1, 1)))
test_data = df[(df.year == 2018)]
test_list = generate_sequence_list(train_data, sequence_length=5,
                                   x_selector=lambda df: normalize(np.array(df['doc2vec'].tolist())),
                                   y_selector=lambda df: df['is_dps_cut'].astype('float32').to_numpy().reshape((-1, 1)))
# y_true = test_data['is_dps_cut'].astype('float32').to_numpy().reshape((-1, 1))
# %%
# model = BaselineRNN()
model = RNN1()
model.train_sequences(train_list, input_shape=(5, 10), epochs=1)

# %%
scores = score_sequence_model(model, test_list)
# %%
# y_pred = model.predict(test_list[23])
# y_true = list(test_list[23])[0][1].numpy()
# x_train = normalize(np.array(train_data['doc2vec'].tolist()))
# y_train = train_data['is_dps_cut'].astype('float32').to_numpy().reshape((-1, 1))

# test_data = df[df.year == 2018]
# x_test = normalize(np.array(test_data['doc2vec'].tolist()))
# y_true = test_data['is_dps_cut'].astype(np.int).to_numpy().reshape((-1, 1))
