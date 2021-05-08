import numpy as np
import pandas as pd
from tensorflow.keras import preprocessing as kprocessing
from sklearn.preprocessing import normalize

from models.baseline_nn import BaselineNN
from models.dnn1 import DNN1
from models.rnn1 import RNN1
from models.baseline_rnn import BaselineRNN
from util import load_merged_data, score_model, score_sequence_model, generate_sequence_list, get_args
from processing.news_processor import ConstructTfidf, ConstructBOW

# %%
args = get_args()
print(args)

# %%
df = load_merged_data()

# %% Build tfidf/bow
train_data = df[(df.year == 2016) | (df.year == 2017)].copy()
test_data = df[df.year == 2018].copy()
# vectorizer = ConstructBOW()
vectorizer = ConstructTfidf()
train_data['vector'] = vectorizer.fit_transform(train_data['body'])
# x_train = np.array(train_data['vector'].tolist())
test_data['vector'] = vectorizer.transform(test_data['body'])
# x_test = np.array(test_data['vector'].tolist())

# %% Prepare data
x_selector = lambda df: normalize(np.array(df['vector'].tolist()))
y_selector = lambda df: df['is_dps_cut'].astype('float32').to_numpy().reshape((-1, 1))[0]

x_train, y_train = generate_sequence_list(train_data, sequence_length=args.sequencelength,
                                          x_selector=x_selector,
                                          y_selector=y_selector,
                                          overlap=args.overlap)
# x_train_slice = normalize(np.array(train_data['doc2vec'].tolist()))
# y_train = train_data['is_dps_cut'].astype('float32').to_numpy().reshape((-1, 1))

# x_test_slice = normalize(np.array(test_data['doc2vec'].tolist()))
# y_true = test_data['is_dps_cut'].astype(np.int).to_numpy().reshape((-1, 1))
x_test, y_test = generate_sequence_list(test_data, sequence_length=args.sequencelength,
                                        x_selector=x_selector,
                                        y_selector=y_selector,
                                        overlap=args.overlap)

print("Train:", x_train.shape)
print("Test:", x_test.shape)

# %% Train Model

if args.model == 0:
    model = BaselineRNN()
else:
    model = RNN1()
model.train_sequences(x_train, y_train, input_shape=(args.sequencelength, len(train_data.iloc[0]['vector'])),
                      epochs=args.epochs)

# %%
# print(model.predict(x_test))
print("Train:", x_train.shape)
print("Test:", x_test.shape)
print(args)
scores = score_model(model, x_test, y_test)
