import numpy as np
import pandas as pd

from sklearn.preprocessing import normalize

from models.baseline_nn import BaselineNN
from models.dnn1 import DNN1
from models.baseline_rnn import BaselineRNN
from util import load_merged_data, score_model, get_args
from processing.news_processor import ConstructTfidf, ConstructBOW

# %%
args = get_args()

# %%
df = load_merged_data()

#%% Prepare data
train_data = df[(df.year == 2016) | (df.year == 2017)].copy()
# x_train_slice = normalize(np.array(train_data['doc2vec'].tolist()))
y_train = train_data['is_dps_cut'].astype('float32').to_numpy().reshape((-1, 1))

test_data = df[df.year == 2018].copy()
# x_test_slice = normalize(np.array(test_data['doc2vec'].tolist()))
y_true = test_data['is_dps_cut'].astype(np.int).to_numpy().reshape((-1, 1))


# %% Build tfidf/bow
if args.vectorizer == 0:
    vectorizer = ConstructBOW()
else:
    vectorizer = ConstructTfidf()
train_data['vector'] = vectorizer.fit_transform(train_data['body'])
x_train = np.array(train_data['vector'].tolist())
test_data['vector'] = vectorizer.transform(test_data['body'])
x_test = np.array(test_data['vector'].tolist())


# %% Train Model

if args.model == 0:
    model = BaselineNN()
else:
    model = DNN1()
model.train(x_train, y_train, epochs=args.epochs)

print(args)
score_model(model, x_test, y_true)
