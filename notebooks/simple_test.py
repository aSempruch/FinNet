import numpy as np
import pandas as pd

from sklearn.preprocessing import normalize

from models.baseline_nn import BaselineNN
from models.dnn1 import DNN1
from util import load_merged_data, score_model
from processing.news_processor import construct_doc2vec

#%% Load Data
df = load_merged_data()

#%% Construct doc2vec embeddings
df['doc2vec'] = construct_doc2vec(df['body'], {
    'vector_size': 10,
    'window': 2,
    'min_count': 1,
    'workers': 8
})

#%% Prepare data
train_data = df[(df.year == 2016) | (df.year == 2017)]
x_train = normalize(np.array(train_data['doc2vec'].tolist()))
y_train = train_data['is_dps_cut'].astype('float32').to_numpy().reshape((-1, 1))

test_data = df[df.year == 2018]
x_test = normalize(np.array(test_data['doc2vec'].tolist()))
y_true = test_data['is_dps_cut'].astype(np.int).to_numpy().reshape((-1, 1))

#%% Train Model
# model = BaselineNN()
model = DNN1()
model.train(x_train, y_train, epochs=3000)


#%% Score model
score_model(model, x_test, y_true)
