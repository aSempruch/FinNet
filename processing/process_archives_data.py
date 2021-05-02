import pandas as pd
from gensim.models import Word2Vec, KeyedVectors

from util import preprocess_text, search_ticker

from time import sleep
import pickle
import ast
import json

#%% Const
VECTOR_SIZE = 200
DF_FILE = '../Datasets/archive/2018_05_112b52537b67659ad3609a234388c50a/loaded-1000.csv'

#%% Load Data
df = pd.read_csv(DF_FILE)

#%% Preprocess Steps
df['text'] = preprocess_text(df['text'])

#%% Word2Vec model
vec_model = Word2Vec(df['text'], vector_size=VECTOR_SIZE)

#%% Get Tickers
ticker_map = dict()
search_miss_set = set()

#%% Run Ticker Search
for entry_str in df['entities.organizations'].values:
    for org_dict in ast.literal_eval(entry_str):

        org_name = org_dict['name'].lower().strip()
        search_result = [False]

        if org_name not in ticker_map and org_name not in search_miss_set:
            print("Searching", org_name)
            while not search_result[0]:
                search_result = search_ticker(org_name)

                if search_result[0]:  # Request went through

                    if search_result[1] != 0:  # Successful search
                        ticker_map[org_name] = search_result[1]
                        print("\t", search_result[1])

                    elif search_result[1] == 0:  # No results
                        search_miss_set.add(org_name)
                        print("\tno results")

                    sleep(4)

                else:  # Other error
                    if search_result[1] == 429:
                        print("Too many requests. Sleeping...")
                        sleep(20)
                    else:
                        print("Unexpected error: ", search_result)
                        sleep(60)
    # for org_obj in :
    #     print(org_obj)
        # org_name = org_obj.name
        # print(org_name)
#%% Save to file
with open('../Datasets/ticker_map.pkl', 'wb') as file:
    pickle.dump(ticker_map, file)
