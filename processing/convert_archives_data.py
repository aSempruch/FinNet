import os
import numpy as np
import json
import pandas as pd

#%% Consts
MAX_FILES = 1000

#%% Empty DF
df = None

#%% Traverse all files and add to DF
count = 0
dir_path, _, filenames = next(os.walk('../Datasets/archive/2018_05_112b52537b67659ad3609a234388c50a/'))
for filename in filenames:
    with open(f'{dir_path}\\{filename}', 'r', encoding='utf-8') as file:
        data_json = json.load(file)
        if df is not None:
            df = df.append(pd.json_normalize(data_json))
        else:
            df = pd.json_normalize(data_json)
    count += 1
    if count >= MAX_FILES:
        break
# df = pd.read_json('Datasets/archive/2018_01_112b52537b67659ad3609a234388c50a/blogs_0003905.json')

#%% Save Dataframe
df.to_csv(dir_path + '\\' + f'loaded-{count}.csv')
