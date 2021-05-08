import os
import pandas as pd
import numpy as np
import re
import requests
import ast
import argparse
from colorama import Fore
from gensim.parsing.preprocessing import preprocess_string, strip_tags, strip_punctuation, remove_stopwords

from sklearn.metrics import f1_score, cohen_kappa_score

from secrets import OPENFIGI_API_KEY
#%%
TEST = bool(os.getenv("TEST"))
VERBOSE = bool(os.getenv("VERBOSE"))
TEST_SAMPLE_FRAC = 0.05
if TEST: print(Fore.LIGHTCYAN_EX + "WARNING: RUNNING IN TEST MODE" + Fore.RESET)


def get_args():
    parser = argparse.ArgumentParser()
    parser.add_argument('-e', '--epochs', default=10, type=int)
    parser.add_argument('-s', '--sequencelength', default=50, type=int)
    parser.add_argument('-o', '--overlap', default=30, type=int)
    parser.add_argument('-m', '--model', default=0, type=int)
    parser.add_argument('-v', '--vectorizer', default=0, type=int)
    return parser.parse_args()


def score_model(model, x_test, y_true, verbose=True):
    y_pred_probs = model.predict(x_test)
    y_pred = (y_pred_probs[:, 0] > 0.5).astype(np.int)
    scores = {
        'F1': f1_score(y_true, y_pred),
        'CK': cohen_kappa_score(y_true, y_pred)
    }
    if verbose:
        print(scores)

    return scores


def score_sequence_model(model, test_sequence_list, verbose=True):
    y_true = list()
    y_pred = list()

    for sequence in test_sequence_list:
        y_true_list = list(sequence)[0][1].numpy()
        y_pred_list = model.predict(sequence)

        y_true_i = y_true_list[0][0].astype(int)
        y_pred_i = y_pred_list.mean().round().astype(int)

        y_true.append(y_true_i)
        y_pred.append(y_pred_i)

    y_true = np.array(y_true)
    y_pred = np.array(y_pred)

    scores = {
        'F1': f1_score(y_true, y_pred),
        'CK': cohen_kappa_score(y_true, y_pred)
    }
    if verbose:
        print(scores)

    return scores


def load_sec_data(target='is_dps_cut', test=TEST):
    #%%
    file_path = os.getenv('VOYA_PATH_DATA') + 'processed_data.csv'
    df = pd.read_csv(file_path, usecols=['cik', 'ticker_x', 'filing_date',
        'year_x', 'filing_year_x', 'perm_id', 'ticker_y', 'year_y',
        'company_name', 'is_dividend_payer', 'dps_change', 'is_dps_cut',
        'z_environmental', 'd_environmental', 'sector', 'filing_year_y'])
    if test: df = df.sample(frac=TEST_SAMPLE_FRAC)
    #%%
    df['year_x'] = df['year_x'].astype(int)

    if target == 'is_dps_cut':
        df.query("is_dividend_payer == 1 and not is_dps_cut.isnull()", inplace=True)
        df['is_dps_cut'] = df['is_dps_cut'].astype(int)

    return df


def load_motley_data(file_path=None, test=TEST):
    #%% Load Data
    if file_path is None:
        file_path = os.getenv('PATH_DATA') + '/motleyfool/processed.csv'
    df = pd.read_csv(file_path)
    if test: df = df.sample(frac=TEST_SAMPLE_FRAC)

    #%% Cast datetime
    df['datetime'] = pd.to_datetime(df['datetime'])
    #%% Parse companies
    df['companies'] = df['companies'].apply(lambda company: ast.literal_eval(company))
    #%%
    return df


def load_merged_data(motley=None, test=TEST):
    cache_file_path = os.getenv('PATH_CACHE') + '/data/motley_merged.csv'
    if os.path.isfile(cache_file_path):
        result = load_motley_data(file_path=cache_file_path, test=test)
        if test: result = result.sample(frac=TEST_SAMPLE_FRAC)
        print("Loaded merged data from cache")
        return result

    #%%
    filings = load_sec_data(test=test)
    if motley is None:
        motley = process_motley(test=test)
    #%%
    sec_tickers = dict()
    for idx, filing_row in filings.iterrows():
        ticker = filing_row.ticker_x.strip()
        if ticker not in sec_tickers:
            sec_tickers[ticker] = dict()
        sec_tickers[ticker][filing_row.year_x] = filing_row.is_dps_cut
    #%%
    result = pd.DataFrame(columns=list(motley.columns)+['is_dps_cut'])
    for idx, article in motley.iterrows():
        ticker = article.ticker
        if ticker in sec_tickers:
            year = article.year
            if year in sec_tickers[ticker]:
                article['is_dps_cut'] = sec_tickers[ticker][year]
                result.loc[len(result)] = article

    if VERBOSE:
        print("Merged tickers.\tFrom", motley.shape[0], "to", result.shape[0], "size")
    if test:
        result = result.sample(frac=TEST_SAMPLE_FRAC)
    return result
    #%%


def process_motley(filter_multi_org=True, test=TEST):
    """
    Process motley fool data
    :param filter_multi_org: ignore articles that mention multiple companies
    :return:
    """
    df = load_motley_data(test=test)
    if VERBOSE: print("Starting with", df.shape[0], "articles")

    df['body'] = df['body'].apply(lambda x: x.replace('\n---', ''))

    if filter_multi_org:  # Filter out articles that mention more than 1 company
        df = df.loc[df['companies'].apply(lambda x: len(x) == 1)]
        if VERBOSE: print("Removed multi_org articles. Left with", df.shape[0], "articles")
    else:  # Filter out articles that don't mention any companies
        df = df.loc[df['companies'].apply(lambda x: len(x) > 0)]

    df['ticker'] = df['companies'].apply(lambda x: x[0][0].split(':')[1])
    df['year'] = df['datetime'].apply(lambda x: x.year)

    return df


def process_motley_raw():
    #%% Load raw data
    file_path = os.getenv('PATH_DATA') + '/motleyfool/scraped.csv'
    df = pd.read_csv(file_path)

    #%% Remove empty article bodies
    df.query("not body.isnull()", inplace=True)
    df = df.loc[df['body'].apply(lambda x: len(x) > 0)]

    #%% Convert dates
    df['datetime'] = df['date'].apply(
        lambda date: date.split('---')[0].replace('Updated: ', '') if '---' in date else date)
    df['datetime'] = df['datetime'].apply(
        lambda date: date.replace('Updated: ', '') if 'Updated:' in date else date)
    df['datetime'] = pd.to_datetime(df['datetime'], format='%b %d, %Y at %I:%M%p')

    #%% Save Output
    df.to_csv(os.getenv('PATH_DATA') + '/motleyfool/processed.csv', index=None)


def generate_sequence_list(df, sequence_length, x_selector, y_selector, overlap=0):
    x_sequences = list()
    y_vals = list()
    df_sorted = df.sort_values(by='datetime')

    for ticker in df_sorted.ticker.unique():
        df_single_ticker = df_sorted[df_sorted.ticker == ticker]
        for year in df_single_ticker.year.unique():
            df_single_year = df_single_ticker[df_single_ticker.year == year]

            # print(len(df_single_year))
            x = x_selector(df_single_year)
            y = y_selector(df_single_year)
            # x_sequence = kprocessing.timeseries_dataset_from_array(x, targets=y, sequence_length=sequence_length)

            for x_idx in range(0, len(x), sequence_length-overlap):
                seq = np.zeros((sequence_length, len(x[0])))
                # x_offset = sequence_length*x_idx
                for seq_idx in range(sequence_length):
                    if (x_idx + seq_idx) < len(x):
                        seq[seq_idx] = x[x_idx + seq_idx]
                # for seq_idx, val in enumerate(x[x_offset:(x_offset+sequence_length)]):
                #     seq[(x_offset+seq_idx)]
                x_sequences.append(seq)
                y_vals.append(y[0])
                # y_vals.append(y)

            # x_sequence = x[:sequence_length]
            # x_sequences.append(x_sequence)
            # y_vals.append(y[0])
            # sequences.append(x_sequence)-

    return np.array(x_sequences), np.array(y_vals)


def _clean_text(text):
    text = re.sub(r"[^A-Za-z0-9^,!.\/'+-=]", " ", text)
    text = re.sub(r"what's", "what is ", text)
    text = re.sub(r"\'s", " ", text)
    text = re.sub(r"\'ve", " have ", text)
    text = re.sub(r"can't", "cannot ", text)
    text = re.sub(r"n't", " not ", text)
    text = re.sub(r"i'm", "i am ", text)
    text = re.sub(r"\'re", " are ", text)
    text = re.sub(r"\'d", " would ", text)
    text = re.sub(r"\'ll", " will ", text)
    text = re.sub(r",", " ", text)
    text = re.sub(r"\.", " ", text)
    text = re.sub(r"!", " ! ", text)
    text = re.sub(r"\/", " ", text)
    text = re.sub(r"\+", " + ", text)
    text = re.sub(r"\-", " - ", text)
    text = re.sub(r"\=", " = ", text)
    text = re.sub(r"'", " ", text)
    text = re.sub(r"(\d+)(k)", r"\g<1>000", text)
    text = re.sub(r":", " : ", text)
    text = re.sub(r" e g ", " eg ", text)
    text = re.sub(r" b g ", " bg ", text)
    text = re.sub(r" u s ", " american ", text)
    text = re.sub(r" 9 11 ", "911", text)
    text = re.sub(r"e - mail", "email", text)
    text = re.sub(r"j k", "jk", text)
    text = re.sub(r"\s{2,}", " ", text)

    return text


def preprocess_text(series):
    """
    Preprocess text
    :param series:
    :param remove_stopwords:
    :return:
    """
    FILTERS = [
        lambda x: x.lower(),
        _clean_text,
        strip_tags,
        strip_punctuation,
        remove_stopwords
    ]

    result = list()
    for string in series:
        result.append(preprocess_string(string, FILTERS))

    return pd.Series(result)


def search_ticker(company_name):
    """
    Search for company ticker by company name through OpenFigi API
    :param company_name: Name of company
    """
    r = requests.post(
        url='https://api.openfigi.com/v3/search',
        headers={
            "Content-Type": "application/json",
            "X-OPENFIGI-APIKEY": OPENFIGI_API_KEY
        },
        json={
            'query': company_name,
            "securityType": "Common Stock"
        }
    )

    if r.status_code != 200:  # Invalid response
        error_map = {
            400: "Bad request",
            401: "Invalid API key",
            404: "Invalid URL",
            429: "Too many requests"
        }
        if r.status_code in error_map:
            error = (r.status_code, error_map[r.status_code])
        else:
            error = [r.status_code]
        return False, (*error)

    data = r.json()['data']
    if len(data) > 0:
        return True, data[0]['ticker']  # Get ticker for first result
    else:
        return True, 0, "No results"  # No results found
