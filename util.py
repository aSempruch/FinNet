import pandas as pd
import re
import requests
from gensim.parsing.preprocessing import preprocess_string, strip_tags, strip_punctuation, remove_stopwords

from secrets import OPENFIGI_API_KEY

#%%
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


#%%
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
        return False, *error

    data = r.json()['data']
    if len(data) > 0:
        return True, data[0]['ticker']  # Get ticker for first result
    else:
        return True, 0, "No results"  # No results found
