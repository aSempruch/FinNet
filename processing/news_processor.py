import pandas as pd
import numpy as np
import gensim
from sklearn.feature_extraction.text import TfidfVectorizer, CountVectorizer

from util import load_merged_data, load_sec_data, preprocess_text, VERBOSE


class ConstructTfidf:

    def __init__(self, params=None):
        params = dict() if params is None else params

        self.vectorizer = TfidfVectorizer(
            tokenizer=lambda x: x,
            preprocessor=lambda x: x,
            **params
        )

    def fit_transform(self, text_series):
        lst_corpus = preprocess_text(text_series)
        return self.vectorizer.fit_transform(lst_corpus).toarray().tolist()

    def transform(self, text_series):
        lst_corpus = preprocess_text(text_series)
        return self.vectorizer.transform(lst_corpus).toarray().tolist()


class ConstructBOW:

    def __init__(self, params=None):
        params = dict() if params is None else params

        self.vectorizer = CountVectorizer(
            tokenizer=lambda x: x,
            preprocessor=lambda x: x,
            **params
        )

    def fit_transform(self, text_series):
        lst_corpus = preprocess_text(text_series)
        return self.vectorizer.fit_transform(lst_corpus).toarray().tolist()

    def transform(self, text_series):
        lst_corpus = preprocess_text(text_series)
        return self.vectorizer.transform(lst_corpus).toarray().tolist()

def construct_word2vec(text_series, params):
    #%%
    lst_corpus = preprocess_text(text_series)
    # bigrams_detector = gensim.models.phrases.Phrases(lst_corpus,
    #                                                  delimiter=" ".encode(), min_count=5, threshold=10)
    # bigrams_detector = gensim.models.phrases.Phraser(bigrams_detector)
    # trigrams_detector = gensim.models.phrases.Phrases(bigrams_detector[lst_corpus],
    #                                                   delimiter=" ".encode(), min_count=5, threshold=10)
    # trigrams_detector = gensim.models.phrases.Phraser(trigrams_detector)
    w2v = gensim.models.word2vec.Word2Vec(lst_corpus, **params)
    return []
    #%%


def construct_doc2vec(text_series, params):
    lst_corpus = preprocess_text(text_series)
    documents = [gensim.models.doc2vec.TaggedDocument(doc, [i]) for i, doc in enumerate(lst_corpus)]
    model = gensim.models.doc2vec.Doc2Vec(documents, **params)
    vectors = [model.dv[idx] for idx in range(len(documents))]
    return vectors


if __name__ == '__main__':
    df = load_merged_data()
    df['doc2vec'] = construct_doc2vec(df['body'], {
        'vector_size': 10,
        'window': 2,
        'min_count': 1,
        'workers': 8
    })

    # df['word2vec'] = construct_word2vec(df['body'], {
    #     'vector_size': 300,
    #     'window': 8,
    #     'min_count': 1,
    #     'sg': 1,
    #     'workers': 8
    # })

