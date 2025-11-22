from sklearn.feature_extraction.text import TfidfVectorizer
import numpy as np


def build_vectorizer(texts, max_features=5000):
    vect = TfidfVectorizer(max_features=max_features, ngram_range=(1, 2))
    vect.fit(texts)
    return vect


def transform_texts(vect, texts):
    return vect.transform(texts)


def add_meta_features(df):
    meta = []
    for _, row in df.iterrows():
        subj = str(row["subject"])
        body = str(row["body"])
        meta.append([
            len(subj.split()),
            len(body.split()),
            subj.count("!") + body.count("!")
        ])
    return np.array(meta)
