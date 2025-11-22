import re
import pandas as pd
import nltk
from nltk.corpus import stopwords
from nltk.stem import PorterStemmer

nltk.download('stopwords', quiet=True)

STOPWORDS = set(stopwords.words("english"))
STEMMER = PorterStemmer()


def clean_text(text: str) -> str:
    if not isinstance(text, str):
        return ""
    text = text.lower()
    text = re.sub(r"http\S+|www\S+", " ", text)
    text = re.sub(r"\S+@\S+", " ", text)
    text = re.sub(r"[^a-z0-9\s]", " ", text)
    text = re.sub(r"\s+", " ", text).strip()
    return text


def tokenize_and_stem(text: str) -> str:
    tokens = [w for w in text.split() if w not in STOPWORDS]
    stems = [STEMMER.stem(t) for t in tokens]
    return " ".join(stems)


def preprocess_dataframe(df: pd.DataFrame):
    df = df.copy()
    df["subject"] = df["subject"].fillna("")
    df["body"] = df["body"].fillna("")
    df["raw_text"] = df["subject"] + " " + df["body"]
    df["clean_text"] = df["raw_text"].apply(clean_text)
    df["processed_text"] = df["clean_text"].apply(tokenize_and_stem)
    return df
