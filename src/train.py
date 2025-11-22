import argparse
import pandas as pd
import numpy as np
from sklearn.utils import resample
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder
from sklearn.naive_bayes import MultinomialNB
from sklearn.linear_model import LogisticRegression
from scipy.sparse import csr_matrix, hstack
from data_preprocessing import preprocess_dataframe
from feature_engineering import build_vectorizer, transform_texts, add_meta_features
from model import MultiTargetModel


def balance(df, col):
    max_count = df[col].value_counts().max()
    balanced_parts = [
        resample(df[df[col] == c], replace=True, n_samples=max_count, random_state=42)
        for c in df[col].unique()
    ]
    return pd.concat(balanced_parts).sample(frac=1, random_state=42)


def main(args):
    df = pd.read_csv(args.data)
    df = preprocess_dataframe(df)

    train_df, test_df = train_test_split(df, test_size=0.2, stratify=df["spam"], random_state=42)

    train_df = balance(train_df, "spam")

    vect = build_vectorizer(train_df["processed_text"])

    X_train_text = transform_texts(vect, train_df["processed_text"])
    X_test_text = transform_texts(vect, test_df["processed_text"])

    X_train_meta = csr_matrix(add_meta_features(train_df))
    X_test_meta = csr_matrix(add_meta_features(test_df))

    targets = {
        "spam": "spam",
        "department": "label_department",
        "priority": "label_priority"
    }

    encoders = {}
    models = {}

    for key, col in targets.items():
        le = LabelEncoder()
        y_train = le.fit_transform(train_df[col])
        y_test = le.transform(test_df[col])
        encoders[key] = le

        X_train = hstack([X_train_text, X_train_meta])

        if key == "spam":
            clf = MultinomialNB()
        else:
            clf = LogisticRegression(max_iter=1000)

        clf.fit(X_train, y_train)
        models[key] = clf

    mtm = MultiTargetModel(vectorizer=vect, models=models, encoders=encoders)
    mtm.save(args.out_dir)
    print("\nModel training complete. Saved to:", args.out_dir)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--data", required=True)
    parser.add_argument("--out_dir", default="models")
    args = parser.parse_args()
    main(args)
