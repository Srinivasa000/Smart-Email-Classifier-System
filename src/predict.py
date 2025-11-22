import argparse
import numpy as np
from scipy.sparse import csr_matrix
from data_preprocessing import clean_text, tokenize_and_stem
from model import MultiTargetModel


def prepare(subject, body, vectorizer):
    processed = tokenize_and_stem(clean_text(subject + " " + body))
    tf = vectorizer.transform([processed])
    meta = csr_matrix([[len(subject.split()), len(body.split()), subject.count("!") + body.count("!")]])
    return tf, meta


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--subject", required=True)
    parser.add_argument("--body", required=True)
    parser.add_argument("--model_dir", default="models")
    args = parser.parse_args()

    mt = MultiTargetModel.load(args.model_dir)

    tf, meta = prepare(args.subject, args.body, mt.vectorizer)
    results = mt.predict(tf, meta)

    print("\n==== RESULTS ====")
    for k, v in results.items():
        print(f"\n{k.upper()}: {v['prediction']} (conf: {v['probability']})")
        for cls, prob in zip(v["classes"], v["probs"]):
            print(f"  {cls}: {round(prob, 3)}")
