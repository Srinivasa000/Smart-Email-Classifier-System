import joblib
import os
from scipy.sparse import hstack


class MultiTargetModel:
    def __init__(self, vectorizer=None, models=None, encoders=None):
        self.vectorizer = vectorizer
        self.models = models or {}
        self.encoders = encoders or {}

    def save(self, out_dir):
        os.makedirs(out_dir, exist_ok=True)
        joblib.dump(self.vectorizer, f"{out_dir}/vectorizer.joblib")
        joblib.dump(self.models, f"{out_dir}/models.joblib")
        joblib.dump(self.encoders, f"{out_dir}/encoders.joblib")

    @classmethod
    def load(cls, model_dir):
        vectorizer = joblib.load(f"{model_dir}/vectorizer.joblib")
        models = joblib.load(f"{model_dir}/models.joblib")
        encoders = joblib.load(f"{model_dir}/encoders.joblib")
        return cls(vectorizer=vectorizer, models=models, encoders=encoders)

    def predict(self, text_tfidf, meta_sparse):
        X = hstack([text_tfidf, meta_sparse])
        output = {}

        for target, model in self.models.items():
            probs = model.predict_proba(X)
            idx = probs.argmax(axis=1)[0]
            pred_encoded = model.classes_[idx]
            prob = probs[0][idx]

            encoder = self.encoders[target]
            pred_label = encoder.inverse_transform([pred_encoded])[0]

            output[target] = {
                "prediction": pred_label,
                "probability": float(prob),
                "classes": list(encoder.inverse_transform(model.classes_)),
                "probs": probs[0].tolist()
            }

        return output
