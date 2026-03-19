import sys, os
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import pickle
import pandas as pd
from sklearn.svm import LinearSVC
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.pipeline import Pipeline
from sklearn.metrics import classification_report
from sklearn.utils.class_weight import compute_class_weight
import numpy as np


class TfidfSVM:
    """
    TF-IDF + LinearSVC pipeline.
    LinearSVC is faster than SVC with linear kernel — same math,
    optimized for large feature spaces like TF-IDF vectors.
    class_weight='balanced' handles the neutral-heavy imbalance.
    """

    def __init__(self):
        self.pipeline = Pipeline([
            ("tfidf", TfidfVectorizer(
                ngram_range=(1, 2),   # unigrams + bigrams
                max_features=30000,   # top 30k features
                sublinear_tf=True,    # log normalization
                min_df=2,             # ignore terms appearing only once
                strip_accents="unicode",
            )),
            ("svm", LinearSVC(
                C=0.5,
                class_weight="balanced",   # handles class imbalance
                max_iter=2000,
            ))
        ])

    def train(self, train_df, text_col="clean_text", label_col="label"):
        X = train_df[text_col].tolist()
        y = train_df[label_col].tolist()
        self.pipeline.fit(X, y)
        print("TF-IDF + SVM trained successfully")

    def predict(self, texts):
        return self.pipeline.predict(texts).tolist()

    def save(self, path="models/saved/tfidf_svm.pkl"):
        os.makedirs(os.path.dirname(path), exist_ok=True)
        with open(path, "wb") as f:
            pickle.dump(self.pipeline, f)
        print(f"Model saved to {path}")

    def load(self, path="models/saved/tfidf_svm.pkl"):
        with open(path, "rb") as f:
            self.pipeline = pickle.load(f)
        print(f"Model loaded from {path}")

    def get_top_features(self, n=15):
        """
        Show top weighted words for each class.
        Only works with LinearSVC — useful for interpretability.
        """
        vectorizer = self.pipeline.named_steps["tfidf"]
        svm        = self.pipeline.named_steps["svm"]
        features   = vectorizer.get_feature_names_out()
        class_map  = {0: "bearish", 1: "neutral", 2: "bullish"}

        print("\nTop discriminative features per class:")
        for i, coef in enumerate(svm.coef_):
            top_indices = coef.argsort()[-n:][::-1]
            top_words   = [features[j] for j in top_indices]
            print(f"\n  {class_map[i]}: {', '.join(top_words)}")


if __name__ == "__main__":
    from preprocessing.dataset import load_phrasebank, preprocess_dataframe, split_data

    df    = load_phrasebank()
    df    = preprocess_dataframe(df)
    train, val, test = split_data(df)

    model = TfidfSVM()
    model.train(train)

    preds = model.predict(test["clean_text"].tolist())
    true  = test["label"].tolist()

    print("\nTF-IDF + SVM — Test Set Results")
    print("=" * 45)
    print(classification_report(
        true, preds,
        target_names=["bearish", "neutral", "bullish"]
    ))

    model.get_top_features()
    model.save()
