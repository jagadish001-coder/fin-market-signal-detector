import sys, os
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import re
import pandas as pd
from sklearn.metrics import classification_report

BULLISH_WORDS = {
    "profit", "growth", "increase", "increased", "increases", "gain",
    "gains", "rose", "rise", "rises", "surged", "surge", "beat",
    "beats", "exceeded", "exceed", "strong", "record", "high",
    "improved", "improve", "improvement", "positive", "up", "raised",
    "raises", "expand", "expanded", "expansion", "outperform",
    "upgrade", "upgraded", "buy", "bullish", "rally", "recover",
    "recovery", "boost", "boosted", "higher", "upside", "revenue",
    "earnings", "margin", "dividend", "acquisition", "deal"
}

BEARISH_WORDS = {
    "loss", "losses", "decline", "declined", "declines", "fell",
    "fall", "falls", "drop", "dropped", "drops", "miss", "missed",
    "misses", "weak", "weakness", "cut", "cuts", "cutting", "reduced",
    "reduce", "reduction", "down", "lower", "lowered", "below",
    "concern", "concerns", "risk", "risks", "warning", "warn",
    "warns", "downgrade", "downgraded", "sell", "bearish", "crash",
    "fear", "fears", "debt", "deficit", "layoff", "layoffs",
    "restructuring", "bankruptcy", "fraud", "investigation", "penalty",
    "fine", "lawsuit", "recall", "shortage", "disruption", "delay"
}

NEGATION_WORDS = {"not", "no", "never", "neither", "nor", "without"}


class KeywordBaseline:
    """
    Rule-based keyword baseline.
    Counts bullish vs bearish words with basic negation handling.
    Predicts: 0=bearish, 1=neutral, 2=bullish
    """

    def __init__(self, negation_window=2):
        self.negation_window = negation_window

    def _get_negated_positions(self, tokens):
        negated = set()
        for i, token in enumerate(tokens):
            if token in NEGATION_WORDS:
                for j in range(i+1, min(i+1+self.negation_window, len(tokens))):
                    negated.add(j)
        return negated

    def predict_one(self, text: str) -> int:
        tokens = text.lower().split()
        negated = self._get_negated_positions(tokens)

        bullish_score = 0
        bearish_score = 0

        for i, token in enumerate(tokens):
            token_clean = re.sub(r"[^\w]", "", token)
            is_negated = i in negated

            if token_clean in BULLISH_WORDS:
                if is_negated:
                    bearish_score += 1
                else:
                    bullish_score += 1

            elif token_clean in BEARISH_WORDS:
                if is_negated:
                    bullish_score += 1
                else:
                    bearish_score += 1

        if bullish_score == 0 and bearish_score == 0:
            return 1   # neutral — no signal words found
        if bullish_score > bearish_score:
            return 2   # bullish
        if bearish_score > bullish_score:
            return 0   # bearish
        return 1       # tie → neutral

    def predict(self, texts):
        return [self.predict_one(t) for t in texts]


if __name__ == "__main__":
    from preprocessing.dataset import load_phrasebank, preprocess_dataframe, split_data

    df    = load_phrasebank()
    df    = preprocess_dataframe(df)
    train, val, test = split_data(df)

    model = KeywordBaseline()
    preds = model.predict(test["clean_text"].tolist())
    true  = test["label"].tolist()

    print("\nKeyword Baseline — Test Set Results")
    print("=" * 45)
    print(classification_report(
        true, preds,
        target_names=["bearish", "neutral", "bullish"]
    ))

    # show some examples
    print("\nSample predictions:")
    for i in range(5):
        text   = test["text"].iloc[i]
        pred   = preds[i]
        actual = true[i]
        label_map = {0: "bearish", 1: "neutral", 2: "bullish"}
        status = "OK" if pred == actual else "WRONG"
        print(f"  [{status}] actual={label_map[actual]} pred={label_map[pred]}")
        print(f"       {text[:80]}...")
        print()
