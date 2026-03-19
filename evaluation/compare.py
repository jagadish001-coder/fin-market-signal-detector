import sys, os
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import time
import json
import torch
import pandas as pd
from sklearn.metrics import classification_report, confusion_matrix
from preprocessing.dataset import load_phrasebank, preprocess_dataframe, split_data

# import at module level so pickle can find them
from models.baseline_keywords import KeywordBaseline
from models.tfidf_svm import TfidfSVM
from models.bilstm import BiLSTMClassifier, Vocabulary, collate_fn
from models.finbert import FinBERTClassifier, FinBERTDataset
from preprocessing.dataset import FinancialDataset

LABEL_NAMES = ["bearish", "neutral", "bullish"]


def evaluate_model(name, predict_fn, texts, labels):
    start   = time.time()
    preds   = predict_fn(texts)
    elapsed = time.time() - start

    report = classification_report(
        labels, preds,
        target_names=LABEL_NAMES,
        output_dict=True
    )
    cm = confusion_matrix(labels, preds)

    results = {
        "model":           name,
        "accuracy":        round(report["accuracy"], 4),
        "macro_f1":        round(report["macro avg"]["f1-score"], 4),
        "macro_precision": round(report["macro avg"]["precision"], 4),
        "macro_recall":    round(report["macro avg"]["recall"], 4),
        "inference_sec":   round(elapsed, 3),
        "ms_per_sample":   round(1000 * elapsed / len(texts), 3),
        "per_class": {
            cls: {
                "precision": round(report[cls]["precision"], 4),
                "recall":    round(report[cls]["recall"],    4),
                "f1":        round(report[cls]["f1-score"],  4),
            }
            for cls in LABEL_NAMES
        },
        "confusion_matrix": cm.tolist(),
        "predictions":      preds
    }
    return results


def load_all_models(device):

    # 1. Keyword baseline
    keyword_model = KeywordBaseline()

    # 2. TF-IDF + SVM
    svm_model = TfidfSVM()
    svm_model.load()

    # 3. BiLSTM
    from torch.utils.data import DataLoader
    vocab        = Vocabulary.load()
    glove_matrix = vocab.load_glove()

    bilstm_model = BiLSTMClassifier(
        vocab_size=len(vocab.word2idx),
        embed_dim=100,
        hidden_dim=128,
        num_layers=2,
        num_classes=3,
        dropout=0.3,
        pretrained_embeddings=glove_matrix,
        freeze_embeddings=False
    ).to(device)
    bilstm_model.load_state_dict(
        torch.load("models/saved/bilstm_best.pt", map_location=device)
    )
    bilstm_model.eval()

    def bilstm_predict(texts):
        encode    = lambda t: vocab.encode(t, max_len=64)
        dummy_df  = pd.DataFrame({"clean_text": texts, "label": [0]*len(texts)})
        ds        = FinancialDataset(dummy_df, text_transform=encode)
        loader    = DataLoader(ds, batch_size=32, collate_fn=collate_fn)
        preds     = []
        with torch.no_grad():
            for batch_texts, lengths, _ in loader:
                logits = bilstm_model(batch_texts.to(device), lengths.to(device))
                preds.extend(logits.argmax(dim=1).cpu().tolist())
        return preds

    # 4. FinBERT
    finbert = FinBERTClassifier(num_labels=3, device=device)
    finbert.load_best()
    finbert.model.eval()

    def finbert_predict(texts):
        dummy_df = pd.DataFrame({"clean_text": texts, "label": [0]*len(texts)})
        _, _, loader = finbert.get_loaders(dummy_df, dummy_df, dummy_df)
        preds = []
        with torch.no_grad():
            for batch in loader:
                outputs = finbert.model(
                    input_ids=batch["input_ids"].to(device),
                    attention_mask=batch["attention_mask"].to(device)
                )
                preds.extend(outputs.logits.argmax(dim=1).cpu().tolist())
        return preds

    return {
        "Keyword Baseline": keyword_model.predict,
        "TF-IDF + SVM":     svm_model.predict,
        "BiLSTM + GloVe":   bilstm_predict,
        "FinBERT":          finbert_predict,
    }


def run_comparison():
    device = torch.device("cpu")

    df = load_phrasebank()
    df = preprocess_dataframe(df)
    _, _, test_df = split_data(df)

    texts  = test_df["clean_text"].tolist()
    labels = test_df["label"].tolist()

    print("Loading all models...")
    models = load_all_models(device)

    all_results = []
    for name, predict_fn in models.items():
        print(f"\nEvaluating: {name}...")
        result = evaluate_model(name, predict_fn, texts, labels)
        all_results.append(result)
        print(f"  Accuracy:  {result['accuracy']}")
        print(f"  Macro F1:  {result['macro_f1']}")
        print(f"  Inference: {result['ms_per_sample']} ms/sample")

    print("\n" + "=" * 65)
    print(f"{'Model':<20} {'Acc':>6} {'MacroF1':>8} {'Bearish':>8} {'Neutral':>8} {'Bullish':>8} {'ms/sample':>10}")
    print("-" * 65)
    for r in all_results:
        print(
            f"{r['model']:<20} "
            f"{r['accuracy']:>6.3f} "
            f"{r['macro_f1']:>8.3f} "
            f"{r['per_class']['bearish']['f1']:>8.3f} "
            f"{r['per_class']['neutral']['f1']:>8.3f} "
            f"{r['per_class']['bullish']['f1']:>8.3f} "
            f"{r['ms_per_sample']:>10.1f}"
        )
    print("=" * 65)

    os.makedirs("evaluation", exist_ok=True)
    save_data = [{k: v for k, v in r.items() if k != "predictions"} for r in all_results]
    with open("evaluation/results.json", "w") as f:
        json.dump(save_data, f, indent=2)
    print(f"\nResults saved to evaluation/results.json")

    return all_results


if __name__ == "__main__":
    run_comparison()
