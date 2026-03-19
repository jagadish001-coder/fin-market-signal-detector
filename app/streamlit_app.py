import sys, os
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import streamlit as st
import json
import torch
import pandas as pd
import plotly.graph_objects as go
import plotly.express as px
from preprocessing.dataset import load_phrasebank, preprocess_dataframe, split_data
from preprocessing.cleaner import FinancialCleaner

st.set_page_config(
    page_title="FinSignal — Market Signal Detector",
    page_icon="📈",
    layout="wide"
)

SIGNAL_COLOR = {"bearish": "#ef4444", "neutral": "#94a3b8", "bullish": "#22c55e"}
LABEL_MAP    = {0: "bearish", 1: "neutral", 2: "bullish"}
cleaner      = FinancialCleaner()


# ─── Load models (cached) ─────────────────────────────────────────────────────
@st.cache_resource
@st.cache_resource
def load_models():
    import models.bilstm as _bilstm_module
    sys.modules['__main__'].Vocabulary = _bilstm_module.Vocabulary

    from models.baseline_keywords import KeywordBaseline
    from models.tfidf_svm import TfidfSVM
    from models.bilstm import BiLSTMClassifier, Vocabulary, collate_fn
    from models.finbert import FinBERTClassifier
    from preprocessing.dataset import FinancialDataset
    from torch.utils.data import DataLoader

    device = torch.device("cpu")

    keyword = KeywordBaseline()

    svm = TfidfSVM()
    svm.load()

    vocab        = Vocabulary.load()
    glove_matrix = vocab.load_glove()

    bilstm = BiLSTMClassifier(
        vocab_size=len(vocab.word2idx), embed_dim=100,
        hidden_dim=128, num_layers=2, num_classes=3,
        dropout=0.3, pretrained_embeddings=glove_matrix,
        freeze_embeddings=False
    ).to(device)
    bilstm.load_state_dict(torch.load("models/saved/bilstm_best.pt", map_location=device))
    bilstm.eval()

    finbert = FinBERTClassifier(num_labels=3, device=device)
    finbert.load_best()
    finbert.model.eval()

    return keyword, svm, bilstm, vocab, collate_fn, finbert, device, DataLoader, FinancialDataset


@st.cache_data
def load_results():
    with open("evaluation/results.json") as f:
        return json.load(f)


def predict_all(headline):
    keyword, svm, bilstm, vocab, collate_fn, finbert, device, DataLoader, FinancialDataset = load_models()
    cleaned = cleaner.clean(headline)

    results = {}

    # keyword
    results["Keyword Baseline"] = LABEL_MAP[keyword.predict_one(cleaned)]

    # svm
    results["TF-IDF + SVM"] = LABEL_MAP[svm.predict([cleaned])[0]]

    # bilstm
    encode   = lambda t: vocab.encode(t, max_len=64)
    dummy_df = pd.DataFrame({"clean_text": [cleaned], "label": [0]})
    ds       = FinancialDataset(dummy_df, text_transform=encode)
    loader   = DataLoader(ds, batch_size=1, collate_fn=collate_fn)
    with torch.no_grad():
        for texts, lengths, _ in loader:
            logits = bilstm(texts.to(device), lengths.to(device))
            results["BiLSTM + GloVe"] = LABEL_MAP[logits.argmax(dim=1).item()]

    # finbert
    dummy_df2 = pd.DataFrame({"clean_text": [cleaned], "label": [0]})
    _, _, loader2 = finbert.get_loaders(dummy_df2, dummy_df2, dummy_df2)
    with torch.no_grad():
        for batch in loader2:
            outputs = finbert.model(
                input_ids=batch["input_ids"].to(device),
                attention_mask=batch["attention_mask"].to(device)
            )
            results["FinBERT"] = LABEL_MAP[outputs.logits.argmax(dim=1).item()]
            break

    return results, cleaned


# ─── UI ───────────────────────────────────────────────────────────────────────
st.title("📈 FinSignal — Financial Market Signal Detector")
st.caption("Classify financial news headlines as bullish, bearish or neutral using 4 NLP models")

tab1, tab2, tab3 = st.tabs(["Live Prediction", "Model Comparison", "Test Set Explorer"])


# ── Tab 1: Live Prediction ────────────────────────────────────────────────────
with tab1:
    st.subheader("Paste a financial headline")

    examples = [
        "Operating profit rose to EUR 13.1 mn from EUR 8.7 mn",
        "The company reported a net loss amid supply chain disruptions",
        "The board decided to maintain current dividend policy",
        "Revenue exceeded analyst expectations by 15 percent",
        "Restructuring charges impacted quarterly earnings significantly"
    ]

    selected = st.selectbox("Or pick an example:", [""] + examples)
    headline = st.text_input("Headline:", value=selected)

    if headline:
        with st.spinner("Running all 4 models..."):
            predictions, cleaned = predict_all(headline)

        st.markdown(f"**Cleaned text:** `{cleaned}`")
        st.divider()

        cols = st.columns(4)
        for col, (model_name, signal) in zip(cols, predictions.items()):
            color = SIGNAL_COLOR[signal]
            col.markdown(f"**{model_name}**")
            col.markdown(
                f"<div style='background:{color};color:white;padding:10px;"
                f"border-radius:8px;text-align:center;font-size:18px;"
                f"font-weight:bold'>{signal.upper()}</div>",
                unsafe_allow_html=True
            )

        # agreement indicator
        signals   = list(predictions.values())
        agreement = len(set(signals)) == 1
        st.divider()
        if agreement:
            st.success(f"All 4 models agree: **{signals[0].upper()}**")
        else:
            st.warning(f"Models disagree — majority signal: **{max(set(signals), key=signals.count).upper()}**")


# ── Tab 2: Model Comparison ───────────────────────────────────────────────────
with tab2:
    results = load_results()
    df_res  = pd.DataFrame([{
        "Model":      r["model"],
        "Accuracy":   r["accuracy"],
        "Macro F1":   r["macro_f1"],
        "Bearish F1": r["per_class"]["bearish"]["f1"],
        "Neutral F1": r["per_class"]["neutral"]["f1"],
        "Bullish F1": r["per_class"]["bullish"]["f1"],
        "ms/sample":  r["ms_per_sample"]
    } for r in results])

    st.subheader("Overall performance")
    col1, col2 = st.columns(2)

    with col1:
        fig = px.bar(
            df_res, x="Model", y="Macro F1",
            color="Macro F1", color_continuous_scale="Greens",
            title="Macro F1 by model"
        )
        fig.update_layout(showlegend=False)
        st.plotly_chart(fig, use_container_width=True)

    with col2:
        fig2 = px.bar(
            df_res, x="Model", y="ms/sample",
            color="ms/sample", color_continuous_scale="Reds",
            title="Inference time (ms per sample)"
        )
        fig2.update_layout(showlegend=False)
        st.plotly_chart(fig2, use_container_width=True)

    st.subheader("Per-class F1 breakdown")
    fig3 = go.Figure()
    for cls, color in [("Bearish F1","#ef4444"),("Neutral F1","#94a3b8"),("Bullish F1","#22c55e")]:
        fig3.add_trace(go.Bar(name=cls, x=df_res["Model"], y=df_res[cls], marker_color=color))
    fig3.update_layout(barmode="group", title="F1 score per class per model")
    st.plotly_chart(fig3, use_container_width=True)

    st.subheader("Full results table")
    st.dataframe(df_res.set_index("Model"), use_container_width=True)


# ── Tab 3: Test Set Explorer ──────────────────────────────────────────────────
with tab3:
    st.subheader("Browse test set predictions")

    @st.cache_data
    def get_test_predictions():
        from models.baseline_keywords import KeywordBaseline
        from models.tfidf_svm import TfidfSVM

        df   = load_phrasebank()
        df   = preprocess_dataframe(df)
        _, _, test_df = split_data(df)

        keyword = KeywordBaseline()
        svm     = TfidfSVM(); svm.load()

        test_df = test_df.copy()
        test_df["keyword_pred"] = [LABEL_MAP[p] for p in keyword.predict(test_df["clean_text"].tolist())]
        test_df["svm_pred"]     = [LABEL_MAP[p] for p in svm.predict(test_df["clean_text"].tolist())]
        test_df["true_signal"]  = test_df["label"].map(LABEL_MAP)
        return test_df

    test_df = get_test_predictions()

    filter_signal = st.selectbox("Filter by true signal:", ["All", "bullish", "neutral", "bearish"])
    filter_agree  = st.checkbox("Show only disagreements between Keyword and SVM")

    display_df = test_df.copy()
    if filter_signal != "All":
        display_df = display_df[display_df["true_signal"] == filter_signal]
    if filter_agree:
        display_df = display_df[display_df["keyword_pred"] != display_df["svm_pred"]]

    st.dataframe(
        display_df[["text", "true_signal", "keyword_pred", "svm_pred"]].head(50),
        use_container_width=True
    )
