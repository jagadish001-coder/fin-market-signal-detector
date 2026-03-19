# FinSignal — Financial News Market Signal Detector

An NLP pipeline that classifies financial news headlines as
bullish, bearish, or neutral using 4 comparative models.

## Models
- Keyword baseline (rule-based)
- TF-IDF + SVM (classical ML)
- BiLSTM + GloVe (deep learning)
- FinBERT fine-tuned (transformer)

## Dataset
Financial PhraseBank — 4500 sentences labeled by finance experts

## Tech Stack
Python · PyTorch · HuggingFace · spaCy · FastAPI · Streamlit
