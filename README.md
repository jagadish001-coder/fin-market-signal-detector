# FinSignal — Financial Market Signal Detector

An end-to-end NLP pipeline that classifies financial news headlines 
as **bullish**, **bearish**, or **neutral** using 4 comparative models.

## Results

| Model | Macro F1 | Accuracy | ms/sample |
|---|---|---|---|
| Keyword Baseline | 0.64 | 0.75 | 0.02 |
| TF-IDF + SVM | 0.90 | 0.92 | 0.04 |
| BiLSTM + GloVe | 0.78 | 0.84 | 0.61 |
| FinBERT (fine-tuned) | 0.98 | 0.98 | 94.1 |

## Models
- **Keyword Baseline** — rule-based lexicon with negation handling
- **TF-IDF + SVM** — classical ML with LinearSVC, bigram features
- **BiLSTM + GloVe** — bidirectional LSTM with pretrained GloVe embeddings
- **FinBERT** — ProsusAI/finbert fine-tuned, 13.5% parameters trainable

## Key Finding
SVM outperformed BiLSTM (0.90 vs 0.78 macro F1) on this 1584-sample 
dataset — confirming that deep learning requires scale to outperform 
classical ML. FinBERT's financial domain pretraining justified its 
complexity cost, reaching 0.98 macro F1.

## Dataset
Financial PhraseBank — 2264 sentences labeled by finance experts  
(sentences_allagree subset, all annotators in agreement)

## Tech Stack
Python · PyTorch · HuggingFace Transformers · scikit-learn · 
NLTK · Streamlit · Plotly

## Setup
```bash
python -m venv venv
source venv/bin/activate
pip install -r requirements.txt
```

## Run Dashboard
```bash
streamlit run app/streamlit_app.py
```

## Run Model Comparison
```bash
python -m evaluation.compare
```

## Project Structure
```
fin-market-signal-detector/
├── preprocessing/    # cleaner, dataset loader
├── models/           # 4 models
├── evaluation/       # comparison engine + results
├── app/              # Streamlit dashboard
└── data/             # Financial PhraseBank + GloVe
```