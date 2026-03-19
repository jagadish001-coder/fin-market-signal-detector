import sys, os
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import torch
from torch.utils.data import Dataset, DataLoader
from sklearn.model_selection import train_test_split
from preprocessing.cleaner import FinancialCleaner
import pandas as pd

LABEL2ID = {"negative": 0, "neutral": 1, "positive": 2}
ID2LABEL  = {0: "negative", 1: "neutral", 2: "positive"}
SIGNAL_MAP = {0: "bearish", 1: "neutral", 2: "bullish"}

def load_phrasebank(path="data/raw/FinancialPhraseBank-v1.0/Sentences_AllAgree.txt"):
    rows = []
    with open(path, encoding="latin-1") as f:
        for line in f:
            line = line.strip()
            if "@" not in line:
                continue
            text, label = line.rsplit("@", 1)
            label = label.strip().lower()
            if label not in LABEL2ID:
                continue
            rows.append({"text": text.strip(), "label": LABEL2ID[label]})
    df = pd.DataFrame(rows)
    df["signal"] = df["label"].map({0: "bearish", 1: "neutral", 2: "bullish"})
    print(f"Loaded {len(df)} samples")
    print(f"Label distribution:\n{df['signal'].value_counts()}")
    return df

def preprocess_dataframe(df, remove_stopwords=False):
    cleaner = FinancialCleaner(expand_abbrev=True, remove_stopwords=remove_stopwords)
    df = df.copy()
    df["clean_text"] = df["text"].apply(cleaner.clean)
    return df

def split_data(df, test_size=0.2, val_size=0.1, random_state=42):
    train_val, test = train_test_split(df, test_size=test_size, stratify=df["label"], random_state=random_state)
    val_ratio = val_size / (1 - test_size)
    train, val = train_test_split(train_val, test_size=val_ratio, stratify=train_val["label"], random_state=random_state)
    print(f"Train: {len(train)} | Val: {len(val)} | Test: {len(test)}")
    return train.reset_index(drop=True), val.reset_index(drop=True), test.reset_index(drop=True)

class FinancialDataset(Dataset):
    def __init__(self, df, text_col="clean_text", text_transform=None):
        self.texts     = df[text_col].tolist()
        self.labels    = df["label"].tolist()
        self.transform = text_transform

    def __len__(self):
        return len(self.texts)

    def __getitem__(self, idx):
        text  = self.texts[idx]
        label = self.labels[idx]
        if self.transform:
            text = self.transform(text)
        return text, torch.tensor(label, dtype=torch.long)

def get_dataloaders(train_df, val_df, test_df, text_transform=None, batch_size=32, text_col="clean_text"):
    train_ds = FinancialDataset(train_df, text_col, text_transform)
    val_ds   = FinancialDataset(val_df,   text_col, text_transform)
    test_ds  = FinancialDataset(test_df,  text_col, text_transform)
    train_loader = DataLoader(train_ds, batch_size=batch_size, shuffle=True)
    val_loader   = DataLoader(val_ds,   batch_size=batch_size, shuffle=False)
    test_loader  = DataLoader(test_ds,  batch_size=batch_size, shuffle=False)
    return train_loader, val_loader, test_loader

if __name__ == "__main__":
    df = load_phrasebank()
    df = preprocess_dataframe(df)
    train, val, test = split_data(df)
    print("\nSample cleaned rows:")
    print(train[["text", "clean_text", "signal"]].head(3).to_string())
    ds = FinancialDataset(train)
    text, label = ds[0]
    print(f"\nDataset sample — text: '{text}' | label: {label}")
