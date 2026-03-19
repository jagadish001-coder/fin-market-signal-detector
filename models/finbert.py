import sys, os
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader
from transformers import AutoTokenizer, AutoModelForSequenceClassification, get_linear_schedule_with_warmup
from sklearn.metrics import classification_report
import numpy as np

MODEL_NAME = "ProsusAI/finbert"
MAX_LEN    = 128
BATCH_SIZE = 16


# ─── Dataset ──────────────────────────────────────────────────────────────────
class FinBERTDataset(Dataset):
    """
    Tokenizes text using FinBERT tokenizer.
    Returns input_ids, attention_mask, label.
    """

    def __init__(self, texts, labels, tokenizer, max_len=MAX_LEN):
        self.texts     = texts
        self.labels    = labels
        self.tokenizer = tokenizer
        self.max_len   = max_len

    def __len__(self):
        return len(self.texts)

    def __getitem__(self, idx):
        encoding = self.tokenizer(
            self.texts[idx],
            max_length=self.max_len,
            padding="max_length",
            truncation=True,
            return_tensors="pt"
        )
        return {
            "input_ids":      encoding["input_ids"].squeeze(0),
            "attention_mask": encoding["attention_mask"].squeeze(0),
            "label":          torch.tensor(self.labels[idx], dtype=torch.long)
        }


# ─── Trainer ──────────────────────────────────────────────────────────────────
class FinBERTClassifier:
    """
    Fine-tunes ProsusAI/finbert for 3-class financial sentiment.
    FinBERT is already pretrained on financial text —
    we only fine-tune the classification head + last 2 transformer layers.
    """

    def __init__(self, num_labels=3, device=None):
        self.device    = device or torch.device("cpu")
        self.tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME)
        self.model     = AutoModelForSequenceClassification.from_pretrained(
            MODEL_NAME,
            num_labels=num_labels,
            ignore_mismatched_sizes=True
        ).to(self.device)

        # freeze all layers except last 2 transformer blocks + classifier
        # this speeds up training significantly on CPU
        self._freeze_layers()

    def _freeze_layers(self):
        # freeze embeddings
        for param in self.model.bert.embeddings.parameters():
            param.requires_grad = False

        # freeze first 10 of 12 transformer layers
        for i, layer in enumerate(self.model.bert.encoder.layer):
            if i < 10:
                for param in layer.parameters():
                    param.requires_grad = False

        total     = sum(p.numel() for p in self.model.parameters())
        trainable = sum(p.numel() for p in self.model.parameters() if p.requires_grad)
        print(f"Total params: {total:,} | Trainable: {trainable:,} ({100*trainable/total:.1f}%)")

    def get_loaders(self, train_df, val_df, test_df):
        def make_loader(df, shuffle):
            ds = FinBERTDataset(
                df["clean_text"].tolist(),
                df["label"].tolist(),
                self.tokenizer
            )
            return DataLoader(ds, batch_size=BATCH_SIZE, shuffle=shuffle)

        return (
            make_loader(train_df, shuffle=True),
            make_loader(val_df,   shuffle=False),
            make_loader(test_df,  shuffle=False)
        )

    def train(self, train_loader, val_loader, epochs=4):
        # class weights for imbalance
        class_weights = torch.tensor([2.0, 0.5, 1.0]).to(self.device)
        criterion     = nn.CrossEntropyLoss(weight=class_weights)

        optimizer = torch.optim.AdamW(
            filter(lambda p: p.requires_grad, self.model.parameters()),
            lr=2e-5,
            weight_decay=0.01
        )

        total_steps = len(train_loader) * epochs
        scheduler   = get_linear_schedule_with_warmup(
            optimizer,
            num_warmup_steps=total_steps // 10,
            num_training_steps=total_steps
        )

        best_val_acc = 0
        for epoch in range(epochs):
            # train
            self.model.train()
            total_loss, correct, total = 0, 0, 0

            for batch in train_loader:
                input_ids      = batch["input_ids"].to(self.device)
                attention_mask = batch["attention_mask"].to(self.device)
                labels         = batch["label"].to(self.device)

                optimizer.zero_grad()
                outputs = self.model(input_ids=input_ids, attention_mask=attention_mask)
                loss    = criterion(outputs.logits, labels)
                loss.backward()
                nn.utils.clip_grad_norm_(self.model.parameters(), 1.0)
                optimizer.step()
                scheduler.step()

                total_loss += loss.item()
                preds       = outputs.logits.argmax(dim=1)
                correct    += (preds == labels).sum().item()
                total      += labels.size(0)

            train_acc = correct / total

            # validate
            val_preds, val_labels = self.evaluate(val_loader)
            val_acc = sum(p == l for p, l in zip(val_preds, val_labels)) / len(val_labels)

            print(f"Epoch {epoch+1}/{epochs} | loss: {total_loss/len(train_loader):.4f} | train_acc: {train_acc:.3f} | val_acc: {val_acc:.3f}")

            if val_acc > best_val_acc:
                best_val_acc = val_acc
                os.makedirs("models/saved", exist_ok=True)
                torch.save(self.model.state_dict(), "models/saved/finbert_best.pt")
                print(f"  Saved best model (val_acc: {val_acc:.3f})")

    def evaluate(self, loader):
        self.model.eval()
        all_preds, all_labels = [], []

        with torch.no_grad():
            for batch in loader:
                input_ids      = batch["input_ids"].to(self.device)
                attention_mask = batch["attention_mask"].to(self.device)
                labels         = batch["label"]
                outputs        = self.model(input_ids=input_ids, attention_mask=attention_mask)
                preds          = outputs.logits.argmax(dim=1).cpu().tolist()
                all_preds.extend(preds)
                all_labels.extend(labels.tolist())

        return all_preds, all_labels

    def load_best(self):
        self.model.load_state_dict(
            torch.load("models/saved/finbert_best.pt", map_location=self.device)
        )


if __name__ == "__main__":
    from preprocessing.dataset import load_phrasebank, preprocess_dataframe, split_data

    device = torch.device("cpu")
    print(f"Using device: {device}")

    df = load_phrasebank()
    df = preprocess_dataframe(df)
    train_df, val_df, test_df = split_data(df)

    classifier = FinBERTClassifier(num_labels=3, device=device)

    train_loader, val_loader, test_loader = classifier.get_loaders(
        train_df, val_df, test_df
    )

    print("\nFine-tuning FinBERT...")
    classifier.train(train_loader, val_loader, epochs=4)

    print("\nLoading best checkpoint...")
    classifier.load_best()

    test_preds, test_labels = classifier.evaluate(test_loader)

    print("\nFinBERT — Test Set Results")
    print("=" * 45)
    print(classification_report(
        test_labels, test_preds,
        target_names=["bearish", "neutral", "bullish"]
    ))
