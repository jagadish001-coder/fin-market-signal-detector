import sys, os
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import torch
import torch.nn as nn
import numpy as np
from collections import Counter
from sklearn.metrics import classification_report
import pickle

GLOVE_PATH = "data/embeddings/glove.6B.100d.txt"
EMBED_DIM  = 100
PAD_TOKEN  = "<PAD>"
UNK_TOKEN  = "<UNK>"


# ─── Vocabulary ───────────────────────────────────────────────────────────────
class Vocabulary:
    """
    Builds word → index mapping from training data.
    Loads GloVe vectors for known words.
    Unknown words get the mean of all GloVe vectors.
    """

    def __init__(self, min_freq=1):
        self.min_freq  = min_freq
        self.word2idx  = {PAD_TOKEN: 0, UNK_TOKEN: 1}
        self.idx2word  = {0: PAD_TOKEN, 1: UNK_TOKEN}
        self.embeddings = None

    def build(self, texts):
        counter = Counter()
        for text in texts:
            counter.update(text.lower().split())

        for word, freq in counter.items():
            if freq >= self.min_freq and word not in self.word2idx:
                idx = len(self.word2idx)
                self.word2idx[word] = idx
                self.idx2word[idx]  = word

        print(f"Vocabulary size: {len(self.word2idx)}")

    def load_glove(self, glove_path=GLOVE_PATH):
        print(f"Loading GloVe from {glove_path}...")
        glove = {}
        with open(glove_path, encoding="utf-8") as f:
            for line in f:
                parts = line.split()
                word  = parts[0]
                vec   = np.array(parts[1:], dtype=np.float32)
                glove[word] = vec

        vocab_size = len(self.word2idx)
        matrix     = np.zeros((vocab_size, EMBED_DIM), dtype=np.float32)

        # mean vector for unknown words
        all_vecs  = np.stack(list(glove.values()))
        mean_vec  = all_vecs.mean(axis=0)

        found = 0
        for word, idx in self.word2idx.items():
            if word in glove:
                matrix[idx] = glove[word]
                found += 1
            else:
                matrix[idx] = mean_vec

        print(f"GloVe coverage: {found}/{vocab_size} words ({100*found/vocab_size:.1f}%)")
        self.embeddings = matrix
        return matrix

    def encode(self, text, max_len=64):
        tokens = text.lower().split()[:max_len]
        ids    = [self.word2idx.get(t, 1) for t in tokens]  # 1 = UNK
        return ids

    def save(self, path="models/saved/vocabulary.pkl"):
        os.makedirs(os.path.dirname(path), exist_ok=True)
        with open(path, "wb") as f:
            pickle.dump(self, f)

    @staticmethod
    def load(path="models/saved/vocabulary.pkl"):
        with open(path, "rb") as f:
            return pickle.load(f)


# ─── Collate function for variable length sequences ───────────────────────────
def collate_fn(batch):
    texts, labels = zip(*batch)
    lengths = [len(t) for t in texts]
    max_len = max(lengths)
    padded  = [t + [0] * (max_len - len(t)) for t in texts]
    return (
        torch.tensor(padded,  dtype=torch.long),
        torch.tensor(lengths, dtype=torch.long),
        torch.tensor(labels,  dtype=torch.long)
    )


# ─── BiLSTM Model ─────────────────────────────────────────────────────────────
class BiLSTMClassifier(nn.Module):
    """
    Bidirectional LSTM for financial sentiment classification.
    Architecture:
        Embedding (GloVe pretrained, frozen initially)
        → BiLSTM (2 layers)
        → Dropout
        → Linear classifier
    """

    def __init__(self, vocab_size, embed_dim=100, hidden_dim=128,
                 num_layers=2, num_classes=3, dropout=0.3,
                 pretrained_embeddings=None, freeze_embeddings=True):

        super().__init__()

        # embedding layer
        self.embedding = nn.Embedding(vocab_size, embed_dim, padding_idx=0)
        if pretrained_embeddings is not None:
            self.embedding.weight.data.copy_(
                torch.tensor(pretrained_embeddings, dtype=torch.float)
            )
            if freeze_embeddings:
                self.embedding.weight.requires_grad = False

        # BiLSTM — bidirectional=True doubles the output size
        self.lstm = nn.LSTM(
            input_size=embed_dim,
            hidden_size=hidden_dim,
            num_layers=num_layers,
            bidirectional=True,
            dropout=dropout if num_layers > 1 else 0,
            batch_first=True
        )

        self.dropout    = nn.Dropout(dropout)

        # hidden_dim * 2 because bidirectional concatenates forward + backward
        self.classifier = nn.Linear(hidden_dim * 2, num_classes)

    def forward(self, x, lengths):
        embedded = self.dropout(self.embedding(x))

        # pack for efficiency — ignores padding during LSTM computation
        packed = nn.utils.rnn.pack_padded_sequence(
            embedded, lengths.cpu(), batch_first=True, enforce_sorted=False
        )
        output, (hidden, _) = self.lstm(packed)

        # take final forward and backward hidden states
        # hidden shape: [num_layers * 2, batch, hidden_dim]
        forward_final  = hidden[-2]   # last layer, forward direction
        backward_final = hidden[-1]   # last layer, backward direction
        sentence_vec   = torch.cat([forward_final, backward_final], dim=1)

        sentence_vec = self.dropout(sentence_vec)
        logits       = self.classifier(sentence_vec)
        return logits


# ─── Training + Evaluation ────────────────────────────────────────────────────
def train_epoch(model, loader, optimizer, criterion, device):
    model.train()
    total_loss, correct, total = 0, 0, 0

    for texts, lengths, labels in loader:
        texts, lengths, labels = texts.to(device), lengths.to(device), labels.to(device)
        optimizer.zero_grad()
        logits = model(texts, lengths)
        loss   = criterion(logits, labels)
        loss.backward()
        nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
        optimizer.step()

        total_loss += loss.item()
        preds       = logits.argmax(dim=1)
        correct    += (preds == labels).sum().item()
        total      += labels.size(0)

    return total_loss / len(loader), correct / total


def evaluate(model, loader, device):
    model.eval()
    all_preds, all_labels = [], []

    with torch.no_grad():
        for texts, lengths, labels in loader:
            texts, lengths = texts.to(device), lengths.to(device)
            logits = model(texts, lengths)
            preds  = logits.argmax(dim=1).cpu().tolist()
            all_preds.extend(preds)
            all_labels.extend(labels.tolist())

    return all_preds, all_labels


if __name__ == "__main__":
    from torch.utils.data import DataLoader
    from preprocessing.dataset import load_phrasebank, preprocess_dataframe, split_data, FinancialDataset

    device = torch.device("cpu")
    print(f"Using device: {device}")

    # load data
    df = load_phrasebank()
    df = preprocess_dataframe(df)
    train_df, val_df, test_df = split_data(df)

    # build vocabulary
    vocab = Vocabulary(min_freq=1)
    vocab.build(train_df["clean_text"].tolist())
    glove_matrix = vocab.load_glove()
    vocab.save()

    # encode function for dataset
    def encode(text):
        return vocab.encode(text, max_len=64)

    # datasets and loaders
    train_ds = FinancialDataset(train_df, text_transform=encode)
    val_ds   = FinancialDataset(val_df,   text_transform=encode)
    test_ds  = FinancialDataset(test_df,  text_transform=encode)

    train_loader = DataLoader(train_ds, batch_size=32, shuffle=True,  collate_fn=collate_fn)
    val_loader   = DataLoader(val_ds,   batch_size=32, shuffle=False, collate_fn=collate_fn)
    test_loader  = DataLoader(test_ds,  batch_size=32, shuffle=False, collate_fn=collate_fn)

    # build model
    model = BiLSTMClassifier(
        vocab_size=len(vocab.word2idx),
        embed_dim=EMBED_DIM,
        hidden_dim=128,
        num_layers=2,
        num_classes=3,
        dropout=0.3,
        pretrained_embeddings=glove_matrix,
        freeze_embeddings=True
    ).to(device)

    # class weights to handle imbalance
    class_weights = torch.tensor([2.0, 0.5, 1.0], dtype=torch.float)
    criterion  = nn.CrossEntropyLoss(weight=class_weights)
    optimizer  = torch.optim.Adam(
        filter(lambda p: p.requires_grad, model.parameters()),
        lr=1e-3
    )
    scheduler  = torch.optim.lr_scheduler.StepLR(optimizer, step_size=3, gamma=0.5)

    # training loop
    print("\nTraining BiLSTM...")
    best_val_acc = 0
    for epoch in range(10):
        train_loss, train_acc = train_epoch(model, train_loader, optimizer, criterion, device)
        val_preds, val_labels = evaluate(model, val_loader, device)
        val_acc = sum(p == l for p, l in zip(val_preds, val_labels)) / len(val_labels)
        scheduler.step()

        print(f"Epoch {epoch+1:2d} | loss: {train_loss:.4f} | train_acc: {train_acc:.3f} | val_acc: {val_acc:.3f}")

        if val_acc > best_val_acc:
            best_val_acc = val_acc
            os.makedirs("models/saved", exist_ok=True)
            torch.save(model.state_dict(), "models/saved/bilstm_best.pt")

    # load best and evaluate on test
    model.load_state_dict(torch.load("models/saved/bilstm_best.pt"))
    test_preds, test_labels = evaluate(model, test_loader, device)

    print("\nBiLSTM — Test Set Results")
    print("=" * 45)
    print(classification_report(
        test_labels, test_preds,
        target_names=["bearish", "neutral", "bullish"]
    ))
