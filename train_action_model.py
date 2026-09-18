"""
train_action_model.py
----------------------
Trains the optional LSTM action classifier on a labeled skeleton-sequence
dataset (see dataset.py for the expected folder layout).

Usage:
    python -m src.train_action_model \
        --csv data/action_dataset/labels.csv \
        --seq_dir data/action_dataset/sequences \
        --epochs 30 --batch_size 16 --out models/action_lstm.pt

This step is OPTIONAL. The project runs fully with the rule-based
recognizer without any training. Train this only if you have (or collect)
labeled action clips and want higher accuracy / more action classes.
"""

import argparse
import torch
from torch.utils.data import DataLoader, random_split
from torch import nn, optim

from .dataset import PoseSequenceDataset
from .action_recognizer import LSTMActionClassifier, ACTIONS


def train(args):
    device = "cuda" if torch.cuda.is_available() else "cpu"
    dataset = PoseSequenceDataset(args.csv, args.seq_dir, seq_len=args.seq_len)

    val_size = max(1, int(0.15 * len(dataset)))
    train_size = len(dataset) - val_size
    train_ds, val_ds = random_split(dataset, [train_size, val_size])

    train_loader = DataLoader(train_ds, batch_size=args.batch_size, shuffle=True)
    val_loader = DataLoader(val_ds, batch_size=args.batch_size)

    model = LSTMActionClassifier(num_classes=len(ACTIONS)).to(device)
    optimizer = optim.Adam(model.parameters(), lr=args.lr)
    criterion = nn.CrossEntropyLoss()

    best_val_acc = 0.0
    for epoch in range(1, args.epochs + 1):
        model.train()
        total_loss = 0.0
        for x, y in train_loader:
            x, y = x.to(device), y.to(device)
            optimizer.zero_grad()
            logits = model(x)
            loss = criterion(logits, y)
            loss.backward()
            optimizer.step()
            total_loss += loss.item() * x.size(0)

        train_loss = total_loss / len(train_ds)
        val_acc = evaluate(model, val_loader, device)
        print(f"Epoch {epoch:03d} | train_loss={train_loss:.4f} | val_acc={val_acc:.3f}")

        if val_acc >= best_val_acc:
            best_val_acc = val_acc
            torch.save(model.state_dict(), args.out)
            print(f"  -> saved new best checkpoint to {args.out}")

    print(f"Training complete. Best val accuracy: {best_val_acc:.3f}")


@torch.no_grad()
def evaluate(model, loader, device):
    model.eval()
    correct, total = 0, 0
    for x, y in loader:
        x, y = x.to(device), y.to(device)
        preds = torch.argmax(model(x), dim=1)
        correct += (preds == y).sum().item()
        total += y.size(0)
    return correct / max(total, 1)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--csv", required=True)
    parser.add_argument("--seq_dir", required=True)
    parser.add_argument("--seq_len", type=int, default=30)
    parser.add_argument("--epochs", type=int, default=30)
    parser.add_argument("--batch_size", type=int, default=16)
    parser.add_argument("--lr", type=float, default=1e-3)
    parser.add_argument("--out", default="models/action_lstm.pt")
    args = parser.parse_args()
    train(args)
