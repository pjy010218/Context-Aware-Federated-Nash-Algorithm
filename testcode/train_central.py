"""
train_central.py
Centralized baseline training for CAFN experiments
---------------------------------------------------
Trains a single global (central) model on the pooled dataset
(created from all .dat files combined via run_experiment.py).

This gives the cooperative upper bound Φ_coop for CAFN evaluation.
"""

import argparse, os, json
import torch
import numpy as np
import pandas as pd
from models import MLPBackbone, Head
from data_utils import make_dataloader
from sklearn.datasets import load_svmlight_file
from sklearn.model_selection import train_test_split

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

def libsvm_to_dataframe(dat_path):
    X, y = load_svmlight_file(dat_path)
    X = X.toarray()
    df = pd.DataFrame(X, columns=[f"f{i}" for i in range(X.shape[1])])
    df["label"] = y.astype(int)
    return df

def combine_dat_files(dat_files, combined_path):
    dfs = [libsvm_to_dataframe(f) for f in dat_files]
    df = pd.concat(dfs, ignore_index=True)
    df.to_csv(combined_path, index=False)
    print(f"Combined {len(dat_files)} .dat files into {combined_path}")
    return combined_path

def train_central_model(train_csv, val_csv, test_csv,
                        input_dim=128, hidden_dims=[256,128], embed_dim=128,
                        epochs=100, batch_size=64, lr=1e-3, outpath="central_results.json"):
    print("🚀 Starting centralized baseline training")

    backbone = MLPBackbone(input_dim=input_dim, hidden_dims=hidden_dims, embed_dim=embed_dim).to(device)
    num_labels = len(pd.read_csv(train_csv)['label'].unique())
    head = Head(embed_dim=embed_dim, out_dim=num_labels).to(device)

    train_loader, _ = make_dataloader(train_csv, batch_size=batch_size, shuffle=True)
    val_loader, _ = make_dataloader(val_csv, batch_size=batch_size, shuffle=False)
    test_loader, _ = make_dataloader(test_csv, batch_size=batch_size, shuffle=False)

    opt = torch.optim.Adam(list(backbone.parameters()) + list(head.parameters()), lr=lr)
    loss_fn = torch.nn.CrossEntropyLoss()

    best_val = 0.0
    for epoch in range(epochs):
        backbone.train(); head.train()
        total_loss = 0.0
        for xb, yb in train_loader:
            xb, yb = xb.to(device), yb.to(device)
            opt.zero_grad()
            logits = head(backbone(xb))
            loss = loss_fn(logits, yb)
            loss.backward()
            opt.step()
            total_loss += loss.item() * len(yb)
        total_loss /= len(train_loader.dataset)
        # validation
        correct, total = 0, 0
        backbone.eval(); head.eval()
        with torch.no_grad():
            for xb, yb in val_loader:
                xb, yb = xb.to(device), yb.to(device)
                preds = head(backbone(xb)).argmax(dim=1)
                correct += (preds == yb).sum().item()
                total += len(yb)
        val_acc = correct / total if total > 0 else 0.0
        if val_acc > best_val:
            best_val = val_acc
            torch.save({"backbone": backbone.state_dict(), "head": head.state_dict()}, "best_central_model.pt")
        if epoch % 10 == 0 or epoch == epochs-1:
            print(f"[Epoch {epoch}] train_loss={total_loss:.4f}, val_acc={val_acc:.4f}")

    # reload best model
    ckpt = torch.load("best_central_model.pt", map_location=device)
    backbone.load_state_dict(ckpt["backbone"])
    head.load_state_dict(ckpt["head"])

    # test performance
    backbone.eval(); head.eval()
    correct, total = 0, 0
    with torch.no_grad():
        for xb, yb in test_loader:
            xb, yb = xb.to(device), yb.to(device)
            preds = head(backbone(xb)).argmax(dim=1)
            correct += (preds == yb).sum().item()
            total += len(yb)
    test_acc = correct / total if total > 0 else 0.0
    print(f"✅ Centralized Test Accuracy: {test_acc:.4f}")

    results = {"val_acc_best": best_val, "test_acc": test_acc}
    with open(outpath, "w") as f:
        json.dump(results, f, indent=2)
    print(f"📁 Saved centralized results to {outpath}")
    return results


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--dat_files", nargs="*", type=str, help="List of .dat files to combine")
    parser.add_argument("--data_dir", type=str, default="./data")
    parser.add_argument("--epochs", type=int, default=100)
    parser.add_argument("--batch_size", type=int, default=64)
    parser.add_argument("--lr", type=float, default=1e-3)
    parser.add_argument("--outpath", type=str, default="./results/central_results.json")
    args = parser.parse_args()

    os.makedirs(args.data_dir, exist_ok=True)
    combined_path = os.path.join(args.data_dir, "combined.dat")
    if args.dat_files:
        combine_dat_files(args.dat_files, combined_path)
    else:
        print("No .dat files provided, using existing combined.dat")
    # Reuse same preprocessing (shared label space)
    df = pd.read_csv(combined_path)
    df["label"] = df["label"] - df["label"].min()
    train, test = train_test_split(df, test_size=0.2, stratify=df["label"], random_state=42)
    val, test = train_test_split(test, test_size=0.5, stratify=test["label"], random_state=42)
    train_csv = os.path.join(args.data_dir, "central_train.csv")
    val_csv = os.path.join(args.data_dir, "central_val.csv")
    test_csv = os.path.join(args.data_dir, "central_test.csv")
    train.to_csv(train_csv, index=False)
    val.to_csv(val_csv, index=False)
    test.to_csv(test_csv, index=False)

    train_central_model(train_csv, val_csv, test_csv,
                        epochs=args.epochs, batch_size=args.batch_size, lr=args.lr,
                        outpath=args.outpath)
