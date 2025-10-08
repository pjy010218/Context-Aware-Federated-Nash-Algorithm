"""
diagnostics.py — CAFN pre-training diagnostics
Usage:
    python diagnostics.py --data_dir ./data --dat_files batch1.dat batch2.dat ... batch10.dat
Outputs:
    - CSV sample counts per client
    - Feature mean/variance histograms
    - Context vector variance plot
"""

import argparse, os
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from cafn import compute_context, gaussian_dp_noise
from models import MLPBackbone
from data_utils import make_dataloader
from sklearn.datasets import load_svmlight_file

def preprocess_dat_to_csv(dat_path, out_dir):
    """Reproduce your data split to inspect balance."""
    df = libsvm_to_dataframe(dat_path)
    unique_labels = sorted(df['label'].unique())
    splits = {
        "clientA": unique_labels[:2],
        "clientB": unique_labels[2:4],
        "clientC": unique_labels[4:],
    }
    os.makedirs(out_dir, exist_ok=True)
    for cname, lbls in splits.items():
        subdf = df[df['label'].isin(lbls)]
        subdf.to_csv(os.path.join(out_dir, f"{cname}_full.csv"), index=False)
    return splits

def analyze_client_data(csv_path, cname):
    df = pd.read_csv(csv_path)
    counts = len(df)
    label_counts = df['label'].value_counts().to_dict()
    stats = df.describe().T[['mean','std']]
    print(f"\n📊 {cname}: {counts} samples, label distribution {label_counts}")
    return stats

def plot_feature_variance(stats_dict):
    plt.figure(figsize=(8,4))
    for cname, stats in stats_dict.items():
        plt.plot(stats['std'].values, label=cname)
    plt.title("Feature-wise std deviation per client")
    plt.xlabel("Feature index")
    plt.ylabel("Std")
    plt.legend(); plt.tight_layout()
    plt.savefig("feature_variance.png")
    print("Saved feature_variance.png")

def context_variance_plot(csv_dir, dp_sigma=0.05):
    contexts = {}
    for cname in ['clientA','clientB','clientC']:
        df = pd.read_csv(f"{csv_dir}/{cname}_full.csv")
        input_dim = df.shape[1] - 1
        backbone = MLPBackbone(input_dim=input_dim, hidden_dims=[256, 128], embed_dim=128)
        dl, _ = make_dataloader(f"{csv_dir}/{cname}_full.csv", batch_size=64, shuffle=False)
        ctx = compute_context(backbone, dl)
        ctx = ctx / np.linalg.norm(ctx)
        ctx = gaussian_dp_noise(ctx, sigma=dp_sigma)
        contexts[cname] = ctx
    embed_dim = 128
    embeddings = {c: ctx[-embed_dim:] for c, ctx in contexts.items()}
    arr = np.stack(list(embeddings.values()))
    mean_ctx = arr.mean(axis=0)
    dists = {c: np.linalg.norm(embeddings[c] - mean_ctx) for c in embeddings}
    print("Context distances from mean:", dists)
    plt.figure(figsize=(6,4))
    plt.bar(dists.keys(), dists.values())
    plt.title("Context distance from mean (DP σ={})".format(dp_sigma))
    plt.ylabel("L2 distance"); plt.tight_layout()
    plt.savefig("context_dispersion.png")
    print("Saved context_dispersion.png")

def libsvm_to_dataframe(dat_path):
    X, y = load_svmlight_file(dat_path)
    X = X.toarray()
    df = pd.DataFrame(X, columns=[f"f{i}" for i in range(X.shape[1])])
    df["label"] = y.astype(int)
    return df

if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--data_dir", required=True)
    ap.add_argument("--dat_files", nargs="+", required=True)
    args = ap.parse_args()

    combined = os.path.join(args.data_dir, "combined.dat")
    # Combine all dats into one for quick balance check
    with open(combined, 'w') as fout:
        for f in args.dat_files:
            with open(f) as fin: fout.write(fin.read())

    splits = preprocess_dat_to_csv(combined, args.data_dir)
    stats_dict = {c: analyze_client_data(f"{args.data_dir}/{c}_full.csv", c) for c in splits}
    plot_feature_variance(stats_dict)
    context_variance_plot(args.data_dir)
