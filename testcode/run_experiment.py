import argparse, os, json, copy, glob
import torch, random, numpy as np
import pandas as pd
from models import MLPBackbone, Head
from cafn import client_local_update, aggregate_attention, aggregate_fedavg, gaussian_dp_noise, compute_context
from collections import defaultdict
from data_utils import make_dataloader
from sklearn.model_selection import train_test_split
from sklearn.datasets import load_svmlight_file
import os, time, math
import tempfile


def libsvm_to_dataframe(dat_path):
    X, y = load_svmlight_file(dat_path)
    X = X.toarray()
    df = pd.DataFrame(X, columns=[f"f{i}" for i in range(X.shape[1])])
    df["label"] = y.astype(int)
    return df

def combine_dat_files(dat_files, combined_path):
    dfs = []
    for f in dat_files:
        df = libsvm_to_dataframe(f)
        dfs.append(df)
    combined_df = pd.concat(dfs, ignore_index=True)
    combined_df.to_csv(combined_path, index=False)
    print(f"Combined {len(dat_files)} .dat files into {combined_path}")

def preprocess_dat_to_csv(dat_path, out_dir):
    """Reads a .dat LibSVM-style dataset and writes per-client CSV splits."""

    os.makedirs(out_dir, exist_ok=True)
    df = pd.read_csv(dat_path)

    # Create 3 domain splits (by label subsets or random splits)
    # Example: each client gets a different subset of gas labels
    unique_labels = sorted(df['label'].unique())
    splits = {
        "clientA": unique_labels[:2],
        "clientB": unique_labels[2:4],
        "clientC": unique_labels[4:],
    }

    for cname, lbls in splits.items():
        subdf = df[df['label'].isin(lbls)].copy()
        # Remap labels to 0...num_labels-1 for each client
        label_map = {orig: i for i, orig in enumerate(sorted(lbls))}
        subdf['label'] = subdf['label'].map(label_map)
        if len(subdf) < 5:
            train, val, test = subdf, subdf.iloc[[]], subdf.iloc[[]]
        else:
            train, test = train_test_split(subdf, test_size=0.2, random_state=42)
            if len(test) < 2:
                val, test = test, test.iloc[[]]
            else:
                val, test = train_test_split(test, test_size=0.5, random_state=42)
        train.to_csv(os.path.join(out_dir, f"{cname}_train.csv"), index=False)
        val.to_csv(os.path.join(out_dir, f"{cname}_val.csv"), index=False)
        test.to_csv(os.path.join(out_dir, f"{cname}_test.csv"), index=False)
        print(f"✅ Saved {cname} train/val/test CSVs.")

def epsilon_for_gaussian_noise(sigma, sensitivity=1.0, delta=1e-5):
    if sigma <= 0:
        return float('inf')
    eps = (sensitivity * math.sqrt(2.0 * math.log(1.25 / delta))) / sigma
    return float(eps)

ctx_dim = 128
proj_dim = 32

Wk = np.random.normal(scale=0.01, size=(proj_dim, ctx_dim)).astype(np.float32)
q_vec = np.random.normal(scale=0.01, size=(proj_dim,)).astype(np.float32)
q_vec = q_vec / (np.linalg.norm(q_vec) + 1e-12)
q_update_lr = 0.1

def evaluate_loss_fn(backbone, head, val_loader):
    backbone.eval(); head.eval()
    loss_fn = torch.nn.CrossEntropyLoss()
    total_loss = 0.0
    total = 0
    with torch.no_grad():
        for xb, yb in val_loader:
            logits = head(backbone(xb))
            loss = loss_fn(logits, yb)
            total_loss += loss.item() * len(yb)
            total += len(yb)
    return total_loss / total if total > 0 else float('nan')

def apply_delta(backbone, agg_delta):
    sd = backbone.state_dict()
    for k in sd.keys():
        sd[k] = sd[k] + torch.from_numpy(agg_delta[k]).to(sd[k].device)
    backbone.load_state_dict(sd)

def evaluate_central_backbone(backbone, clients, heads):
    """Measure central AI generalization and transferability."""

    # 1. Global pooled validation accuracy
    all_paths = [clients[c]["val"] for c in clients]
    pooled_df = pd.concat([pd.read_csv(p) for p in all_paths], ignore_index=True)
    pooled_path = os.path.join(os.path.dirname(all_paths[0]), "pooled_val.csv")
    pooled_df.to_csv(pooled_path, index=False)
    pooled_loader, _ = make_dataloader(pooled_path, batch_size=64, shuffle=False)

    backbone.eval()
    total, correct = 0, 0
    with torch.no_grad():
        for xb, yb in pooled_loader:
            # choose any head (labels may differ across clients, so handle with caution)
            logits = list(heads.values())[0](backbone(xb))
            preds = logits.argmax(dim=1)
            correct += (preds == (yb-1)).sum().item()
            total += len(yb)
    global_acc = correct / total if total > 0 else 0
    print(f"🌍 Central Backbone (pooled) Accuracy: {global_acc:.4f}")

    # 2. Personalization / head retraining test (per client)
    def fine_tune_head(backbone, train_csv, test_csv, num_labels=2):
        head = Head(embed_dim=128, out_dim=num_labels)
        opt = torch.optim.Adam(head.parameters(), lr=1e-3)
        loss_fn = torch.nn.CrossEntropyLoss()
        train_loader, _ = make_dataloader(train_csv, batch_size=32, shuffle=True)
        test_loader, _ = make_dataloader(test_csv, batch_size=64, shuffle=False)
        backbone.eval()
        for epoch in range(5):  # few-shot fine-tuning
            for xb, yb in train_loader:
                with torch.no_grad(): emb = backbone(xb)
                logits = head(emb)
                loss = loss_fn(logits, yb)
                opt.zero_grad(); loss.backward(); opt.step()
        correct, total = 0, 0
        with torch.no_grad():
            for xb, yb in test_loader:
                logits = head(backbone(xb))
                preds = logits.argmax(dim=1)
                correct += (preds == (yb-1)).sum().item()
                total += len(yb)
        return correct / total if total > 0 else 0.0

    for cname in clients:
        acc = fine_tune_head(backbone, clients[cname]["train"], clients[cname]["test"], clients[cname]["num_labels"])
        print(f"🧠 Personalization ({cname}): fine-tune acc = {acc:.4f}")


def main(args):
    global q_vec
    data_dir = args.data_dir

    if hasattr(args, "dat_files") and args.dat_files:
        combined_dat_path = os.path.join(args.data_dir, "combined.dat")
        combine_dat_files(args.dat_files, combined_dat_path)
        dat_path = combined_dat_path
    else:
        dat_path = os.path.join(args.data_dir, "batch7.dat")  # or your default
    
    preprocess_dat_to_csv(dat_path, args.data_dir)
    clients = {
        "clientA": {"train": f"{data_dir}/clientA_train.csv", "val": f"{data_dir}/clientA_val.csv", "test": f"{data_dir}/clientA_test.csv", "num_labels": 2},
        "clientB": {"train": f"{data_dir}/clientB_train.csv", "val": f"{data_dir}/clientB_val.csv", "test": f"{data_dir}/clientB_test.csv", "num_labels": 2},
        "clientC": {"train": f"{data_dir}/clientC_train.csv", "val": f"{data_dir}/clientC_val.csv", "test": f"{data_dir}/clientC_test.csv", "num_labels": 2}
    }

    # Prepare validation loaders and train csv dicts
    client_val_loaders = {}
    client_train_csv = {}
    for cname in clients:
        dl, _ = make_dataloader(clients[cname]["val"], batch_size=64, shuffle=False)
        client_val_loaders[cname] = dl
        client_train_csv[cname] = clients[cname]["train"]

    lambda_prox = args.lambda_prox
    local_epochs = args.local_epochs
    batch_size = args.batch_size
    lr = args.lr
    beta = 0.1  # You can make this an argument if desired
    dp_sigma = args.dp_sigma
    tau = 1.0   # You can make this an argument if desired

    results = {}
    for alpha in args.alpha_list:
        print("Running alpha =", alpha)
        backbone = MLPBackbone(input_dim=128, hidden_dims=[256,128], embed_dim=128)
        heads = {c: Head(embed_dim=128, out_dim=clients[c]["num_labels"]) for c in clients}
        rounds = args.rounds

        for r in range(rounds):
            deltas = []
            contexts = []

            # Compute per-client validation losses for cooperative penalty
            per_client_val_losses = []
            for cname in clients:
                val_loss = evaluate_loss_fn(backbone, heads[cname], client_val_loaders[cname])
                per_client_val_losses.append(float(val_loss))
            server_avg_loss = float(np.mean(per_client_val_losses))

            for cname in clients:
                delta, ctx, new_head = client_local_update(
                    theta_global=backbone,
                    head=heads[cname],
                    train_csv=client_train_csv[cname],
                    alpha=alpha,
                    lambda_prox=lambda_prox,
                    local_epochs=local_epochs,
                    batch_size=batch_size,
                    lr=lr,
                    server_avg_loss=server_avg_loss,
                    beta=beta,
                    seed=r
                )

                if ctx is not None:
                    ctx = ctx / (np.linalg.norm(ctx) + 1e-12)  # Normalize before noise
                    ctx_noisy = gaussian_dp_noise(ctx, sigma=dp_sigma, clip_norm=1.0)
                else:
                    ctx_noisy = None

                deltas.append(delta)
                contexts.append(ctx_noisy)
                heads[cname] = new_head

            agg_delta, weights = aggregate_attention(deltas, contexts, Wk=Wk, q_vec=q_vec, tau=tau)
            apply_delta(backbone, agg_delta)

            num_hist = len([c for c in contexts if c is not None][0]) - Wk.shape[1]
            projected_ctxs = [Wk.dot(c[num_hist:]) if c is not None else None for c in contexts]
            valid_proj = np.stack([p for p in projected_ctxs if p is not None])

            if valid_proj.shape[0] > 0:
                w_stack = np.array(weights[:valid_proj.shape[0]])
                w_stack /= w_stack.sum() + 1e-12
                mean_proj = (w_stack.reshape(-1,1) * valid_proj).sum(axis=0)
                mean_proj /= np.linalg.norm(mean_proj) + 1e-12
                q_vec = (1 - q_update_lr) * q_vec + q_update_lr * mean_proj
                q_vec /= np.linalg.norm(q_vec) + 1e-12

        perf = {}
        for cname in clients:
            dl, _ = make_dataloader(clients[cname]["test"], batch_size=64, shuffle=False)
            backbone.eval(); heads[cname].eval()
            correct = 0; total = 0
            with torch.no_grad():
                for xb, yb in dl:
                    logits = heads[cname](backbone(xb))
                    preds = logits.argmax(dim=1).cpu().numpy()
                    correct += (preds == (yb.numpy()-1)).sum()
                    total += len(yb)
            acc = correct / total if total > 0 else 0.0
            perf[cname+"_acc"] = acc
        results[alpha] = perf
        print("alpha", alpha, "perf", perf)

    print("\n=== Verifying Nash Equilibrium ===")
    _ = verify_nash_equilibrium(
        backbone,
        heads,
        client_train_csv,
        client_val_loaders,
        alpha=max(args.alpha_list),      # choose one α for NE test
        server_avg_loss=server_avg_loss,
        beta=0.1
    )
    evaluate_central_backbone(backbone, clients, heads)

    out_path = args.outpath
    with open(out_path, "w") as f:
        json.dump(results, f, indent=2)
    print("Saved results to", out_path)
    eps = epsilon_for_gaussian_noise(dp_sigma, sensitivity=1.0, delta=1e-5)
    print(f"DP ε ≈ {eps:.3f} | attention weights: {weights}")

def verify_nash_equilibrium(backbone, heads, client_train_csv, client_val_loaders, alpha, server_avg_loss, beta):
    improvements = {}
    for cname in heads:
        print(f"\n[NE test] Client {cname} best-response retrain...")
        theta_ref = copy.deepcopy(backbone)
        head_ref = copy.deepcopy(heads[cname])

        base_loss = evaluate_loss_fn(theta_ref, head_ref, client_val_loaders[cname])
        base_U = alpha * base_loss + beta * (base_loss - server_avg_loss) ** 2

        delta, _, head_new = client_local_update(
            theta_global=theta_ref,
            head=head_ref,
            train_csv=client_train_csv[cname],
            alpha=alpha,
            lambda_prox=0.0,
            local_epochs=20,
            batch_size=32,
            lr=1e-3,
            server_avg_loss=server_avg_loss,
            beta=beta,
            seed=42
        )
        new_loss = evaluate_loss_fn(theta_ref, head_new, client_val_loaders[cname])
        new_U = alpha * new_loss + beta * (new_loss - server_avg_loss) ** 2
        improvements[cname] = float(base_U - new_U)
        print(f"ΔU_{cname} = {improvements[cname]:.6f}")

    print("\nApproximate Nash verification results:")
    for c, val in improvements.items():
        print(f"  Client {c}: ΔU = {val:.6f}")
    return improvements

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--dat_files", nargs="*", type=str, help="List of .dat files to combine")
    parser.add_argument("--data_dir", type=str, default="/mnt/data/gas_clients")
    parser.add_argument("--outpath", type=str, default="cafn_results.json")  # <-- add this line
    parser.add_argument("--alpha_list", nargs="+", type=float, default=[0.0,0.1,0.25,0.5,0.75,1.0])
    parser.add_argument("--rounds", type=int, default=30)
    parser.add_argument("--local_epochs", type=int, default=3)
    parser.add_argument("--batch_size", type=int, default=32)
    parser.add_argument("--lr", type=float, default=1e-3)
    parser.add_argument("--server_lr", type=float, default=1.0)
    parser.add_argument("--lambda_prox", type=float, default=0.1)
    parser.add_argument("--aggregation", type=str, default="attention")
    parser.add_argument("--dp_sigma", type=float, default=0.1)
    args = parser.parse_args()
    main(args)
