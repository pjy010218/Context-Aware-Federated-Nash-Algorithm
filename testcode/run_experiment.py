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
    """Create per-client splits with shared label space but domain bias."""
    os.makedirs(out_dir, exist_ok=True)
    df = pd.read_csv(dat_path)
    df['label'] = df['label'] - df['label'].min()  # Ensure labels start at 0

    df.loc[:, df.columns != 'label'] = (df.loc[:, df.columns != 'label'] - df.loc[:, df.columns != 'label'].mean()) / df.loc[:, df.columns != 'label'].std()

    train, test = train_test_split(df, test_size=0.2, stratify=df['label'], random_state=42)
    val, test = train_test_split(test, test_size=0.5, stratify=test['label'], random_state=42)

    def domain_filter(df, bias_label):
        # Oversample bias_label, but also sample from all classes (with possible repeats)
        # Ensures every class is present in the client data
        all_labels = df['label'].unique()
        # Get all samples of the bias label
        bias_samples = df[df['label'] == bias_label]
        # Sample 30% of the full data (could include bias_label again)
        mixed_samples = df.sample(frac=0.3, random_state=42)
        # Concatenate and shuffle
        biased = pd.concat([bias_samples, mixed_samples])
        # Ensure at least one sample from every class (class coverage ≥ 70%)
        for lbl in all_labels:
            if (biased['label'] == lbl).sum() == 0:
                # Add one random sample of this class
                extra = df[df['label'] == lbl].sample(n=1, random_state=42)
                biased = pd.concat([biased, extra])
        return biased.sample(frac=1.0, random_state=42)

    # Each client gets all labels, but with a different bias
    clientA = domain_filter(train, 1)
    clientB = domain_filter(train, 3)
    clientC = domain_filter(train, 5)

    # All clients get the same val/test sets (shared label space)
    for cname, ctrain in zip(['clientA', 'clientB', 'clientC'], [clientA, clientB, clientC]):
        ctrain.to_csv(os.path.join(out_dir, f"{cname}_train.csv"), index=False)
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

def apply_delta(backbone, agg_delta, lr_global=1.0):
    sd = backbone.state_dict()
    for k in sd.keys():
        sd[k] = sd[k] + torch.from_numpy(agg_delta[k]).to(sd[k].device) * lr_global
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
    num_labels = len(pd.read_csv(dat_path)['label'].unique())

    clients = {
        "clientA": {"train": f"{data_dir}/clientA_train.csv", "val": f"{data_dir}/clientA_val.csv", "test": f"{data_dir}/clientA_test.csv", "num_labels": num_labels},
        "clientB": {"train": f"{data_dir}/clientB_train.csv", "val": f"{data_dir}/clientB_val.csv", "test": f"{data_dir}/clientB_test.csv", "num_labels": num_labels},
        "clientC": {"train": f"{data_dir}/clientC_train.csv", "val": f"{data_dir}/clientC_val.csv", "test": f"{data_dir}/clientC_test.csv", "num_labels": num_labels}
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
    lr_local = args.lr_local
    lr_global = args.lr_global
    beta = args.beta
    dp_sigma = args.dp_sigma
    tau = args.tau

    results = {
        "alpha_results": alpha_results,
        "NE_results": ne_results,   # <- ensure this exact key name
        "central_acc": central_acc,
        "personalization": personalizations,
        "attention_weights": attention_weights.tolist(),
        "dp_epsilon": dp_epsilon
    }
    log = []
    log_path = os.path.join(data_dir, "cafn_training_log.jsonl")

    for alpha in args.alpha_list:
        print("Running alpha =", alpha)
        backbone = MLPBackbone(input_dim=128, hidden_dims=[256,128], embed_dim=128)
        heads = {c: Head(embed_dim=128, out_dim=num_labels) for c in clients}
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
                    lr=lr_local,
                    server_avg_loss=server_avg_loss,
                    beta=beta,
                    seed=r
                )

                if ctx is not None:
                    # Normalize
                    ctx = ctx / (np.linalg.norm(ctx) + 1e-12)
                    # Whiten (zero mean, unit variance)
                    ctx = (ctx - ctx.mean()) / (ctx.std() + 1e-12)
                    ctx_noisy = gaussian_dp_noise(ctx, sigma=dp_sigma, clip_norm=1.0)
                else:
                    ctx_noisy = None

                deltas.append(delta)
                contexts.append(ctx_noisy)
                heads[cname] = new_head

            agg_delta, weights = aggregate_attention(deltas, contexts, Wk=Wk, q_vec=q_vec, tau=tau)
            apply_delta(backbone, agg_delta, lr_global=lr_global)

            embed_dim = 128
            projected_ctxs = [Wk.dot(c[-embed_dim:]) if c is not None else None for c in contexts]
            valid_proj = np.stack([p for p in projected_ctxs if p is not None])

            if valid_proj.shape[0] > 0:
                w_stack = np.array(weights[:valid_proj.shape[0]])
                w_stack /= w_stack.sum() + 1e-12
                mean_proj = (w_stack.reshape(-1,1) * valid_proj).sum(axis=0)
                mean_proj /= np.linalg.norm(mean_proj) + 1e-12
                q_vec = (1 - q_update_lr) * q_vec + q_update_lr * mean_proj
                q_vec /= np.linalg.norm(q_vec) + 1e-12

            # After aggregation and before next round
            # Compute per-client training losses for logging
            client_losses = {}
            for cname in clients:
                train_dl, _ = make_dataloader(clients[cname]["train"], batch_size=64, shuffle=False)
                loss_val = evaluate_loss_fn(backbone, heads[cname], train_dl)
                client_losses[cname] = float(loss_val)

            log.append({
                "round": r,
                "alpha": alpha,
                "server_loss": float(server_avg_loss),
                "attention": weights.tolist(),
                "client_losses": client_losses,
            })

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

        # After all rounds
        with open(log_path, "w") as f:
            for entry in log:
                f.write(json.dumps(entry) + "\n")
        print(f"Saved per-round training log to {log_path}")

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

        # === Freeze backbone parameters ===
        for param in theta_ref.parameters():
            param.requires_grad = False

        # Only optimize the head
        head_ref.train()
        opt = torch.optim.Adam(head_ref.parameters(), lr=1e-3)
        loss_fn = torch.nn.CrossEntropyLoss()
        train_loader, _ = make_dataloader(client_train_csv[cname], batch_size=32, shuffle=True)
        for epoch in range(5):  # or 10 for more thorough retrain
            for xb, yb in train_loader:
                xb = xb.to(next(theta_ref.parameters()).device)
                yb = yb.to(next(theta_ref.parameters()).device)
                logits = head_ref(theta_ref(xb))
                loss = loss_fn(logits, yb)
                opt.zero_grad()
                loss.backward()
                opt.step()

        new_loss = evaluate_loss_fn(theta_ref, head_ref, client_val_loaders[cname])
        new_U = alpha * new_loss + beta * (new_loss - server_avg_loss) ** 2
        improvements[cname] = float(base_U - new_U)
        print(f"ΔU_{cname} = {improvements[cname]:.6f}")

        print("client", cname, "base_loss", base_loss, "new_loss", new_loss, "base_U", base_U, "new_U", new_U)

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
    parser.add_argument("--beta", type=float, default=0.5)
    parser.add_argument("--tau", type=float, default=0.5)
    parser.add_argument("--rounds", type=int, default=30)
    parser.add_argument("--local_epochs", type=int, default=3)
    parser.add_argument("--batch_size", type=int, default=32)
    parser.add_argument("--lr_local", type=float, default=0.005, help="Local client learning rate")
    parser.add_argument("--lr_global", type=float, default=0.001, help="Global server learning rate")
    parser.add_argument("--lambda_prox", type=float, default=0.1)
    parser.add_argument("--aggregation", type=str, default="attention")
    parser.add_argument("--dp_sigma", type=float, default=0.01)
    args = parser.parse_args()
    main(args)
