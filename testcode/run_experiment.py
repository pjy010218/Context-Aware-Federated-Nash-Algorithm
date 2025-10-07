import argparse, os, json
import torch, random, numpy as np
from models import MLPBackbone, Head
from cafn import client_local_update, aggregate_attention, aggregate_fedavg, gaussian_dp_noise, apply_delta
from data_utils import make_dataloader
import time

def main(args):
    data_dir = args.data_dir
    clients = {
        "clientA": {"train": os.path.join(data_dir, "clientA_train.csv"), "val": os.path.join(data_dir, "clientA_val.csv"), "test": os.path.join(data_dir, "clientA_test.csv"), "num_labels": 2},
        "clientB": {"train": os.path.join(data_dir, "clientB_train.csv"), "val": os.path.join(data_dir, "clientB_val.csv"), "test": os.path.join(data_dir, "clientB_test.csv"), "num_labels": 2},
        "clientC": {"train": os.path.join(data_dir, "clientC_train.csv"), "val": os.path.join(data_dir, "clientC_val.csv"), "test": os.path.join(data_dir, "clientC_test.csv"), "num_labels": 2}
    }
    results = {}
    for alpha in args.alpha_list:
        print("Running alpha =", alpha)
        backbone = MLPBackbone(input_dim=128, hidden_dims=[256,128], embed_dim=128)
        heads = {c: Head(embed_dim=128, out_dim=clients[c]["num_labels"]) for c in clients}
        rounds = args.rounds
        for r in range(rounds):
            deltas = []
            contexts = []
            for cname in clients:
                delta, ctx, new_head = client_local_update(backbone, heads[cname], clients[cname]["train"],
                                                           alpha=alpha, lambda_prox=args.lambda_prox,
                                                           local_epochs=args.local_epochs, batch_size=args.batch_size,
                                                           lr=args.lr, seed=r)
                if ctx is not None:
                    ctx = gaussian_dp_noise(ctx, sigma=args.dp_sigma)
                deltas.append(delta)
                contexts.append(ctx)
                heads[cname] = new_head
            if args.aggregation == "attention":
                agg_delta, weights = aggregate_attention(backbone, deltas, contexts)
            else:
                agg_delta = aggregate_fedavg(deltas)
            apply_delta(backbone, agg_delta, lr=args.server_lr)
        perf = {}
        for cname in clients:
            dl, ds = make_dataloader(clients[cname]["test"], batch_size=64, shuffle=False)
            backbone.eval(); heads[cname].eval()
            correct = 0; total = 0
            with torch.no_grad():
                for xb, yb in dl:
                    logits = heads[cname](backbone(xb))
                    preds = logits.argmax(dim=1).cpu().numpy()
                    correct += (preds == (yb.numpy()-1)).sum()
                    total += len(yb)
            acc = correct / total if total>0 else 0.0
            perf[cname+"_acc"] = acc
        results[alpha] = perf
        print("alpha", alpha, "perf", perf)
    out_path = os.path.join(data_dir, "cafn_results.json")
    with open(out_path, "w") as f:
        json.dump(results, f, indent=2)
    print("Saved results to", out_path)

if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument("--data_dir", type=str, default="/mnt/data/gas_clients")
    parser.add_argument("--alpha_list", nargs="+", type=float, default=[0.0,0.25,0.5,0.75,1.0])
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
