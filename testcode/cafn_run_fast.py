# cafn_run_fast.py
# Minimal, fast CAFN + FedAvg experiment for the gas dataset (libSVM-style input)
import json, random, time, argparse
from pathlib import Path
import numpy as np
import torch, torch.nn as nn, torch.optim as optim
from sklearn.model_selection import train_test_split

def parse_libsvm(path):
    lines = Path(path).read_text().strip().splitlines()
    rows=[]; labels=[]; max_idx=0
    for line in lines:
        parts=line.strip().split()
        if not parts: continue
        lab=int(parts[0]); labels.append(lab)
        vec={}
        for tok in parts[1:]:
            if ':' not in tok: continue
            i,v=tok.split(':',1)
            try: idx=int(i); val=float(v)
            except: continue
            vec[idx]=val
            if idx>max_idx: max_idx=idx
        rows.append(vec)
    n=len(rows)
    X=np.zeros((n,max_idx), dtype=np.float32)
    for i,vec in enumerate(rows):
        for idx,val in vec.items(): X[i,idx-1]=val
    y=np.array(labels,dtype=int)
    return X, y

# Simple models
class Backbone(nn.Module):
    def __init__(self,input_dim=128,embed_dim=16):
        super().__init__()
        self.fc = nn.Linear(input_dim, embed_dim)
        self.act = nn.ReLU()
    def forward(self,x): return self.act(self.fc(x))

class Head(nn.Module):
    def __init__(self,embed_dim=16,out=2):
        super().__init__()
        self.fc = nn.Linear(embed_dim, out)
    def forward(self,x): return self.fc(x)

def make_loader(X,y,batch_size=128, shuffle=True):
    import torch.utils.data as data
    if len(y)==0:
        return data.DataLoader(data.TensorDataset(torch.zeros((0,X.shape[1])), torch.zeros((0,),dtype=torch.long)), batch_size=1)
    return data.DataLoader(data.TensorDataset(torch.from_numpy(X).float(), torch.from_numpy(y).long()), batch_size=batch_size, shuffle=shuffle)

def client_update(theta_model, head_model, X_train, y_train, alpha=0.5, lambda_prox=0.1, local_epochs=1, batch_size=128, lr=1e-3, seed=0, device='cpu', coop_loss=None, beta=0.0):
    torch.manual_seed(seed); np.random.seed(seed); random.seed(seed)
    n = X_train.shape[0]
    if n==0:
        return None, None, head_model
    loader = make_loader(X_train, y_train, batch_size=batch_size, shuffle=True)
    backbone = Backbone().to(device); backbone.load_state_dict(theta_model.state_dict())
    head = Head(out=head_model.fc.out_features).to(device); head.load_state_dict(head_model.state_dict())
    backbone.train(); head.train()
    opt = optim.Adam(list(backbone.parameters()) + list(head.parameters()), lr=lr)
    loss_fn = nn.CrossEntropyLoss()
    for epoch in range(local_epochs):
        for xb,yb in loader:
            xb = xb.to(device); yb = yb.to(device)
            logits = head(backbone(xb))
            loss_local = loss_fn(logits, yb)
            prox = 0.0
            for p_new, p_old in zip(backbone.parameters(), theta_model.parameters()):
                prox += ((p_new - p_old).pow(2).sum())
            prox = (lambda_prox/2.0)*prox
            loss = alpha*loss_local + (1.0-alpha)*prox
            if coop_loss is not None and beta > 0:
                loss = loss + beta * (loss_local - coop_loss)**2
            opt.zero_grad(); loss.backward(); opt.step()
    # delta
    delta = {}
    for name,p in backbone.state_dict().items():
        delta[name] = (p.cpu().numpy() - theta_model.state_dict()[name].cpu().numpy())
    # context: histogram + mean embedding on a small sample (up to 32)
    sample_idx = np.random.choice(np.arange(n), size=min(32,n), replace=False)
    xb_sample = torch.from_numpy(X_train[sample_idx]).float().to(device)
    with torch.no_grad():
        emb = backbone(xb_sample).cpu().numpy()
    mean_emb = emb.mean(axis=0)
    hist = np.bincount(y_train[sample_idx], minlength=len(np.unique(y_train))).astype(np.float32)
    if hist.sum()>0: hist = hist / hist.sum()
    ctx = np.concatenate([hist, mean_emb.astype(np.float32)])
    return delta, ctx, head

def aggregate_fedavg(deltas):
    keys=list(deltas[0].keys())
    agg={}
    for k in keys:
        agg[k] = sum(d[k] for d in deltas) / len(deltas)
    return agg

def aggregate_attention(deltas, contexts, dp_sigma=0.1, tau=1.0):
    maxlen = max([0 if c is None else len(c) for c in contexts])
    if maxlen==0:
        return aggregate_fedavg(deltas), [1.0/len(deltas)]*len(deltas)
    C = np.stack([np.zeros(maxlen) if c is None else c for c in contexts])
    scores = C.sum(axis=1) / (tau + 1e-9)
    ex = np.exp(scores - np.max(scores)); a = ex / ex.sum()
    agg={}
    for k in deltas[0].keys():
        agg[k] = sum(a[i]*deltas[i][k] for i in range(len(deltas)))
    return agg, a

def apply_delta(theta_model, agg_delta):
    sd = theta_model.state_dict()
    for k in sd.keys():
        sd[k] = sd[k] + torch.from_numpy(agg_delta[k]).to(sd[k].device)
    theta_model.load_state_dict(sd)

def evaluate(backbone, head, X_test, y_test, device='cpu'):
    if len(y_test)==0:
        return float('nan')
    dl = make_loader(X_test, y_test, batch_size=256, shuffle=False)
    backbone.eval(); head.eval()
    correct=0; total=0
    with torch.no_grad():
        for xb,yb in dl:
            xb = xb.to(device)
            logits = head(backbone(xb)).cpu().numpy()
            preds = logits.argmax(axis=1)
            correct += (preds == yb.numpy()).sum()
            total += len(yb)
    return correct/total if total>0 else float('nan')

def run_experiment(datapath, outpath, fast=True, device='cpu'):
    X,y = parse_libsvm(datapath)
    client_groups = {"clientA":[1,2],"clientB":[3,4],"clientC":[5,6]}
    client_data={}
    for cname,labs in client_groups.items():
        mask = np.isin(y, labs)
        Xc = X[mask]; yc = y[mask]
        unique=np.unique(yc); lab_map={lab:i for i,lab in enumerate(unique)}
        client_data[cname] = {"X": Xc.copy(), "y_raw": yc.copy(), "label_map": lab_map}
    # splits
    for cname in client_data:
        Xc = client_data[cname]["X"]; yc = client_data[cname]["y_raw"]
        train_idx, temp_idx = train_test_split(np.arange(len(yc)), test_size=0.30, stratify=yc, random_state=42)
        val_idx, test_idx = train_test_split(temp_idx, test_size=0.6667, stratify=yc[temp_idx], random_state=42)
        client_data[cname]["trainX"] = Xc[train_idx]
        client_data[cname]["trainy"] = np.array([client_data[cname]["label_map"][lab] for lab in yc[train_idx]])
        client_data[cname]["testX"] = Xc[test_idx]
        client_data[cname]["testy"] = np.array([client_data[cname]["label_map"][lab] for lab in yc[test_idx]])
    # global standardize
    all_train = np.vstack([client_data[c]["trainX"] for c in client_data])
    mean = all_train.mean(axis=0); std = all_train.std(axis=0); std[std==0]=1.0
    for c in client_data:
        client_data[c]["trainX"] = (client_data[c]["trainX"] - mean)/std
        client_data[c]["testX"] = (client_data[c]["testX"] - mean)/std
    # fast mode params
    alphas = [0.0, 0.1, 0.2, 0.3, 0.4, 0.5, 1.0] if fast else [0.0, 0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9, 1.0]
    rounds = 10 if fast else 20
    local_epochs = 1
    batch_size = 128
    lr = 1e-3
    lambda_prox = 0.1
    dp_sigma = 0.01
    results={}
    for alpha in alphas:
        name = "FedAvg" if alpha==0.0 else f"CAFN_alpha_{alpha}"
        print("Running", name)
        backbone = Backbone().to(device)
        heads = {c: Head(out=len(np.unique(client_data[c]["trainy"]))).to(device) for c in client_data}
        for r in range(rounds):
            deltas=[]; contexts=[]
            local_losses=[]
            # First pass: compute local losses without cooperative penalty
            for cname in client_data:
                Xc = client_data[cname]["trainX"]
                yc = client_data[cname]["trainy"]
                backbone.eval(); heads[cname].eval()
                dl = make_loader(Xc, yc, batch_size=batch_size, shuffle=False)
                loss_fn = nn.CrossEntropyLoss()
                total_loss = 0; total = 0
                with torch.no_grad():
                    for xb, yb in dl:
                        xb = xb.to(device); yb = yb.to(device)
                        logits = heads[cname](backbone(xb))
                        total_loss += loss_fn(logits, yb).item() * len(yb)
                        total += len(yb)
                local_losses.append(total_loss / total if total > 0 else 0)
            mean_loss = np.mean(local_losses)
            # Second pass: client updates with cooperative penalty
            for idx, cname in enumerate(client_data):
                delta, ctx, new_head = client_update(
                    backbone, heads[cname],
                    client_data[cname]["trainX"], client_data[cname]["trainy"],
                    alpha=(1.0 if alpha==0.0 else alpha), lambda_prox=lambda_prox,
                    local_epochs=local_epochs, batch_size=batch_size, lr=lr, seed=r+1, device=device,
                    coop_loss=mean_loss, beta=0.1  # Set beta as desired
                )
                if ctx is not None:
                    ctx = ctx / np.linalg.norm(ctx)
                    ctx = ctx + np.random.normal(scale=dp_sigma, size=ctx.shape).astype(np.float32)
                deltas.append(delta); contexts.append(ctx); heads[cname]=new_head
            if alpha==0.0:
                agg = aggregate_fedavg(deltas)
            else:
                agg, weights = aggregate_attention(deltas, contexts, dp_sigma=dp_sigma)
            apply_delta(backbone, agg)
        perf={}
        for cname in client_data:
            perf[cname+"_acc"] = evaluate(backbone, heads[cname], client_data[cname]["testX"], client_data[cname]["testy"], device=device)
        results[name] = perf
        print(name, "perf:", perf)
    results["meta"] = {"rounds": rounds, "local_epochs": local_epochs}
    Path(outpath).write_text(json.dumps(results, indent=2))
    print("Saved results to", outpath)
    return results

if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument("--datapath", type=str, default="batch7.dat")
    parser.add_argument("--outpath", type=str, default="cafn_full_results.json")
    parser.add_argument("--fast", action="store_true", default=True)
    parser.add_argument("--device", type=str, default="cpu")
    args = parser.parse_args()
    run_experiment(args.datapath, args.outpath, fast=args.fast, device=args.device)
