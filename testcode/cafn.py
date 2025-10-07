import torch
import torch.nn as nn
import torch.optim as optim
import numpy as np
from models import MLPBackbone, Head
from data_utils import make_dataloader
import copy
import math

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

def compute_context(backbone, dl, pca=None, num_samples=32):
    backbone.eval()
    xs = []
    ys = []
    with torch.no_grad():
        cnt = 0
        for xb, yb in dl:
            emb = backbone(xb.to(device)).cpu().numpy()
            xs.append(emb)
            ys.append(yb.numpy())
            cnt += len(yb)
            if cnt >= num_samples:
                break
    if len(xs) == 0:
        return None
    X = np.vstack(xs)[:num_samples]
    Y = np.hstack(ys)[:num_samples]
    hist = np.bincount(Y - 1)
    hist = hist.astype(np.float32)
    if hist.sum() > 0:
        hist = hist / hist.sum()
    mean_emb = X.mean(axis=0)
    if pca is not None:
        mean_emb = pca.transform(mean_emb.reshape(1,-1)).flatten()
    ctx = np.concatenate([hist, mean_emb.astype(np.float32)])
    return ctx

def gaussian_dp_noise(vec, sigma):
    if vec is None:
        return None
    return vec + np.random.normal(scale=sigma, size=vec.shape).astype(np.float32)

def client_local_update(theta_global, head, train_csv, alpha=0.5, lambda_prox=0.1, 
                        local_epochs=5, batch_size=32, lr=1e-3, context_holdout_ratio=0.02, seed=0):
    torch.manual_seed(seed)
    import pandas as pd
    df = pd.read_csv(train_csv)
    holdout = df.sample(frac=context_holdout_ratio, random_state=seed)
    holdout_loader = None
    if len(holdout) > 0:
        holdout.to_csv("/tmp/tmp_holdout.csv", index=False)
        from data_utils import make_dataloader
        holdout_loader, _ = make_dataloader("/tmp/tmp_holdout.csv", batch_size=len(holdout), shuffle=False)
    train_loader, ds = make_dataloader(train_csv, batch_size=batch_size, shuffle=True)
    backbone = copy.deepcopy(theta_global).to(device)
    backbone.train()
    head = head.to(device)
    opt = optim.Adam(list(backbone.parameters()) + list(head.parameters()), lr=lr)
    loss_fn = nn.CrossEntropyLoss()
    for epoch in range(local_epochs):
        for xb, yb in train_loader:
            xb = xb.to(device); yb = yb.to(device) - 1  # convert to 0-based
            logits = head(backbone(xb))
            loss_local = loss_fn(logits, yb)
            prox = 0.0
            for p_new, p_old in zip(backbone.parameters(), theta_global.parameters()):
                prox = prox + ((p_new - p_old).pow(2).sum())
            prox = (lambda_prox / 2.0) * prox
            loss = alpha * loss_local + (1.0 - alpha) * prox
            opt.zero_grad(); loss.backward(); opt.step()
    # compute delta
    delta = {}
    for name, p in backbone.named_parameters():
        delta[name] = (p.detach().cpu().numpy() - dict(theta_global.named_parameters())[name].detach().cpu().numpy())
    ctx = None
    if holdout_loader is not None:
        ctx = compute_context(backbone, holdout_loader)
    return delta, ctx, head

def aggregate_attention(theta, deltas, contexts, tau=1.0):
    maxlen = max([0 if c is None else len(c) for c in contexts])
    if maxlen == 0:
        agg = {}
        for k in deltas[0].keys():
            agg[k] = sum([d[k] for d in deltas]) / len(deltas)
        return agg, None
    C = np.stack([np.zeros(maxlen) if c is None else c for c in contexts])
    scores = C.sum(axis=1)
    scores = scores / (tau + 1e-9)
    a = np.exp(scores - np.max(scores))
    a = a / a.sum()
    agg = {}
    for k in deltas[0].keys():
        agg[k] = sum(a[i] * deltas[i][k] for i in range(len(deltas)))
    return agg, a

def apply_delta(theta, delta, lr=1.0):
    for name, p in theta.named_parameters():
        p.data = p.data + torch.from_numpy(delta[name]).to(p.device) * lr

def aggregate_fedavg(deltas):
    agg = {}
    for k in deltas[0].keys():
        agg[k] = sum(d[k] for d in deltas) / len(deltas)
    return agg
