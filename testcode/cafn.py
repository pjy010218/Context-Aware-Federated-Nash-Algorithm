import torch
import torch.nn as nn
import torch.optim as optim
import numpy as np
from models import MLPBackbone, Head
from data_utils import make_dataloader
import copy
import math
import pandas as pd
import tempfile, os

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

def compute_context(backbone, dl, pca=None, num_samples=32, clip_norm=1.0):
    """
    Compute context vector: normalized label-histogram + mean embedding.
    Clips L2 norm to clip_norm (sensitivity control). Returns numpy array.
    """
    backbone.eval()
    xs = []
    ys = []
    with torch.no_grad():
        cnt = 0
        for xb, yb in dl:
            emb = backbone(xb.to(next(backbone.parameters()).device)).cpu().numpy()
            xs.append(emb)
            ys.append(yb.numpy())
            cnt += len(yb)
            if cnt >= num_samples:
                break
    if len(xs) == 0:
        return None
    X = np.vstack(xs)[:num_samples]
    Y = np.hstack(ys)[:num_samples]
    # label histogram: map labels (already local) to 0..k-1 range externally
    hist = np.bincount(Y, minlength=int(Y.max()-Y.min()+1)).astype(np.float32)
    if hist.sum() > 0:
        hist = hist / hist.sum()
    mean_emb = X.mean(axis=0).astype(np.float32)
    if pca is not None:
        mean_emb = pca.transform(mean_emb.reshape(1,-1)).flatten().astype(np.float32)
    ctx = np.concatenate([hist, mean_emb]).astype(np.float32)
    # clip L2 norm to clip_norm for DP sensitivity control
    norm = np.linalg.norm(ctx)
    if norm > clip_norm:
        ctx = ctx * (clip_norm / norm)
    # normalize to unit norm for attention (pre-DP)
    n = np.linalg.norm(ctx) + 1e-12
    ctx = ctx / n
    return ctx


# ---------------------------
# Gaussian DP noise with L2-sensitivity clipping
# ---------------------------
def gaussian_dp_noise(vec, sigma, clip_norm=1.0):
    """
    vec: numpy array (already clipped and normalized by compute_context)
    sigma: gaussian noise scale (stddev)
    clip_norm: for safety, re-clip before noise
    Returns noised numpy array
    """
    if vec is None:
        return None
    norm = np.linalg.norm(vec)
    if norm > clip_norm:
        vec = vec * (clip_norm / norm)
    noise = np.random.normal(loc=0.0, scale=sigma, size=vec.shape).astype(np.float32)
    return vec + noise
# ---------------------------

# ---------------------------
# Client local update (NEW signature includes server_avg_loss & beta)
# ---------------------------
def client_local_update(theta_global, head, train_csv, alpha=0.5, lambda_prox=0.1,
                        local_epochs=5, batch_size=32, lr=1e-3, context_holdout_ratio=0.02,
                        server_avg_loss=0.0, beta=0.0, seed=0):
    """
    Updated client local training:
    - includes cooperative regularizer: beta * (local_loss - server_avg_loss)^2
    - returns delta (numpy dict), ctx (numpy), updated head.
    """
    
    torch.manual_seed(seed)
    # prepare loaders as before (same code you already have)
    df = pd.read_csv(train_csv)
    holdout = df.sample(frac=context_holdout_ratio, random_state=seed)
    holdout_loader = None
    if len(holdout) > 0:
        tmp_path = os.path.join(tempfile.gettempdir(), "tmp_holdout.csv")
        holdout.to_csv(tmp_path, index=False)
        holdout_loader, _ = make_dataloader(tmp_path, batch_size=len(holdout), shuffle=False)
    train_loader, ds = make_dataloader(train_csv, batch_size=batch_size, shuffle=True)

    # copy backbone
    backbone = copy.deepcopy(theta_global).to(next(theta_global.parameters()).device)
    backbone.train()
    head = head.to(next(theta_global.parameters()).device)
    opt = optim.Adam(list(backbone.parameters()) + list(head.parameters()), lr=lr)
    loss_fn = torch.nn.CrossEntropyLoss()

    for epoch in range(local_epochs):
        for xb, yb in train_loader:
            xb = xb.to(next(theta_global.parameters()).device)
            yb = yb.to(next(theta_global.parameters()).device)
            # NOTE: yb should already be local indices 0..k-1
            logits = head(backbone(xb))
            loss_local = loss_fn(logits, yb)

            # prox term (global surrogate)
            prox = 0.0
            for p_new, p_old in zip(backbone.parameters(), theta_global.parameters()):
                prox = prox + ((p_new - p_old).pow(2).sum())
            prox = (lambda_prox / 2.0) * prox

            # cooperative variance penalty (client uses server-provided server_avg_loss)
            coop_pen = 0.0
            if beta > 0.0:
                # server_avg_loss is a scalar estimate of mean local loss across clients
                coop_pen = beta * (loss_local - server_avg_loss) ** 2

            loss = alpha * loss_local + (1.0 - alpha) * prox + coop_pen
            opt.zero_grad(); loss.backward(); opt.step()

    # compute delta
    delta = {}
    for name, p in backbone.state_dict().items():
        delta[name] = (p.detach().cpu().numpy() - dict(theta_global.state_dict())[name].detach().cpu().numpy())
    # compute context from holdout (already normalized inside compute_context)
    ctx = None
    if holdout_loader is not None:
        ctx = compute_context(backbone, holdout_loader)
    return delta, ctx, head
# ---------------------------

# ---------------------------
# Attention aggregation using projection + cosine similarity
# (stateless scoring; server can maintain projection params externally)
# ---------------------------
def aggregate_attention(deltas, contexts, Wk=None, q_vec=None, tau=1.0):
    """
    deltas: list of dicts (param-name -> numpy arr)
    contexts: list of numpy arrays (already normalized), may be None for some clients
    Wk: numpy matrix (proj_dim x ctx_dim) or None (if None use identity)
    q_vec: numpy vector (proj_dim,) representing query vector (server maintained)
    Returns (agg_delta, weights)
    """
    # prepare contexts matrix
    C_list = []
    for c in contexts:
        if c is None:
            C_list.append(None)
        else:
            C_list.append(c.astype(np.float32))
    # if no contexts => fallback to uniform average
    if all(c is None for c in C_list):
        agg = {}
        for k in deltas[0].keys():
            agg[k] = sum(d[k] for d in deltas) / len(deltas)
        weights = np.ones(len(deltas), dtype=np.float32) / len(deltas)
        return agg, weights

    # project contexts (Wk @ ctx) and compute cosine with q_vec
    projected = []
    embed_dim = Wk.shape[1]  # e.g., 128
    for c in C_list:
        if c is None:
            projected.append(None)
        else:
            mean_emb = c[-embed_dim:]  # Always take the last embed_dim elements
            if Wk is None:
                pv = mean_emb
            else:
                pv = Wk.dot(mean_emb)
            n = np.linalg.norm(pv) + 1e-12
            pv = pv / n
            projected.append(pv)

    if q_vec is None:
        # default q: mean of projected contexts
        q_vec = np.mean([p for p in projected if p is not None], axis=0)
        q_vec = q_vec / (np.linalg.norm(q_vec) + 1e-12)

    # compute scores
    scores = np.zeros(len(projected), dtype=np.float32)
    for i, p in enumerate(projected):
        if p is None:
            scores[i] = -1e9
        else:
            scores[i] = float(np.dot(q_vec, p))

    # softmax
    s = scores / (tau + 1e-12)
    ex = np.exp(s - np.max(s))
    a = ex / (ex.sum() + 1e-12)
    # aggregated delta
    agg = {}
    for k in deltas[0].keys():
        agg[k] = sum(a[i] * deltas[i][k] for i in range(len(deltas)))
    return agg, a

def apply_delta(theta, delta, lr=1.0):
    # Only update backbone parameters (exclude head parameters)
    for name, p in theta.named_parameters():
        if "fc" in name or "head" in name:  # assuming head params have 'fc' or 'head' in their name
            continue  # skip head parameters
        p.data = p.data + torch.from_numpy(delta[name]).to(p.device) * lr

def aggregate_fedavg(deltas):
    agg = {}
    for k in deltas[0].keys():
        agg[k] = sum(d[k] for d in deltas) / len(deltas)
    return agg
