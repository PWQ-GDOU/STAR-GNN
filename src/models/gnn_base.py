"""
gnn_base.py — Shared GNN model definitions and training utilities.

All active scripts (benchmark_clean, ablation_infer, quick_verify,
run_experiments_v2) import from this single module instead of duplicating
GCN, GraphSAGE, GNNC, train_gnn, and utility functions.

Canonical default parameters: epochs=200, lr=0.001, patience=30.
"""

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.optim import Adam
from sklearn.metrics import f1_score
from sklearn.preprocessing import StandardScaler
from sklearn.neighbors import NearestNeighbors

# Default k-NN parameter
KNN_K = 10


# ═══════════════════════════════════════════════════════════════
# Utility functions
# ═══════════════════════════════════════════════════════════════

def t2n(t):
    """Convert torch tensor to numpy array safely."""
    if t.numel() == 0:
        return np.array([])
    return np.array(t.cpu().tolist())


def build_knn_graph(X, k=None):
    """Build a row-normalized k-NN graph adjacency matrix from feature matrix X.

    Uses cosine distance and row-normalization (out-degree = 1 for each node).

    Args:
        X: Feature matrix (n_samples, n_features)
        k: Number of neighbors (default: KNN_K=10)

    Returns:
        Row-normalized sparse COO adjacency tensor.
    """
    if k is None:
        k = KNN_K
    n = len(X)
    k = min(k, n - 1)
    Xs = StandardScaler().fit_transform(X.astype(np.float64))
    nn_model = NearestNeighbors(n_neighbors=k + 1, metric='cosine', n_jobs=-1).fit(Xs)
    _, idx = nn_model.kneighbors(Xs)
    src, dst = [], []
    for i in range(n):
        for j in idx[i]:
            if i != j:
                src.append(i)
                dst.append(j)
    adj = torch.sparse_coo_tensor(
        torch.tensor([src, dst], dtype=torch.long),
        torch.ones(len(src)), (n, n)
    ).coalesce()
    # Row-normalize
    dg = torch.sparse.sum(adj, 1).to_dense()
    dg[dg == 0] = 1
    di = torch.arange(n)
    return torch.sparse.mm(
        torch.sparse_coo_tensor(torch.stack([di, di]), 1.0 / dg, (n, n)), adj
    ).coalesce()


def permute_graph(adj, n_nodes):
    """Randomly rewire destination nodes while preserving out-degree.

    Note: This preserves each node's out-degree exactly, but does NOT
    preserve in-degree. The resulting graph has the same out-degree
    distribution but a randomized in-degree distribution.

    Args:
        adj: Sparse COO adjacency tensor
        n_nodes: Number of nodes

    Returns:
        Row-normalized sparse COO adjacency tensor with shuffled edges.
    """
    indices = adj.coalesce().indices()
    src = indices[0].numpy() if hasattr(indices[0], 'numpy') else np.array(indices[0].tolist())
    dst = indices[1].numpy() if hasattr(indices[1], 'numpy') else np.array(indices[1].tolist())
    # Shuffle destination nodes (preserves out-degree only, not in-degree)
    np.random.shuffle(dst)
    new_adj = torch.sparse_coo_tensor(
        torch.tensor(np.stack([src, dst]), dtype=torch.long),
        torch.ones(len(src)), (n_nodes, n_nodes)
    ).coalesce()
    # Row-normalize
    dg = torch.sparse.sum(new_adj, 1).to_dense()
    dg[dg == 0] = 1
    di = torch.arange(n_nodes)
    return torch.sparse.mm(
        torch.sparse_coo_tensor(torch.stack([di, di]), 1.0 / dg, (n_nodes, n_nodes)), new_adj
    ).coalesce()


# ═══════════════════════════════════════════════════════════════
# GNN Model Definitions
# ═══════════════════════════════════════════════════════════════

class GCN(nn.Module):
    """2-layer Graph Convolutional Network with sparse adjacency.

    Args:
        d_in: Input feature dimension
        h: Hidden dimension (default: 128)
        o: Output embedding dimension (default: 64)
        dp: Dropout rate (default: 0.3)
    """

    def __init__(self, d_in, h=128, o=64, dp=0.3):
        super().__init__()
        self.l1 = nn.Linear(d_in, h)
        self.l2 = nn.Linear(h, o)
        self.bn = nn.BatchNorm1d(h)
        self.dp = nn.Dropout(dp)

    def forward(self, x, adj):
        x = F.relu(self.bn(self.l1(x)))
        x = self.dp(x)
        x = torch.sparse.mm(adj, x)
        x = self.l2(x)
        x = torch.sparse.mm(adj, x)
        return x


class GraphSAGE(nn.Module):
    """2-layer GraphSAGE with concatenation-based neighbor aggregation.

    Uses mean aggregation via row-normalized adjacency and concatenates
    self-features with neighbor features at each layer.

    Args:
        d_in: Input feature dimension
        h: Hidden dimension (default: 128)
        o: Output embedding dimension (default: 64)
        dp: Dropout rate (default: 0.3)
    """

    def __init__(self, d_in, h=128, o=64, dp=0.3):
        super().__init__()
        self.f1 = nn.Linear(d_in * 2, h)
        self.f2 = nn.Linear(h * 2, o)
        self.bn = nn.BatchNorm1d(h)
        self.dp = nn.Dropout(dp)

    def forward(self, x, adj):
        n = torch.sparse.mm(adj, x)
        h = torch.cat([x, n], -1)
        h = F.relu(self.bn(self.f1(h)))
        h = self.dp(h)
        n2 = torch.sparse.mm(adj, h)
        h = torch.cat([h, n2], -1)
        return self.f2(h)


class GNNC(nn.Module):
    """GNN Classifier wrapping a GCN or GraphSAGE encoder with MLP head.

    Args:
        t: Encoder type — 'GCN' or 'GraphSAGE'
        d_in: Input feature dimension
        h: Hidden dimension (default: 128)
        o: Output embedding dimension (default: 64)
        nc: Number of classes (default: 2)
        dp: Dropout rate (default: 0.3)
    """

    def __init__(self, t, d_in, h=128, o=64, nc=2, dp=0.3):
        super().__init__()
        self.enc = GCN(d_in, h, o, dp) if t == 'GCN' else GraphSAGE(d_in, h, o, dp)
        self.cls = nn.Sequential(
            nn.Linear(o, 32), nn.ReLU(), nn.Dropout(dp), nn.Linear(32, nc)
        )

    def forward(self, x, adj):
        return self.cls(self.enc(x, adj))


# ═══════════════════════════════════════════════════════════════
# Training
# ═══════════════════════════════════════════════════════════════

def train_gnn(m, x, adj, y, tr, va, device='cpu', ep=200, lr=0.001, pa=30):
    """Train a GNN model with early stopping on validation F1.

    Transductive setting: all nodes participate in message passing but
    only training nodes contribute to the loss.

    Args:
        m: GNNC model instance
        x: Node features (n_nodes, d_in)
        adj: Row-normalized sparse adjacency (n_nodes, n_nodes)
        y: Labels (n_nodes,)
        tr: Training mask (n_nodes,) boolean
        va: Validation mask (n_nodes,) boolean
        device: Torch device string
        ep: Maximum epochs (default: 200)
        lr: Learning rate (default: 0.001)
        pa: Early stopping patience in validation checks (default: 30)

    Returns:
        The model with best checkpoint weights loaded.
    """
    m = m.to(device)
    x = x.to(device)
    adj = adj.to(device)
    y = y.to(device)
    np_ = y[tr].sum().item()
    pw = (tr.sum().item() - np_) / max(np_, 1)
    cr = nn.CrossEntropyLoss(weight=torch.tensor([1., pw], device=device))
    op = Adam(m.parameters(), lr=lr, weight_decay=5e-4)
    bf, best, cnt = 0., {k: v.cpu().clone() for k, v in m.state_dict().items()}, 0
    for e in range(ep):
        m.train()
        op.zero_grad()
        loss = cr(m(x, adj)[tr], y[tr])
        loss.backward()
        torch.nn.utils.clip_grad_norm_(m.parameters(), 1.)
        op.step()
        if e % 5 == 0:
            m.eval()
            with torch.no_grad():
                vf = f1_score(t2n(y[va]), t2n(m(x, adj)[va].argmax(1)), zero_division=0)
            if vf > bf:
                bf = vf
                best = {k: v.cpu().clone() for k, v in m.state_dict().items()}
                cnt = 0
            else:
                cnt += 1
            if cnt >= pa:
                break
    m.load_state_dict(best)
    return m
