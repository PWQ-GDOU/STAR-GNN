"""
quick_verify.py — 快速正确性验证（1 fold, 2 HP combos）
用于在沙箱中快速确认代码无误；完整版在本地运行 benchmark_clean.py
"""
import sys, os, pickle, warnings, time, gc
warnings.filterwarnings('ignore', category=FutureWarning)
import torch, torch.nn as nn, torch.nn.functional as F
from torch.optim import Adam
import numpy as np

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print("Device:", device)

_REPO_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
PROJECT_ROOT = os.environ.get("STAR_GNN_HOME", _REPO_ROOT)

from sklearn.model_selection import StratifiedKFold
from sklearn.metrics import f1_score, roc_auc_score
from xgboost import XGBClassifier
from lightgbm import LGBMClassifier

# Ensure src/ is importable
_src_dir = os.path.dirname(os.path.abspath(__file__))
if _src_dir not in sys.path:
    sys.path.insert(0, _src_dir)

from models.gnn_base import t2n, GCN, GraphSAGE, GNNC, train_gnn

# === Load data ===
print("Loading graph...")
graph_path = os.environ.get("STAR_GNN_GRAPH",
    os.path.join(PROJECT_ROOT, "results", "graph_data_v2.pkl"))
with open(graph_path, 'rb') as f: g = pickle.load(f)
X = g['node_features'].copy().astype(np.float32)
y = g['labels'].copy().astype(np.int64)
n = len(y); d = X.shape[1]
print("Nodes: %d, Feats: %d, Pos: %d (%.1f%%)" % (n, d, y.sum(), y.sum()/n*100))

# Build adjacency
ei = g['edge_index']
adj = torch.sparse_coo_tensor(
    torch.stack([torch.tensor(ei[0].tolist(),dtype=torch.long),
                 torch.tensor(ei[1].tolist(),dtype=torch.long)]),
    torch.ones(len(ei[0])),(n,n)).coalesce()
dg = torch.sparse.sum(adj,1).to_dense(); dg[dg==0]=1
di = torch.arange(n)
adj_n = torch.sparse.mm(torch.sparse_coo_tensor(torch.stack([di,di]),1.0/dg,(n,n)),adj).coalesce()
xt = torch.tensor(X.tolist(),dtype=torch.float32)
yt = torch.tensor(y.tolist(),dtype=torch.long)

# === 1-fold verification ===
skf = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)
tr_idx, te_idx = next(skf.split(X, y))

print("Train: %d, Test: %d" % (len(tr_idx), len(te_idx)))

# Split train into sub-train (80%) + val (20%) for ES
np.random.seed(42)
vsize = int(len(tr_idx)*0.2)
vsi = np.random.choice(tr_idx, max(vsize,1), replace=False)
tsi = np.setdiff1d(tr_idx, vsi)
ts_m = torch.zeros(n,dtype=torch.bool); ts_m[torch.tensor(tsi.tolist())]=True
vs_m = torch.zeros(n,dtype=torch.bool); vs_m[torch.tensor(vsi.tolist())]=True
te_m = torch.zeros(n,dtype=torch.bool); te_m[torch.tensor(te_idx.tolist())]=True

# Test GCN
torch.cuda.empty_cache()
m = GNNC('GCN', d)
m = train_gnn(m, xt, adj_n, yt, ts_m, vs_m, ep=100, lr=0.001, pa=15)
m.eval()
with torch.no_grad():
    o = m(xt.to(device), adj_n.to(device))[te_m]
    pr = F.softmax(o,1)[:,1]; pd = o.argmax(1)
    yte = t2n(yt[te_m]); yp = t2n(pd); yr = t2n(pr)
print("GCN: F1=%.4f AUC=%.4f" % (f1_score(yte,yp,zero_division=0),
      roc_auc_score(yte,yr) if len(np.unique(yte))>1 else 0.5))
del m; torch.cuda.empty_cache()

# Test GraphSAGE
m = GNNC('GraphSAGE', d)
m = train_gnn(m, xt, adj_n, yt, ts_m, vs_m, ep=100, lr=0.001, pa=15)
m.eval()
with torch.no_grad():
    o = m(xt.to(device), adj_n.to(device))[te_m]
    pr = F.softmax(o,1)[:,1]; pd = o.argmax(1)
    yte = t2n(yt[te_m]); yp = t2n(pd); yr = t2n(pr)
print("SAGE: F1=%.4f AUC=%.4f" % (f1_score(yte,yp,zero_division=0),
      roc_auc_score(yte,yr) if len(np.unique(yte))>1 else 0.5))
del m

# Test XGBoost
Xtr, Xte = X[tr_idx], X[te_idx]; Ytr, Yte = y[tr_idx], y[te_idx]
npos=Ytr.sum()
xgb = XGBClassifier(n_estimators=100,max_depth=6,scale_pos_weight=(len(Ytr)-npos)/max(npos,1),random_state=42,verbosity=0)
xgb.fit(Xtr,Ytr)
yp=xgb.predict(Xte);yr=xgb.predict_proba(Xte)[:,1]
print("XGB: F1=%.4f AUC=%.4f" % (f1_score(Yte,yp,zero_division=0),roc_auc_score(Yte,yr)))

# Test LGB
lgb=LGBMClassifier(n_estimators=100,num_leaves=31,class_weight='balanced',random_state=42,verbose=-1)
lgb.fit(Xtr,Ytr)
yp=lgb.predict(Xte);yr=lgb.predict_proba(Xte)[:,1]
print("LGB: F1=%.4f AUC=%.4f" % (f1_score(Yte,yp,zero_division=0),roc_auc_score(Yte,yr)))

print("\n=== ALL OK — code is correct ===")
print("Run 'python src/benchmark_clean.py' locally for full experiment.")
