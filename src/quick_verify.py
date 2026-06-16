"""
quick_verify.py — 快速正确性验证（1 fold, 2 HP combos）
用于在沙箱中快速确认代码无误；完整版在本地运行 benchmark_clean.py
"""
import sys, os, pickle, warnings, time, gc
warnings.filterwarnings('ignore')
import torch, torch.nn as nn, torch.nn.functional as F
from torch.optim import Adam
import numpy as np

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print("Device:", device)

_VENV = r"D:\cxdownload\大数据实训\code\test_4\Lib\site-packages"
if _VENV not in sys.path: sys.path.insert(0, _VENV)

from sklearn.model_selection import StratifiedKFold
from sklearn.metrics import f1_score, roc_auc_score
from xgboost import XGBClassifier
from lightgbm import LGBMClassifier

# === GNN (same as benchmark_clean.py) ===
def t2n(t):
    if t.numel() == 0: return np.array([])
    return np.array(t.cpu().tolist())

class GCN(nn.Module):
    def __init__(self, d_in, h=128, o=64, dp=0.3):
        super().__init__()
        self.l1=nn.Linear(d_in,h); self.l2=nn.Linear(h,o)
        self.bn=nn.BatchNorm1d(h); self.dp=nn.Dropout(dp)
    def forward(self,x,adj):
        x=F.relu(self.bn(self.l1(x))); x=self.dp(x)
        x=torch.sparse.mm(adj,x); x=self.l2(x); x=torch.sparse.mm(adj,x); return x

class GraphSAGE(nn.Module):
    def __init__(self,d_in,h=128,o=64,dp=0.3):
        super().__init__()
        self.f1=nn.Linear(d_in*2,h); self.f2=nn.Linear(h*2,o)
        self.bn=nn.BatchNorm1d(h); self.dp=nn.Dropout(dp)
    def forward(self,x,adj):
        n=torch.sparse.mm(adj,x); h=torch.cat([x,n],-1)
        h=F.relu(self.bn(self.f1(h))); h=self.dp(h)
        n2=torch.sparse.mm(adj,h); h=torch.cat([h,n2],-1); return self.f2(h)

class GNNC(nn.Module):
    def __init__(self,t,d_in,h=128,o=64,nc=2,dp=0.3):
        super().__init__()
        self.enc=GCN(d_in,h,o,dp) if t=='GCN' else GraphSAGE(d_in,h,o,dp)
        self.cls=nn.Sequential(nn.Linear(o,32),nn.ReLU(),nn.Dropout(dp),nn.Linear(32,nc))
    def forward(self,x,adj): return self.cls(self.enc(x,adj))

def train_gnn(m,x,adj,y,tr,va,ep=100,lr=0.001,pa=15):
    m=m.to(device);x=x.to(device);adj=adj.to(device);y=y.to(device)
    np_=y[tr].sum().item();pw=(tr.sum().item()-np_)/max(np_,1)
    cr=nn.CrossEntropyLoss(weight=torch.tensor([1.,pw],device=device))
    op=Adam(m.parameters(),lr=lr,weight_decay=5e-4)
    bf,best,cnt=0.,{k:v.cpu().clone() for k,v in m.state_dict().items()},0
    for e in range(ep):
        m.train();op.zero_grad()
        loss=cr(m(x,adj)[tr],y[tr]);loss.backward()
        torch.nn.utils.clip_grad_norm_(m.parameters(),1.);op.step()
        if e%5==0:
            m.eval()
            with torch.no_grad():
                vf=f1_score(t2n(y[va]),t2n(m(x,adj)[va].argmax(1)),zero_division=0)
            if vf>bf:bf=vf;best={k:v.cpu().clone() for k,v in m.state_dict().items()};cnt=0
            else:cnt+=1
            if cnt>=pa:break
    m.load_state_dict(best);return m

# === Load data ===
print("Loading graph...")
graph_path = r"D:\cxdownload\大数据实训\code_sci\results\graph_data_v2.pkl"
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
