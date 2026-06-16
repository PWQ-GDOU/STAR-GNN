"""
ablation_infer.py — 特征消融 + 推理效率 (Enterprise only)
"""
import pickle, warnings, time, json, os, gc
warnings.filterwarnings('ignore')
import numpy as np

PROJECT_ROOT = os.environ.get("STAR_GNN_HOME", r"D:\cxdownload\大数据实训\code_sci")
RESULTS_DIR = os.environ.get("STAR_GNN_RESULTS", os.path.join(PROJECT_ROOT, "results", "benchmark"))
os.makedirs(RESULTS_DIR, exist_ok=True)
GRAPH_PATH = os.path.join(PROJECT_ROOT, "results", "graph_data_v2.pkl")

import torch, torch.nn as nn, torch.nn.functional as F
from torch.optim import Adam

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print("Device:", device)
RANDOM_SEED = 42

from sklearn.model_selection import StratifiedKFold
from sklearn.metrics import f1_score, roc_auc_score
from xgboost import XGBClassifier
import matplotlib; matplotlib.use('Agg')
import matplotlib.pyplot as plt

# Feature levels — defined BEFORE use; populated after loading data
FEATURE_LEVELS = None

# ================ Load data ================
print("Loading: %s" % GRAPH_PATH)
with open(GRAPH_PATH, 'rb') as f:
    g = pickle.load(f)
X = g['node_features'].copy().astype(np.float32)
y = g['labels'].copy().astype(np.int64)

# Feature ranking
n_pos = y.sum()
clf = XGBClassifier(n_estimators=100, max_depth=6,
                     scale_pos_weight=(len(y)-n_pos)/max(n_pos,1),
                     random_state=42, verbosity=0).fit(X, y)
feat_order = np.argsort(clf.feature_importances_)[::-1]

n_feats = X.shape[1]; n_nodes = len(y)
print("Data: %d nodes x %d features" % (n_nodes, n_feats))

# Dynamically set feature levels
max_dims = X.shape[1]
if FEATURE_LEVELS is None:
    base_levels = [max_dims, 50, 30, 20, 15, 10, 7, 5, 3]
    FEATURE_LEVELS = sorted(set([l for l in base_levels if l <= max_dims]), reverse=True)
    if max_dims not in FEATURE_LEVELS:
        FEATURE_LEVELS.insert(0, max_dims)
print("Feature levels: %s" % FEATURE_LEVELS)

# Build graph adjacency
ei = g['edge_index']
adj = torch.sparse_coo_tensor(
    torch.stack([torch.tensor(ei[0].tolist(), dtype=torch.long),
                 torch.tensor(ei[1].tolist(), dtype=torch.long)]),
    torch.ones(len(ei[0])), (n_nodes, n_nodes)).coalesce()
deg = torch.sparse.sum(adj, 1).to_dense(); deg[deg==0] = 1
di = torch.arange(n_nodes)
adj_norm = torch.sparse.mm(
    torch.sparse_coo_tensor(torch.stack([di,di]), 1.0/deg, (n_nodes,n_nodes)), adj).coalesce()

# ================ GNN ================
class GraphSAGE(nn.Module):
    def __init__(self, d_in, h=128, o=64, dp=0.3):
        super().__init__()
        self.f1 = nn.Linear(d_in*2, h); self.f2 = nn.Linear(h*2, o)
        self.bn = nn.BatchNorm1d(h); self.dp = nn.Dropout(dp)
    def forward(self, x, adj):
        n = torch.sparse.mm(adj, x); h = torch.cat([x, n], -1)
        h = F.relu(self.bn(self.f1(h))); h = self.dp(h)
        n2 = torch.sparse.mm(adj, h); h = torch.cat([h, n2], -1)
        return self.f2(h)

class GNNC(nn.Module):
    def __init__(self, d_in, h=128, o=64, nc=2, dp=0.3):
        super().__init__()
        self.enc = GraphSAGE(d_in, h, o, dp)
        self.cls = nn.Sequential(nn.Linear(o,32), nn.ReLU(), nn.Dropout(dp), nn.Linear(32,nc))
    def forward(self, x, adj): return self.cls(self.enc(x, adj))

def t2n(t):
    if t.numel() == 0: return np.array([])
    return np.array(t.cpu().tolist())

def train_gnn(m, x, adj, y, tr, va, ep=200, lr=0.002, pa=30):
    m = m.to(device); x = x.to(device); adj = adj.to(device); y = y.to(device)
    np_ = y[tr].sum().item(); pw = (tr.sum().item() - np_) / max(np_, 1)
    cr = nn.CrossEntropyLoss(weight=torch.tensor([1., pw], device=device))
    op = Adam(m.parameters(), lr=lr, weight_decay=5e-4)
    bf, best, cnt = 0., {k: v.cpu().clone() for k,v in m.state_dict().items()}, 0
    for e in range(ep):
        m.train(); op.zero_grad()
        loss = cr(m(x,adj)[tr], y[tr]); loss.backward()
        torch.nn.utils.clip_grad_norm_(m.parameters(), 1.); op.step()
        if e % 5 == 0:
            m.eval()
            with torch.no_grad():
                vf = f1_score(t2n(y[va]), t2n(m(x, adj)[va].argmax(1)), zero_division=0)
            if vf > bf: bf = vf; best = {k: v.cpu().clone() for k,v in m.state_dict().items()}; cnt = 0
            else: cnt += 1
            if cnt >= pa: break
    m.load_state_dict(best); return m

def eval_gnn(m, x, adj, y, mask):
    m.eval(); x = x.to(device); adj = adj.to(device); y = y.to(device)
    with torch.no_grad():
        o = m(x, adj)[mask]; pr = F.softmax(o, 1)[:, 1]; pd = o.argmax(1)
        yt = t2n(y[mask]); yp = t2n(pd); yr = t2n(pr)
    return {'f1': f1_score(yt, yp, zero_division=0),
            'auc': roc_auc_score(yt, yr) if len(np.unique(yt)) > 1 else 0.5}

# ================ Experiment ================
skf = StratifiedKFold(n_splits=3, shuffle=True, random_state=RANDOM_SEED)
N_WARMUP = 50

results = {d: {'xgb_f1': [], 'xgb_auc': [], 'xgb_time': [],
                'sage_f1': [], 'sage_auc': [], 'sage_time': []} for d in FEATURE_LEVELS}

print("\n" + "="*60)
print("  Feature Ablation: %s -> %s dims" % (FEATURE_LEVELS[0], FEATURE_LEVELS[-1]))
print("="*60)

for n_dim in FEATURE_LEVELS:
    print("\n--- %d features ---" % n_dim)
    sel_idx = feat_order[:n_dim]
    X_sel = X[:, sel_idx]
    feat_t = torch.tensor(X_sel.tolist(), dtype=torch.float32)
    d = n_dim
    y_t = torch.tensor(y.tolist(), dtype=torch.long)

    for fold, (tr_idx, te_idx) in enumerate(skf.split(X_sel, y)):
        te_m = torch.zeros(n_nodes, dtype=torch.bool)
        te_m[torch.tensor(te_idx.tolist())] = True
        Xtr, Xte = X_sel[tr_idx], X_sel[te_idx]
        Ytr, Yte = y[tr_idx], y[te_idx]
        npos = Ytr.sum()

        # ES split: train-subset (80%) + val-subset (20%)
        np.random.seed(RANDOM_SEED + fold)
        vsize = int(len(tr_idx) * 0.2)
        vsi = np.random.choice(tr_idx, max(vsize, 1), replace=False).tolist()
        tsi = [x for x in tr_idx if x not in set(vsi)]
        ts_m = torch.zeros(n_nodes, dtype=torch.bool)
        vs_m = torch.zeros(n_nodes, dtype=torch.bool)
        ts_m[torch.tensor(tsi)] = True
        vs_m[torch.tensor(vsi)] = True

        # XGBoost
        xgb = XGBClassifier(n_estimators=100, max_depth=6,
                            scale_pos_weight=(len(Ytr)-npos)/max(npos,1),
                            random_state=42, verbosity=0)
        t0 = time.time(); xgb.fit(Xtr, Ytr); train_time = time.time() - t0
        for _ in range(N_WARMUP): _ = xgb.predict(Xte[:10])
        t0 = time.time()
        for _ in range(100): _ = xgb.predict(Xte)
        infer_time = (time.time() - t0) / 100 * 1000
        yp = xgb.predict(Xte); ypr = xgb.predict_proba(Xte)[:, 1]
        results[n_dim]['xgb_f1'].append(f1_score(Yte, yp, zero_division=0))
        results[n_dim]['xgb_auc'].append(roc_auc_score(Yte, ypr))
        results[n_dim]['xgb_time'].append(infer_time)

        # GNN
        torch.cuda.empty_cache(); gc.collect()
        m = GNNC(d, h=128)
        m = train_gnn(m, feat_t, adj_norm, y_t, ts_m, vs_m, ep=200, lr=0.002, pa=30)
        with torch.no_grad():
            ft = feat_t.to(device); at = adj_norm.to(device)
            for _ in range(N_WARMUP): _ = m(ft, at)
            torch.cuda.synchronize()
            t0 = time.time()
            for _ in range(100): _ = m(ft, at)
            torch.cuda.synchronize()
            gnn_time = (time.time() - t0) / 100 * 1000
        r = eval_gnn(m, feat_t, adj_norm, y_t, te_m)
        results[n_dim]['sage_f1'].append(r['f1']); results[n_dim]['sage_auc'].append(r['auc'])
        results[n_dim]['sage_time'].append(gnn_time)
        del m

        print("  %dd, fold %d: XGB F1=%.4f (%.1fms) | SAGE F1=%.4f (%.1fms)" %
              (n_dim, fold+1, results[n_dim]['xgb_f1'][-1], infer_time,
               results[n_dim]['sage_f1'][-1], gnn_time))

# ================ Summary ================
print("\n" + "="*60)
print("  ABLATION SUMMARY")
print("="*60)
print("%-8s %12s %12s %12s %12s %12s" % ("Dims", "XGB F1", "SAGE F1", "Delta", "XGB ms", "SAGE ms"))
print("-"*70)

summary = []
for n_dim in FEATURE_LEVELS:
    r = results[n_dim]
    xgb_f1 = np.mean(r['xgb_f1']); sage_f1 = np.mean(r['sage_f1'])
    delta = sage_f1 - xgb_f1
    xgb_t = np.mean(r['xgb_time']); sage_t = np.mean(r['sage_time'])
    print("%-8d %12.4f %12.4f %+12.4f %10.1f %10.1f" % (n_dim, xgb_f1, sage_f1, delta, xgb_t, sage_t))
    summary.append({'dims': n_dim, 'xgb_f1': float(xgb_f1), 'sage_f1': float(sage_f1),
                    'delta': float(delta), 'xgb_time': float(xgb_t), 'sage_time': float(sage_t)})

# Crossover
for s in summary:
    if s['delta'] > 0:
        print("  CROSSOVER at ~%d dimensions! (Delta = %+.4f)" % (s['dims'], s['delta'])); break
else:
    best = min(summary, key=lambda s: abs(s['delta']))
    print("  No crossover. Closest to zero at %d dims (Delta = %+.4f)" % (best['dims'], best['delta']))

xgb_avg_t = np.mean([s['xgb_time'] for s in summary])
sage_avg_t = np.mean([s['sage_time'] for s in summary])
speedup = sage_avg_t / max(xgb_avg_t, 1e-6)
print("  Avg speed: XGB=%.1fms | SAGE=%.1fms | SAGE is %.1fx" % (xgb_avg_t, sage_avg_t, speedup))

# ================ Plot ================
fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 5))
dims = [s['dims'] for s in summary]
xgb_f1s = [s['xgb_f1'] for s in summary]
sage_f1s = [s['sage_f1'] for s in summary]
deltas = [s['delta'] for s in summary]

ax1.plot(dims, xgb_f1s, 'o-', color='#3498db', lw=2, markersize=8, label='XGBoost')
ax1.plot(dims, sage_f1s, 's--', color='#2ecc71', lw=2, markersize=8, label='GraphSAGE')
for i, d in enumerate(deltas):
    if d > -0.01:
        ax1.axvline(x=dims[i], color='#e74c3c', ls=':', alpha=0.7)
        ax1.annotate('Gap closes\nat %dd' % dims[i], xy=(dims[i], (xgb_f1s[i]+sage_f1s[i])/2),
                    xytext=(dims[i]+5, (xgb_f1s[i]+sage_f1s[i])/2),
                    arrowprops=dict(arrowstyle='->', color='#e74c3c'), fontsize=9, color='#e74c3c')
ax1.set_xlabel('Number of Features'); ax1.set_ylabel('F1 Score')
ax1.set_title('Feature Ablation: XGB vs GraphSAGE'); ax1.legend(); ax1.grid(True, alpha=0.3)

ax2.bar(np.array(dims)-2, [s['xgb_time'] for s in summary], width=4, color='#3498db', alpha=0.85, label='XGBoost')
ax2.bar(np.array(dims)+2, [s['sage_time'] for s in summary], width=4, color='#2ecc71', alpha=0.85, label='GraphSAGE')
ax2.set_xlabel('Number of Features'); ax2.set_ylabel('Inference Time (ms)')
ax2.set_title('Inference Efficiency (SAGE/XGB = %.1fx)' % speedup); ax2.legend()

plt.tight_layout()
out_png = os.path.join(RESULTS_DIR, "ablation_crossover.png")
plt.savefig(out_png, dpi=150, bbox_inches='tight'); plt.close()
print("\nSaved:", out_png)

out_json = os.path.join(RESULTS_DIR, "ablation_results.json")
with open(out_json, 'w') as f: json.dump(summary, f, indent=2)
print("Saved:", out_json)
print("\n=== DONE ===")
