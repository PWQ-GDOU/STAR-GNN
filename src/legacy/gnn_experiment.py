"""
gnn_experiment.py — GNN vs Baseline (memory-optimized)
稀疏邻接矩阵 + GPU 缓存清理
"""
import sys, os, pickle, warnings, time, json, gc
warnings.filterwarnings('ignore')

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.optim import Adam
import numpy as np
import pandas as pd

print("torch:", torch.__version__, "| numpy:", np.__version__)
print("CUDA:", torch.cuda.is_available())
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print("Device:", device)

# venv packages
_VENV_SP = r"D:\cxdownload\大数据实训\code\test_4\Lib\site-packages"
if _VENV_SP not in sys.path:
    sys.path.insert(0, _VENV_SP)

from sklearn.model_selection import StratifiedKFold
from sklearn.metrics import f1_score, recall_score, precision_score, roc_auc_score
from xgboost import XGBClassifier
from lightgbm import LGBMClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import matplotlib.font_manager as fm

for fp in fm.findSystemFonts():
    if 'msyh' in fp.lower() or 'simhei' in fp.lower():
        fm.fontManager.addfont(fp)
plt.rcParams['font.sans-serif'] = ['Microsoft YaHei','SimHei','DejaVu Sans']
plt.rcParams['axes.unicode_minus'] = False

# ==================== Safe torch<->numpy bridge ====================
def torch_to_np(t):
    if t.numel() == 0: return np.array([])
    return np.array(t.cpu().tolist())

# ==================== Config ====================
GRAPH_PATH = r"D:\cxdownload\大数据实训\code_sci\results\graph_data_v2.pkl"
OUTPUT_DIR = r"D:\cxdownload\大数据实训\code_sci\results\experiments"
os.makedirs(OUTPUT_DIR, exist_ok=True)

N_FOLDS = 5; RANDOM_SEED = 42; EPOCHS = 200; LR = 0.001; PATIENCE = 20

# ==================== Load ====================
print("\n" + "="*60)
print("  Loading graph")
print("="*60)
with open(GRAPH_PATH, 'rb') as f:
    graph = pickle.load(f)

X_np = graph['node_features'].copy().astype(np.float32)
y_np = graph['labels'].copy().astype(np.int64)

# torch tensors (safe path)
node_features = torch.tensor(graph['node_features'].tolist(), dtype=torch.float32)
labels_t = torch.tensor(graph['labels'].tolist(), dtype=torch.long)
ei = graph['edge_index']
edge_index = torch.tensor(ei.tolist(), dtype=torch.long)

n_nodes = len(labels_t)
n_features = node_features.shape[1]
print("Nodes: %d  Feats: %d  Pos: %d (%.1f%%)" % (n_nodes, n_features, labels_t.sum().item(), labels_t.sum().item()/n_nodes*100))

# Build SPARSE normalized adjacency (critical for memory!)
print("Building sparse adjacency...")
adj = torch.sparse_coo_tensor(
    torch.stack([edge_index[0], edge_index[1]]),
    torch.ones(len(edge_index[0])),
    (n_nodes, n_nodes)
).coalesce()

# Row-sum normalization: we'll do D^{-1} @ x during message passing
deg = torch.sparse.sum(adj, dim=1).to_dense()
deg_inv = 1.0 / (deg + 1e-8)
deg_inv[torch.isinf(deg_inv)] = 0
# Store as sparse diagonal
diag_idx = torch.arange(n_nodes)
adj_norm_sp = torch.sparse_coo_tensor(
    torch.stack([diag_idx, diag_idx]),
    deg_inv,
    (n_nodes, n_nodes)
)
# Pre-compute: norm_adj = D^{-1} @ adj (sparse)
adj_final = torch.sparse.mm(adj_norm_sp, adj).coalesce()
print("  Sparse adj: %d nnz, %.2f MB" % (adj_final._nnz(), adj_final._nnz() * 8 / 1e6))

def sparse_message_pass(adj_sp, x):
    """Sparse: y_i = sum_j (adj_ij * x_j)"""
    return torch.sparse.mm(adj_sp, x)

# ==================== GNN Models (sparse-compatible) ====================

class GCN(nn.Module):
    def __init__(self, in_dim, hidden=128, out_dim=64, dropout=0.3):
        super().__init__()
        self.lin1 = nn.Linear(in_dim, hidden)
        self.lin2 = nn.Linear(hidden, out_dim)
        self.bn = nn.BatchNorm1d(hidden)
        self.dropout = nn.Dropout(dropout)

    def forward(self, x, adj_sp):
        x = F.relu(self.bn(self.lin1(x)))
        x = self.dropout(x)
        x = sparse_message_pass(adj_sp, x)
        x = self.lin2(x)
        x = sparse_message_pass(adj_sp, x)
        return x


class GraphSAGE(nn.Module):
    def __init__(self, in_dim, hidden=128, out_dim=64, dropout=0.3):
        super().__init__()
        self.fc1 = nn.Linear(in_dim * 2, hidden)
        self.fc2 = nn.Linear(hidden * 2, out_dim)
        self.bn = nn.BatchNorm1d(hidden)
        self.dropout = nn.Dropout(dropout)

    def forward(self, x, adj_sp):
        neigh = sparse_message_pass(adj_sp, x)
        h = torch.cat([x, neigh], dim=-1)
        h = F.relu(self.bn(self.fc1(h)))
        h = self.dropout(h)
        neigh2 = sparse_message_pass(adj_sp, h)
        h = torch.cat([h, neigh2], dim=-1)
        return self.fc2(h)


class GNNClassifier(nn.Module):
    def __init__(self, enc_type, in_dim, hidden=128, gnn_out=64, n_classes=2, dropout=0.3):
        super().__init__()
        if enc_type == 'GCN':
            self.encoder = GCN(in_dim, hidden, gnn_out, dropout)
        elif enc_type == 'GraphSAGE':
            self.encoder = GraphSAGE(in_dim, hidden, gnn_out, dropout)
        else:
            raise ValueError(enc_type)
        self.cls = nn.Sequential(
            nn.Linear(gnn_out, 32), nn.ReLU(), nn.Dropout(dropout),
            nn.Linear(32, n_classes)
        )
        self.enc_type = enc_type

    def forward(self, x, adj_sp):
        return self.cls(self.encoder(x, adj_sp))


# ==================== Training ====================

def train_gnn(model, x, adj_sp, y, train_mask, val_mask, epochs=200, lr=0.001, patience=20):
    model = model.to(device)
    x = x.to(device)
    adj_sp = adj_sp.to(device)
    y = y.to(device)

    n_pos_tr = y[train_mask].sum().item()
    n_neg_tr = train_mask.sum().item() - n_pos_tr
    pw = n_neg_tr / max(n_pos_tr, 1)
    criterion = nn.CrossEntropyLoss(weight=torch.tensor([1.0, pw], device=device))
    opt = Adam(model.parameters(), lr=lr, weight_decay=5e-4)
    best_f1, best_state, count = 0.0, {k: v.cpu().clone() for k, v in model.state_dict().items()}, 0
    eval_every = 5  # check every 5 epochs

    for ep in range(epochs):
        model.train()
        opt.zero_grad()
        out = model(x, adj_sp)
        loss = criterion(out[train_mask], y[train_mask])
        loss.backward()
        opt.step()

        if ep % eval_every == 0 or ep == epochs - 1:
            model.eval()
            with torch.no_grad():
                vo = model(x, adj_sp)
                vp = vo[val_mask].argmax(1)
                vt = y[val_mask]
                vf1 = f1_score(torch_to_np(vt), torch_to_np(vp), zero_division=0)
            if vf1 > best_f1:
                best_f1 = vf1
                best_state = {k: v.cpu().clone() for k, v in model.state_dict().items()}
                count = 0
            else:
                count += 1
            if count >= patience:
                break

    model.load_state_dict(best_state)
    return model

def eval_gnn(model, x, adj_sp, y, mask):
    model.eval()
    with torch.no_grad():
        out = model(x, adj_sp)[mask]
        prob = F.softmax(out, dim=1)[:, 1]
        pred = out.argmax(1)
        yt = torch_to_np(y[mask])
        yp = torch_to_np(pred)
        ypr = torch_to_np(prob)
    return {
        'f1': f1_score(yt, yp, zero_division=0),
        'prec': precision_score(yt, yp, zero_division=0),
        'rec': recall_score(yt, yp, zero_division=0),
        'auc': roc_auc_score(yt, ypr) if len(np.unique(yt)) > 1 else 0.5,
    }

# ==================== Baseline ====================

def train_baseline(name, X_tr, y_tr, X_te, y_te):
    n_pos = y_tr.sum(); n_neg = len(y_tr) - n_pos
    if name == 'XGBoost':
        m = XGBClassifier(n_estimators=100, max_depth=6, learning_rate=0.1,
                          scale_pos_weight=n_neg/max(n_pos,1), random_state=RANDOM_SEED, verbosity=0)
    elif name == 'LightGBM':
        m = LGBMClassifier(n_estimators=100, num_leaves=31, learning_rate=0.1,
                           class_weight='balanced', random_state=RANDOM_SEED, verbose=-1)
    elif name == 'RF':
        m = RandomForestClassifier(n_estimators=100, max_depth=10,
                                   class_weight='balanced', random_state=RANDOM_SEED, n_jobs=-1)
    elif name == 'LR':
        m = LogisticRegression(max_iter=1000, class_weight='balanced', random_state=RANDOM_SEED)
    else: raise ValueError(name)
    m.fit(X_tr, y_tr)
    yp = m.predict(X_te)
    ypr = m.predict_proba(X_te)[:,1] if hasattr(m,'predict_proba') else None
    r = {'f1': f1_score(y_te,yp,zero_division=0),
         'prec': precision_score(y_te,yp,zero_division=0),
         'rec': recall_score(y_te,yp,zero_division=0)}
    r['auc'] = roc_auc_score(y_te,ypr) if ypr is not None and len(np.unique(y_te))>1 else 0.5
    return r

# ==================== CV ====================

def run_cv():
    print("\n" + "="*60)
    print("  %d-Fold CV" % N_FOLDS)
    print("="*60)

    skf = StratifiedKFold(n_splits=N_FOLDS, shuffle=True, random_state=RANDOM_SEED)
    results = {m: {'f1':[],'prec':[],'rec':[],'auc':[]}
               for m in ['GCN','GraphSAGE','XGBoost','LightGBM','RF','LR']}

    for fold, (tr_idx, te_idx) in enumerate(skf.split(X_np, y_np)):
        print("\n--- Fold %d/%d ---" % (fold+1, N_FOLDS))
        t0 = time.time()

        tr_mask = torch.zeros(n_nodes, dtype=torch.bool)
        te_mask = torch.zeros(n_nodes, dtype=torch.bool)
        tr_mask[torch.tensor(tr_idx.tolist())] = True
        te_mask[torch.tensor(te_idx.tolist())] = True

        Xt, Xe = X_np[tr_idx], X_np[te_idx]
        Yt, Ye = y_np[tr_idx], y_np[te_idx]
        print("  Train: %d (pos=%d) Test: %d (pos=%d)" % (len(tr_idx), Yt.sum(), len(te_idx), Ye.sum()))

        # GNNs
        x_t = node_features.to(device)
        adj_t = adj_final.to(device)
        y_t = labels_t.to(device)

        for g in ['GCN', 'GraphSAGE']:
            torch.cuda.empty_cache()
            m = GNNClassifier(g, n_features)
            m = train_gnn(m, x_t, adj_t, y_t, tr_mask, te_mask)
            r = eval_gnn(m, x_t, adj_t, y_t, te_mask)
            for k in ['f1','prec','rec','auc']:
                results[g][k].append(r[k])
            print("  %-12s F1=%.4f AUC=%.4f Rec=%.4f" % (g, r['f1'], r['auc'], r['rec']))
            del m; gc.collect(); torch.cuda.empty_cache()

        # Baselines
        for b in ['XGBoost','LightGBM','RF','LR']:
            r = train_baseline(b, Xt, Yt, Xe, Ye)
            for k in ['f1','prec','rec','auc']:
                results[b][k].append(r[k])
            print("  %-12s F1=%.4f AUC=%.4f" % (b, r['f1'], r['auc']))

        print("  Time: %.1f min" % ((time.time()-t0)/60))

    # Summary
    print("\n" + "="*60)
    print("  RESULTS (%d-fold)" % N_FOLDS)
    print("="*60)
    print("%-14s %10s %10s %10s %10s" % ("Model","F1","AUC","Precision","Recall"))
    print("-"*58)

    summary = {}
    order = ['LR','RF','LightGBM','XGBoost','GCN','GraphSAGE']
    for m in order:
        if m not in results: continue
        r = results[m]
        sm = {k+'_mean': np.mean(r[k]) for k in ['f1','prec','rec','auc']}
        sm.update({k+'_std': np.std(r[k]) for k in ['f1','prec','rec','auc']})
        summary[m] = sm
        print("%-14s %.4f±%.3f %.4f±%.3f %.4f±%.3f %.4f±%.3f" %
              (m, sm['f1_mean'], sm['f1_std'], sm['auc_mean'], sm['auc_std'],
               sm['prec_mean'], sm['prec_std'], sm['rec_mean'], sm['rec_std']))

    return summary

# ==================== Ablation ====================

def run_ablation():
    print("\n" + "="*60)
    print("  ABLATION")
    print("="*60)

    skf = StratifiedKFold(n_splits=3, shuffle=True, random_state=RANDOM_SEED)  # fewer folds for speed
    variants = {
        'GCN_full (79feat)':     node_features,
        'GCN_base_only (7feat)': node_features[:, :7],
        'GCN_structure_only':    torch.eye(n_nodes),
    }
    results = {k: {'f1':[],'auc':[]} for k in variants}

    for fold, (tr_idx, te_idx) in enumerate(skf.split(X_np, y_np)):
        print("\n--- Ablation Fold %d ---" % (fold+1))
        tr_mask = torch.zeros(n_nodes, dtype=torch.bool)
        te_mask = torch.zeros(n_nodes, dtype=torch.bool)
        tr_mask[torch.tensor(tr_idx.tolist())] = True
        te_mask[torch.tensor(te_idx.tolist())] = True

        for name, feat in variants.items():
            torch.cuda.empty_cache()
            m = GNNClassifier('GCN', feat.shape[1])
            m = train_gnn(m, feat.to(device), adj_final.to(device), labels_t.to(device),
                         tr_mask, te_mask, epochs=200, lr=0.001, patience=20)
            r = eval_gnn(m, feat.to(device), adj_final.to(device), labels_t.to(device), te_mask)
            results[name]['f1'].append(r['f1'])
            results[name]['auc'].append(r['auc'])
            print("  %-25s F1=%.4f AUC=%.4f" % (name, r['f1'], r['auc']))
            del m; gc.collect(); torch.cuda.empty_cache()

    print("\n--- Ablation Summary ---")
    for n in results:
        f = results[n]['f1']
        print("  %-25s F1=%.4f±%.3f" % (n, np.mean(f), np.std(f)))
    return results

# ==================== Plot ====================

def plot_results(summary, ablation):
    order = ['LR','RF','LightGBM','XGBoost','GCN','GraphSAGE']
    models = [m for m in order if m in summary]
    colors = ['#e74c3c' if m in ('GCN','GraphSAGE') else '#3498db' for m in models]

    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 5))
    for ax, met in [(ax1, 'f1'), (ax2, 'auc')]:
        means = [summary[m][met+'_mean'] for m in models]
        stds = [summary[m][met+'_std'] for m in models]
        bars = ax.bar(models, means, yerr=stds, color=colors, capsize=5, alpha=0.85)
        ax.set_ylabel(met.upper())
        ax.set_title('Model Comparison: %s' % met.upper())
        lo = max(0, min(means)-0.15)
        ax.set_ylim(lo, min(1.02, max(means)+0.15))
        for b, v in zip(bars, means):
            ax.text(b.get_x()+b.get_width()/2, b.get_height()+0.01, '%.3f'%v,
                    ha='center', va='bottom', fontsize=8)

    plt.tight_layout()
    p = os.path.join(OUTPUT_DIR, 'gnn_vs_baseline.png')
    plt.savefig(p, dpi=150, bbox_inches='tight'); plt.close()
    print("Chart:", p)

    # Ablation
    fig, ax = plt.subplots(figsize=(10, 4))
    names = list(ablation.keys())
    f1m = [np.mean(ablation[n]['f1']) for n in names]
    f1s = [np.std(ablation[n]['f1']) for n in names]
    ax.barh(names, f1m, xerr=f1s, color=['#2ecc71','#f39c12','#e74c3c'], capsize=4, alpha=0.85)
    ax.set_xlabel('F1 Score')
    ax.set_title('Ablation: GCN Feature Contribution')
    for i, v in enumerate(f1m):
        ax.text(v+0.005, i, '%.3f'%v, va='center', fontsize=11)
    plt.tight_layout()
    p = os.path.join(OUTPUT_DIR, 'ablation.png')
    plt.savefig(p, dpi=150, bbox_inches='tight'); plt.close()
    print("Ablation:", p)

# ==================== Main ====================

def main():
    print("\n" + "="*60)
    print("  GNN vs Baseline | Device: %s" % device)
    print("="*60)

    summary = run_cv()
    ablation = run_ablation()
    plot_results(summary, ablation)

    out = {
        'summary': {k: {kk: float(vv) for kk, vv in v.items()} for k, v in summary.items()},
        'ablation': {k: {kk: [float(x) for x in vv] for kk, vv in v.items()} for k, v in ablation.items()}
    }
    jp = os.path.join(OUTPUT_DIR, 'results.json')
    with open(jp, 'w') as f:
        json.dump(out, f, indent=2)
    print("\nSaved:", jp)

if __name__ == '__main__':
    main()
