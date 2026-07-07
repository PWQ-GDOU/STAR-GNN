"""
gnn_embed_xgb.py — 方案B: GNN嵌入 + 表格特征 → XGBoost
"""
import sys, os, pickle, warnings, time, json, gc
warnings.filterwarnings('ignore')

import torch, torch.nn as nn, torch.nn.functional as F
from torch.optim import Adam
import numpy as np, pandas as pd

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print("Device:", device)

_VENV_SP = r"D:\cxdownload\大数据实训\code\test_4\Lib\site-packages"
if _VENV_SP not in sys.path: sys.path.insert(0, _VENV_SP)

from sklearn.model_selection import StratifiedKFold
from sklearn.metrics import f1_score, recall_score, precision_score, roc_auc_score, accuracy_score
from xgboost import XGBClassifier
from lightgbm import LGBMClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression
import matplotlib; matplotlib.use('Agg')
import matplotlib.pyplot as plt

def torch_to_np(t):
    if t.numel() == 0: return np.array([])
    return np.array(t.cpu().tolist())

# ============ Load graph ============
GRAPH_PATH = r"D:\cxdownload\大数据实训\code_sci\results\graph_data_v2.pkl"
OUTPUT_DIR = r"D:\cxdownload\大数据实训\code_sci\results\experiments"
os.makedirs(OUTPUT_DIR, exist_ok=True)

with open(GRAPH_PATH, 'rb') as f:
    graph = pickle.load(f)

X_np = graph['node_features'].copy().astype(np.float32)
y_np = graph['labels'].copy().astype(np.int64)

node_features = torch.tensor(graph['node_features'].tolist(), dtype=torch.float32)
labels_t = torch.tensor(graph['labels'].tolist(), dtype=torch.long)
ei = graph['edge_index']
edge_index = torch.tensor(ei.tolist(), dtype=torch.long)
n_nodes = len(labels_t); n_features = node_features.shape[1]

print("Nodes: %d  Feats: %d  Pos: %d (%.1f%%)" % (n_nodes, n_features, labels_t.sum().item(), labels_t.sum().item()/n_nodes*100))

# Build sparse normalized adjacency
adj = torch.sparse_coo_tensor(
    torch.stack([edge_index[0], edge_index[1]]),
    torch.ones(len(edge_index[0])), (n_nodes, n_nodes)
).coalesce()
deg = torch.sparse.sum(adj, dim=1).to_dense()
deg_inv = 1.0 / (deg + 1e-8); deg_inv[torch.isinf(deg_inv)] = 0
diag_idx = torch.arange(n_nodes)
adj_norm_sp = torch.sparse_coo_tensor(
    torch.stack([diag_idx, diag_idx]), deg_inv, (n_nodes, n_nodes)
)
adj_final = torch.sparse.mm(adj_norm_sp, adj).coalesce()

def sparse_mp(adj_sp, x):
    return torch.sparse.mm(adj_sp, x)

# ============ GNN Encoder (unsupervised, for embedding) ============

class GNNEncoder(nn.Module):
    def __init__(self, enc_type, in_dim, hidden=256, out_dim=128, dropout=0.3):
        super().__init__()
        self.enc_type = enc_type
        if enc_type == 'GCN':
            self.lin1 = nn.Linear(in_dim, hidden)
            self.lin2 = nn.Linear(hidden, out_dim)
            self.bn = nn.BatchNorm1d(hidden)
        elif enc_type == 'GraphSAGE':
            self.fc1 = nn.Linear(in_dim * 2, hidden)
            self.fc2 = nn.Linear(hidden * 2, out_dim)
            self.bn = nn.BatchNorm1d(hidden)
        self.dropout = nn.Dropout(dropout)

    def forward(self, x, adj_sp):
        if self.enc_type == 'GCN':
            h = F.relu(self.bn(self.lin1(x)))
            h = self.dropout(h)
            h = sparse_mp(adj_sp, h)
            h = self.lin2(h)
            h = sparse_mp(adj_sp, h)
            return h
        else:
            neigh = sparse_mp(adj_sp, x)
            h = torch.cat([x, neigh], dim=-1)
            h = F.relu(self.bn(self.fc1(h)))
            h = self.dropout(h)
            neigh2 = sparse_mp(adj_sp, h)
            h = torch.cat([h, neigh2], dim=-1)
            return self.fc2(h)

class LinkPredGNN(nn.Module):
    """Train GNN with link prediction loss + supervised loss"""
    def __init__(self, enc_type, in_dim, hidden=256, out_dim=128, n_classes=2):
        super().__init__()
        self.encoder = GNNEncoder(enc_type, in_dim, hidden, out_dim)
        self.cls = nn.Linear(out_dim, n_classes)

    def forward(self, x, adj_sp):
        emb = self.encoder(x, adj_sp)
        return emb, self.cls(emb)


def train_gnn_supervised(model, x, adj_sp, y, train_mask, val_mask,
                         epochs=300, lr=0.002, patience=40):
    model = model.to(device)
    x, adj_sp, y = x.to(device), adj_sp.to(device), y.to(device)

    n_pos = y[train_mask].sum().item()
    n_neg = train_mask.sum().item() - n_pos
    pw = n_neg / max(n_pos, 1)
    crit = nn.CrossEntropyLoss(weight=torch.tensor([1.0, pw], device=device))
    opt = Adam(model.parameters(), lr=lr, weight_decay=1e-4)
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(opt, mode='max', factor=0.5, patience=15)

    best_f1, best_state, count = 0.0, None, 0

    for ep in range(epochs):
        model.train()
        opt.zero_grad()
        emb, out = model(x, adj_sp)
        loss = crit(out[train_mask], y[train_mask])
        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
        opt.step()

        if ep % 5 == 0:
            model.eval()
            with torch.no_grad():
                _, out_v = model(x, adj_sp)
                vp = out_v[val_mask].argmax(1)
                vt = y[val_mask]
                vf1 = f1_score(torch_to_np(vt), torch_to_np(vp), zero_division=0)
            scheduler.step(vf1)

            if vf1 > best_f1:
                best_f1 = vf1
                best_state = {k: v.cpu().clone() for k, v in model.state_dict().items()}
                count = 0
            else:
                count += 1
            if count >= patience:
                break

    model.load_state_dict(best_state)
    return model, best_f1

# ============ Extract embeddings ============

def extract_embeddings(model, x, adj_sp):
    model.eval()
    with torch.no_grad():
        emb, _ = model(x.to(device), adj_sp.to(device))
    return torch_to_np(emb)

# ============ Cross Validation (all methods) ============

def run_cv(gnn_type='GraphSAGE', emb_dim=128, epochs=300):
    print("\n" + "="*60)
    print("  GNN Embedding + Tabular → XGBoost/LightGBM")
    print("  GNN: %s | Emb dim: %d" % (gnn_type, emb_dim))
    print("="*60)

    skf = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)
    methods = {
        'XGB_tab_only':      [],   # pure tabular
        'LGB_tab_only':      [],
        'XGB_tab+emb':        [],   # tabular + GNN embedding
        'LGB_tab+emb':       [],
        'XGB_emb_only':       [],   # pure embedding
        'LGB_emb_only':      [],
    }
    for m in methods:
        methods[m] = {'f1':[], 'auc':[], 'prec':[], 'rec':[]}
    emb_results_raw = []

    for fold, (tr_idx, te_idx) in enumerate(skf.split(X_np, y_np)):
        print("\n--- Fold %d/5 ---" % (fold+1))

        tr_mask = torch.zeros(n_nodes, dtype=torch.bool)
        te_mask = torch.zeros(n_nodes, dtype=torch.bool)
        tr_mask[torch.tensor(tr_idx.tolist())] = True
        te_mask[torch.tensor(te_idx.tolist())] = True

        # Train GNN on train fold
        torch.cuda.empty_cache()
        m = LinkPredGNN(gnn_type, n_features, hidden=256, out_dim=emb_dim)
        m, val_f1 = train_gnn_supervised(m, node_features, adj_final, labels_t,
                                          tr_mask, te_mask, epochs=epochs)
        print("  GNN val F1: %.4f" % val_f1)

        # Extract embeddings for ALL nodes
        emb_all = extract_embeddings(m, node_features, adj_final)
        del m; gc.collect(); torch.cuda.empty_cache()

        # --- Prepare feature sets ---
        X_tr_tab, X_te_tab = X_np[tr_idx], X_np[te_idx]
        Y_tr, Y_te = y_np[tr_idx], y_np[te_idx]

        X_tr_emb, X_te_emb = emb_all[tr_idx], emb_all[te_idx]
        X_tr_both = np.hstack([X_tr_tab, X_tr_emb])
        X_te_both = np.hstack([X_te_tab, X_te_emb])

        # Evaluate all combinations
        for label, (X_tr, X_te) in [
            ('XGB_tab_only', (X_tr_tab, X_te_tab)),
            ('LGB_tab_only', (X_tr_tab, X_te_tab)),
            ('XGB_tab+emb', (X_tr_both, X_te_both)),
            ('LGB_tab+emb', (X_tr_both, X_te_both)),
            ('XGB_emb_only', (X_tr_emb, X_te_emb)),
            ('LGB_emb_only', (X_tr_emb, X_te_emb)),
        ]:
            n_pos = Y_tr.sum(); n_neg = len(Y_tr) - n_pos
            if 'XGB' in label:
                clf = XGBClassifier(n_estimators=200, max_depth=6, learning_rate=0.05,
                                    scale_pos_weight=n_neg/max(n_pos,1),
                                    random_state=42, verbosity=0)
            else:
                clf = LGBMClassifier(n_estimators=200, num_leaves=31, learning_rate=0.05,
                                     class_weight='balanced', random_state=42, verbose=-1)

            clf.fit(X_tr, Y_tr)
            yp = clf.predict(X_te)
            ypr = clf.predict_proba(X_te)[:,1]

            methods[label]['f1'].append(f1_score(Y_te, yp, zero_division=0))
            methods[label]['auc'].append(roc_auc_score(Y_te, ypr))
            methods[label]['prec'].append(precision_score(Y_te, yp, zero_division=0))
            methods[label]['rec'].append(recall_score(Y_te, yp, zero_division=0))

        print("  XGB tab-only F1=%.4f  |  XGB tab+emb F1=%.4f" %
              (methods['XGB_tab_only']['f1'][-1], methods['XGB_tab+emb']['f1'][-1]))
        print("  LGB tab-only F1=%.4f  |  LGB tab+emb F1=%.4f" %
              (methods['LGB_tab_only']['f1'][-1], methods['LGB_tab+emb']['f1'][-1]))

    # Summary
    print("\n" + "="*60)
    print("  GNN EMBEDDING + TABULAR RESULTS")
    print("="*60)
    summary = {}
    for label in methods:
        r = methods[label]
        s = {k: (np.mean(r[k]), np.std(r[k])) for k in ['f1','auc','prec','rec']}
        summary[label] = s
        print("  %-16s  F1=%.4f±%.3f  AUC=%.4f±%.3f" % (label, s['f1'][0], s['f1'][1], s['auc'][0], s['auc'][1]))

    # Delta
    for clf in ['XGB', 'LGB']:
        base = summary['%s_tab_only' % clf]['f1'][0]
        aug = summary['%s_tab+emb' % clf]['f1'][0]
        delta = aug - base
        print("  Δ %s: %.4f (tab+emb - tab_only)" % (clf, delta))

    return summary, methods

# ============ Main ============

if __name__ == '__main__':
    all_summaries = {}
    for gnn_type in ['GraphSAGE', 'GCN']:
        summary, methods = run_cv(gnn_type)
        all_summaries[gnn_type] = summary

    # Plot
    fig, axes = plt.subplots(1, 2, figsize=(16, 5))
    for i, gnn_type in enumerate(['GraphSAGE', 'GCN']):
        ax = axes[i]
        s = all_summaries[gnn_type]
        labels_show = ['XGB_tab_only','XGB_tab+emb','XGB_emb_only',
                       'LGB_tab_only','LGB_tab+emb','LGB_emb_only']
        display = ['XGB\ntab','XGB\n+emb','XGB\nemb',
                   'LGB\ntab','LGB\n+emb','LGB\nemb']
        colors = ['#3498db','#2ecc71','#95a5a6','#e67e22','#27ae60','#bdc3c7']
        f1_vals = [float(s[l]['f1'][0]) for l in labels_show]
        f1_errs = [float(s[l]['f1'][1]) for l in labels_show]

        bars = ax.bar(display, f1_vals, yerr=f1_errs, color=colors, capsize=4)
        ax.set_title('%s Embedding Augmentation' % gnn_type)
        ax.set_ylabel('F1 Score')
        ax.set_ylim(0.5, 0.9)
        for b, v in zip(bars, f1_vals):
            ax.text(b.get_x()+b.get_width()/2, b.get_height()+0.005,
                    '%.4f'%v, ha='center', fontsize=8)

    plt.tight_layout()
    p = os.path.join(OUTPUT_DIR, 'gnn_embed_xgb.png')
    plt.savefig(p, dpi=150, bbox_inches='tight'); plt.close()
    print("\nChart:", p)

    # Save
    out_json = {k: {kk: {'f1_mean': float(vv['f1'][0]), 'f1_std': float(vv['f1'][1]),
                         'auc_mean': float(vv['auc'][0]), 'auc_std': float(vv['auc'][1])}
                    for kk, vv in v.items()}
                for k, v in all_summaries.items()}
    jp = os.path.join(OUTPUT_DIR, 'embed_aug_results.json')
    with open(jp, 'w') as f:
        json.dump(out_json, f, indent=2)
    print("Saved:", jp)
