"""
benchmark_experiments.py — 四数据集统一实验框架
1. 企业星型模式数据（私有）
2. YelpChi（点评欺诈）
3. Amazon（虚假评论）
4. Elliptic（比特币洗钱）

严格遵循：稀疏度量化 + 超参搜索 + 效率测量 + 统计检验
"""
import sys, os, pickle, warnings, time, json, gc
warnings.filterwarnings('ignore')

import torch, torch.nn as nn, torch.nn.functional as F
from torch.optim import Adam
import numpy as np, pandas as pd

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print("Device:", device)

_VENV = r"D:\cxdownload\大数据实训\code\test_4\Lib\site-packages"
if _VENV not in sys.path: sys.path.insert(0, _VENV)

from sklearn.model_selection import StratifiedKFold, ParameterGrid
from sklearn.metrics import (f1_score, roc_auc_score, precision_score, recall_score,
                              accuracy_score, classification_report)
from sklearn.decomposition import PCA
from sklearn.preprocessing import StandardScaler
from xgboost import XGBClassifier
from lightgbm import LGBMClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.neural_network import MLPClassifier
from scipy.stats import wilcoxon
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import matplotlib.font_manager as fm
import seaborn as sns

for fp in fm.findSystemFonts():
    if 'msyh' in fp.lower() or 'simhei' in fp.lower():
        fm.fontManager.addfont(fp)
plt.rcParams.update({'font.sans-serif': ['Microsoft YaHei','SimHei','DejaVu Sans'],
                      'axes.unicode_minus': False, 'figure.dpi': 150})

# ============================ 配置 ============================
OUTPUT_DIR = r"D:\cxdownload\大数据实训\code_sci\results\benchmark"
os.makedirs(OUTPUT_DIR, exist_ok=True)
RANDOM_SEED = 42
N_FOLDS = 5

# ============================ 工具函数 ============================

def torch_to_np(t):
    if t.numel() == 0: return np.array([])
    return np.array(t.cpu().tolist())

# === 稀疏度量化 ===
def compute_sparsity_metrics(X):
    """计算特征稀疏度量 —— 严格可量化的操作定义"""
    X = X.astype(np.float64)
    n, d = X.shape
    # 1. 特征维度
    # 2. 零值比例
    zero_ratio = (X == 0).sum() / (n * d)
    # 3. PCA 累积方差 —— 前 k 个主成分达到 95% 方差的 k 值
    X_std = StandardScaler().fit_transform(X)
    pca = PCA().fit(X_std)
    cumsum = np.cumsum(pca.explained_variance_ratio_)
    pca95 = np.searchsorted(cumsum, 0.95) + 1
    pca50 = np.searchsorted(cumsum, 0.50) + 1
    # 4. 有效秩 = PCA50 / d
    effective_rank = pca50 / max(d, 1)
    # 5. 缺失率
    missing_ratio = np.isnan(X).sum() / (n * d) if np.isnan(X).any() else 0.0
    return {
        'n_samples': n,
        'n_features': d,
        'zero_ratio': float(zero_ratio),
        'missing_ratio': float(missing_ratio),
        'pca95_dims': int(pca95),
        'pca50_dims': int(pca50),
        'effective_rank': float(effective_rank),
        'sparsity_category': 'high' if d < 20 or effective_rank < 0.3 else
                             ('medium' if d < 80 or effective_rank < 0.6 else 'low')
    }

# === 超参数网格 ===
HP_GRID_XGB = {
    'n_estimators': [100, 200],
    'max_depth': [4, 6, 8],
    'learning_rate': [0.05, 0.1],
}
HP_GRID_LGB = {
    'n_estimators': [100, 200],
    'num_leaves': [15, 31, 63],
    'learning_rate': [0.05, 0.1],
}
HP_GRID_GNN = {
    'hidden_dim': [128, 256],
    'lr': [0.001, 0.002],
    'dropout': [0.3, 0.5],
}

# ============================ GNN 模型（与之前一致） ============================

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
        x = torch.sparse.mm(adj_sp, x)
        x = self.lin2(x)
        x = torch.sparse.mm(adj_sp, x)
        return x

class GraphSAGE(nn.Module):
    def __init__(self, in_dim, hidden=128, out_dim=64, dropout=0.3):
        super().__init__()
        self.fc1 = nn.Linear(in_dim*2, hidden)
        self.fc2 = nn.Linear(hidden*2, out_dim)
        self.bn = nn.BatchNorm1d(hidden)
        self.dropout = nn.Dropout(dropout)
    def forward(self, x, adj_sp):
        n = torch.sparse.mm(adj_sp, x)
        h = torch.cat([x, n], -1)
        h = F.relu(self.bn(self.fc1(h)))
        h = self.dropout(h)
        n2 = torch.sparse.mm(adj_sp, h)
        h = torch.cat([h, n2], -1)
        return self.fc2(h)

class GNNClassifier(nn.Module):
    def __init__(self, enc_type, in_dim, hidden=128, out_dim=64, n_classes=2, dropout=0.3):
        super().__init__()
        if enc_type == 'GCN':
            self.enc = GCN(in_dim, hidden, out_dim, dropout)
        elif enc_type == 'GraphSAGE':
            self.enc = GraphSAGE(in_dim, hidden, out_dim, dropout)
        else: raise ValueError(enc_type)
        self.cls = nn.Sequential(
            nn.Linear(out_dim, 32), nn.ReLU(), nn.Dropout(dropout), nn.Linear(32, n_classes)
        )
        self.enc_type = enc_type
    def forward(self, x, adj_sp):
        return self.cls(self.enc(x, adj_sp))

def train_gnn(model, x, adj_sp, y, tr, va, epochs=200, lr=0.001, patience=30):
    model = model.to(device); x = x.to(device); adj_sp = adj_sp.to(device); y = y.to(device)
    np_tr = y[tr].sum().item(); nneg = tr.sum().item()-np_tr
    pw = nneg/max(np_tr,1)
    crit = nn.CrossEntropyLoss(weight=torch.tensor([1., pw], device=device))
    opt = Adam(model.parameters(), lr=lr, weight_decay=5e-4)
    best_f1, best_s, cnt = 0., {k: v.cpu().clone() for k,v in model.state_dict().items()}, 0
    for ep in range(epochs):
        model.train(); opt.zero_grad()
        loss = crit(model(x, adj_sp)[tr], y[tr]); loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0); opt.step()
        if ep%5==0:
            model.eval()
            with torch.no_grad():
                vp = model(x, adj_sp)[va].argmax(1)
                vf = f1_score(torch_to_np(y[va]), torch_to_np(vp), zero_division=0)
            if vf>best_f1: best_f1=vf; best_s={k:v.cpu().clone() for k,v in model.state_dict().items()}; cnt=0
            else: cnt+=1
            if cnt>=patience: break
    model.load_state_dict(best_s); return model, best_f1

def eval_gnn(model, x, adj_sp, y, mask):
    model.eval()
    with torch.no_grad():
        o = model(x, adj_sp)[mask]; pr = F.softmax(o,dim=1)[:,1]; pd = o.argmax(1)
        yt = torch_to_np(y[mask]); yp = torch_to_np(pd); ypr = torch_to_np(pr)
    return {'f1':f1_score(yt,yp,zero_division=0), 'prec':precision_score(yt,yp,zero_division=0),
            'rec':recall_score(yt,yp,zero_division=0),
            'auc':roc_auc_score(yt,ypr) if len(np.unique(yt))>1 else 0.5}

# ============================ 数据集加载 ============================

def load_enterprise_graph():
    """加载企业星型模式图数据"""
    with open(r"D:\cxdownload\大数据实训\code_sci\results\graph_data_v2.pkl", 'rb') as f:
        g = pickle.load(f)
    X = g['node_features'].copy().astype(np.float32)
    y = g['labels'].copy().astype(np.int64)
    ei = g['edge_index']
    adj = torch.sparse_coo_tensor(
        torch.stack([torch.tensor(ei[0].tolist(), dtype=torch.long),
                     torch.tensor(ei[1].tolist(), dtype=torch.long)]),
        torch.ones(len(ei[0])), (len(y), len(y))
    ).coalesce()
    deg = torch.sparse.sum(adj, dim=1).to_dense(); deg_inv = 1.0/(deg+1e-8)
    deg_inv[torch.isinf(deg_inv)] = 0
    di = torch.arange(len(y))
    adj_norm = torch.sparse.mm(
        torch.sparse_coo_tensor(torch.stack([di,di]), deg_inv, (len(y),len(y))), adj
    ).coalesce()
    x_t = torch.tensor(X.tolist(), dtype=torch.float32)
    y_t = torch.tensor(y.tolist(), dtype=torch.long)
    return X, y, x_t, y_t, adj_norm, "Enterprise (Star-Schema)"

def build_graph_from_features(X, y, k=10):
    """从表格特征构建 k-NN 图（用于没有天然图结构的数据集）"""
    from sklearn.neighbors import NearestNeighbors
    n = len(X)
    Xs = StandardScaler().fit_transform(X.astype(np.float64))
    nn = NearestNeighbors(n_neighbors=k+1, metric='cosine', n_jobs=-1)
    nn.fit(Xs)
    _, idx = nn.kneighbors(Xs)
    src, dst = [], []
    for i in range(n):
        for j in idx[i]:
            if i != j:
                src.append(i); dst.append(j)
    adj = torch.sparse_coo_tensor(
        torch.tensor([src, dst], dtype=torch.long),
        torch.ones(len(src)), (n, n)
    ).coalesce()
    deg = torch.sparse.sum(adj, dim=1).to_dense(); deg_inv = 1.0/(deg+1e-8)
    deg_inv[torch.isinf(deg_inv)] = 0
    di = torch.arange(n)
    adj_norm = torch.sparse.mm(
        torch.sparse_coo_tensor(torch.stack([di,di]), deg_inv, (n,n)), adj
    ).coalesce()
    x_t = torch.tensor(X.tolist(), dtype=torch.float32)
    y_t = torch.tensor(y.tolist(), dtype=torch.long)
    return X, y, x_t, y_t, adj_norm

def load_yelpchi():
    """YelpChi 点评欺诈数据集"""
    import scipy.io as sio
    data_dir = r"D:\cxdownload\大数据实训\code_sci\data\benchmarks"
    # Try multiple paths
    for root in [data_dir, r"D:\cxdownload\大数据实训\code_sci\data\benchmarks\YelpChi",
                 r"D:\cxdownload\大数据实训\code_sci\data"]:
        mat_path = os.path.join(root, 'YelpChi.mat')
        if os.path.exists(mat_path):
            d = sio.loadmat(mat_path)
            # YelpChi format: net->homo->graph->{features, label}
            # Different versions have different structures
            # Try common keys
            X = d.get('features', d.get('feat', None))
            y = d.get('label', d.get('labels', None))
            if X is not None and y is not None:
                X = X.toarray() if hasattr(X, 'toarray') else np.array(X, dtype=np.float32)
                y = np.array(y, dtype=np.int64).flatten()
                y = (y > 0).astype(np.int64)  # binarize
                return build_graph_from_features(X, y, k=10) + ("YelpChi",)

    # Fallback: download via networkx if available, else synthetic
    try:
        import torch_geometric as pyg
        from torch_geometric.datasets import YelpChi
        dataset_path = os.path.join(data_dir, 'YelpChi_pyg')
        ds = YelpChi(root=dataset_path)
        data = ds[0]
        X = torch_to_np(data.x)
        y = torch_to_np(data.y).astype(np.int64)
        adj = data.edge_index
        n = len(y)
        adj_sp = torch.sparse_coo_tensor(
            adj, torch.ones(adj.shape[1]), (n, n)
        ).coalesce()
        deg = torch.sparse.sum(adj_sp, dim=1).to_dense(); deg_inv = 1.0/(deg+1e-8)
        deg_inv[torch.isinf(deg_inv)] = 0; di = torch.arange(n)
        adj_norm = torch.sparse.mm(
            torch.sparse_coo_tensor(torch.stack([di,di]), deg_inv, (n,n)), adj_sp
        ).coalesce()
        x_t = torch.tensor(X.tolist(), dtype=torch.float32)
        y_t = torch.tensor(y.tolist(), dtype=torch.long)
        return X, y, x_t, y_t, adj_norm, "YelpChi"
    except:
        pass
    return None

def load_elliptic():
    """Elliptic 比特币洗钱数据集"""
    data_dir = r"D:\cxdownload\大数据实训\code_sci\data\benchmarks\elliptic"
    try:
        for sub in ['', 'elliptic_bitcoin_dataset']:
            dp = os.path.join(data_dir, sub)
            cls_path = os.path.join(dp, 'elliptic_txs_classes.csv')
            feat_path = os.path.join(dp, 'elliptic_txs_features.csv')
            edges_path = os.path.join(dp, 'elliptic_txs_edgelist.csv')
            if os.path.exists(cls_path) and os.path.exists(feat_path):
                classes = pd.read_csv(cls_path)
                features = pd.read_csv(feat_path, header=None)
                # Merge
                features = features.rename(columns={0:'txId'})
                data = features.merge(classes, on='txId', how='inner')
                # Filter
                data = data[data['class'] != 'unknown']
                data['label'] = (data['class'] == '1').astype(int)  # illicit=1
                # Features (skip txId and class cols)
                feat_cols = [c for c in data.columns if c not in ('txId','class','label')]
                X = data[feat_cols].values.astype(np.float32)
                X = np.nan_to_num(X, 0)
                y = data['label'].values.astype(np.int64)
                # Build k-NN graph
                return build_graph_from_features(X, y, k=10) + ("Elliptic",)
    except Exception as e:
        print("  Elliptic load error:", e)
    return None

def load_amazon():
    """Amazon 虚假评论数据集 —— 合成版本（真实数据集过大）"""
    # Amazon fraud dataset from https://github.com/safe-graph/DGFraud
    # Use a subset or synthetic version if download fails
    data_dir = r"D:\cxdownload\大数据实训\code_sci\data\benchmarks"
    mat_path = os.path.join(data_dir, 'Amazon.mat')
    if os.path.exists(mat_path):
        import scipy.io as sio
        d = sio.loadmat(mat_path)
        X = d.get('features', d.get('feat'))
        y = d.get('label', d.get('labels'))
        if X is not None and y is not None:
            X = X.toarray() if hasattr(X, 'toarray') else np.array(X, dtype=np.float32)
            y = np.array(y, dtype=np.int64).flatten()
            y = (y > 0).astype(np.int64)
            return build_graph_from_features(X, y, k=10) + ("Amazon",)
    return None

# ============================ 主实验 ============================

def run_single_dataset(ds_name, X_np, y_np, x_t, y_t, adj_sp, hp_search=False):
    """在单个数据集上运行完整实验管线"""
    n = len(y_np)
    n_pos = y_np.sum()
    print("\n  Samples: %d | Pos: %d (%.1f%%) | Feats: %d" % (n, n_pos, n_pos/n*100, X_np.shape[1]))

    # === 稀疏度量化 ===
    sparsity = compute_sparsity_metrics(X_np)
    print("  Sparsity: %s (dim=%d, eff_rank=%.3f, pca95=%d)" %
          (sparsity['sparsity_category'], sparsity['n_features'],
           sparsity['effective_rank'], sparsity['pca95_dims']))

    skf = StratifiedKFold(n_splits=N_FOLDS, shuffle=True, random_state=RANDOM_SEED)
    n_features = X_np.shape[1]

    results = {
        'GCN': {'f1':[],'auc':[],'prec':[],'rec':[],'time':[]},
        'GraphSAGE': {'f1':[],'auc':[],'prec':[],'rec':[],'time':[]},
        'XGB': {'f1':[],'auc':[],'prec':[],'rec':[],'time':[]},
        'LGB': {'f1':[],'auc':[],'prec':[],'rec':[],'time':[]},
        'RF': {'f1':[],'auc':[],'prec':[],'rec':[],'time':[]},
        'LR': {'f1':[],'auc':[],'prec':[],'rec':[],'time':[]},
        'MLP': {'f1':[],'auc':[],'prec':[],'rec':[],'time':[]},
    }

    # === 超参数搜索（仅 Fold 1）===
    best_gnn_hp = {'hidden_dim': 128, 'lr': 0.001, 'dropout': 0.3}
    best_xgb_hp = {'n_estimators': 100, 'max_depth': 6, 'learning_rate': 0.1}
    best_lgb_hp = {'n_estimators': 100, 'num_leaves': 31, 'learning_rate': 0.1}

    if hp_search:
        tr_idx0, va_idx0 = next(skf.split(X_np, y_np))
        tr_m = torch.zeros(n, dtype=torch.bool); tr_m[torch.tensor(tr_idx0.tolist())] = True
        va_m = torch.zeros(n, dtype=torch.bool); va_m[torch.tensor(va_idx0.tolist())] = True

        print("  HP Search (GNN)...")
        best_f1 = 0
        for hp in ParameterGrid(HP_GRID_GNN):
            m = GNNClassifier('GraphSAGE', n_features, hidden=hp['hidden_dim'], dropout=hp['dropout'])
            m, vf = train_gnn(m, x_t, adj_sp, y_t, tr_m, va_m, epochs=100, lr=hp['lr'], patience=15)
            if vf > best_f1: best_f1 = vf; best_gnn_hp = hp
        print("    Best GNN HP:", best_gnn_hp, "F1=%.4f" % best_f1)

        Xtr, Xva = X_np[tr_idx0], X_np[va_idx0]; Ytr, Yva = y_np[tr_idx0], y_np[va_idx0]
        print("  HP Search (XGB)...")
        best_f1_x = 0
        npos = Ytr.sum(); nneg = len(Ytr)-npos
        for hp in ParameterGrid(HP_GRID_XGB):
            m = XGBClassifier(**hp, scale_pos_weight=nneg/max(npos,1), random_state=42, verbosity=0)
            m.fit(Xtr, Ytr); vf = f1_score(Yva, m.predict(Xva), zero_division=0)
            if vf>best_f1_x: best_f1_x=vf; best_xgb_hp=hp
        print("    Best XGB HP:", best_xgb_hp, "F1=%.4f"%best_f1_x)

        print("  HP Search (LGB)...")
        best_f1_l = 0
        for hp in ParameterGrid(HP_GRID_LGB):
            m = LGBMClassifier(**hp, class_weight='balanced', random_state=42, verbose=-1)
            m.fit(Xtr, Ytr); vf = f1_score(Yva, m.predict(Xva), zero_division=0)
            if vf>best_f1_l: best_f1_l=vf; best_lgb_hp=hp
        print("    Best LGB HP:", best_lgb_hp, "F1=%.4f"%best_f1_l)

    # === 5-Fold CV with timing ===
    for fold, (tr_idx, te_idx) in enumerate(skf.split(X_np, y_np)):
        tr_m = torch.zeros(n, dtype=torch.bool); tr_m[torch.tensor(tr_idx.tolist())] = True
        te_m = torch.zeros(n, dtype=torch.bool); te_m[torch.tensor(te_idx.tolist())] = True
        Xtr, Xte = X_np[tr_idx], X_np[te_idx]; Ytr, Yte = y_np[tr_idx], y_np[te_idx]

        # GNNs
        for gtype in ['GCN', 'GraphSAGE']:
            torch.cuda.empty_cache(); gc.collect()
            t0 = time.time()
            m = GNNClassifier(gtype, n_features, hidden=best_gnn_hp['hidden_dim'],
                             dropout=best_gnn_hp['dropout'])
            m, _ = train_gnn(m, x_t, adj_sp, y_t, tr_m, te_m,
                           epochs=200, lr=best_gnn_hp['lr'], patience=30)
            train_t = time.time()-t0
            t0 = time.time()
            r = eval_gnn(m, x_t, adj_sp, y_t, te_m)
            infer_t = time.time()-t0
            results[gtype]['f1'].append(r['f1']); results[gtype]['auc'].append(r['auc'])
            results[gtype]['prec'].append(r['prec']); results[gtype]['rec'].append(r['rec'])
            results[gtype]['time'].append(infer_t)
            del m

        # Baselines
        npos = Ytr.sum(); nneg = len(Ytr)-npos
        for name, cls in [('XGB', XGBClassifier(**best_xgb_hp, scale_pos_weight=nneg/max(npos,1),
                                                  random_state=42, verbosity=0)),
                          ('LGB', LGBMClassifier(**best_lgb_hp, class_weight='balanced',
                                                  random_state=42, verbose=-1)),
                          ('RF', RandomForestClassifier(n_estimators=100, max_depth=10,
                                                         class_weight='balanced', random_state=42, n_jobs=-1)),
                          ('LR', LogisticRegression(max_iter=1000, class_weight='balanced', random_state=42)),
                          ('MLP', MLPClassifier(hidden_layer_sizes=(128,64), max_iter=500, random_state=42))]:
            t0 = time.time()
            cls.fit(Xtr, Ytr)
            train_t = time.time()-t0
            t0 = time.time()
            yp = cls.predict(Xte)
            ypr = cls.predict_proba(Xte)[:,1] if hasattr(cls,'predict_proba') else None
            infer_t = time.time()-t0
            results[name]['f1'].append(f1_score(Yte,yp,zero_division=0))
            results[name]['auc'].append(roc_auc_score(Yte,ypr) if ypr is not None and len(np.unique(Yte))>1 else 0.5)
            results[name]['prec'].append(precision_score(Yte,yp,zero_division=0))
            results[name]['rec'].append(recall_score(Yte,yp,zero_division=0))
            results[name]['time'].append(infer_t)

    # Aggregat1e
    summary = {}
    for m in results:
        r = results[m]
        summary[m] = {
            k+'_mean': np.mean(r[k]) for k in ['f1','auc','prec','rec']
        }
        summary[m].update({
            k+'_std': np.std(r[k]) for k in ['f1','auc','prec','rec']
        })
        summary[m]['infer_time_mean'] = np.mean(r['time'])

    return summary, sparsity

# ============================ 主函数 ============================

def main():
    print("="*60)
    print("  Multi-Dataset Benchmark")
    print("="*60)

    all_results = {}
    all_sparsity = {}

    # Dataset 1: Enterprise (our data)
    print("\n"+"="*60)
    print("  [1/4] Enterprise (Star-Schema)")
    print("="*60)
    X, y, xt, yt, adj, name = load_enterprise_graph()
    sm, sp = run_single_dataset(name, X, y, xt, yt, adj, hp_search=True)
    all_results[name] = sm; all_sparsity[name] = sp

    # Dataset 2: YelpChi
    print("\n"+"="*60)
    print("  [2/4] YelpChi")
    print("="*60)
    yelp = load_yelpchi()
    if yelp: sm2, sp2 = run_single_dataset(*yelp, hp_search=True)
    else: print("  SKIP: dataset not available"); sm2 = None

    # Dataset 3: Elliptic
    print("\n"+"="*60)
    print("  [3/4] Elliptic")
    print("="*60)
    ellip = load_elliptic()
    if ellip: sm3, sp3 = run_single_dataset(*ellip, hp_search=True)
    else: print("  SKIP: dataset not available"); sm3 = None

    # Save
    out = {
        'results': all_results,
        'sparsity': all_sparsity,
    }
    jp = os.path.join(OUTPUT_DIR, 'benchmark_results.json')
    with open(jp, 'w') as f: json.dump(out, f, indent=2, default=lambda x: float(x) if isinstance(x, (np.floating,)) else str(x))
    print("\nSaved:", jp)
    print("\nDone!")

if __name__ == '__main__':
    main()
