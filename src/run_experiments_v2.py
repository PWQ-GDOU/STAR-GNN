"""
run_experiments_v2.py — 全量实验: 排列检验 + 边重连 + 嵌入拼接 + 推理效率 + 消融
基于 benchmark_clean.py，补全所有缺失数据点。
输出: results/experiments_v2/ 下所有 JSON + 图表
"""
import sys, os, pickle, warnings, time, json, gc, itertools
from pathlib import Path
warnings.filterwarnings('ignore')
os.environ['LOKY_MAX_CPU_COUNT'] = '4'

PROJECT_ROOT = Path(os.environ.get("STAR_GNN_HOME", r"D:\cxdownload\大数据实训\code_sci"))
DATA_DIR = Path(os.environ.get("STAR_GNN_DATA", str(PROJECT_ROOT / "data")))
RESULTS_DIR = Path(os.environ.get("STAR_GNN_RESULTS", str(PROJECT_ROOT / "results" / "experiments_v2")))
GRAPH_PATH = PROJECT_ROOT / "results" / "graph_data_v2.pkl"
RESULTS_DIR.mkdir(parents=True, exist_ok=True)

import torch, torch.nn as nn, torch.nn.functional as F
from torch.optim import Adam
import numpy as np, pandas as pd

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print("Device:", device)

_VENV = r"D:\cxdownload\大数据实训\code\test_4\Lib\site-packages"
if _VENV not in sys.path: sys.path.insert(0, _VENV)

from sklearn.model_selection import StratifiedKFold
from sklearn.metrics import f1_score, roc_auc_score, precision_score, recall_score
from sklearn.preprocessing import StandardScaler, LabelEncoder
from sklearn.decomposition import PCA
from sklearn.neighbors import NearestNeighbors
from xgboost import XGBClassifier
from lightgbm import LGBMClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.neural_network import MLPClassifier
import matplotlib; matplotlib.use('Agg')
import matplotlib.pyplot as plt

RANDOM_SEED = 42; N_OUTER_FOLDS = 5; N_INNER_FOLDS = 2; N_SUBSAMPLE = 6000
N_PERM = 20  # full 20 permutations

# ==================== Utils ====================
def t2n(t):
    if t.numel() == 0: return np.array([])
    return np.array(t.cpu().tolist())

def build_knn_graph(X, k=10):
    n = len(X); k = min(k, n-1)
    Xs = StandardScaler().fit_transform(X.astype(np.float64))
    nn = NearestNeighbors(n_neighbors=k+1, metric='cosine', n_jobs=-1).fit(Xs)
    _, idx = nn.kneighbors(Xs)
    src, dst = [], []
    for i in range(n):
        for j in idx[i]:
            if i != j: src.append(i); dst.append(j)
    adj = torch.sparse_coo_tensor(
        torch.tensor([src,dst], dtype=torch.long), torch.ones(len(src)), (n,n)).coalesce()
    dg = torch.sparse.sum(adj, 1).to_dense(); dg[dg==0] = 1
    di = torch.arange(n)
    return torch.sparse.mm(
        torch.sparse_coo_tensor(torch.stack([di,di]), 1.0/dg, (n,n)), adj).coalesce()

def rewire_edges(adj):
    """Fast degree-preserving edge randomization via column permutation.
    Shuffles destination nodes randomly while preserving each node's
    out-degree exactly. Destroys meaningful topology, keeps degree distribution."""
    adj_coo = adj.coalesce()
    row = adj_coo.indices()[0].numpy().copy()
    col = adj_coo.indices()[1].numpy().copy()
    m = len(row)
    np.random.seed(RANDOM_SEED + 999)
    np.random.shuffle(col)
    print(f"  Rewire: {m} edges permuted (degree-preserving)")
    n = adj.shape[0]
    adj_r = torch.sparse_coo_tensor(
        torch.tensor([row.tolist(), col.tolist()], dtype=torch.long),
        torch.ones(m), (n, n)).coalesce()
    adj_r = (adj_r + adj_r.transpose(0, 1)).coalesce()
    adj_r.values().clamp_(0, 1)
    dg = torch.sparse.sum(adj_r, 1).to_dense(); dg[dg==0] = 1
    di = torch.arange(n)
    adj_r = torch.sparse.mm(
        torch.sparse_coo_tensor(torch.stack([di, di]), 1.0/dg, (n, n)), adj_r).coalesce()
    return adj_r

# ==================== GNN Models ====================
class GNNEncoder(nn.Module):
    """Pure encoder, returns embeddings"""
    def __init__(self, gnn_type, d_in, h=128, o=64, dp=0.3):
        super().__init__()
        self.gnn_type = gnn_type
        if gnn_type == 'GCN':
            self.l1 = nn.Linear(d_in, h); self.l2 = nn.Linear(h, o)
            self.bn = nn.BatchNorm1d(h); self.dp = nn.Dropout(dp)
        else:
            self.f1 = nn.Linear(d_in*2, h); self.f2 = nn.Linear(h*2, o)
            self.bn = nn.BatchNorm1d(h); self.dp = nn.Dropout(dp)
    def forward(self, x, adj):
        if self.gnn_type == 'GCN':
            h = F.relu(self.bn(self.l1(x))); h = self.dp(h)
            h = torch.sparse.mm(adj, h); h = self.l2(h)
            h = torch.sparse.mm(adj, h)
            return h
        else:
            n = torch.sparse.mm(adj, x); h = torch.cat([x, n], -1)
            h = F.relu(self.bn(self.f1(h))); h = self.dp(h)
            n2 = torch.sparse.mm(adj, h); h = torch.cat([h, n2], -1)
            return self.f2(h)

class GNNC(nn.Module):
    def __init__(self, t, d_in, h=128, o=64, nc=2, dp=0.3):
        super().__init__()
        self.enc = GNNEncoder(t, d_in, h, o, dp)
        self.cls = nn.Sequential(nn.Linear(o, 32), nn.ReLU(), nn.Dropout(dp), nn.Linear(32, nc))
    def forward(self, x, adj): return self.cls(self.enc(x, adj))

def train_gnn(m, x, adj, y, tr, va, ep=200, lr=0.001, pa=30):
    m = m.to(device); x = x.to(device); adj = adj.to(device); y = y.to(device)
    np_ = y[tr].sum().item(); pw = (tr.sum().item() - np_) / max(np_, 1)
    cr = nn.CrossEntropyLoss(weight=torch.tensor([1., pw], device=device))
    op = Adam(m.parameters(), lr=lr, weight_decay=5e-4)
    bf, best, cnt = 0., {k: v.cpu().clone() for k,v in m.state_dict().items()}, 0
    for e in range(ep):
        m.train(); op.zero_grad()
        loss = cr(m(x, adj)[tr], y[tr]); loss.backward()
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
    return {'f1': f1_score(yt, yp, zero_division=0), 'prec': precision_score(yt, yp, zero_division=0),
            'rec': recall_score(yt, yp, zero_division=0),
            'auc': roc_auc_score(yt, yr) if len(np.unique(yt)) > 1 else 0.5}

def extract_embeddings(m, x, adj):
    """Extract encoder output embeddings for all nodes"""
    m.eval(); x = x.to(device); adj = adj.to(device)
    with torch.no_grad():
        emb = m.enc(x, adj)
    return t2n(emb)

# ==================== Datasets ====================
def load_enterprise():
    with open(GRAPH_PATH, 'rb') as f: g = pickle.load(f)
    X = g['node_features'].copy().astype(np.float32); y = g['labels'].copy().astype(np.int64)
    ei = g['edge_index']
    adj = torch.sparse_coo_tensor(
        torch.stack([torch.tensor(ei[0].tolist(), dtype=torch.long),
                     torch.tensor(ei[1].tolist(), dtype=torch.long)]),
        torch.ones(len(ei[0])), (len(y), len(y))).coalesce()
    dg = torch.sparse.sum(adj, 1).to_dense(); dg[dg==0] = 1
    di = torch.arange(len(y))
    adj_n = torch.sparse.mm(torch.sparse_coo_tensor(torch.stack([di,di]), 1.0/dg, (len(y),len(y))), adj).coalesce()
    return X, y, "Enterprise", torch.tensor(X.tolist(), dtype=torch.float32), torch.tensor(y.tolist(), dtype=torch.long), adj_n

def load_creditcard():
    df = pd.read_csv(DATA_DIR / "creditcard.csv")
    y = df['Class'].values.astype(np.int64)
    X = df.drop(['Class','Time'], axis=1).values.astype(np.float32)
    X = StandardScaler().fit_transform(X)
    fraud = np.where(y==1)[0]; legit = np.where(y==0)[0]
    n_fraud = len(fraud); n_legit_target = min(len(legit), max(N_SUBSAMPLE - n_fraud, n_fraud*10))
    np.random.seed(RANDOM_SEED)
    sel = np.r_[fraud, np.random.choice(legit, n_legit_target, replace=False)]
    X, y = X[sel], y[sel]
    adj = build_knn_graph(X)
    return X, y, "CreditCard", torch.tensor(X.tolist(), dtype=torch.float32), torch.tensor(y.tolist(), dtype=torch.long), adj

def load_adult():
    cols = ['age','workclass','fnlwgt','education','edu_num','marital','occupation',
            'rel','race','sex','cap_gain','cap_loss','hpw','country','income']
    tr = pd.read_csv(DATA_DIR/'adult.data', header=None, names=cols, skipinitialspace=True)
    te = pd.read_csv(DATA_DIR/'adult.test', header=None, names=cols, skipinitialspace=True, skiprows=1)
    df = pd.concat([tr,te], ignore_index=True)
    y = (df['income'].str.strip().str.rstrip('.') == '>50K').astype(np.int64).values
    cat_cols = ['workclass','education','marital','occupation','rel','race','sex','country']
    for c in cat_cols: df[c] = LabelEncoder().fit_transform(df[c].astype(str))
    num_cols = ['age','fnlwgt','edu_num','cap_gain','cap_loss','hpw']
    X = StandardScaler().fit_transform(df[num_cols+cat_cols].values.astype(np.float32))
    pos = np.where(y==1)[0]; neg = np.where(y==0)[0]
    n_pos_sample = min(len(pos), N_SUBSAMPLE//2)
    n_neg_sample = min(len(neg), N_SUBSAMPLE - n_pos_sample)
    np.random.seed(RANDOM_SEED)
    sel = np.r_[np.random.choice(pos, n_pos_sample, replace=False),
                 np.random.choice(neg, n_neg_sample, replace=False)]
    X, y = X[sel], y[sel]
    adj = build_knn_graph(X)
    return X, y, "Adult", torch.tensor(X.tolist(), dtype=torch.float32), torch.tensor(y.tolist(), dtype=torch.long), adj

def load_covtype():
    import gzip as gz
    with gz.open(DATA_DIR/'covtype.data.gz', 'rt') as f: df = pd.read_csv(f, header=None)
    y_all = df.iloc[:,-1].values
    y = ((y_all==1)|(y_all==2)).astype(np.int64)
    X = StandardScaler().fit_transform(df.iloc[:,:-1].values.astype(np.float32))
    pos = np.where(y==1)[0]; neg = np.where(y==0)[0]
    n_each = min(len(pos), len(neg), N_SUBSAMPLE//2)
    np.random.seed(RANDOM_SEED)
    sel = np.r_[np.random.choice(pos,n_each,replace=False), np.random.choice(neg,n_each,replace=False)]
    X, y = X[sel], y[sel]
    adj = build_knn_graph(X)
    return X, y, "Covtype", torch.tensor(X.tolist(), dtype=torch.float32), torch.tensor(y.tolist(), dtype=torch.long), adj

# ==================== HP Search ====================
HP_GNN = {'hidden_dim':[128,256], 'lr':[0.001,0.002], 'dropout':[0.3,0.5]}
HP_XGB = {'n_estimators':[100,200], 'max_depth':[4,6,8], 'learning_rate':[0.05,0.1]}
HP_LGB = {'n_estimators':[100,200], 'num_leaves':[15,31,63], 'learning_rate':[0.05,0.1]}

def nested_hp_search(X, y, xt, yt, adj, n, d, outer_tr_idx):
    inner_skf = StratifiedKFold(n_splits=N_INNER_FOLDS, shuffle=True, random_state=RANDOM_SEED)
    inner_tr, inner_va = next(inner_skf.split(X[outer_tr_idx], y[outer_tr_idx]))
    itr_idx = outer_tr_idx[inner_tr]; iva_idx = outer_tr_idx[inner_va]
    itr_m = torch.zeros(n, dtype=torch.bool); itr_m[torch.tensor(itr_idx.tolist())] = True
    iva_m = torch.zeros(n, dtype=torch.bool); iva_m[torch.tensor(iva_idx.tolist())] = True
    Xtr_in, Xva_in = X[itr_idx], X[iva_idx]; Ytr_in = y[itr_idx]; Yva_in = y[iva_idx]
    npos = Ytr_in.sum(); nneg = len(Ytr_in)-npos
    
    best_gnn = {'hidden_dim':128,'lr':0.001,'dropout':0.3}; bf=0
    gnn_combos = list(itertools.product(HP_GNN['hidden_dim'], HP_GNN['lr'], HP_GNN['dropout']))
    for combo in gnn_combos:
        torch.cuda.empty_cache()
        m = GNNC('GraphSAGE', d, combo[0], 64, 2, combo[2])
        m = train_gnn(m, xt, adj, yt, itr_m, iva_m, ep=100, lr=combo[1], pa=15)
        r = eval_gnn(m, xt, adj, yt, iva_m)
        if r['f1']>bf: bf=r['f1']; best_gnn={'hidden_dim':combo[0],'lr':combo[1],'dropout':combo[2]}
        del m
    
    best_xgb = {'n_estimators':100,'max_depth':6,'learning_rate':0.1}; bf=0
    xgb_combos = list(itertools.product(HP_XGB['n_estimators'], HP_XGB['max_depth'], HP_XGB['learning_rate']))
    for combo in xgb_combos:
        m = XGBClassifier(n_estimators=combo[0], max_depth=combo[1], learning_rate=combo[2],
                          scale_pos_weight=nneg/max(npos,1), random_state=42, verbosity=0)
        m.fit(Xtr_in, Ytr_in)
        vf = f1_score(Yva_in, m.predict(Xva_in), zero_division=0)
        if vf>bf: bf=vf; best_xgb={'n_estimators':combo[0],'max_depth':combo[1],'learning_rate':combo[2]}
    
    best_lgb = {'n_estimators':100,'num_leaves':31,'learning_rate':0.1}; bf=0
    lgb_combos = list(itertools.product(HP_LGB['n_estimators'], HP_LGB['num_leaves'], HP_LGB['learning_rate']))
    for combo in lgb_combos:
        m = LGBMClassifier(n_estimators=combo[0], num_leaves=combo[1], learning_rate=combo[2],
                           class_weight='balanced', random_state=42, verbose=-1)
        m.fit(Xtr_in, Ytr_in)
        vf = f1_score(Yva_in, m.predict(Xva_in), zero_division=0)
        if vf>bf: bf=vf; best_lgb={'n_estimators':combo[0],'num_leaves':combo[1],'learning_rate':combo[2]}
    
    return best_gnn, best_xgb, best_lgb

# ==================== Permutation Tests ====================
def permutation_test_gnn(xt, adj, yt, train_mask, test_mask, d, hp, n_perm=N_PERM):
    tr_idx = np.array(train_mask.nonzero(as_tuple=True)[0].tolist())
    vsize = int(len(tr_idx) * 0.2)
    np.random.seed(RANDOM_SEED)
    vsi = np.random.choice(tr_idx, max(vsize, 1), replace=False)
    tsi = np.setdiff1d(tr_idx, vsi)
    nn = len(yt)
    trs_m = torch.zeros(nn, dtype=torch.bool); trs_m[torch.tensor(tsi.tolist())] = True
    vs_m  = torch.zeros(nn, dtype=torch.bool); vs_m[torch.tensor(vsi.tolist())] = True

    m = GNNC('GraphSAGE', d, hp['hidden_dim'], 64, 2, hp['dropout'])
    m = train_gnn(m, xt, adj, yt, trs_m, vs_m, ep=100, lr=hp['lr'], pa=15)
    real_r = eval_gnn(m, xt, adj, yt, test_mask)
    del m; torch.cuda.empty_cache()

    ys = yt.clone()
    perm_f1s = []
    for _ in range(n_perm):
        pi = torch.randperm(len(ys)); ys = ys[pi]
        m = GNNC('GraphSAGE', d, hp['hidden_dim'], 64, 2, hp['dropout'])
        m = train_gnn(m, xt, adj, ys, trs_m, vs_m, ep=100, lr=hp['lr'], pa=15)
        r = eval_gnn(m, xt, adj, ys, test_mask)
        perm_f1s.append(r['f1']); del m; torch.cuda.empty_cache()
    return {'real_f1': real_r['f1'], 'real_auc': real_r['auc'],
            'perm_f1_mean': float(np.mean(perm_f1s)), 'perm_f1_std': float(np.std(perm_f1s)),
            'perm_f1_list': [float(f) for f in perm_f1s],
            'p_value': 1.0/(n_perm+1) if real_r['f1'] > max(perm_f1s) else
                       sum(1 for f in perm_f1s if f >= real_r['f1'])/n_perm}

def permutation_test_xgb(Xtr, Ytr, Xva, Yva, hp, n_perm=N_PERM):
    npos=Ytr.sum(); nneg=len(Ytr)-npos
    clf = XGBClassifier(**hp, scale_pos_weight=nneg/max(npos,1), random_state=42, verbosity=0)
    clf.fit(Xtr, Ytr)
    real_f1 = f1_score(Yva, clf.predict(Xva), zero_division=0)
    real_auc = roc_auc_score(Yva, clf.predict_proba(Xva)[:,1])
    perm_f1s = []; perm_aucs = []
    yp = Ytr.copy()
    for _ in range(n_perm):
        np.random.shuffle(yp)
        clf.fit(Xtr, yp)
        perm_f1s.append(f1_score(Yva, clf.predict(Xva), zero_division=0))
        perm_aucs.append(roc_auc_score(Yva, clf.predict_proba(Xva)[:,1]))
    return {'real_f1': float(real_f1), 'real_auc': float(real_auc),
            'perm_f1_mean': float(np.mean(perm_f1s)), 'perm_f1_std': float(np.std(perm_f1s)),
            'perm_f1_list': [float(f) for f in perm_f1s],
            'perm_auc_mean': float(np.mean(perm_aucs)),
            'p_value': 1.0/(n_perm+1) if real_f1 > max(perm_f1s) else
                       sum(1 for f in perm_f1s if f >= real_f1)/n_perm}

# ==================== Run One Dataset ====================
def run_one(name, loader):
    print("\n" + "="*60)
    print("  [%s]" % name)
    print("="*60)
    
    X, y, dn, xt, yt, adj = loader()
    n, d = X.shape
    print("  Samples: %d | Feats: %d | Pos: %d (%.1f%%)" % (n, d, y.sum(), y.sum()/n*100))
    
    outer_cv = StratifiedKFold(n_splits=N_OUTER_FOLDS, shuffle=True, random_state=RANDOM_SEED)
    models = ['GCN','GraphSAGE','XGB','LGB','RF','LR','MLP']
    results = {m:{'f1':[],'auc':[],'prec':[],'rec':[]} for m in models}

    perm_results = {'xgb': {}, 'gnn': {}}
    rewire_results = {}
    embed_results = {}  # embedding concatenation on fold 0
    timing_results = {}  # inference timing on fold 0
    
    for fold, (tr_idx, te_idx) in enumerate(outer_cv.split(X, y)):
        print("\n  --- Fold %d/%d ---" % (fold+1, N_OUTER_FOLDS))
        
        best_gnn, best_xgb, best_lgb = nested_hp_search(X, y, xt, yt, adj, n, d, tr_idx)
        
        tr_m = torch.zeros(n, dtype=torch.bool); tr_m[torch.tensor(tr_idx.tolist())] = True
        te_m = torch.zeros(n, dtype=torch.bool); te_m[torch.tensor(te_idx.tolist())] = True
        Xtr, Xte = X[tr_idx], X[te_idx]; Ytr, Yte = y[tr_idx], y[te_idx]
        np_ = Ytr.sum(); nneg_ = len(Ytr)-np_
        
        # === Permutation tests + Edge Rewiring + Embedding Concat (fold 0 only) ===
        if fold == 0:
            # --- Permutation Tests ---
            print("\n  [Permutation Tests: %d permutations]" % N_PERM)
            gnn_perm = permutation_test_gnn(xt, adj, yt, tr_m, te_m, d, best_gnn, N_PERM)
            xgb_perm = permutation_test_xgb(Xtr, Ytr, Xte, Yte, best_xgb, N_PERM)
            perm_results['gnn'] = gnn_perm
            perm_results['xgb'] = xgb_perm
            print("  GNN Perm: real F1=%.4f vs shuffled %.4f±%.3f (p=%.3f)" %
                  (gnn_perm['real_f1'], gnn_perm['perm_f1_mean'], gnn_perm['perm_f1_std'], gnn_perm['p_value']))
            print("  XGB Perm: real F1=%.4f vs shuffled %.4f±%.3f (p=%.3f)" %
                  (xgb_perm['real_f1'], xgb_perm['perm_f1_mean'], xgb_perm['perm_f1_std'], xgb_perm['p_value']))
            
            # --- Edge Rewiring ---
            print("\n  [Edge Rewiring]")
            val_sz = int(len(tr_idx) * 0.2)
            np.random.seed(RANDOM_SEED)
            vsi2 = np.random.choice(tr_idx, max(val_sz,1), replace=False)
            tsi2 = np.setdiff1d(tr_idx, vsi2)
            ts2 = torch.zeros(n, dtype=torch.bool); ts2[torch.tensor(tsi2.tolist())] = True
            vs2 = torch.zeros(n, dtype=torch.bool); vs2[torch.tensor(vsi2.tolist())] = True
            adj_rewired = rewire_edges(adj)
            m_r = GNNC('GraphSAGE', d, best_gnn['hidden_dim'], 64, 2, best_gnn['dropout'])
            m_r = train_gnn(m_r, xt, adj_rewired, yt, ts2, vs2, ep=200, lr=best_gnn['lr'], pa=30)
            r_r = eval_gnn(m_r, xt, adj_rewired, yt, te_m)
            rewire_results = {
                'orig_f1': float(gnn_perm['real_f1']),
                'rewired_f1': float(r_r['f1']),
                'rewired_auc': float(r_r['auc']),
                'delta_f1': float(gnn_perm['real_f1'] - r_r['f1']),
            }
            del m_r; torch.cuda.empty_cache()
            print("  Original F1=%.4f | Rewired F1=%.4f | Δ=%.4f" %
                  (rewire_results['orig_f1'], rewire_results['rewired_f1'], rewire_results['delta_f1']))
            
            # --- Embedding Concatenation ---
            print("\n  [Embedding Concatenation]")
            val_sub_idx = np.random.choice(tr_idx, val_sz, replace=False)
            train_sub_idx = np.setdiff1d(tr_idx, val_sub_idx)
            train_sub_m = torch.zeros(n, dtype=torch.bool); train_sub_m[torch.tensor(train_sub_idx.tolist())] = True
            val_sub_m = torch.zeros(n, dtype=torch.bool); val_sub_m[torch.tensor(val_sub_idx.tolist())] = True
            
            for gnn_type in ['GCN', 'GraphSAGE']:
                torch.cuda.empty_cache(); gc.collect()
                m = GNNC(gnn_type, d, best_gnn['hidden_dim'], 64, 2, best_gnn['dropout'])
                m = train_gnn(m, xt, adj, yt, train_sub_m, val_sub_m, ep=200, lr=best_gnn['lr'], pa=30)
                emb = extract_embeddings(m, xt, adj)
                del m
                
                # Pure tabular baseline
                xgb_tab = XGBClassifier(**best_xgb, scale_pos_weight=nneg_/max(np_,1), random_state=42, verbosity=0)
                xgb_tab.fit(Xtr, Ytr)
                tab_f1 = f1_score(Yte, xgb_tab.predict(Xte), zero_division=0)
                
                lgb_tab = LGBMClassifier(**best_lgb, class_weight='balanced', random_state=42, verbose=-1)
                lgb_tab.fit(Xtr, Ytr)
                lgb_tab_f1 = f1_score(Yte, lgb_tab.predict(Xte), zero_division=0)
                
                # Tab + Emb
                emb_all = emb
                X_tr_both = np.hstack([Xtr, emb_all[tr_idx]])
                X_te_both = np.hstack([Xte, emb_all[te_idx]])
                
                xgb_both = XGBClassifier(**best_xgb, scale_pos_weight=nneg_/max(np_,1), random_state=42, verbosity=0)
                xgb_both.fit(X_tr_both, Ytr)
                both_f1 = f1_score(Yte, xgb_both.predict(X_te_both), zero_division=0)
                
                lgb_both = LGBMClassifier(**best_lgb, class_weight='balanced', random_state=42, verbose=-1)
                lgb_both.fit(X_tr_both, Ytr)
                lgb_both_f1 = f1_score(Yte, lgb_both.predict(X_te_both), zero_division=0)
                
                # Emb only
                X_tr_emb = emb_all[tr_idx]; X_te_emb = emb_all[te_idx]
                xgb_emb = XGBClassifier(**best_xgb, scale_pos_weight=nneg_/max(np_,1), random_state=42, verbosity=0)
                xgb_emb.fit(X_tr_emb, Ytr)
                emb_only_f1 = f1_score(Yte, xgb_emb.predict(X_te_emb), zero_division=0)
                
                lgb_emb = LGBMClassifier(**best_lgb, class_weight='balanced', random_state=42, verbose=-1)
                lgb_emb.fit(X_tr_emb, Ytr)
                lgb_emb_only_f1 = f1_score(Yte, lgb_emb.predict(X_te_emb), zero_division=0)
                
                embed_results[gnn_type] = {
                    'xgb_tab_f1': float(tab_f1),
                    'xgb_tab_emb_f1': float(both_f1),
                    'xgb_emb_only_f1': float(emb_only_f1),
                    'xgb_delta': float(both_f1 - tab_f1),
                    'lgb_tab_f1': float(lgb_tab_f1),
                    'lgb_tab_emb_f1': float(lgb_both_f1),
                    'lgb_emb_only_f1': float(lgb_emb_only_f1),
                    'lgb_delta': float(lgb_both_f1 - lgb_tab_f1),
                }
                print("  %s: tab F1=%.4f → +emb F1=%.4f (Δ=%+.4f) | emb-only F1=%.4f" %
                      (gnn_type, tab_f1, both_f1, both_f1-tab_f1, emb_only_f1))
            
            # --- Inference Timing ---
            print("\n  [Inference Timing]")
            N_WARM = 30; N_RUNS = 200
            
            # XGBoost inference
            xgb_clf = XGBClassifier(**best_xgb, scale_pos_weight=nneg_/max(np_,1), random_state=42, verbosity=0)
            xgb_clf.fit(Xtr, Ytr)
            for _ in range(N_WARM): _ = xgb_clf.predict(Xte[:10])
            t0 = time.perf_counter()
            for _ in range(N_RUNS): _ = xgb_clf.predict(Xte)
            xgb_latency = (time.perf_counter() - t0) / N_RUNS * 1000
            
            # GraphSAGE inference (full graph forward pass)
            torch.cuda.empty_cache()
            m_t = GNNC('GraphSAGE', d, best_gnn['hidden_dim'], 64, 2, best_gnn['dropout'])
            m_t = train_gnn(m_t, xt, adj, yt, train_sub_m, val_sub_m, ep=200, lr=best_gnn['lr'], pa=30)
            m_t.eval()
            xt_d = xt.to(device); adj_d = adj.to(device)
            for _ in range(N_WARM): _ = m_t(xt_d, adj_d)
            torch.cuda.synchronize()
            t0 = time.perf_counter()
            for _ in range(N_RUNS): _ = m_t(xt_d, adj_d)
            torch.cuda.synchronize()
            gnn_latency = (time.perf_counter() - t0) / N_RUNS * 1000
            del m_t; torch.cuda.empty_cache()
            
            timing_results = {
                'xgb_ms': round(xgb_latency, 2),
                'gnn_ms': round(gnn_latency, 2),
                'speedup': round(gnn_latency / max(xgb_latency, 1e-6), 1),
                'n_nodes': n, 'n_edges': adj.coalesce().indices().shape[1],
            }
            print("  XGB: %.2f ms | GraphSAGE: %.2f ms | SAGE/XGB = %.1fx" %
                  (xgb_latency, gnn_latency, gnn_latency/max(xgb_latency,1e-6)))
        
        # === Main GNN training (all folds) ===
        val_size = int(len(tr_idx) * 0.2)
        np.random.seed(RANDOM_SEED + fold)
        val_sub_idx = np.random.choice(tr_idx, val_size, replace=False)
        train_sub_idx = np.setdiff1d(tr_idx, val_sub_idx)
        train_sub_m = torch.zeros(n, dtype=torch.bool); train_sub_m[torch.tensor(train_sub_idx.tolist())] = True
        val_sub_m = torch.zeros(n, dtype=torch.bool); val_sub_m[torch.tensor(val_sub_idx.tolist())] = True

        for gt in ['GCN','GraphSAGE']:
            torch.cuda.empty_cache(); gc.collect()
            m = GNNC(gt, d, best_gnn['hidden_dim'], 64, 2, best_gnn['dropout'])
            m = train_gnn(m, xt, adj, yt, train_sub_m, val_sub_m, ep=200, lr=best_gnn['lr'], pa=30)
            r = eval_gnn(m, xt, adj, yt, te_m)
            for k in ['f1','auc','prec','rec']: results[gt][k].append(r[k])
            del m
        
        for label, clf in [
            ('XGB', XGBClassifier(**best_xgb, scale_pos_weight=nneg_/max(np_,1), random_state=42, verbosity=0)),
            ('LGB', LGBMClassifier(**best_lgb, class_weight='balanced', random_state=42, verbose=-1)),
            ('RF', RandomForestClassifier(n_estimators=100, max_depth=10, class_weight='balanced', random_state=42, n_jobs=-1)),
            ('LR', LogisticRegression(max_iter=1000, class_weight='balanced', random_state=42)),
            ('MLP', MLPClassifier(hidden_layer_sizes=(128,64), max_iter=500, random_state=42)),
        ]:
            clf.fit(Xtr, Ytr)
            yp = clf.predict(Xte); ypr = clf.predict_proba(Xte)[:,1] if hasattr(clf,'predict_proba') else None
            results[label]['f1'].append(f1_score(Yte,yp,zero_division=0))
            results[label]['auc'].append(roc_auc_score(Yte,ypr) if ypr is not None and len(np.unique(Yte))>1 else 0.5)
            results[label]['prec'].append(precision_score(Yte,yp,zero_division=0))
            results[label]['rec'].append(recall_score(Yte,yp,zero_division=0))
    
    summary = {}
    for m in models:
        r = results[m]
        summary[m] = {k+'_mean': float(np.mean(r[k])) for k in ['f1','auc','prec','rec']}
        summary[m].update({k+'_std': float(np.std(r[k])) for k in ['f1','auc','prec','rec']})
    
    gnn_best = max(summary[m]['f1_mean'] for m in ['GCN','GraphSAGE'])
    tab_best = max(summary[m]['f1_mean'] for m in ['XGB','LGB','RF','LR','MLP'])
    print("\n  Best Tab: %.4f | Best GNN: %.4f | Δ: %+.4f" % (tab_best, gnn_best, gnn_best-tab_best))
    
    return summary, perm_results, rewire_results, embed_results, timing_results

# ==================== Main ====================
def main():
    print("="*60)
    print("  Full Experiments v2 (PermTest + Rewire + Embed + Timing)")
    print("="*60)
    
    datasets = [
        ('Enterprise', load_enterprise),
        ('CreditCard', load_creditcard),
        ('Adult', load_adult),
        ('Covtype', load_covtype),
    ]
    
    all_results = {}
    for ds_name, loader in datasets:
        try:
            sm, perm, rewire, embed, timing = run_one(ds_name, loader)
            all_results[ds_name] = {
                'benchmark': sm,
                'permutation': perm,
                'edge_rewire': rewire,
                'embed_concat': embed,
                'inference_timing': timing,
            }
        except Exception as e:
            print("  ERROR [%s]: %s" % (ds_name, e))
            import traceback; traceback.print_exc()
    
    # ============= Save =============
    def convert(o):
        if isinstance(o, (np.floating,)): return float(o)
        if isinstance(o, (np.integer,)): return int(o)
        if isinstance(o, dict): return {k:convert(v) for k,v in o.items()}
        if isinstance(o, (list,tuple)): return [convert(i) for i in o]
        return o
    
    out_json = RESULTS_DIR / 'all_results.json'
    with open(out_json, 'w') as f:
        json.dump(convert(all_results), f, indent=2)
    print("\nSaved:", out_json)
    
    # ============= Generate Figures =============
    n_ds = len(all_results)
    if n_ds == 0: print("No results"); return
    
    # Figure 4: F1 bar chart (4 datasets)
    fig, axes = plt.subplots(1, n_ds, figsize=(5*n_ds, 4.5))
    if n_ds == 1: axes = [axes]
    for idx, ds in enumerate(all_results):
        ax = axes[idx]; sm = all_results[ds]['benchmark']
        order = ['LR','RF','MLP','XGB','LGB','GCN','GraphSAGE']
        mdl = [m for m in order if m in sm]
        f1v = [sm[m]['f1_mean'] for m in mdl]; f1e = [sm[m]['f1_std'] for m in mdl]
        colors = ['#2ecc71' if m in ('GCN','GraphSAGE') else '#3498db' for m in mdl]
        ax.bar(mdl, f1v, yerr=f1e, color=colors, capsize=3, alpha=0.85)
        ax.set_title(ds, fontsize=11, fontweight='bold')
        ax.set_ylim(0, 1.08); ax.tick_params(axis='x', rotation=30)
        ax.set_ylabel('F1 Score')
    plt.suptitle('Model Comparison (5-Fold CV)', fontweight='bold', fontsize=13)
    plt.tight_layout()
    fig4 = RESULTS_DIR / 'figure4_model_comparison.png'
    plt.savefig(fig4, dpi=150, bbox_inches='tight'); plt.close()
    print("Figure 4:", fig4)
    
    # Figure 5: ΔF1 cross-dataset
    fig, ax = plt.subplots(figsize=(8, 5))
    ds_names = []; deltas = []
    for ds in all_results:
        sm = all_results[ds]['benchmark']
        gnn_best = max(sm[m]['f1_mean'] for m in ['GCN','GraphSAGE'])
        tab_best = max(sm[m]['f1_mean'] for m in ['XGB','LGB','RF','LR','MLP'])
        ds_names.append(ds); deltas.append(gnn_best - tab_best)
    bars = ax.bar(ds_names, deltas, color=['#e74c3c' if d<0 else '#2ecc71' for d in deltas], alpha=0.85)
    ax.axhline(y=0, color='black', linestyle='-', linewidth=1)
    ax.set_ylabel('ΔF1 (GNN − Best Tabular)'); ax.set_title('Cross-Dataset ΔF1: Feature Ceiling Effect')
    for bar, d in zip(bars, deltas):
        ax.text(bar.get_x()+bar.get_width()/2, bar.get_height()-0.003,
                '%+.3f'%d, ha='center', va='top' if d<0 else 'bottom', fontsize=11, fontweight='bold')
    plt.tight_layout()
    fig5 = RESULTS_DIR / 'figure5_delta_cross_dataset.png'
    plt.savefig(fig5, dpi=150, bbox_inches='tight'); plt.close()
    print("Figure 5:", fig5)
    
    # Figure 7: Permutation test violin (Enterprise only)
    if 'Enterprise' in all_results and all_results['Enterprise']['permutation'].get('gnn', {}):
        fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 5))
        for i, (ds, label) in enumerate([('Enterprise', 'Enterprise')]):
            pass  # Process all datasets later
        # Enterprise GNN perm
        perm_g = all_results['Enterprise']['permutation']['gnn']
        perm_x = all_results['Enterprise']['permutation']['xgb']
        if 'perm_f1_list' in perm_g:
            # GNN
            parts1 = ax1.violinplot([perm_g['perm_f1_list']], positions=[0], showmeans=True)
            ax1.scatter([0], [perm_g['real_f1']], color='#e74c3c', s=100, zorder=5, marker='D', label='Real F1=%.3f'%perm_g['real_f1'])
            ax1.set_xticks([0]); ax1.set_xticklabels(['GraphSAGE'])
            ax1.set_ylabel('F1 Score'); ax1.set_title('GNN Permutation Test (p=%.3f)' % perm_g['p_value'])
            ax1.legend(); ax1.grid(axis='y', alpha=0.3)
            # XGB
            parts2 = ax2.violinplot([perm_x['perm_f1_list']], positions=[0], showmeans=True)
            ax2.scatter([0], [perm_x['real_f1']], color='#e74c3c', s=100, zorder=5, marker='D', label='Real F1=%.3f'%perm_x['real_f1'])
            ax2.set_xticks([0]); ax2.set_xticklabels(['XGBoost'])
            ax2.set_ylabel('F1 Score'); ax2.set_title('XGBoost Permutation Test (p=%.3f)' % perm_x['p_value'])
            ax2.legend(); ax2.grid(axis='y', alpha=0.3)
            plt.suptitle('Enterprise Permutation Tests (%d permutations)' % N_PERM, fontweight='bold')
            plt.tight_layout()
            fig7 = RESULTS_DIR / 'figure7_permutation_test.png'
            plt.savefig(fig7, dpi=150, bbox_inches='tight'); plt.close()
            print("Figure 7:", fig7)
    
    print("\n=== ALL EXPERIMENTS COMPLETE ===")
    print("Results:", out_json)

if __name__ == '__main__':
    main()
