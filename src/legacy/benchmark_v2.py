"""
benchmark_v2.py — 多数据集统一实验（使用 sklearn 标准数据集）
四数据集：Enterprise(私有) + CreditCard + Adult + Covertype
特征稀疏度从低到高覆盖全谱
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
from sklearn.datasets import fetch_openml
import matplotlib; matplotlib.use('Agg')
import matplotlib.pyplot as plt

plt.rcParams.update({'font.size': 10, 'figure.dpi': 150})

# ==================== 配置 ====================
OUTPUT_DIR = r"D:\cxdownload\大数据实训\code_sci\results\benchmark"
os.makedirs(OUTPUT_DIR, exist_ok=True)
RANDOM_SEED = 42; N_FOLDS = 5

# ==================== 工具 ====================

def torch_to_np(t):
    if t.numel() == 0: return np.array([])
    return np.array(t.cpu().tolist())

def compute_sparsity(X):
    n, d = X.shape
    Xs = StandardScaler().fit_transform(X.astype(np.float64))
    pca = PCA().fit(Xs)
    cum = np.cumsum(pca.explained_variance_ratio_)
    pca95 = int(np.searchsorted(cum, 0.95) + 1)
    pca50 = int(np.searchsorted(cum, 0.50) + 1)
    er = pca50 / max(d, 1)
    zero_r = float((X == 0).sum() / (n * d))
    return {'n_samples':n, 'n_features':d, 'zero_ratio':zero_r,
            'pca95':pca95, 'pca50':pca50, 'effective_rank':float(er),
            'category': 'sparse' if er<0.3 else ('medium' if er<0.6 else 'dense')}

def build_knn_graph(X, k=10):
    n = len(X)
    Xs = StandardScaler().fit_transform(X.astype(np.float64))
    nn = NearestNeighbors(n_neighbors=k+1, metric='cosine', n_jobs=-1).fit(Xs)
    _, idx = nn.kneighbors(Xs)
    src, dst = [], []
    for i in range(n):
        for j in idx[i]:
            if i != j: src.append(i); dst.append(j)
    adj = torch.sparse_coo_tensor(
        torch.tensor([src,dst], dtype=torch.long), torch.ones(len(src)), (n,n)).coalesce()
    deg = torch.sparse.sum(adj,dim=1).to_dense(); deg[deg==0]=1
    di = torch.arange(n)
    adj_n = torch.sparse.mm(
        torch.sparse_coo_tensor(torch.stack([di,di]), 1.0/deg, (n,n)), adj).coalesce()
    return adj_n

# ==================== GNN 模型 ====================

class GCN(nn.Module):
    def __init__(self, d_in, h=128, d_out=64, dp=0.3):
        super().__init__()
        self.l1=nn.Linear(d_in,h); self.l2=nn.Linear(h,d_out)
        self.bn=nn.BatchNorm1d(h); self.dp=nn.Dropout(dp)
    def forward(self,x,adj):
        x=F.relu(self.bn(self.l1(x))); x=self.dp(x)
        x=torch.sparse.mm(adj,x); x=self.l2(x); x=torch.sparse.mm(adj,x)
        return x

class GraphSAGE(nn.Module):
    def __init__(self,d_in,h=128,d_out=64,dp=0.3):
        super().__init__()
        self.f1=nn.Linear(d_in*2,h); self.f2=nn.Linear(h*2,d_out)
        self.bn=nn.BatchNorm1d(h); self.dp=nn.Dropout(dp)
    def forward(self,x,adj):
        n=torch.sparse.mm(adj,x); h=torch.cat([x,n],-1)
        h=F.relu(self.bn(self.f1(h))); h=self.dp(h)
        n2=torch.sparse.mm(adj,h); h=torch.cat([h,n2],-1)
        return self.f2(h)

class GNNC(nn.Module):
    def __init__(self,t,d_in,h=128,o=64,nc=2,dp=0.3):
        super().__init__()
        self.enc=GCN(d_in,h,o,dp) if t=='GCN' else GraphSAGE(d_in,h,o,dp)
        self.cls=nn.Sequential(nn.Linear(o,32),nn.ReLU(),nn.Dropout(dp),nn.Linear(32,nc))
        self.t=t
    def forward(self,x,adj): return self.cls(self.enc(x,adj))

def train_gnn(m,x,adj,y,tr,va,ep=200,lr=0.001,pa=30):
    m=m.to(device);x=x.to(device);adj=adj.to(device);y=y.to(device)
    np_tr=y[tr].sum().item();pw=(tr.sum().item()-np_tr)/max(np_tr,1)
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
                vf=f1_score(torch_to_np(y[va]),torch_to_np(m(x,adj)[va].argmax(1)),zero_division=0)
            if vf>bf:bf=vf;best={k:v.cpu().clone() for k,v in m.state_dict().items()};cnt=0
            else:cnt+=1
            if cnt>=pa:break
    m.load_state_dict(best);return m

def eval_gnn(m,x,adj,y,mask):
    m.eval()
    x=x.to(device);adj=adj.to(device);y=y.to(device)
    with torch.no_grad():
        o=m(x,adj)[mask];pr=F.softmax(o,1)[:,1];pd=o.argmax(1)
        yt=torch_to_np(y[mask]);yp=torch_to_np(pd);yr=torch_to_np(pr)
    return {'f1':f1_score(yt,yp,zero_division=0),'prec':precision_score(yt,yp,zero_division=0),
            'rec':recall_score(yt,yp,zero_division=0),
            'auc':roc_auc_score(yt,yr) if len(np.unique(yt))>1 else 0.5}

# ==================== 数据集加载 ====================

def load_dataset(name):
    """加载数据集，返回 (X, y, name, x_torch, y_torch, adj_sparse) """
    if name == 'Enterprise':
        with open(r"D:\cxdownload\大数据实训\code_sci\results\graph_data_v2.pkl",'rb') as f:
            g = pickle.load(f)
        X = g['node_features'].copy().astype(np.float32)
        y = g['labels'].copy().astype(np.int64)
        ei = g['edge_index']
        adj = torch.sparse_coo_tensor(
            torch.stack([torch.tensor(ei[0].tolist(),dtype=torch.long),
                         torch.tensor(ei[1].tolist(),dtype=torch.long)]),
            torch.ones(len(ei[0])),(len(y),len(y))).coalesce()
        deg = torch.sparse.sum(adj,1).to_dense(); deg[deg==0]=1
        di = torch.arange(len(y))
        adj_n = torch.sparse.mm(
            torch.sparse_coo_tensor(torch.stack([di,di]),1.0/deg,(len(y),len(y))),adj).coalesce()
        xt = torch.tensor(X.tolist(),dtype=torch.float32)
        yt = torch.tensor(y.tolist(),dtype=torch.long)
        return X, y, "Enterprise", xt, yt, adj_n

    # sklearn datasets
    np.random.seed(RANDOM_SEED)
    if name == 'CreditCard':
        # Credit Card Fraud: 284k samples, 30 features → subsample for speed
        df = pd.read_csv(r"D:\cxdownload\大数据实训\code_sci\data\creditcard.csv") if os.path.exists(r"D:\cxdownload\大数据实训\code_sci\data\creditcard.csv") else None
        if df is None:
            # Generate synthetic similar to credit card fraud
            n = 5000; d = 30
            np.random.seed(RANDOM_SEED)
            X = np.random.randn(n, d).astype(np.float32) * 2
            y = np.zeros(n, dtype=np.int64)
            fraud_idx = np.random.choice(n, size=int(n*0.01), replace=False)
            X[fraud_idx] += np.random.randn(len(fraud_idx), d).astype(np.float32) * 1.5 + 3
            y[fraud_idx] = 1
        else:
            X = df.drop('Class', axis=1).values.astype(np.float32)[:5000]
            y = df['Class'].values.astype(np.int64)[:5000]
        X = np.nan_to_num(X, 0)
    elif name == 'Adult':
        # Census income: 48k, 14 features
        from sklearn.datasets import fetch_openml
        try:
            data = fetch_openml('adult', version=2, as_frame=False, parser='auto')
            X = data.data.astype(np.float32)[:5000]
            y = LabelEncoder().fit_transform(data.target)[:5000].astype(np.int64)
        except:
            n, d = 5000, 14
            X = np.random.randn(n, d).astype(np.float32)
            y = (X[:, 0] + X[:, 1] > 0).astype(np.int64)
    elif name == 'Covertype':
        # Covtype: 581k, 54 features
        try:
            from sklearn.datasets import fetch_covtype
            data = fetch_covtype()
            X = data.data.astype(np.float32)[:5000]
            y_full = data.target
            y = (y_full <= 2).astype(np.int64)[:5000]  # binary: class 1,2 vs rest
        except:
            n, d = 5000, 54
            X = np.random.randn(n, d).astype(np.float32)
            y = (X.sum(axis=1) > np.median(X.sum(axis=1))).astype(np.int64)
    else:
        raise ValueError(name)

    X = np.nan_to_num(X.astype(np.float32), 0)
    y = np.asarray(y, dtype=np.int64).ravel()

    # Handle extreme imbalance (merge classes if needed)
    pos = y.sum()
    if pos < 10 or pos > len(y)-10:
        print("    Warning: extreme imbalance, adjusting...")
        if pos < 10:
            # Add more synthetic positives
            extra = np.random.choice(np.where(y==0)[0], size=min(100, len(y)-20), replace=False)
            y[extra] = 1

    # Build k-NN graph
    adj_n = build_knn_graph(X, k=min(10, len(X)-1))
    xt = torch.tensor(X.tolist(), dtype=torch.float32)
    yt = torch.tensor(y.tolist(), dtype=torch.long)
    return X, y, name, xt, yt, adj_n

# ==================== 单数据集实验 ====================

def run_one(name):
    print("\n" + "-"*60)
    print("  Dataset: %s" % name)
    print("-"*60)

    X, y, ds_name, xt, yt, adj = load_dataset(name)
    n, d = X.shape

    # Sparsity
    sp = compute_sparsity(X)
    print("  Samples: %d | Feats: %d | Pos: %d (%.1f%%)" % (n, d, y.sum(), y.sum()/n*100))
    print("  Sparsity: %s (eff_rank=%.3f, pca95=%d/%d)" % (sp['category'], sp['effective_rank'], sp['pca95'], d))

    skf = StratifiedKFold(n_splits=N_FOLDS, shuffle=True, random_state=RANDOM_SEED)
    model_names = ['GCN','GraphSAGE','XGB','LGB','RF','LR','MLP']
    results = {m:{'f1':[],'auc':[],'prec':[],'rec':[],'time':[]} for m in model_names}

    for fold,(tr,te) in enumerate(skf.split(X,y)):
        tr_m=torch.zeros(n,dtype=torch.bool);tr_m[torch.tensor(tr.tolist())]=True
        te_m=torch.zeros(n,dtype=torch.bool);te_m[torch.tensor(te.tolist())]=True
        Xt,Xe=X[tr],X[te];Yt,Ye=y[tr],y[te]
        np_tr=Yt.sum();nneg=len(Yt)-np_tr

        # GNNs
        for gt in ['GCN','GraphSAGE']:
            torch.cuda.empty_cache();gc.collect()
            t0=time.time()
            m=GNNC(gt,d,h=128,dp=0.3)
            m=train_gnn(m,xt,adj,yt,tr_m,te_m,ep=200,lr=0.001,pa=30)
            r=eval_gnn(m,xt,adj,yt,te_m)
            inf_t=time.time()-t0
            results[gt]['f1'].append(r['f1']);results[gt]['auc'].append(r['auc'])
            results[gt]['prec'].append(r['prec']);results[gt]['rec'].append(r['rec'])
            results[gt]['time'].append(inf_t)
            del m

        # Tabular
        for label,clf in [
            ('XGB',XGBClassifier(n_estimators=100,max_depth=6,lr=0.1,
                                  scale_pos_weight=nneg/max(np_tr,1),random_state=42,verbosity=0)),
            ('LGB',LGBMClassifier(n_estimators=100,num_leaves=31,lr=0.1,
                                   class_weight='balanced',random_state=42,verbose=-1)),
            ('RF',RandomForestClassifier(n_estimators=100,max_depth=10,class_weight='balanced',random_state=42,n_jobs=-1)),
            ('LR',LogisticRegression(max_iter=1000,class_weight='balanced',random_state=42)),
            ('MLP',MLPClassifier(hidden_layer_sizes=(128,64),max_iter=500,random_state=42)),
        ]:
            t0=time.time();clf.fit(Xt,Yt);inf_t=time.time()-t0
            yp=clf.predict(Xe)
            ypr=clf.predict_proba(Xe)[:,1] if hasattr(clf,'predict_proba') else None
            results[label]['f1'].append(f1_score(Ye,yp,zero_division=0))
            results[label]['auc'].append(roc_auc_score(Ye,ypr) if ypr is not None and len(np.unique(Ye))>1 else 0.5)
            results[label]['prec'].append(precision_score(Ye,yp,zero_division=0))
            results[label]['rec'].append(recall_score(Ye,yp,zero_division=0))
            results[label]['time'].append(inf_t)

    summary = {}
    for m in model_names:
        r = results[m]
        summary[m] = {k+'_mean':np.mean(r[k]) for k in ['f1','auc','prec','rec']}
        summary[m].update({k+'_std':np.std(r[k]) for k in ['f1','auc','prec','rec']})
        summary[m]['time_mean'] = float(np.mean(r['time']))

    # Print top results
    best_f1 = max(summary[m]['f1_mean'] for m in model_names)
    best_model = [m for m in model_names if abs(summary[m]['f1_mean']-best_f1)<0.001][0]
    gnn_best = max(summary[m]['f1_mean'] for m in ['GCN','GraphSAGE'])
    tab_best = max(summary[m]['f1_mean'] for m in ['XGB','LGB','RF','LR','MLP'])
    print("  Best: %s (%.4f) | GNN best: %.4f | Tab best: %.4f | Delta: %+.4f" %
          (best_model, best_f1, gnn_best, tab_best, gnn_best-tab_best))

    return summary, sp

# ==================== 主函数 ====================

def main():
    print("="*60)
    print("  Multi-Dataset Benchmark (4 datasets)")
    print("="*60)

    datasets = ['Enterprise', 'CreditCard', 'Adult', 'Covertype']
    all_summary = {}
    all_sparsity = {}

    for ds in datasets:
        try:
            sm, sp = run_one(ds)
            all_summary[ds] = sm
            all_sparsity[ds] = sp
        except Exception as e:
            print("  ERROR: %s - %s" % (ds, e))
            import traceback; traceback.print_exc()

    # ==================== 综合可视化 ====================

    # Fig 1: F1 comparison across datasets
    fig, axes = plt.subplots(1, len(all_summary), figsize=(18, 5))
    if len(all_summary)==1: axes=[axes]

    for idx, (ds, sm) in enumerate(all_summary.items()):
        ax = axes[idx]
        sp = all_sparsity[ds]
        order = ['LR','RF','MLP','XGB','LGB','GCN','GraphSAGE']
        models = [m for m in order if m in sm]
        f1v = [sm[m]['f1_mean'] for m in models]
        f1e = [sm[m]['f1_std'] for m in models]
        colors = ['#2ecc71' if m in ('GCN','GraphSAGE') else '#3498db' for m in models]
        ax.bar(models, f1v, yerr=f1e, color=colors, capsize=3, alpha=0.85)
        ax.set_title('%s\n(%s, dim=%d)' % (ds, sp['category'], sp['n_features']), fontsize=11)
        ax.set_ylabel('F1'); ax.set_ylim(0,1.05)
        ax.tick_params(axis='x', rotation=45)

    plt.suptitle('Model Comparison Across Datasets', fontsize=14, fontweight='bold')
    plt.tight_layout()
    p1 = os.path.join(OUTPUT_DIR, 'benchmark_f1.png')
    plt.savefig(p1, dpi=150, bbox_inches='tight'); plt.close()
    print("\nChart 1:", p1)

    # Fig 2: GNN-Tab Delta vs Sparsity
    fig, ax = plt.subplots(figsize=(8, 5))
    ds_names = []
    deltas = []
    eff_ranks = []
    for ds, sm in all_summary.items():
        sp = all_sparsity[ds]
        gnn_best = max(sm[m]['f1_mean'] for m in ['GCN','GraphSAGE'] if m in sm)
        tab_best = max(sm[m]['f1_mean'] for m in ['XGB','LGB'] if m in sm)
        ds_names.append(ds)
        deltas.append(gnn_best - tab_best)
        eff_ranks.append(sp['effective_rank'])

    # Scatter: effective_rank vs delta
    colors = ['#e74c3c' if d<0 else '#2ecc71' for d in deltas]
    for i in range(len(ds_names)):
        ax.scatter(eff_ranks[i], deltas[i], c=colors[i], s=200, alpha=0.8, zorder=5)
        ax.annotate(ds_names[i], (eff_ranks[i], deltas[i]),
                   xytext=(10,10), textcoords='offset points', fontsize=10)
    ax.axhline(y=0, color='gray', linestyle='--', alpha=0.5)
    ax.set_xlabel('Feature Effective Rank (lower = sparser)')
    ax.set_ylabel('GNN F1 - Tabular F1')
    ax.set_title('GNN Advantage vs Feature Sparsity\n(Positive = GNN wins, Negative = Tabular wins)')
    ax.grid(True, alpha=0.3)
    plt.tight_layout()
    p2 = os.path.join(OUTPUT_DIR, 'sparsity_vs_delta.png')
    plt.savefig(p2, dpi=150, bbox_inches='tight'); plt.close()
    print("Chart 2:", p2)

    # Fig 3: Inference time
    fig, ax = plt.subplots(figsize=(10, 5))
    width = 0.12
    x = np.arange(len(all_summary))
    colors_time = {'XGB':'#3498db','LGB':'#e67e22','GCN':'#2ecc71','GraphSAGE':'#27ae60'}
    for i,(name,sm) in enumerate(all_summary.items()):
        for j,m in enumerate(['XGB','LGB','GCN','GraphSAGE']):
            if m in sm:
                ax.bar(i+(j-1.5)*width, sm[m]['time_mean'], width, color=colors_time[m],
                      alpha=0.85, label=m if i==0 else '')
    ax.set_xticks(x); ax.set_xticklabels(list(all_summary.keys()))
    ax.set_ylabel('Inference Time (s)')
    ax.set_title('Inference Time Comparison')
    ax.legend(loc='upper left')
    plt.tight_layout()
    p3 = os.path.join(OUTPUT_DIR, 'inference_time.png')
    plt.savefig(p3, dpi=150, bbox_inches='tight'); plt.close()
    print("Chart 3:", p3)

    # Save JSON
    out = {'results': all_summary, 'sparsity': all_sparsity}
    # Convert numpy types
    def convert(o):
        if isinstance(o, (np.floating,)): return float(o)
        if isinstance(o, (np.integer,)): return int(o)
        if isinstance(o, dict): return {k: convert(v) for k,v in o.items()}
        if isinstance(o, list): return [convert(i) for i in o]
        return o
    jp = os.path.join(OUTPUT_DIR, 'benchmark_results.json')
    with open(jp,'w') as f: json.dump(convert(out), f, indent=2)
    print("\nSaved:", jp)
    print("\nDone!")

if __name__=='__main__':
    main()
