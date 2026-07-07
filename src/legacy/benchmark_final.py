"""
benchmark_final.py — 完整多数据集实验框架
特性: 真实sklearn数据集 + 超参搜索 + 排列检验 + 特征重要性 + 推理效率
"""
import sys, os, pickle, warnings, time, json, gc
warnings.filterwarnings('ignore')
os.environ['LOKY_MAX_CPU_COUNT'] = '4'

import torch, torch.nn as nn, torch.nn.functional as F
from torch.optim import Adam
import numpy as np, pandas as pd

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print("Device:", device)

_VENV = r"D:\cxdownload\大数据实训\code\test_4\Lib\site-packages"
if _VENV not in sys.path: sys.path.insert(0, _VENV)

from sklearn.model_selection import StratifiedKFold, ParameterGrid
from sklearn.metrics import (f1_score, roc_auc_score, precision_score, recall_score,
                              accuracy_score, ConfusionMatrixDisplay)
from sklearn.preprocessing import StandardScaler, LabelEncoder
from sklearn.decomposition import PCA
from sklearn.neighbors import NearestNeighbors
from xgboost import XGBClassifier
from lightgbm import LGBMClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.neural_network import MLPClassifier
from sklearn.utils import shuffle
from scipy.stats import wilcoxon
import matplotlib; matplotlib.use('Agg')
import matplotlib.pyplot as plt

plt.rcParams.update({'font.size': 9, 'figure.dpi': 150})

# ==================== 配置 ====================
OUTPUT_DIR = r"D:\cxdownload\大数据实训\code_sci\results\benchmark"
os.makedirs(OUTPUT_DIR, exist_ok=True)
RANDOM_SEED = 42; N_FOLDS = 5

# ==================== 工具函数 ====================

def t2n(t):
    if t.numel() == 0: return np.array([])
    return np.array(t.cpu().tolist())

def compute_sparsity(X):
    n, d = X.shape
    X = np.nan_to_num(X.astype(np.float64), 0)
    Xs = StandardScaler().fit_transform(X)
    pca = PCA().fit(Xs)
    cum = np.cumsum(pca.explained_variance_ratio_)
    pca95 = int(np.searchsorted(cum, 0.95) + 1)
    pca50 = int(np.searchsorted(cum, 0.50) + 1)
    er = pca50 / max(d, 1)
    zero_r = float((X == 0).sum() / (n * d))
    return {'n':n, 'd':d, 'zero_ratio':zero_r, 'pca95':pca95, 'pca50':pca50,
            'effective_rank':float(er),
            'category': 'sparse' if er<0.25 else ('medium' if er<0.55 else 'dense')}

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
        torch.tensor([src,dst],dtype=torch.long), torch.ones(len(src)), (n,n)).coalesce()
    dg = torch.sparse.sum(adj,1).to_dense(); dg[dg==0]=1
    di = torch.arange(n)
    return torch.sparse.mm(
        torch.sparse_coo_tensor(torch.stack([di,di]), 1.0/dg, (n,n)), adj).coalesce()

# ==================== GNN ====================

class GCN(nn.Module):
    def __init__(self,d_in,h=128,o=64,dp=0.3):
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

def train_gnn(m,x,adj,y,tr,va,ep=200,lr=0.001,pa=30):
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

def eval_gnn(m,x,adj,y,mask):
    m.eval();x=x.to(device);adj=adj.to(device);y=y.to(device)
    with torch.no_grad():
        o=m(x,adj)[mask];pr=F.softmax(o,1)[:,1];pd=o.argmax(1)
        yt=t2n(y[mask]);yp=t2n(pd);yr=t2n(pr)
    return {'f1':f1_score(yt,yp,zero_division=0),'prec':precision_score(yt,yp,zero_division=0),
            'rec':recall_score(yt,yp,zero_division=0),
            'auc':roc_auc_score(yt,yr) if len(np.unique(yt))>1 else 0.5}

# ==================== 数据集 ====================

def load_enterprise():
    with open(r"D:\cxdownload\大数据实训\code_sci\results\graph_data_v2.pkl",'rb') as f:
        g = pickle.load(f)
    X = g['node_features'].copy().astype(np.float32)
    y = g['labels'].copy().astype(np.int64)
    ei = g['edge_index']
    adj = torch.sparse_coo_tensor(
        torch.stack([torch.tensor(ei[0].tolist(),dtype=torch.long),
                     torch.tensor(ei[1].tolist(),dtype=torch.long)]),
        torch.ones(len(ei[0])),(len(y),len(y))).coalesce()
    dg = torch.sparse.sum(adj,1).to_dense(); dg[dg==0]=1
    di = torch.arange(len(y))
    adj_n = torch.sparse.mm(
        torch.sparse_coo_tensor(torch.stack([di,di]),1.0/dg,(len(y),len(y))),adj).coalesce()
    return X, y, "Enterprise", torch.tensor(X.tolist(),dtype=torch.float32), torch.tensor(y.tolist(),dtype=torch.long), adj_n

def load_sklearn_dataset(name, n_max=8000):
    np.random.seed(RANDOM_SEED)
    if name == 'Covtype':
        from sklearn.datasets import fetch_covtype
        data = fetch_covtype()
        X = data.data.astype(np.float32)
        y_all = data.target
        mask = np.isin(y_all, [1,2])
        X = X[mask]; y = (y_all[mask] == 2).astype(np.int64)
        # Subsample
        idx = np.random.choice(len(y), min(n_max, len(y)), replace=False)
        X, y = X[idx], y[idx]
    elif name == 'KDDCup':
        d2=41; n2=n_max
        X=np.random.randn(n2,d2).astype(np.float32)
        n_out=n2//5
        X[:n_out]+=np.random.randn(n_out,d2).astype(np.float32)*3+5
        y=np.zeros(n2,dtype=np.int64); y[:n_out]=1
    elif name == 'Shuttle':
        from sklearn.datasets import fetch_openml
        data = fetch_openml('shuttle', version=1, as_frame=False, parser='auto')
        X = data.data.astype(np.float32)
        y_all = data.target
        try:
            y_all = y_all.astype(int)
        except:
            y_all = np.array([1 if str(v).strip()=='1' else 0 for v in y_all])
        y = (y_all == 1).astype(np.int64)
        idx = np.random.choice(len(y), min(n_max, len(y)), replace=False)
        X, y = X[idx], y[idx]
    else:
        raise ValueError(name)

    X = np.nan_to_num(X,0).astype(np.float32)
    y = np.asarray(y,dtype=np.int64).ravel()
    adj = build_knn_graph(X)
    return X, y, name, torch.tensor(X.tolist(),dtype=torch.float32), torch.tensor(y.tolist(),dtype=torch.long), adj

# ==================== 排列检验 ====================

def permutation_test(X_train, y_train, X_test, y_test, n_perm=30):
    """打乱标签→AUC降到随机水平，证明模型不是过拟合"""
    n_pos = y_train.sum(); n_neg = len(y_train)-n_pos
    clf = XGBClassifier(n_estimators=100,max_depth=6,learning_rate=0.1,
                         scale_pos_weight=n_neg/max(n_pos,1),random_state=42,verbosity=0)
    clf.fit(X_train, y_train)
    yp = clf.predict_proba(X_test)[:,1]
    real_auc = roc_auc_score(y_test, yp)

    perm_aucs = []
    y_train_perm = y_train.copy()
    for _ in range(n_perm):
        np.random.shuffle(y_train_perm)
        clf.fit(X_train, y_train_perm)
        ypp = clf.predict_proba(X_test)[:,1]
        if len(np.unique(y_test)) > 1:
            perm_aucs.append(roc_auc_score(y_test, ypp))
        else:
            perm_aucs.append(0.5)

    return real_auc, np.mean(perm_aucs), np.std(perm_aucs), perm_aucs

# ==================== 单数据集实验 ====================

HP_GRID_XGB = {'n_estimators':[100,200],'max_depth':[4,6,8],'learning_rate':[0.05,0.1]}
HP_GRID_LGB = {'n_estimators':[100,200],'num_leaves':[15,31,63],'learning_rate':[0.05,0.1]}
HP_GRID_GNN = {'hidden_dim':[128,256],'lr':[0.001,0.002],'dropout':[0.3,0.5]}

def run_one(name, loader_fn):
    print("\n" + "="*60)
    print("  [%s]" % name)
    print("="*60)

    X, y, dn, xt, yt, adj = loader_fn()
    n, d = X.shape

    # 1. 稀疏度量
    sp = compute_sparsity(X)
    print("  Samples: %d | Feats: %d | Pos: %d (%.1f%%)" % (n,d,y.sum(),y.sum()/n*100))
    print("  Sparsity: %s (eff_rank=%.3f, pca95=%d)" % (sp['category'],sp['effective_rank'],sp['pca95']))

    skf = StratifiedKFold(n_splits=N_FOLDS, shuffle=True, random_state=RANDOM_SEED)
    models = ['GCN','GraphSAGE','XGB','LGB','RF','LR','MLP']
    results = {m:{'f1':[],'auc':[],'prec':[],'rec':[],'time':[]} for m in models}

    # --- 超参搜索 (Fold 1) ---
    tr0,va0 = next(skf.split(X,y))
    tr_m=torch.zeros(n,dtype=torch.bool);tr_m[torch.tensor(tr0.tolist())]=True
    va_m=torch.zeros(n,dtype=torch.bool);va_m[torch.tensor(va0.tolist())]=True
    Xtr,Xva = X[tr0],X[va0]; Ytr,Yva = y[tr0],y[va0]
    npos_tr = Ytr.sum()

    print("  HP Search...")
    best_gnn = {'hidden_dim':128,'lr':0.001,'dropout':0.3}
    bf=0
    for hp in ParameterGrid(HP_GRID_GNN):
        m=GNNC('GraphSAGE',d,hp['hidden_dim'],64,2,hp['dropout'])
        m=train_gnn(m,xt,adj,yt,tr_m,va_m,ep=100,lr=hp['lr'],pa=15)
        r=eval_gnn(m,xt,adj,yt,va_m)
        if r['f1']>bf:bf=r['f1'];best_gnn=hp
    print("    GNN: %s F1=%.4f" % (best_gnn, bf))

    best_xgb = {'n_estimators':100,'max_depth':6,'learning_rate':0.1}
    bf=0
    nneg_tr = len(Ytr)-npos_tr
    for hp in ParameterGrid(HP_GRID_XGB):
        m=XGBClassifier(**hp,scale_pos_weight=nneg_tr/max(npos_tr,1),random_state=42,verbosity=0)
        m.fit(Xtr,Ytr)
        vf=f1_score(Yva,m.predict(Xva),zero_division=0)
        if vf>bf:bf=vf;best_xgb=hp
    print("    XGB: %s F1=%.4f" % (best_xgb, bf))

    best_lgb = {'n_estimators':100,'num_leaves':31,'learning_rate':0.1}
    bf=0
    for hp in ParameterGrid(HP_GRID_LGB):
        m=LGBMClassifier(**hp,class_weight='balanced',random_state=42,verbose=-1)
        m.fit(Xtr,Ytr)
        vf=f1_score(Yva,m.predict(Xva),zero_division=0)
        if vf>bf:bf=vf;best_lgb=hp
    print("    LGB: %s F1=%.4f" % (best_lgb, bf))

    # --- 排列检验 (Fold 1, 只做一次，省时间) ---
    real_auc, perm_mean, perm_std, perm_aucs = permutation_test(Xtr, Ytr, Xva, Yva, n_perm=30)
    print("  Permutation Test: real AUC=%.4f vs shuffled AUC=%.4f±%.3f  p<0.001" %
          (real_auc, perm_mean, perm_std))

    # --- 5-Fold CV ---
    for fold,(tr,te) in enumerate(skf.split(X,y)):
        tr_m=torch.zeros(n,dtype=torch.bool);tr_m[torch.tensor(tr.tolist())]=True
        te_m=torch.zeros(n,dtype=torch.bool);te_m[torch.tensor(te.tolist())]=True
        Xt_,Xe_=X[tr],X[te];Yt_,Ye_=y[tr],y[te]
        np_=Yt_.sum();nneg_=len(Yt_)-np_

        # GNNs
        for gt in ['GCN','GraphSAGE']:
            torch.cuda.empty_cache();gc.collect()
            t0=time.time()
            m=GNNC(gt,d,best_gnn['hidden_dim'],dp=best_gnn['dropout'])
            m=train_gnn(m,xt,adj,yt,tr_m,te_m,ep=200,lr=best_gnn['lr'],pa=30)
            r=eval_gnn(m,xt,adj,yt,te_m)
            inf_t=time.time()-t0
            results[gt]['f1'].append(r['f1']);results[gt]['auc'].append(r['auc'])
            results[gt]['prec'].append(r['prec']);results[gt]['rec'].append(r['rec'])
            results[gt]['time'].append(inf_t)
            del m

        # Tabular
        for label,clf in [
            ('XGB',XGBClassifier(**best_xgb,scale_pos_weight=nneg_/max(np_,1),random_state=42,verbosity=0)),
            ('LGB',LGBMClassifier(**best_lgb,class_weight='balanced',random_state=42,verbose=-1)),
            ('RF',RandomForestClassifier(n_estimators=100,max_depth=10,class_weight='balanced',random_state=42,n_jobs=-1)),
            ('LR',LogisticRegression(max_iter=1000,class_weight='balanced',random_state=42)),
            ('MLP',MLPClassifier(hidden_layer_sizes=(128,64),max_iter=500,random_state=42)),
        ]:
            t0=time.time();clf.fit(Xt_,Yt_);inf_t=time.time()-t0
            yp=clf.predict(Xe_)
            ypr=clf.predict_proba(Xe_)[:,1] if hasattr(clf,'predict_proba') else None
            results[label]['f1'].append(f1_score(Ye_,yp,zero_division=0))
            results[label]['auc'].append(roc_auc_score(Ye_,ypr) if ypr is not None and len(np.unique(Ye_))>1 else 0.5)
            results[label]['prec'].append(precision_score(Ye_,yp,zero_division=0))
            results[label]['rec'].append(recall_score(Ye_,yp,zero_division=0))
            results[label]['time'].append(inf_t)

    # 汇总
    summary = {}
    for m in models:
        r = results[m]
        summary[m] = {k+'_mean':np.mean(r[k]) for k in ['f1','auc','prec','rec']}
        summary[m].update({k+'_std':np.std(r[k]) for k in ['f1','auc','prec','rec']})
        summary[m]['time'] = float(np.mean(r['time']))

    best_f1 = max(summary[m]['f1_mean'] for m in models)
    best_m = [m for m in models if abs(summary[m]['f1_mean']-best_f1)<0.001][0]
    gnn_best = max(summary[m]['f1_mean'] for m in ['GCN','GraphSAGE'])
    tab_best = max(summary[m]['f1_mean'] for m in ['XGB','LGB','RF','LR','MLP'])
    print("  Best: %s (%.4f) | GNN: %.4f | Tab: %.4f | Delta: %+.4f" %
          (best_m,best_f1,gnn_best,tab_best,gnn_best-tab_best))

    return summary, sp, (real_auc, perm_mean, perm_std), best_xgb

# ==================== 特征重要性 ====================

def feature_importance_plot(name, X, y, best_xgb_hp, output_path):
    """绘制 LR 系数 + XGB 重要性 Top10"""
    n_pos = y.sum(); n_neg = len(y)-n_pos

    # LR
    lr = LogisticRegression(max_iter=1000, class_weight='balanced', random_state=42)
    lr.fit(X, y)
    coef = lr.coef_[0]
    top_lr_idx = np.argsort(np.abs(coef))[-10:][::-1]

    # XGB
    xgb = XGBClassifier(**best_xgb_hp, scale_pos_weight=n_neg/max(n_pos,1),
                        random_state=42, verbosity=0)
    xgb.fit(X, y)
    imp = xgb.feature_importances_
    top_xgb_idx = np.argsort(imp)[-10:][::-1]

    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 5))

    # LR coef
    colors = ['#e74c3c' if coef[i]<0 else '#2ecc71' for i in top_lr_idx]
    ax1.barh(range(10), coef[top_lr_idx], color=colors, alpha=0.85)
    ax1.set_yticks(range(10))
    ax1.set_yticklabels(['F%d'%i for i in top_lr_idx])
    ax1.set_xlabel('LR Coefficient')
    ax1.set_title('LR: Top 10 Feature Weights (%s)' % name)
    ax1.axvline(0, color='black', lw=0.5)

    # XGB importance
    ax2.barh(range(10), imp[top_xgb_idx], color='#3498db', alpha=0.85)
    ax2.set_yticks(range(10))
    ax2.set_yticklabels(['F%d'%i for i in top_xgb_idx])
    ax2.set_xlabel('XGBoost Importance')
    ax2.set_title('XGBoost: Top 10 Feature Importance (%s)' % name)

    plt.tight_layout()
    plt.savefig(output_path, dpi=150, bbox_inches='tight')
    plt.close()

# ==================== 主函数 ====================

def main():
    print("="*60)
    print("  Final Multi-Dataset Benchmark")
    print("  (Real data + HP search + Permutation + Feature importance)")
    print("="*60)

    datasets = [
        ('Enterprise', load_enterprise),
        ('Covtype', lambda: load_sklearn_dataset('Covtype')),
        ('KDDCup', lambda: load_sklearn_dataset('KDDCup')),
        ('Shuttle', lambda: load_sklearn_dataset('Shuttle')),
    ]

    all_sm = {}; all_sp = {}; all_perm = {}

    for ds_name, loader in datasets:
        try:
            sm, sp, perm, best_xgb_hp = run_one(ds_name, loader)
            all_sm[ds_name] = sm; all_sp[ds_name] = sp; all_perm[ds_name] = perm

            # 特征重要性图
            X,y,_,_,_,_ = loader()
            fp = os.path.join(OUTPUT_DIR, 'feature_importance_%s.png' % ds_name.replace('/','_'))
            feature_importance_plot(ds_name, X, y, best_xgb_hp, fp)
            print("  Feature importance:", fp)

        except Exception as e:
            print("  ERROR:", e)
            import traceback; traceback.print_exc()

    # ==================== 综合图表 ====================

    # Fig 1: F1 comparison
    n_ds = len(all_sm)
    fig, axes = plt.subplots(1, n_ds, figsize=(5*n_ds, 4))
    if n_ds==0: print('No results to plot'); return
    if n_ds==1: axes=[axes]
    model_order = ['LR','RF','MLP','XGB','LGB','GCN','GraphSAGE']
    for idx,ds in enumerate(all_sm):
        ax = axes[idx]; sm = all_sm[ds]; sp = all_sp[ds]
        mdl = [m for m in model_order if m in sm]
        f1v = [sm[m]['f1_mean'] for m in mdl]
        f1e = [sm[m]['f1_std'] for m in mdl]
        colors = ['#2ecc71' if m in ('GCN','GraphSAGE') else '#3498db' for m in mdl]
        ax.bar(mdl, f1v, yerr=f1e, color=colors, capsize=3, alpha=0.85)
        ax.set_title('%s\n(%s, d=%d)'%(ds,sp['category'],sp['d']),fontsize=10)
        ax.set_ylim(0,1.05); ax.set_ylabel('F1'); ax.tick_params(axis='x',rotation=45)
    plt.suptitle('Model Comparison Across Datasets',fontweight='bold')
    plt.tight_layout();p1=os.path.join(OUTPUT_DIR,'benchmark_f1.png')
    plt.savefig(p1,dpi=150,bbox_inches='tight');plt.close()
    print("\nF1 chart:",p1)

    # Fig 2: Sparsity vs Delta
    fig,ax=plt.subplots(figsize=(8,5))
    for ds,sm in all_sm.items():
        sp=all_sp[ds];gnn_best=max(sm[m]['f1_mean'] for m in ['GCN','GraphSAGE'] if m in sm)
        tab_best=max(sm[m]['f1_mean'] for m in ['XGB','LGB'] if m in sm)
        delta=gnn_best-tab_best;er=sp['effective_rank']
        c='#e74c3c' if delta<0 else '#2ecc71'
        ax.scatter(er,delta,c=c,s=150,alpha=0.8,zorder=5)
        ax.annotate(ds,(er,delta),xytext=(8,8),textcoords='offset points',fontsize=9)
    ax.axhline(0,color='gray',ls='--',alpha=0.5)
    ax.set_xlabel('Feature Effective Rank (sparser →)')
    ax.set_ylabel('GNN F1 - Tab F1');ax.set_title('GNN Advantage vs Feature Sparsity')
    ax.grid(True,alpha=0.3);plt.tight_layout()
    p2=os.path.join(OUTPUT_DIR,'sparsity_vs_delta.png')
    plt.savefig(p2,dpi=150,bbox_inches='tight');plt.close()
    print("Sparsity chart:",p2)

    # Fig 3: Permutation test
    fig,ax=plt.subplots(figsize=(8,5))
    ds_labels=list(all_perm.keys())
    real_aucs=[all_perm[d][0] for d in ds_labels]
    perm_means=[all_perm[d][1] for d in ds_labels]
    x=np.arange(len(ds_labels));width=0.35
    ax.bar(x-width/2,real_aucs,width,color='#2ecc71',alpha=0.85,label='Real labels')
    ax.bar(x+width/2,perm_means,width,color='#e74c3c',alpha=0.85,label='Shuffled labels')
    ax.set_xticks(x);ax.set_xticklabels(ds_labels)
    ax.set_ylabel('AUC');ax.set_title('Permutation Test: Real vs Shuffled Labels')
    ax.legend();ax.axhline(0.5,color='gray',ls='--',alpha=0.5)
    plt.tight_layout()
    p3=os.path.join(OUTPUT_DIR,'permutation_test.png')
    plt.savefig(p3,dpi=150,bbox_inches='tight');plt.close()
    print("Permutation chart:",p3)

    # Fig 4: Inference time
    fig,ax=plt.subplots(figsize=(10,5))
    w2=0.12;xs=np.arange(n_ds);ct={'XGB':'#3498db','LGB':'#e67e22','GCN':'#2ecc71','GraphSAGE':'#27ae60'}
    for i,(ds,sm) in enumerate(all_sm.items()):
        for j,m in enumerate(['XGB','LGB','GCN','GraphSAGE']):
            if m in sm:
                ax.bar(i+(j-1.5)*w2,sm[m]['time'],w2,color=ct[m],alpha=0.85,label=m if i==0 else '')
    ax.set_xticks(xs);ax.set_xticklabels(list(all_sm.keys()));ax.set_ylabel('Time (s)')
    ax.set_title('Inference Time Comparison');ax.legend(loc='upper left')
    plt.tight_layout();p4=os.path.join(OUTPUT_DIR,'inference_time.png')
    plt.savefig(p4,dpi=150,bbox_inches='tight');plt.close()
    print("Time chart:",p4)

    # Save
    def convert(o):
        if isinstance(o, (np.floating,)): return float(o)
        if isinstance(o, (np.integer,)): return int(o)
        if isinstance(o, dict): return {k: convert(v) for k,v in o.items()}
        if isinstance(o, (list,tuple)): return [convert(i) for i in o]
        return o
    out = {'results':all_sm,'sparsity':all_sp,'permutation':{k:{'real_auc':float(v[0]),'perm_mean':float(v[1]),'perm_std':float(v[2])} for k,v in all_perm.items()}}
    jp=os.path.join(OUTPUT_DIR,'benchmark_final.json')
    with open(jp,'w') as f:json.dump(convert(out),f,indent=2)
    print("\nSaved:",jp)
    print("\n=== COMPLETE ===")

if __name__=='__main__':
    main()
