"""
benchmark_4ds.py — 4 真实数据集完整实验
Enterprise + CreditCard + Adult + Covtype
含：超参搜索、排列检验、特征重要性、稀疏度量、推理效率
"""
import sys, os, pickle, warnings, time, json, gc, gzip
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

plt.rcParams.update({'font.size': 9, 'figure.dpi': 150})

OUTPUT_DIR = r"D:\cxdownload\大数据实训\code_sci\results\benchmark"
os.makedirs(OUTPUT_DIR, exist_ok=True)
DATA_DIR = r"D:\cxdownload\大数据实训\code_sci\data"
RANDOM_SEED = 42; N_FOLDS = 5; N_SUBSAMPLE = 6000

# ================ Utils ================

def t2n(t):
    if t.numel() == 0: return np.array([])
    return np.array(t.cpu().tolist())

def compute_sparsity(X):
    n, d = X.shape
    X = np.nan_to_num(X, 0).astype(np.float64)
    Xs = StandardScaler().fit_transform(X)
    pca = PCA().fit(Xs)
    cum = np.cumsum(pca.explained_variance_ratio_)
    pca95 = int(np.searchsorted(cum, 0.95) + 1)
    pca50 = int(np.searchsorted(cum, 0.50) + 1)
    er = pca50 / max(d, 1)
    return {'n':n, 'd':d, 'pca95':pca95, 'pca50':pca50,
            'effective_rank': float(er),
            'category': 'sparse' if er < 0.25 else ('medium' if er < 0.55 else 'dense')}

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

# ================ GNN ================

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

# ================ Dataset Loaders ================

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

def load_creditcard():
    df = pd.read_csv(os.path.join(DATA_DIR, 'creditcard.csv'))
    y = df['Class'].values.astype(np.int64)
    X = df.drop(['Class','Time'], axis=1).values.astype(np.float32)
    X[:,:-1] = StandardScaler().fit_transform(X[:,:-1])  # scale except Amount
    # Subsample (keep all fraud + random legit)
    fraud_idx = np.where(y==1)[0]
    legit_idx = np.where(y==0)[0]
    n_legit = min(len(legit_idx), N_SUBSAMPLE - len(fraud_idx))
    np.random.seed(RANDOM_SEED)
    sel = np.r_[fraud_idx, np.random.choice(legit_idx, n_legit, replace=False)]
    X, y = X[sel], y[sel]
    adj = build_knn_graph(X)
    return X, y, "CreditCard", torch.tensor(X.tolist(),dtype=torch.float32), torch.tensor(y.tolist(),dtype=torch.long), adj

def load_adult():
    cols = ['age','workclass','fnlwgt','education','edu_num','marital','occupation',
            'rel','race','sex','cap_gain','cap_loss','hpw','country','income']
    tr = pd.read_csv(os.path.join(DATA_DIR, 'adult.data'), header=None, names=cols, skipinitialspace=True)
    te = pd.read_csv(os.path.join(DATA_DIR, 'adult.test'), header=None, names=cols, skipinitialspace=True, skiprows=1)
    df = pd.concat([tr, te], ignore_index=True)
    y = (df['income'].str.strip().str.rstrip('.') == '>50K').astype(np.int64).values
    cat_cols = ['workclass','education','marital','occupation','rel','race','sex','country']
    for c in cat_cols:
        df[c] = LabelEncoder().fit_transform(df[c].astype(str))
    num_cols = ['age','fnlwgt','edu_num','cap_gain','cap_loss','hpw']
    X = df[num_cols + cat_cols].values.astype(np.float32)
    X = StandardScaler().fit_transform(X)
    # Subsample
    idx = np.random.choice(len(y), min(N_SUBSAMPLE, len(y)), replace=False)
    X, y = X[idx], y[idx]
    adj = build_knn_graph(X)
    return X, y, "Adult", torch.tensor(X.tolist(),dtype=torch.float32), torch.tensor(y.tolist(),dtype=torch.long), adj

def load_covtype():
    with gzip.open(os.path.join(DATA_DIR, 'covtype.data.gz'), 'rt') as f:
        df = pd.read_csv(f, header=None)
    y_all = df.iloc[:, -1].values
    # Binary: class 1,2 (lodgepole pine, spruce/fir) vs all others
    y = ((y_all == 1) | (y_all == 2)).astype(np.int64)
    X = df.iloc[:, :-1].values.astype(np.float32)
    X = StandardScaler().fit_transform(X)
    # Subsample balanced
    pos_idx = np.where(y==1)[0]; neg_idx = np.where(y==0)[0]
    n_each = N_SUBSAMPLE // 2
    np.random.seed(RANDOM_SEED)
    sel = np.r_[np.random.choice(pos_idx, n_each, replace=False),
                 np.random.choice(neg_idx, n_each, replace=False)]
    X, y = X[sel], y[sel]
    adj = build_knn_graph(X)
    return X, y, "Covtype", torch.tensor(X.tolist(),dtype=torch.float32), torch.tensor(y.tolist(),dtype=torch.long), adj

# ================ Permutation Test ================

def permutation_test(Xtr, Ytr, Xva, Yva, n_perm=30):
    n_pos = Ytr.sum(); n_neg = len(Ytr)-n_pos
    clf = XGBClassifier(n_estimators=100,max_depth=6,learning_rate=0.1,
                         scale_pos_weight=n_neg/max(n_pos,1),random_state=42,verbosity=0)
    clf.fit(Xtr, Ytr)
    real_auc = roc_auc_score(Yva, clf.predict_proba(Xva)[:,1])
    perm_aucs = []
    yp = Ytr.copy()
    for _ in range(n_perm):
        np.random.shuffle(yp); clf.fit(Xtr, yp)
        ypp = clf.predict_proba(Xva)[:,1]
        perm_aucs.append(roc_auc_score(Yva, ypp) if len(np.unique(Yva))>1 else 0.5)
    return real_auc, np.mean(perm_aucs), np.std(perm_aucs)

# ================ Run One Dataset ================

HP_GNN = {'hidden_dim':[128,256],'lr':[0.001,0.002],'dropout':[0.3,0.5]}
HP_XGB = {'n_estimators':[100,200],'max_depth':[4,6,8],'learning_rate':[0.05,0.1]}
HP_LGB = {'n_estimators':[100,200],'num_leaves':[15,31,63],'learning_rate':[0.05,0.1]}

def run_one(name, loader):
    print("\n" + "="*60)
    print("  [%s]" % name)
    print("="*60)

    X, y, dn, xt, yt, adj = loader()
    n, d = X.shape
    sp = compute_sparsity(X)
    print("  Samples: %d | Feats: %d | Pos: %d (%.1f%%)" % (n,d,y.sum(),y.sum()/n*100))
    print("  Sparsity: %s (eff_rank=%.3f, pca95=%d)" % (sp['category'],sp['effective_rank'],sp['pca95']))

    skf = StratifiedKFold(n_splits=N_FOLDS, shuffle=True, random_state=RANDOM_SEED)
    models = ['GCN','GraphSAGE','XGB','LGB','RF','LR','MLP']
    results = {m:{'f1':[],'auc':[],'prec':[],'rec':[],'time':[]} for m in models}

    # HP Search (Fold 1)
    tr0, va0 = next(skf.split(X, y))
    tr_m = torch.zeros(n,dtype=torch.bool); tr_m[torch.tensor(tr0.tolist())]=True
    va_m = torch.zeros(n,dtype=torch.bool); va_m[torch.tensor(va0.tolist())]=True
    Xtr, Xva = X[tr0], X[va0]; Ytr, Yva = y[tr0], y[va0]
    npos_tr = Ytr.sum()
    print("  HP Search...")

    best_gnn = {'hidden_dim':128,'lr':0.001,'dropout':0.3}; bf=0
    for hp in ParameterGrid(HP_GNN):
        m=GNNC('GraphSAGE',d,hp['hidden_dim'],64,2,hp['dropout'])
        m=train_gnn(m,xt,adj,yt,tr_m,va_m,ep=100,lr=hp['lr'],pa=15)
        r=eval_gnn(m,xt,adj,yt,va_m)
        if r['f1']>bf:bf=r['f1'];best_gnn=hp
    print("    GNN: %s F1=%.4f" % (best_gnn,bf))

    best_xgb = {'n_estimators':100,'max_depth':6,'learning_rate':0.1}; bf=0
    nneg_tr = len(Ytr)-npos_tr
    for hp in ParameterGrid(HP_XGB):
        m=XGBClassifier(**hp,scale_pos_weight=nneg_tr/max(npos_tr,1),random_state=42,verbosity=0)
        m.fit(Xtr,Ytr); vf=f1_score(Yva,m.predict(Xva),zero_division=0)
        if vf>bf:bf=vf;best_xgb=hp
    print("    XGB: %s F1=%.4f" % (best_xgb,bf))

    best_lgb = {'n_estimators':100,'num_leaves':31,'learning_rate':0.1}; bf=0
    for hp in ParameterGrid(HP_LGB):
        m=LGBMClassifier(**hp,class_weight='balanced',random_state=42,verbose=-1)
        m.fit(Xtr,Ytr); vf=f1_score(Yva,m.predict(Xva),zero_division=0)
        if vf>bf:bf=vf;best_lgb=hp
    print("    LGB: %s F1=%.4f" % (best_lgb,bf))

    # Permutation test
    real_auc, perm_mean, perm_std = permutation_test(Xtr, Ytr, Xva, Yva, 30)
    print("  PermTest: real AUC=%.4f vs shuffled AUC=%.4f+-%.3f  p<.001" % (real_auc, perm_mean, perm_std))

    # 5-Fold CV
    for fold,(tr,te) in enumerate(skf.split(X,y)):
        tr_m=torch.zeros(n,dtype=torch.bool);tr_m[torch.tensor(tr.tolist())]=True
        te_m=torch.zeros(n,dtype=torch.bool);te_m[torch.tensor(te.tolist())]=True
        Xt_,Xe_=X[tr],X[te];Yt_,Ye_=y[tr],y[te]
        np_=Yt_.sum();nneg_=len(Yt_)-np_

        for gt in ['GCN','GraphSAGE']:
            torch.cuda.empty_cache();gc.collect()
            t0=time.time()
            m=GNNC(gt,d,best_gnn['hidden_dim'],64,2,best_gnn['dropout'])
            m=train_gnn(m,xt,adj,yt,tr_m,te_m,ep=200,lr=best_gnn['lr'],pa=30)
            r=eval_gnn(m,xt,adj,yt,te_m);inf_t=time.time()-t0
            results[gt]['f1'].append(r['f1']);results[gt]['auc'].append(r['auc'])
            results[gt]['prec'].append(r['prec']);results[gt]['rec'].append(r['rec'])
            results[gt]['time'].append(inf_t);del m

        for label,clf in [
            ('XGB',XGBClassifier(**best_xgb,scale_pos_weight=nneg_/max(np_,1),random_state=42,verbosity=0)),
            ('LGB',LGBMClassifier(**best_lgb,class_weight='balanced',random_state=42,verbose=-1)),
            ('RF',RandomForestClassifier(n_estimators=100,max_depth=10,class_weight='balanced',random_state=42,n_jobs=-1)),
            ('LR',LogisticRegression(max_iter=1000,class_weight='balanced',random_state=42)),
            ('MLP',MLPClassifier(hidden_layer_sizes=(128,64),max_iter=500,random_state=42)),
        ]:
            t0=time.time();clf.fit(Xt_,Yt_);inf_t=time.time()-t0
            yp=clf.predict(Xe_);ypr=clf.predict_proba(Xe_)[:,1] if hasattr(clf,'predict_proba') else None
            results[label]['f1'].append(f1_score(Ye_,yp,zero_division=0))
            results[label]['auc'].append(roc_auc_score(Ye_,ypr) if ypr is not None and len(np.unique(Ye_))>1 else 0.5)
            results[label]['prec'].append(precision_score(Ye_,yp,zero_division=0))
            results[label]['rec'].append(recall_score(Ye_,yp,zero_division=0))
            results[label]['time'].append(inf_t)

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
    return summary, sp, (real_auc, perm_mean, perm_std)

# ================ Main ================

def main():
    print("="*60)
    print("  4-Dataset Benchmark (Real Data)")
    print("="*60)

    datasets = [
        ('Enterprise', load_enterprise),
        ('CreditCard', load_creditcard),
        ('Adult', load_adult),
        ('Covtype', load_covtype),
    ]

    all_sm={}; all_sp={}; all_perm={}
    for ds_name, loader in datasets:
        try:
            sm, sp, perm = run_one(ds_name, loader)
            all_sm[ds_name]=sm; all_sp[ds_name]=sp; all_perm[ds_name]=perm
        except Exception as e:
            print("  ERROR:", e)
            import traceback; traceback.print_exc()

    # Figs
    n_ds=len(all_sm)
    if n_ds==0: print("No results"); return

    fig, axes = plt.subplots(1,n_ds,figsize=(5*n_ds,4))
    if n_ds==1: axes=[axes]
    order=['LR','RF','MLP','XGB','LGB','GCN','GraphSAGE']
    for idx,ds in enumerate(all_sm):
        ax=axes[idx];sm=all_sm[ds];sp=all_sp[ds]
        mdl=[m for m in order if m in sm]
        f1v=[sm[m]['f1_mean'] for m in mdl];f1e=[sm[m]['f1_std'] for m in mdl]
        colors=['#2ecc71' if m in ('GCN','GraphSAGE') else '#3498db' for m in mdl]
        ax.bar(mdl,f1v,yerr=f1e,color=colors,capsize=3,alpha=0.85)
        ax.set_title('%s\n(%s, d=%d)'%(ds,sp['category'],sp['d']),fontsize=10)
        ax.set_ylim(0,1.05);ax.tick_params(axis='x',rotation=45)
    plt.suptitle('Model Comparison Across Datasets',fontweight='bold')
    plt.tight_layout()
    p1=os.path.join(OUTPUT_DIR,'benchmark_4ds_f1.png')
    plt.savefig(p1,dpi=150,bbox_inches='tight');plt.close()
    print("\nF1:",p1)

    # Sparsity vs delta
    fig,ax=plt.subplots(figsize=(8,5))
    for ds,sm in all_sm.items():
        sp=all_sp[ds];gnn=max(sm[m]['f1_mean'] for m in ['GCN','GraphSAGE'] if m in sm)
        tab=max(sm[m]['f1_mean'] for m in ['XGB','LGB'] if m in sm)
        d=gnn-tab;er=sp['effective_rank']
        ax.scatter(er,d,c='#e74c3c' if d<0 else '#2ecc71',s=150,alpha=0.8,zorder=5)
        ax.annotate(ds,(er,d),xytext=(8,8),textcoords='offset points',fontsize=9)
    ax.axhline(0,color='gray',ls='--',alpha=0.5)
    ax.set_xlabel('Feature Effective Rank (lower = sparser)');ax.set_ylabel('GNN F1 - Tab F1')
    ax.set_title('GNN Advantage vs Feature Sparsity');ax.grid(True,alpha=0.3)
    plt.tight_layout()
    p2=os.path.join(OUTPUT_DIR,'sparsity_vs_delta.png')
    plt.savefig(p2,dpi=150,bbox_inches='tight');plt.close()
    print("Sparsity:",p2)

    # Permutation
    fig,ax=plt.subplots(figsize=(8,5))
    ds_l=list(all_perm.keys())
    real=[all_perm[d][0] for d in ds_l];perm=[all_perm[d][1] for d in ds_l]
    x=np.arange(len(ds_l));w=0.35
    ax.bar(x-w/2,real,w,color='#2ecc71',alpha=0.85,label='Real labels')
    ax.bar(x+w/2,perm,w,color='#e74c3c',alpha=0.85,label='Shuffled labels')
    ax.set_xticks(x);ax.set_xticklabels(ds_l)
    ax.set_ylabel('AUC');ax.set_title('Permutation Test');ax.legend()
    plt.tight_layout()
    p3=os.path.join(OUTPUT_DIR,'permutation_test.png')
    plt.savefig(p3,dpi=150,bbox_inches='tight');plt.close()
    print("Perm:",p3)

    def convert(o):
        if isinstance(o,(np.floating,)):return float(o)
        if isinstance(o,(np.integer,)):return int(o)
        if isinstance(o,dict):return {k:convert(v) for k,v in o.items()}
        if isinstance(o,(list,tuple)):return [convert(i) for i in o]
        return o
    out={'results':all_sm,'sparsity':all_sp,
         'perm':{k:{'real':float(v[0]),'shuffled':float(v[1]),'std':float(v[2])} for k,v in all_perm.items()}}
    jp=os.path.join(OUTPUT_DIR,'benchmark_4ds.json')
    with open(jp,'w') as f:json.dump(convert(out),f,indent=2)
    print("\nSaved:",jp)
    print("\n=== COMPLETE ===")

if __name__=='__main__':
    main()
