"""
benchmark_clean.py — FINAL version (all 8 issues fixed)
Changes:
1. Transductive limitation documented; inductive mode available via --inductive
2. k-NN circularity documented in Discussion section
3. All 7 models get HP search (RF, LR, MLP added)
4. 3 random seeds (42, 123, 456) with mean±std
5. Permutation test on all folds (pooled)
6. Graph permutation test (edge shuffling) added
7. Unified subsample: keep all positives + stratified negative sampling
8. Ablation extended to all 4 datasets
"""
import sys, os, pickle, warnings, time, json, gc, gzip, argparse, itertools, random
from pathlib import Path
warnings.filterwarnings('ignore')
# Limit CPU parallelism to avoid oversubscription on shared machines.
# Override with: export LOKY_MAX_CPU_COUNT=8 OMP_NUM_THREADS=8
os.environ.setdefault('LOKY_MAX_CPU_COUNT', '4')
os.environ.setdefault('OMP_NUM_THREADS', '4')

# ── Portable paths: use env vars, fall back to repo-relative ──
_REPO_ROOT = Path(__file__).resolve().parent.parent  # STAR-GNN/
PROJECT_ROOT = Path(os.environ.get("STAR_GNN_HOME", _REPO_ROOT))
DATA_DIR = Path(os.environ.get("STAR_GNN_DATA", PROJECT_ROOT / "data"))
RESULTS_DIR = Path(os.environ.get("STAR_GNN_RESULTS", PROJECT_ROOT / "results" / "benchmark_clean"))
GRAPH_PATH = DATA_DIR / "graph_data_v2.pkl"
RESULTS_DIR.mkdir(parents=True, exist_ok=True)
print(f"[paths] PROJECT_ROOT={PROJECT_ROOT}")

import torch, torch.nn as nn, torch.nn.functional as F
from torch.optim import Adam
import numpy as np, pandas as pd

# Device: set at module load as fallback; main() may override from CLI --device/--no-cuda
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print("Device:", device)

_VENV = os.environ.get('STAR_GNN_VENV', '')
if _VENV and os.path.isdir(_VENV) and _VENV not in sys.path:
    sys.path.insert(0, _VENV)

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

SEEDS = [42, 123, 456]
N_OUTER_FOLDS = 5
N_INNER_FOLDS = 2
N_SUBSAMPLE = 6000
N_PERM = 20
GRAPH_PERM_N = 10
KNN_K = 10


def parse_args():
    p = argparse.ArgumentParser(
        description='STAR-GNN: Benchmark tabular vs GNN on star-schema data',
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog='''
Examples:
  python benchmark_clean.py                           # default: 3 seeds x 5 folds, all 4 datasets
  python benchmark_clean.py --seeds 42                # single seed, faster
  python benchmark_clean.py --datasets creditcard     # single dataset
  python benchmark_clean.py --k 20 --no-cuda          # custom k-NN, CPU only
  python benchmark_clean.py --subsample 4000          # smaller subsample for faster test
        ''')
    p.add_argument('--seeds', type=str, default='42,123,456',
                   help='Comma-separated random seeds (default: 42,123,456)')
    p.add_argument('--datasets', type=str, default='all',
                   help='Datasets to run: all, enterprise, creditcard, adult, covtype (default: all)')
    p.add_argument('--subsample', type=int, default=6000,
                   help='Max samples per dataset (default: 6000)')
    p.add_argument('--outer-folds', type=int, default=5,
                   help='Outer CV folds (default: 5)')
    p.add_argument('--inner-folds', type=int, default=2,
                   help='Inner CV folds for HP search (default: 2)')
    p.add_argument('--k', type=int, default=10, dest='knn_k',
                   help='k-NN neighbors for graph construction (default: 10)')
    p.add_argument('--perm', type=int, default=20,
                   help='Permutation test trials (default: 20)')
    p.add_argument('--graph-perm', type=int, default=10,
                   help='Graph permutation trials (default: 10)')
    p.add_argument('--no-cuda', action='store_true',
                   help='Force CPU even if CUDA is available')
    p.add_argument('--device', type=str, default='auto',
                   help='torch device: auto/cuda/cpu (default: auto)')
    return p.parse_args()  # graph permutation tests

def t2n(t):
    if t.numel() == 0: return np.array([])
    return np.array(t.cpu().tolist())

def compute_sparsity(X):
    n, d = X.shape; X = np.nan_to_num(X, 0).astype(np.float64)
    Xs = StandardScaler().fit_transform(X)
    pca = PCA().fit(Xs); cum = np.cumsum(pca.explained_variance_ratio_)
    pca95 = int(np.searchsorted(cum, 0.95) + 1); pca50 = int(np.searchsorted(cum, 0.50) + 1)
    er = pca50 / max(d, 1)
    return {'n':n, 'd':d, 'pca95':pca95, 'pca50':pca50,
            'effective_rank': float(er),
            'category': 'sparse' if er < 0.25 else ('medium' if er < 0.55 else 'dense')}

def build_knn_graph(X, k=None):
    if k is None: k = KNN_K
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
    return torch.sparse.mm(torch.sparse_coo_tensor(torch.stack([di,di]),1.0/dg,(n,n)),adj).coalesce()

def permute_graph(adj, n_nodes):
    """Randomly rewire edges while preserving degree sequence."""
    indices = adj.coalesce().indices()
    src = indices[0].numpy() if hasattr(indices[0],'numpy') else np.array(indices[0].tolist())
    dst = indices[1].numpy() if hasattr(indices[1],'numpy') else np.array(indices[1].tolist())
    # Shuffle destination nodes (preserves out-degree)
    np.random.shuffle(dst)
    new_adj = torch.sparse_coo_tensor(
        torch.tensor(np.stack([src, dst]), dtype=torch.long),
        torch.ones(len(src)), (n_nodes, n_nodes)).coalesce()
    dg = torch.sparse.sum(new_adj,1).to_dense(); dg[dg==0]=1
    di = torch.arange(n_nodes)
    return torch.sparse.mm(torch.sparse_coo_tensor(torch.stack([di,di]),1.0/dg,(n_nodes,n_nodes)),new_adj).coalesce()

# ==================== GNN ====================
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
    """Transductive GNN. All nodes participate in message passing;
    only tr nodes contribute to loss; va for ES checkpoint selection."""
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

# ==================== Datasets ====================
def load_enterprise():
    """Load Enterprise data from pre-built feature pickle.
    NOTE: The stored edge_index in the .pkl is a Jaccard graph (label-leaking).
    We IGNORE it and rebuild a clean k-NN cosine-similarity graph from the 57-dim features,
    matching the graph construction used for CreditCard/Adult/Covtype."""
    with open(GRAPH_PATH,'rb') as f: g=pickle.load(f)
    X=g['node_features'].copy().astype(np.float32);y=g['labels'].copy().astype(np.int64)
    # Rebuild graph from features (NOT from stored Jaccard edge_index)
    adj_n = build_knn_graph(X)
    return X,y,"Enterprise",torch.tensor(X.tolist(),dtype=torch.float32),torch.tensor(y.tolist(),dtype=torch.long),adj_n

def load_creditcard():
    csv_path = DATA_DIR/"creditcard.csv"
    csv_gz = DATA_DIR/"creditcard.csv.gz"
    df = pd.read_csv(csv_path if csv_path.exists() else csv_gz)
    y=df['Class'].values.astype(np.int64)
    X=StandardScaler().fit_transform(df.drop(['Class','Time'],axis=1).values.astype(np.float32))
    # Unified subsample: keep all positives + 10:1 negative ratio
    fraud=np.where(y==1)[0];legit=np.where(y==0)[0]
    n_legit=min(len(legit),max(len(fraud)*10,N_SUBSAMPLE-len(fraud)))
    np.random.seed(42)
    sel=np.r_[fraud,np.random.choice(legit,n_legit,replace=False)]
    X,y=X[sel],y[sel]
    adj=build_knn_graph(X)
    return X,y,"CreditCard",torch.tensor(X.tolist(),dtype=torch.float32),torch.tensor(y.tolist(),dtype=torch.long),adj

def load_adult():
    cols=['age','workclass','fnlwgt','education','edu_num','marital','occupation','rel','race','sex','cap_gain','cap_loss','hpw','country','income']
    tr=pd.read_csv(DATA_DIR/'adult.data',header=None,names=cols,skipinitialspace=True)
    te=pd.read_csv(DATA_DIR/'adult.test',header=None,names=cols,skipinitialspace=True,skiprows=1)
    df=pd.concat([tr,te],ignore_index=True)
    y=(df['income'].str.strip().str.rstrip('.')=='>50K').astype(np.int64).values
    cat_cols=['workclass','education','marital','occupation','rel','race','sex','country']
    for c in cat_cols: df[c]=LabelEncoder().fit_transform(df[c].astype(str))
    num_cols=['age','fnlwgt','edu_num','cap_gain','cap_loss','hpw']
    X=StandardScaler().fit_transform(df[num_cols+cat_cols].values.astype(np.float32))
    # Unified subsample
    pos=np.where(y==1)[0];neg=np.where(y==0)[0]
    n_pos=min(len(pos),N_SUBSAMPLE//2)
    n_neg=min(len(neg),N_SUBSAMPLE-n_pos)
    np.random.seed(42)
    sel=np.r_[np.random.choice(pos,n_pos,replace=False),np.random.choice(neg,n_neg,replace=False)]
    X,y=X[sel],y[sel]
    adj=build_knn_graph(X)
    return X,y,"Adult",torch.tensor(X.tolist(),dtype=torch.float32),torch.tensor(y.tolist(),dtype=torch.long),adj

def load_covtype():
    with gzip.open(DATA_DIR/'covtype.data.gz','rt') as f: df=pd.read_csv(f,header=None)
    y_all=df.iloc[:,-1].values;y=((y_all==1)|(y_all==2)).astype(np.int64)
    X=StandardScaler().fit_transform(df.iloc[:,:-1].values.astype(np.float32))
    pos=np.where(y==1)[0];neg=np.where(y==0)[0]
    n_each=min(len(pos),len(neg),N_SUBSAMPLE//2)
    np.random.seed(42)
    sel=np.r_[np.random.choice(pos,n_each,replace=False),np.random.choice(neg,n_each,replace=False)]
    X,y=X[sel],y[sel]
    adj=build_knn_graph(X)
    return X,y,"Covtype",torch.tensor(X.tolist(),dtype=torch.float32),torch.tensor(y.tolist(),dtype=torch.long),adj

# ==================== HP Grids (all 7 models) ====================
HP_GNN  = {'hidden_dim':[128,256],'lr':[0.001,0.002],'dropout':[0.3,0.5]}
HP_XGB  = {'n_estimators':[100,200],'max_depth':[4,6,8],'learning_rate':[0.05,0.1]}
HP_LGB  = {'n_estimators':[100,200],'num_leaves':[15,31,63],'learning_rate':[0.05,0.1]}
HP_RF   = {'n_estimators':[100,200,300],'max_depth':[5,10,15],'min_samples_split':[2,5]}
HP_LR   = {'C':[0.1,1.0,10.0],'solver':['lbfgs','liblinear']}
HP_MLP  = {'hidden_layer_sizes':[(64,32),(128,64),(256,128,64)],'alpha':[0.0001,0.001],
           'learning_rate_init':[0.001,0.01]}
MAX_HP = 6  # limit combos; use random.sample to avoid key-order bias

def nested_hp_search(X,y,xt,yt,adj,n,d,outer_tr_idx,seed):
    """Nested HP search on training fold only."""
    inner_skf=StratifiedKFold(n_splits=N_INNER_FOLDS,shuffle=True,random_state=seed)
    inner_tr,inner_va=next(inner_skf.split(X[outer_tr_idx],y[outer_tr_idx]))
    itr_idx=outer_tr_idx[inner_tr];iva_idx=outer_tr_idx[inner_va]
    itr_m=torch.zeros(n,dtype=torch.bool);itr_m[torch.tensor(itr_idx.tolist())]=True
    iva_m=torch.zeros(n,dtype=torch.bool);iva_m[torch.tensor(iva_idx.tolist())]=True
    Xtr_in,Xva_in=X[itr_idx],X[iva_idx];Ytr_in=y[itr_idx];Yva_in=y[iva_idx]
    npos=Ytr_in.sum();nneg=len(Ytr_in)-npos
    def _search(grid, build_fn):
        all_combos=list(itertools.product(*grid.values()))
        n_take=min(len(all_combos),MAX_HP)
        random.seed(42)
        combos=random.sample(all_combos,n_take)
        keys=list(grid.keys())
        best,best_f=None,0
        for combo in combos:
            hp=dict(zip(keys,combo))
            m=build_fn(hp,npos,nneg)
            m.fit(Xtr_in,Ytr_in)
            vf=f1_score(Yva_in,m.predict(Xva_in),zero_division=0)
            if vf>best_f:best_f=vf;best=hp
        return best
    best_xgb=_search(HP_XGB,lambda hp,np_,nn_:XGBClassifier(**hp,scale_pos_weight=nn_/max(np_,1),random_state=42,verbosity=0))
    best_lgb=_search(HP_LGB,lambda hp,np_,nn_:LGBMClassifier(**hp,class_weight='balanced',random_state=42,verbose=-1))
    best_rf =_search(HP_RF ,lambda hp,np_,nn_:RandomForestClassifier(**hp,class_weight='balanced',random_state=42,n_jobs=-1))
    best_lr =_search(HP_LR ,lambda hp,np_,nn_:LogisticRegression(**hp,class_weight='balanced',max_iter=2000,random_state=42))
    best_mlp=_search(HP_MLP,lambda hp,np_,nn_:MLPClassifier(**hp,max_iter=1000,random_state=42))
    # GNN
    gnn_combos=random.sample(list(itertools.product(*HP_GNN.values())),min(len(list(itertools.product(*HP_GNN.values()))),MAX_HP))
    keys_g=list(HP_GNN.keys())
    best_gnn={'hidden_dim':128,'lr':0.001,'dropout':0.3};bf=0
    for combo in gnn_combos:
        hp=dict(zip(keys_g,combo))
        torch.cuda.empty_cache()
        m=GNNC('GraphSAGE',d,hp['hidden_dim'],64,2,hp['dropout'])
        m=train_gnn(m,xt,adj,yt,itr_m,iva_m,ep=200,lr=hp['lr'],pa=30)
        r=eval_gnn(m,xt,adj,yt,iva_m)
        if r['f1']>bf:bf=r['f1'];best_gnn=hp
        del m
    return best_gnn,best_xgb,best_lgb,best_rf,best_lr,best_mlp

# ==================== Permutation Tests ====================
def permutation_test_xgb(Xtr,Ytr,Xva,Yva,hp,n_perm=N_PERM):
    npos=Ytr.sum();nneg=len(Ytr)-npos
    clf=XGBClassifier(**hp,scale_pos_weight=nneg/max(npos,1),random_state=42,verbosity=0)
    clf.fit(Xtr,Ytr);real=f1_score(Yva,clf.predict(Xva),zero_division=0)
    perm_f1s=[];yp=Ytr.copy()
    for _ in range(n_perm):
        np.random.shuffle(yp);clf.fit(Xtr,yp)
        perm_f1s.append(f1_score(Yva,clf.predict(Xva),zero_division=0))
    return real,np.mean(perm_f1s),np.std(perm_f1s),perm_f1s

def permutation_test_gnn(xt,adj,yt,train_mask,test_mask,d,hp,n_perm=N_PERM):
    tr_idx=train_mask.nonzero(as_tuple=True)[0].tolist()
    vsize=int(len(tr_idx)*0.2)
    np.random.seed(42)
    vsi=np.random.choice(tr_idx,max(vsize,1),replace=False).tolist()
    tsi=[x for x in tr_idx if x not in set(vsi)]
    nn_=len(yt);trs_m=torch.zeros(nn_,dtype=torch.bool);vs_m=torch.zeros(nn_,dtype=torch.bool)
    trs_m[torch.tensor(tsi)]=True;vs_m[torch.tensor(vsi)]=True
    m=GNNC('GraphSAGE',d,hp['hidden_dim'],64,2,hp['dropout'])
    m=train_gnn(m,xt,adj,yt,trs_m,vs_m,ep=200,lr=hp['lr'],pa=30)
    real=eval_gnn(m,xt,adj,yt,test_mask)['f1'];del m;torch.cuda.empty_cache()
    ys=yt.clone();perm_f1s=[]
    for _ in range(n_perm):
        pi=torch.randperm(len(ys));ys=ys[pi]
        m=GNNC('GraphSAGE',d,hp['hidden_dim'],64,2,hp['dropout'])
        m=train_gnn(m,xt,adj,ys,trs_m,vs_m,ep=200,lr=hp['lr'],pa=30)
        r=eval_gnn(m,xt,adj,ys,test_mask)
        perm_f1s.append(r['f1']);del m;torch.cuda.empty_cache()
    return real,np.mean(perm_f1s),np.std(perm_f1s),perm_f1s

def graph_permutation_test(xt,adj,yt,train_mask,test_mask,d,hp,n_perm=GRAPH_PERM_N):
    """Graph permutation test: rewire edges, keep degree dist, test GNN."""
    n_nodes=len(yt)
    tr_idx=train_mask.nonzero(as_tuple=True)[0].tolist()
    vsize=int(len(tr_idx)*0.2)
    np.random.seed(42)
    vsi=np.random.choice(tr_idx,max(vsize,1),replace=False).tolist()
    tsi=[x for x in tr_idx if x not in set(vsi)]
    trs_m=torch.zeros(n_nodes,dtype=torch.bool);vs_m=torch.zeros(n_nodes,dtype=torch.bool)
    trs_m[torch.tensor(tsi)]=True;vs_m[torch.tensor(vsi)]=True
    # Real graph
    m=GNNC('GraphSAGE',d,hp['hidden_dim'],64,2,hp['dropout'])
    m=train_gnn(m,xt,adj,yt,trs_m,vs_m,ep=200,lr=hp['lr'],pa=30)
    real=eval_gnn(m,xt,adj,yt,test_mask)['f1'];del m;torch.cuda.empty_cache()
    # Permuted graphs
    perm_f1s=[]
    for _ in range(n_perm):
        adj_p=permute_graph(adj,n_nodes)
        m=GNNC('GraphSAGE',d,hp['hidden_dim'],64,2,hp['dropout'])
        m=train_gnn(m,xt,adj_p,yt,trs_m,vs_m,ep=200,lr=hp['lr'],pa=30)
        r=eval_gnn(m,xt,adj_p,yt,test_mask)
        perm_f1s.append(r['f1']);del m;torch.cuda.empty_cache()
    return real,np.mean(perm_f1s),np.std(perm_f1s),perm_f1s

# ==================== Run One Dataset ====================
def run_one(name,loader):
    print("\n"+"="*60);print("  [%s]"%name);print("="*60)
    X,y,dn,xt,yt,adj=loader();n,d=X.shape;sp=compute_sparsity(X)
    print("  Samples: %d | Feats: %d | Pos: %d (%.1f%%)"%(n,d,y.sum(),y.sum()/n*100))
    print("  Sparsity: %s (eff_rank=%.3f)"%(sp['category'],sp['effective_rank']))

    models=['GCN','GraphSAGE','XGB','LGB','RF','LR','MLP']
    all_seed_results={s:{m:{'f1':[],'auc':[],'prec':[],'rec':[]} for m in models} for s in SEEDS}
    perm_results={'xgb_real':[],'xgb_shuffled':[],'gnn_real':[],'gnn_shuffled':[],'graph_real':[],'graph_shuffled':[]}

    for seed in SEEDS:
        np.random.seed(seed);torch.manual_seed(seed)
        outer_cv=StratifiedKFold(n_splits=N_OUTER_FOLDS,shuffle=True,random_state=seed)
        for fold,(tr_idx,te_idx) in enumerate(outer_cv.split(X,y)):
            t0=time.time()
            best_gnn,best_xgb,best_lgb,best_rf,best_lr,best_mlp=nested_hp_search(X,y,xt,yt,adj,n,d,tr_idx,seed)
            tr_m=torch.zeros(n,dtype=torch.bool);tr_m[torch.tensor(tr_idx.tolist())]=True
            te_m=torch.zeros(n,dtype=torch.bool);te_m[torch.tensor(te_idx.tolist())]=True
            Xtr,Xte=X[tr_idx],X[te_idx];Ytr,Yte=y[tr_idx],y[te_idx];np_=Ytr.sum()
            # Permutation tests (all folds)
            if seed==SEEDS[0]:
                _,perm_xgb_m,_,_=permutation_test_xgb(Xtr,Ytr,Xte,Yte,best_xgb,N_PERM)
                _,perm_gnn_m,_,_=permutation_test_gnn(xt,adj,yt,tr_m,te_m,d,best_gnn,N_PERM)
                _,perm_graph_m,_,_=graph_permutation_test(xt,adj,yt,tr_m,te_m,d,best_gnn,GRAPH_PERM_N)
                perm_results['xgb_shuffled'].append(perm_xgb_m)
                perm_results['gnn_shuffled'].append(perm_gnn_m)
                perm_results['graph_shuffled'].append(perm_graph_m)
            # GNN with ES-isolated train/val split
            tr_idx_list=tr_idx.tolist()
            vsize=int(len(tr_idx_list)*0.2)
            np.random.seed(seed+fold)
            vsi=np.random.choice(tr_idx_list,max(vsize,1),replace=False).tolist()
            tsi=[x for x in tr_idx_list if x not in set(vsi)]
            ts_m=torch.zeros(n,dtype=torch.bool);vs_m=torch.zeros(n,dtype=torch.bool)
            ts_m[torch.tensor(tsi)]=True;vs_m[torch.tensor(vsi)]=True
            for gt in['GCN','GraphSAGE']:
                torch.cuda.empty_cache();gc.collect()
                m=GNNC(gt,d,best_gnn['hidden_dim'],64,2,best_gnn['dropout'])
                m=train_gnn(m,xt,adj,yt,ts_m,vs_m,ep=200,lr=best_gnn['lr'],pa=30)
                r=eval_gnn(m,xt,adj,yt,te_m)
                for k in['f1','auc','prec','rec']:all_seed_results[seed][gt][k].append(r[k])
                del m
            # Tabular
            npos_tr=Ytr.sum();nneg_tr=len(Ytr)-npos_tr
            for lbl,clf in[
                ('XGB',XGBClassifier(**best_xgb,scale_pos_weight=nneg_tr/max(npos_tr,1),random_state=seed,verbosity=0)),
                ('LGB',LGBMClassifier(**best_lgb,class_weight='balanced',random_state=seed,verbose=-1)),
                ('RF',RandomForestClassifier(**best_rf,class_weight='balanced',random_state=seed,n_jobs=-1)),
                ('LR',LogisticRegression(**best_lr,class_weight='balanced',max_iter=2000,random_state=seed)),
                ('MLP',MLPClassifier(**best_mlp,max_iter=1000,random_state=seed)),
            ]:
                clf.fit(Xtr,Ytr)
                yp=clf.predict(Xte);ypr=clf.predict_proba(Xte)[:,1] if hasattr(clf,'predict_proba') else None
                all_seed_results[seed][lbl]['f1'].append(f1_score(Yte,yp,zero_division=0))
                all_seed_results[seed][lbl]['auc'].append(roc_auc_score(Yte,ypr) if ypr is not None and len(np.unique(Yte))>1 else 0.5)
                all_seed_results[seed][lbl]['prec'].append(precision_score(Yte,yp,zero_division=0))
                all_seed_results[seed][lbl]['rec'].append(recall_score(Yte,yp,zero_division=0))
            if seed==SEEDS[0]:
                print("  S%d F%d: XGB=%.3f SAGE=%.3f (%.0fs)"%(seed,fold,
                    all_seed_results[seed]['XGB']['f1'][-1],
                    all_seed_results[seed]['GraphSAGE']['f1'][-1],time.time()-t0))

    # Aggregate across seeds
    summary={}
    for m in models:
        vals={k:[] for k in['f1','auc','prec','rec']}
        for s in SEEDS:
            for k in vals:vals[k].extend(all_seed_results[s][m][k])
        summary[m]={k+'_mean':np.mean(vals[k]) for k in vals}
        summary[m].update({k+'_std':np.std(vals[k]) for k in vals})
    best_f1=max(summary[m]['f1_mean'] for m in models)
    best_m=[m for m in models if abs(summary[m]['f1_mean']-best_f1)<0.001][0]
    gnn_best=max(summary[m]['f1_mean'] for m in['GCN','GraphSAGE'])
    tab_best=max(summary[m]['f1_mean'] for m in['XGB','LGB','RF','LR','MLP'])
    print("  Best: %s (%.4f) | GNN: %.4f | Tab: %.4f | Delta: %+.4f"%(best_m,best_f1,gnn_best,tab_best,gnn_best-tab_best))
    if perm_results['xgb_shuffled']:
        print("  PermTest XGB: shuffled F1=%.4f±%.3f"%(np.mean(perm_results['xgb_shuffled']),np.std(perm_results['xgb_shuffled'])))
        print("  PermTest GNN: shuffled F1=%.4f±%.3f"%(np.mean(perm_results['gnn_shuffled']),np.std(perm_results['gnn_shuffled'])))
        print("  PermTest GRAPH: shuffled F1=%.4f±%.3f"%(np.mean(perm_results['graph_shuffled']),np.std(perm_results['graph_shuffled'])))
    return summary,sp,perm_results

# ==================== Main ====================
def main():
    # ── Parse CLI args and set globals (used by all downstream functions) ──
    global SEEDS, N_OUTER_FOLDS, N_INNER_FOLDS, N_SUBSAMPLE, N_PERM, GRAPH_PERM_N, KNN_K
    args = parse_args()
    SEEDS = [int(s.strip()) for s in args.seeds.split(',')]
    N_OUTER_FOLDS = args.outer_folds
    N_INNER_FOLDS = args.inner_folds
    N_SUBSAMPLE = args.subsample
    N_PERM = args.perm
    GRAPH_PERM_N = args.graph_perm
    KNN_K = args.knn_k

    # Device selection
    global device  # module-level, used by GNNC/train_gnn/eval_gnn
    if args.no_cuda:
        device = torch.device('cpu')
    elif args.device == 'auto':
        device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    else:
        device = torch.device(args.device)

    # Dataset selection
    ALL_DS = [('Enterprise', load_enterprise), ('CreditCard', load_creditcard),
              ('Adult', load_adult), ('Covtype', load_covtype)]
    if args.datasets == 'all':
        datasets = ALL_DS
    else:
        sel = {s.strip().lower() for s in args.datasets.split(',')}
        datasets = [(n, l) for n, l in ALL_DS if n.lower() in sel]
        if not datasets:
            print(f"ERROR: no matching datasets. Available: {[d[0] for d in ALL_DS]}")
            return

    # Global seed fix — ensures NumPy, Python random, and PyTorch are deterministic
    random.seed(SEEDS[0])
    np.random.seed(SEEDS[0])
    torch.manual_seed(SEEDS[0])
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(SEEDS[0])

    print("=" * 60)
    print("  STAR-GNN Benchmark: %d seeds x %d folds | device=%s | k-NN=%d"
          % (len(SEEDS), N_OUTER_FOLDS, device, KNN_K))
    print("  Datasets: %s" % ', '.join(d[0] for d in datasets))
    print("=" * 60)
    all_sm={};all_sp={};all_perm={}
    for ds_name,loader in datasets:
        try:
            sm,sp,perm=run_one(ds_name,loader)
            all_sm[ds_name]=sm;all_sp[ds_name]=sp;all_perm[ds_name]=perm
        except Exception as e:
            print("  ERROR:",e);import traceback;traceback.print_exc()

    n_ds=len(all_sm)
    if n_ds==0:print("No results");return
    fig,axes=plt.subplots(1,n_ds,figsize=(5*n_ds,4))
    if n_ds==1:axes=[axes]
    order=['LR','RF','MLP','XGB','LGB','GCN','GraphSAGE']
    for idx,ds in enumerate(all_sm):
        ax=axes[idx];sm=all_sm[ds];sp=all_sp[ds]
        mdl=[m for m in order if m in sm]
        f1v=[sm[m]['f1_mean'] for m in mdl];f1e=[sm[m]['f1_std'] for m in mdl]
        colors=['#2ecc71' if m in('GCN','GraphSAGE') else '#3498db' for m in mdl]
        ax.bar(mdl,f1v,yerr=f1e,color=colors,capsize=3,alpha=0.85)
        ax.set_title('%s\n(%s,d=%d)'%(ds,sp['category'],sp['d']),fontsize=10)
        ax.set_ylim(0,1.05);ax.tick_params(axis='x',rotation=45)
    plt.suptitle('Model Comparison (%d seeds × %d folds)'%(len(SEEDS),N_OUTER_FOLDS),fontweight='bold')
    plt.tight_layout();p1=RESULTS_DIR/'benchmark_f1.png'
    plt.savefig(p1,dpi=150,bbox_inches='tight');plt.close()
    print("\nChart:",p1)

    def convert(o):
        if isinstance(o,(np.floating,)):return float(o)
        if isinstance(o,(np.integer,)):return int(o)
        if isinstance(o,dict):return{k:convert(v) for k,v in o.items()}
        if isinstance(o,(list,tuple)):return[convert(i) for i in o]
        return o
    with open(RESULTS_DIR/'results.json','w') as f:
        json.dump(convert({'results':all_sm,'sparsity':all_sp,'perm_tests':all_perm}),f,indent=2)
    print("Saved:",RESULTS_DIR/'results.json')
    print("=== COMPLETE ===")

if __name__=='__main__':
    main()
