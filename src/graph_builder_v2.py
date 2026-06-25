"""
graph_builder_v2.py — 集成 features.py 94维特征 + 图构建
纯 numpy, 无需 torch
"""
import numpy as np, pandas as pd
import os, sys, pickle, warnings
from collections import defaultdict
from scipy.sparse import csr_matrix
from sklearn.preprocessing import StandardScaler, LabelEncoder
from sklearn.feature_extraction.text import TfidfVectorizer

warnings.filterwarnings('ignore')

# ── Portable paths: use env vars, fall back to repo-relative ──
_REPO_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))  # STAR-GNN/

# Data directory: first check STAR_GNN_DATA, then default to <repo>/data/
DATA_DIR = os.environ.get("STAR_GNN_DATA", os.path.join(_REPO_ROOT, "data"))
if not os.path.isdir(DATA_DIR):
    # Last-resort fallback for legacy Windows setups (auto-detect CSV dir)
    found = False
    for p in glob.iglob(os.path.join(_REPO_ROOT, '**', 'base_info.csv'), recursive=True):
        d = os.path.dirname(p)
        if os.path.exists(os.path.join(d, 'entprise_info.csv')):
            DATA_DIR = d
            found = True
            break
    if not found:
        msg = (
            f"ERROR: Data directory not found.\n"
            f"  Tried: STAR_GNN_DATA env var (not set)\n"
            f"  Tried: {os.path.join(_REPO_ROOT, 'data')} (does not exist)\n"
            f"  Tried: recursive search for base_info.csv under {_REPO_ROOT} (not found)\n\n"
            f"Please set STAR_GNN_DATA to your data directory:\n"
            f"  Linux/Mac:  export STAR_GNN_DATA=/path/to/data\n"
            f"  Windows:    set STAR_GNN_DATA=C:\\path\\to\\data\n"
            f"Or place CSV files in {os.path.join(_REPO_ROOT, 'data')}/\n"
        )
        raise FileNotFoundError(msg)

OUTPUT_DIR = os.environ.get("STAR_GNN_RESULTS", os.path.join(_REPO_ROOT, "results"))
os.makedirs(OUTPUT_DIR, exist_ok=True)
print(f"[paths] DATA_DIR={DATA_DIR}")
print(f"[paths] OUTPUT_DIR={OUTPUT_DIR}")

# ============ 从 features.py 复制特征工程（自包含，不依赖外部 import）============

BGXMDM_MAP = {
    101:"经营范围变更", 102:"章程备案", 103:"住所变更",
    104:"法定代表人变更", 105:"注册资本变更", 106:"营业期限变更",
    107:"名称变更", 108:"经营场所变更", 109:"投资人变更",
    110:"董事备案", 111:"监事备案", 112:"经理备案",
    113:"高级管理人员备案", 114:"分公司备案", 115:"清算组备案",
    116:"负责人变更", 117:"出资额变更", 118:"出资方式变更",
    119:"出资时间变更", 120:"出资比例变更", 121:"营业期限变更",
    122:"投资人出资额变更", 123:"实收资本变更", 124:"企业类型变更",
    125:"经营期限变更", 126:"分支机构变更", 127:"联络员备案",
    128:"财务负责人备案", 129:"撤销分支机构", 130:"增补证照",
    131:"补发证照", 132:"换发证照", 133:"撤销变更登记",
    134:"注销备案", 135:"迁移变更", 136:"股权变更",
    137:"股东变更", 138:"股东出资变更", 139:"外资变更",
    140:"中外合资变更", 141:"外资转内资", 142:"内资转外资",
    143:"改制变更", 144:"合并变更", 145:"分立变更",
}
HIGH_RISK_CHANGES = {109,110,111,112,136,137,138,143,144,145,104,105,113}


def build_base_features(df_base):
    base = df_base.copy()
    base['regcap_log']   = np.log1p(pd.to_numeric(base['regcap'], errors='coerce').fillna(0))
    base['reccap_log']   = np.log1p(pd.to_numeric(base['reccap'], errors='coerce').fillna(0))
    base['reccap_ratio'] = base['reccap_log'] / (base['regcap_log'] + 1e-6)
    base['reccap_ratio'] = base['reccap_ratio'].clip(0, 5)
    base['fake_capital_flag'] = (
        (pd.to_numeric(base['regcap'], errors='coerce').fillna(0) > 1000) &
        (pd.to_numeric(base['reccap'], errors='coerce').fillna(0) == 0)
    ).astype(int)
    base['opfrom_year'] = pd.to_datetime(base['opfrom'], errors='coerce').dt.year
    base['opto_year']   = pd.to_datetime(base['opto'], errors='coerce').dt.year.fillna(2021)
    base['biz_years']   = (base['opto_year'] - base['opfrom_year']).clip(0, 50)
    base['short_life']  = (base['biz_years'] < 3).astype(int)
    base['empnum'] = pd.to_numeric(base['empnum'], errors='coerce')
    for col in ['state', 'enttype', 'enttypegb', 'oplocdistrict', 'regtype', 'compform', 'opform']:
        if col in base.columns:
            base[col+'_enc'] = LabelEncoder().fit_transform(base[col].astype(str))
    for col in ['enttypeitem', 'enttypeminu']:
        if col in base.columns:
            base[col+'_enc'] = LabelEncoder().fit_transform(
                pd.to_numeric(base[col], errors='coerce').fillna(-1).astype(int).astype(str))
    base['industryco_num'] = pd.to_numeric(base['industryco'], errors='coerce')
    base['industryco_group'] = (base['industryco_num'] // 100).fillna(-1).astype(int)
    base['industryco_group_enc'] = LabelEncoder().fit_transform(base['industryco_group'].astype(str))
    freq = base['industryco_num'].value_counts(normalize=True)
    base['industryco_freq'] = base['industryco_num'].map(freq).fillna(0)
    if 'industryphy' in base.columns:
        base['industryphy_enc'] = LabelEncoder().fit_transform(base['industryphy'].fillna('UNKNOWN').astype(str))
    base['venind'] = pd.to_numeric(base['venind'], errors='coerce').fillna(0)
    base['opscope_len'] = base['opscope'].fillna('').str.len()
    base['dom_len']     = base['dom'].fillna('').str.len()
    for col in ['townsign', 'adbusign']:
        if col in base.columns:
            base[col] = pd.to_numeric(base[col], errors='coerce').fillna(0)
    base['regcap_per_emp'] = base['regcap_log'] / (base['empnum'].fillna(1) + 1)
    base['reccap_deviation'] = abs(base['regcap_log'] - base['reccap_log'])
    keep = ['id','regcap_log','reccap_log','reccap_ratio','fake_capital_flag',
        'biz_years','short_life','empnum',
        'state_enc','enttype_enc','enttypegb_enc','oplocdistrict_enc',
        'regtype_enc','compform_enc','opform_enc',
        'enttypeitem_enc','enttypeminu_enc',
        'industryco_group_enc','industryco_freq','industryphy_enc',
        'venind','townsign','adbusign',
        'opscope_len','dom_len',
        'regcap_per_emp','reccap_deviation']
    return base[[c for c in keep if c in base.columns]]


def build_annual_features(df_annual):
    df = df_annual.copy()
    num_cols = ['FUNDAM','EMPNUM','COLGRANUM','RETSOLNUM','DISPERNUM','UNENUM',
                'COLEMPLNUM','RETEMPLNUM','DISEMPLNUM','UNEEMPLNUM']
    for c in num_cols:
        if c in df.columns:
            df[c] = pd.to_numeric(df[c], errors='coerce')

    # Simple aggregations by hand — avoid mixed agg types
    results = []
    # count & max of year
    results.append(df.groupby('id')['ANCHEYEAR'].agg(n_annual='count', last_annual_yr='max').reset_index())

    # numeric columns
    num_aggs = {}
    for c in ['FUNDAM','EMPNUM']:
        if c in df.columns:
            num_aggs[c+'_mean'] = (c, 'mean')
            num_aggs[c+'_std'] = (c, 'std')
    if 'COLEMPLNUM' in df.columns:
        num_aggs['colemp_mean'] = ('COLEMPLNUM', 'mean')
        num_aggs['colemp_max'] = ('COLEMPLNUM', 'max')
    if 'RETEMPLNUM' in df.columns:
        num_aggs['retemp_mean'] = ('RETEMPLNUM', 'mean')
    if 'DISEMPLNUM' in df.columns:
        num_aggs['disemp_mean'] = ('DISEMPLNUM', 'mean')
    if 'PUBSTATE' in df.columns:
        num_aggs['pubstate_mean'] = ('PUBSTATE', 'mean')
    if num_aggs:
        results.append(df.groupby('id').agg(**num_aggs).reset_index())

    # boolean flags (max=1 if any)
    for c in ['WEBSITSIGN','FORINVESTSIGN','STOCKTRANSIGN']:
        if c in df.columns:
            flag_name = 'has_' + c.lower().replace('sign','')
            r = df.groupby('id')[c].max().reset_index()
            r.columns = ['id', flag_name]
            results.append(r)

    # Merge all
    agg = results[0]
    for r in results[1:]:
        agg = agg.merge(r, on='id', how='left')
    return agg


def build_change_features(df_change):
    df = df_change.copy()
    df['bgrq'] = pd.to_datetime(df['bgrq'], format='%Y%m%d%H%M%S', errors='coerce')
    df['bgxmdm'] = pd.to_numeric(df['bgxmdm'], errors='coerce')
    df['is_high_risk_change'] = df['bgxmdm'].isin(HIGH_RISK_CHANGES).astype(int)

    agg = df.groupby('id').agg(
        n_changes=('bgxmdm','count'),
        n_change_types=('bgxmdm','nunique'),
        n_high_risk_changes=('is_high_risk_change','sum'),
        high_risk_ratio=('is_high_risk_change','mean'),
        first_change=('bgrq','min'),
        last_change=('bgrq','max'),
    ).reset_index()
    agg['change_span'] = (agg['last_change'] - agg['first_change']).dt.days.clip(lower=1)
    agg['change_freq'] = agg['n_changes'] / agg['change_span']
    for code in [111,118,119,120,113,930]:
        name = BGXMDM_MAP.get(code, 'code_%d'%code)
        sub = df[df['bgxmdm']==code].groupby('id').size().reset_index(name='chg_'+name)
        agg = agg.merge(sub, on='id', how='left')
        agg['chg_'+name] = agg['chg_'+name].fillna(0)
    return agg


def build_news_features(df_news):
    df = df_news.copy()
    agg = df.groupby('id').agg(
        n_news=('id','count'),
        n_negative=('positive_negtive', lambda x: (x=='消极').sum()),
        n_positive=('positive_negtive', lambda x: (x=='积极').sum()),
        n_neutral=('positive_negtive', lambda x: (x=='中立').sum()),
    ).reset_index()
    agg['neg_ratio'] = agg['n_negative'] / (agg['n_news'] + 1)
    agg['pos_ratio'] = agg['n_positive'] / (agg['n_news'] + 1)
    return agg


def build_tax_features(df_tax):
    df = df_tax.copy()
    for c in ['TAX_AMOUNT','TAX_RATE','TAXATION_BASIS','DEDUCTION']:
        if c in df.columns:
            df[c] = pd.to_numeric(df[c], errors='coerce')
    agg = df.groupby('id').agg(
        n_tax_records=('TAX_AMOUNT','count') if 'TAX_AMOUNT' in df.columns else ('id','count'),
    ).reset_index()
    if 'TAX_CATEGORIES' in df.columns:
        agg['n_tax_categories'] = df.groupby('id')['TAX_CATEGORIES'].nunique().values
    if 'TAX_ITEMS' in df.columns:
        agg['n_tax_items'] = df.groupby('id')['TAX_ITEMS'].nunique().values
    if 'TAX_AMOUNT' in df.columns:
        grouped = df.groupby('id')['TAX_AMOUNT']
        agg['tax_sum'] = grouped.sum().values
        agg['tax_mean'] = grouped.mean().values
        agg['tax_std'] = grouped.std().fillna(0).values
        agg['tax_max'] = grouped.max().values
    if 'TAX_RATE' in df.columns:
        agg['tax_rate_mean'] = df.groupby('id')['TAX_RATE'].mean().values
    agg['tax_normal'] = (agg.get('tax_sum', 0) > 0).astype(int)
    return agg


def build_other_features(df_other):
    df = df_other.copy()
    for col in ['legal_judgment_num','brand_num','patent_num']:
        if col in df.columns:
            df['has_'+col.split('_')[0]] = df[col].notna().astype(int)
    return df


# ============ 图构建器 ============

class StarSchemaGraphBuilderV2:
    def __init__(self, data_dir=DATA_DIR):
        self.data_dir = data_dir
        self.dfs = {}
        self.enterprise_ids = None
        self.id_to_idx = {}
        self.idx_to_id = {}
        self.edge_index = None
        self.edge_weight = None
        self.node_features = None
        self.feature_names = None
        self.labels = None
        self.tax_adj = None
        self.tax_items = None
        self.text_embeddings = None

    def load_data(self):
        print("[V2] Loading data...")
        files = ['base_info','annual_report_info','change_info','news_info',
                 'tax_info','other_info','entprise_info','entprise_evaluate']
        for name in files:
            path = os.path.join(self.data_dir, name+'.csv')
            if os.path.exists(path):
                self.dfs[name] = pd.read_csv(path)
                print("  OK %-25s %s" % (name, str(self.dfs[name].shape)))
            else:
                print("  MISS %s" % name)

        self.enterprise_ids = sorted(self.dfs['entprise_info']['id'].unique())
        self.id_to_idx = {eid: i for i, eid in enumerate(self.enterprise_ids)}
        self.idx_to_id = {i: eid for eid, i in self.id_to_idx.items()}
        N = len(self.enterprise_ids)
        print("  Nodes: %d" % N)

        self.labels = np.zeros(N, dtype=np.int64)
        label_map = dict(zip(self.dfs['entprise_info']['id'],
                             self.dfs['entprise_info']['label']))
        for i, eid in enumerate(self.enterprise_ids):
            if eid in label_map:
                self.labels[i] = int(label_map[eid])
        n_pos = self.labels.sum()
        print("  Positive: %d (%.1f%%), Negative: %d" % (n_pos, n_pos/N*100, N-n_pos))
        return self

    def build_node_features(self):
        print("\n[V2] Building node features (leakage-mitigated: base+annual+news only)...")
        df_label = self.dfs['entprise_info'][['id','label']].copy()

        # Build per-table features
        base_f   = build_base_features(self.dfs['base_info'])
        annual_f = build_annual_features(self.dfs['annual_report_info'])
        change_f = build_change_features(self.dfs['change_info'])
        news_f   = build_news_features(self.dfs['news_info'])
        tax_f    = build_tax_features(self.dfs['tax_info'])
        other_f  = build_other_features(self.dfs['other_info'])

        # Merge to label table
        data = df_label
        for name, feats in [('base', base_f), ('annual', annual_f),
                              ('change', change_f), ('news', news_f),
                              ('tax', tax_f), ('other', other_f)]:
            data = data.merge(feats, on='id', how='left', suffixes=('','_'+name))

        # Source flags
        data['has_annual'] = (~data.filter(like='annual').iloc[:,0].isna()).astype(int) if 'annual' in str(data.columns) else 0
        for src in ['annual','change','news','tax','other']:
            cols_with_src = [c for c in data.columns if c.startswith('n_'+src)]
            if cols_with_src:
                data['has_'+src] = data[cols_with_src[0]].notna().astype(int)
            else:
                data['has_'+src] = 0
        data['n_sources'] = data[['has_annual','has_change','has_news','has_tax','has_other']].sum(axis=1)

        # Map to node order
        id_to_row = dict(zip(data['id'], range(len(data))))
        N = len(self.enterprise_ids)

        # Drop id, label, and non-numeric columns (datetime, object)
        data_num = data.select_dtypes(include=[np.number])
        feat_cols = [c for c in data_num.columns if c not in ('id','label')]
        self.feature_names = feat_cols
        n_feats = len(feat_cols)
        print("  Numeric feature columns: %d" % n_feats)

        X = np.zeros((N, n_feats), dtype=np.float32)
        for i, eid in enumerate(self.enterprise_ids):
            if eid in id_to_row:
                row = data_num.iloc[id_to_row[eid]]
                vals = row[feat_cols].values
                vals = np.nan_to_num(vals.astype(np.float64), nan=0.0)
                X[i] = vals

        # Standardize
        from sklearn.preprocessing import StandardScaler
        # === Label Leakage Mitigation ===
        # ⚠️  WARNING: Jaccard graph is DEPRECATED for final experiments.           ⚠️
        # ⚠️  The Jaccard change-code graph encodes label signal (high-risk          ⚠️
        # ⚠️  enterprises share more high-risk change codes → label leakage).        ⚠️
        # ⚠️  benchmark_clean.py REBUILDS the graph via k-NN cosine similarity.      ⚠️
        # ⚠️  This file is kept ONLY for the 57-dim feature extraction pipeline.     ⚠️
        # change_info is used for BOTH graph topology (Jaccard edges)
        # and node features (n_high_risk_changes, etc.).
        # To prevent data leakage via graph structure, we exclude
        # change-derived and tax-derived features from node features,
        # keeping only base_info + annual_report_info + news as attributes.
        change_patterns = ['change', 'chg_', 'high_risk', 'n_change', 'chg']
        tax_patterns = ['tax_', 'n_tax', 'tax_sum', 'tax_mean', 'tax_std', 'tax_max',
                        'tax_rate', 'tax_cv', 'tax_normal', 'tax_item']
        exclude = []
        for name in self.feature_names:
            if any(p in name.lower() for p in change_patterns + tax_patterns):
                exclude.append(True)
            else:
                exclude.append(False)
        keep_idx = [i for i, e in enumerate(exclude) if not e]
        X = X[:, keep_idx]
        self.feature_names = [self.feature_names[i] for i in keep_idx]
        n_removed = len(exclude) - len(keep_idx)
        print('  [Leakage fix] Removed %d change/tax features, keeping %d (base+annual+news only)'
              % (n_removed, len(keep_idx)))

        mask = np.std(X, axis=0) > 1e-8
        X[:, mask] = StandardScaler().fit_transform(X[:, mask])
        X[:, ~mask] = 0.0
        self.node_features = X.astype(np.float32)
        print("  Features: %d dims" % n_feats)
        return self

    def build_change_graph(self, threshold=0.3, max_edges=50, min_co=2):
        print("\n[V2] Building change graph (Jaccard>=%.2f, max_edges=%d, min_co=%d)" % (threshold, max_edges, min_co))
        df = self.dfs.get('change_info')
        if df is None:
            print("  SKIP: no change_info")
            return self

        eid_to_changes = defaultdict(set)
        for _, row in df.iterrows():
            eid = row.get('id')
            bgxmdm = str(row.get('bgxmdm','')).strip()
            if eid in self.id_to_idx and bgxmdm:
                eid_to_changes[eid].add(bgxmdm)

        N = len(self.enterprise_ids)
        change_sets = [eid_to_changes.get(eid, set()) for eid in self.enterprise_ids]

        sources, targets, weights = [], [], []
        for i in range(N):
            if i % 3000 == 0:
                print("    progress %d/%d..." % (i, N))
            set_i = change_sets[i]
            if len(set_i) < min_co:
                continue
            sims = []
            for j in range(N):
                if i == j:
                    continue
                set_j = change_sets[j]
                if len(set_j) < min_co:
                    continue
                inter = len(set_i & set_j)
                union = len(set_i | set_j)
                if union == 0:
                    continue
                jaccard = inter / union
                if jaccard >= threshold:
                    sims.append((j, jaccard))
            sims.sort(key=lambda x: x[1], reverse=True)
            for j, w in sims[:max_edges]:
                sources.append(i); targets.append(j); weights.append(w)

        num_edges = len(sources)
        print("  Edges: %d, density: %.6f, avg_degree: %.1f" % (num_edges, num_edges/(N*N), num_edges/N))
        # DESIGN NOTE: Topology-Label Dependency
        # The Jaccard graph connects enterprises with similar change histories.
        # High-risk enterprises tend to share certain change patterns,
        # so the graph topology partially encodes risk signal. This is intentional
        # — the graph captures relational structure — but change-derived features
        # are excluded from node attributes to prevent double-counting.
        # See Section 5 (Discussion/Limitations) in the paper.

        self.edge_index = np.array([sources, targets], dtype=np.int64)
        self.edge_weight = np.array(weights, dtype=np.float32)
        return self

    def build_tax_bipartite(self, min_count=5):
        print("\n[V2] Building tax bipartite (min_count=%d)" % min_count)
        df = self.dfs.get('tax_info')
        if df is None:
            return self

        tax_col = None
        for c in ['TAX_ITEMS','tax_items','TAXITEM']:
            if c in df.columns:
                tax_col = c; break
        if tax_col is None:
            return self

        tax_freq = df[tax_col].value_counts()
        self.tax_items = tax_freq[tax_freq >= min_count].index.tolist()
        T = len(self.tax_items)
        tax_to_idx = {t: i for i, t in enumerate(self.tax_items)}
        N = len(self.enterprise_ids)

        self.tax_adj = np.zeros((N, T), dtype=np.float32)
        for _, row in df.iterrows():
            eid = row.get('id')
            tax = row.get(tax_col)
            if eid in self.id_to_idx and tax in tax_to_idx:
                self.tax_adj[self.id_to_idx[eid], tax_to_idx[tax]] = 1.0
        print("  Tax items: %d, edges: %d" % (T, int(self.tax_adj.sum())))
        return self

    def build_text_embeddings(self):
        print("\n[V2] Building text embeddings...")
        news = self.dfs.get('news_info')
        if news is None:
            return self

        eid_to_text = defaultdict(list)
        for _, row in news.iterrows():
            eid = row.get('id')
            txt = ' '.join(str(v) for v in row.values if isinstance(v, str))
            if eid in self.id_to_idx and txt:
                eid_to_text[eid].append(txt)

        N = len(self.enterprise_ids)
        corpus = [' '.join(eid_to_text.get(eid, [''])) for eid in self.enterprise_ids]
        tfidf = TfidfVectorizer(max_features=128, stop_words=None)
        self.text_embeddings = tfidf.fit_transform(corpus).toarray().astype(np.float32)
        print("  Text dims: %d" % self.text_embeddings.shape[1])
        return self

    def save(self, path=None):
        p = path or os.path.join(OUTPUT_DIR, "graph_data_v2.pkl")
        data = {
            'num_nodes': len(self.enterprise_ids),
            'num_edges': self.edge_index.shape[1] if self.edge_index is not None else 0,
            'num_node_features': self.node_features.shape[1],
            'feature_names': self.feature_names,
            'enterprise_ids': self.enterprise_ids,
            'id_to_idx': self.id_to_idx,
            'idx_to_id': self.idx_to_id,
            'edge_index': self.edge_index,
            'edge_weight': self.edge_weight,
            'node_features': self.node_features,
            'labels': self.labels,
            'tax_adj': self.tax_adj,
            'tax_items': self.tax_items,
            'text_embeddings': self.text_embeddings,
        }
        with open(p, 'wb') as f:
            pickle.dump(data, f, protocol=5)
        fs_mb = os.path.getsize(p) / 1024 / 1024
        print("\n[DONE] Saved: %s (%.1f MB)" % (p, fs_mb))
        self._summary(data)
        return p

    def _summary(self, d):
        s = ["="*55,
             "  GRAPH SUMMARY (v2 - leakage-mitigated features)",
             "="*55,
             "  Nodes:       %d" % d['num_nodes'],
             "  Edges:       %d" % d['num_edges'],
             "  Avg degree:  %.1f" % (d['num_edges']/max(d['num_nodes'],1)),
             "  Features:    %d dims" % d['num_node_features'],
             "  Pos/Neg:     %d / %d (%.1f%%)" % (d['labels'].sum(), len(d['labels'])-d['labels'].sum(), d['labels'].sum()/len(d['labels'])*100),
             "="*55]
        print('\n'.join(s))

    def run(self):
        self.load_data()
        self.build_node_features()
        self.build_change_graph()
        self.build_tax_bipartite()
        self.build_text_embeddings()
        return self.save()


if __name__ == "__main__":
    builder = StarSchemaGraphBuilderV2()
    builder.run()
