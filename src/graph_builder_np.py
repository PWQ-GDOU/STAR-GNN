"""
graph_builder_np.py — 纯 numpy 版，无需 torch
星型模式 → 异构图网络构建
"""
import numpy as np, pandas as pd
import os, sys, pickle, warnings
from collections import defaultdict
from scipy.sparse import csr_matrix
from sklearn.preprocessing import StandardScaler
from sklearn.feature_extraction.text import TfidfVectorizer

warnings.filterwarnings('ignore')

# 路径配置：自动搜索数据目录（避免中文路径编码问题）
def _find_data_dir():
    import glob
    for p in glob.iglob(r'D:\cxdownload\**\base_info.csv', recursive=True):
        d = os.path.dirname(p)
        if os.path.exists(os.path.join(d, 'entprise_info.csv')):
            return d
    raise FileNotFoundError("Cannot find data directory with base_info.csv + entprise_info.csv")

DATA_DIR = _find_data_dir()
OUTPUT_DIR = r"D:\cxdownload\大数据实训\code_sci\results"
os.makedirs(OUTPUT_DIR, exist_ok=True)


class StarSchemaGraphBuilder:
    def __init__(self, data_dir=DATA_DIR):
        self.data_dir = data_dir
        self.dfs = {}
        self.enterprise_ids = None
        self.id_to_idx = {}
        self.idx_to_id = {}
        self.adj_matrix = None
        self.node_features = None
        self.edge_index = None
        self.edge_weight = None
        self.labels = None
        self.tax_adj = None
        self.tax_items = None
        self.text_embeddings = None

    def load_data(self):
        print("[GraphBuilder-NP] Loading data...")
        tbl_files = {
            'base_info': 'base_info.csv',
            'annual_report_info': 'annual_report_info.csv',
            'change_info': 'change_info.csv',
            'news_info': 'news_info.csv',
            'tax_info': 'tax_info.csv',
            'other_info': 'other_info.csv',
            'entprise_info': 'entprise_info.csv',
            'entprise_evaluate': 'entprise_evaluate.csv',
        }
        for name, fname in tbl_files.items():
            path = os.path.join(self.data_dir, fname)
            if os.path.exists(path):
                self.dfs[name] = pd.read_csv(path)
                print(f"  OK {name}: {self.dfs[name].shape}")
            else:
                print(f"  MISS {name}")

        self.enterprise_ids = sorted(self.dfs['entprise_info']['id'].unique())
        self.id_to_idx = {eid: i for i, eid in enumerate(self.enterprise_ids)}
        self.idx_to_id = {i: eid for eid, i in self.id_to_idx.items()}
        N = len(self.enterprise_ids)
        print(f"  Nodes: {N}")

        self.labels = np.zeros(N, dtype=np.int64)
        label_map = dict(zip(
            self.dfs['entprise_info']['id'],
            self.dfs['entprise_info']['label']
        ))
        for i, eid in enumerate(self.enterprise_ids):
            if eid in label_map:
                self.labels[i] = int(label_map[eid])
        n_pos = self.labels.sum()
        print(f"  Positive: {n_pos} ({n_pos/N*100:.1f}%), Negative: {N-n_pos}")
        return self

    def build_node_features(self):
        print("\n[GraphBuilder-NP] Building node features...")
        N = len(self.enterprise_ids)
        all_feats = []

        # base_info numeric
        base = self.dfs.get('base_info')
        base_cols = []
        if base is not None:
            base = base.set_index('id')
            for col in ['regcap', 'reccap', 'empnum']:
                if col in base.columns:
                    base_cols.append(col)
            if base_cols:
                bf = np.zeros((N, len(base_cols)), dtype=np.float32)
                for i, eid in enumerate(self.enterprise_ids):
                    if eid in base.index:
                        for j, col in enumerate(base_cols):
                            val = pd.to_numeric(base.loc[eid, col], errors='coerce')
                            bf[i, j] = np.log1p(val) if pd.notna(val) and val > 0 else 0.0
                all_feats.append(bf)
                print(f"  base_info: {len(base_cols)} dims")

        # annual_report_info aggregation
        annual = self.dfs.get('annual_report_info')
        if annual is not None and 'id' in annual.columns:
            annual_num_cols = []
            for nc in ['business_scope', 'total_equity', 'total_liability',
                       'total_assets', 'total_profit', 'prime_operating_revenue']:
                if nc in annual.columns:
                    annual_num_cols.append(nc)
            n_annual_cols = len(annual_num_cols) + 1  # +1 for report count
            af = np.zeros((N, n_annual_cols), dtype=np.float32)
            for i, eid in enumerate(self.enterprise_ids):
                sub = annual[annual['id'] == eid]
                if len(sub) > 0:
                    for j, nc in enumerate(annual_num_cols):
                        vals = pd.to_numeric(sub[nc], errors='coerce').dropna()
                        af[i, j] = vals.mean() if len(vals) > 0 else 0.0
                    af[i, -1] = len(sub)
            all_feats.append(af)
            print(f"  annual: {n_annual_cols} dims")

        # build full matrix
        if all_feats:
            X = np.hstack(all_feats)
        else:
            X = np.zeros((N, 1))

        scaler = StandardScaler()
        X = scaler.fit_transform(X)
        self.node_features = X.astype(np.float32)
        print(f"  Total features: {self.node_features.shape[1]}")
        return self

    def build_change_graph(self, threshold=0.3, max_edges=50, min_co=2):
        print(f"\n[GraphBuilder-NP] Building change graph (Jaccard>={threshold}, max_edges={max_edges})")
        df = self.dfs.get('change_info')
        if df is None:
            print("  SKIP: no change_info")
            return self

        eid_to_changes = defaultdict(set)
        for _, row in df.iterrows():
            eid = row.get('id')
            bgxmdm = str(row.get('bgxmdm', '')).strip()
            if eid in self.id_to_idx and bgxmdm:
                eid_to_changes[eid].add(bgxmdm)

        N = len(self.enterprise_ids)
        change_sets = [eid_to_changes.get(eid, set()) for eid in self.enterprise_ids]

        sources, targets, weights = [], [], []
        for i in range(N):
            if i % 3000 == 0:
                print(f"    progress {i}/{N}...")
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
                sources.append(i)
                targets.append(j)
                weights.append(w)

        num_edges = len(sources)
        print(f"  Edges: {num_edges}, density: {num_edges/(N*N):.6f}, avg_degree: {num_edges/N:.1f}")
        self.edge_index = np.array([sources, targets], dtype=np.int64)
        self.edge_weight = np.array(weights, dtype=np.float32)
        self.adj_matrix = csr_matrix((weights, (sources, targets)), shape=(N, N))
        return self

    def build_tax_bipartite(self, min_count=5):
        print(f"\n[GraphBuilder-NP] Building tax bipartite (min_count={min_count})")
        df = self.dfs.get('tax_info')
        if df is None:
            print("  SKIP: no tax_info")
            return self

        tax_col = None
        for c in ['TAX_ITEMS', 'tax_items', 'TAXITEM']:
            if c in df.columns:
                tax_col = c
                break
        if tax_col is None:
            print("  SKIP: tax column not found")
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

        print(f"  Tax items: {T}, edges: {int(self.tax_adj.sum())}")
        return self

    def build_text_embeddings(self):
        print("\n[GraphBuilder-NP] Building text embeddings...")
        news = self.dfs.get('news_info')
        if news is None:
            print("  SKIP: no news_info")
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
        print(f"  Text embedding dims: {self.text_embeddings.shape[1]}")
        return self

    def save(self, path=None):
        p = path or os.path.join(OUTPUT_DIR, "graph_data_np.pkl")
        data = {
            'num_nodes': len(self.enterprise_ids),
            'num_edges': self.edge_index.shape[1] if self.edge_index is not None else 0,
            'num_node_features': self.node_features.shape[1],
            'enterprise_ids': self.enterprise_ids,
            'id_to_idx': self.id_to_idx,
            'idx_to_id': self.idx_to_id,
            'edge_index': self.edge_index,
            'edge_weight': self.edge_weight,
            'node_features': self.node_features,
            'labels': self.labels,
            'adj_matrix': self.adj_matrix,
            'tax_adj': self.tax_adj,
            'tax_items': self.tax_items,
            'text_embeddings': self.text_embeddings,
        }
        with open(p, 'wb') as f:
            pickle.dump(data, f, protocol=5)
        fs = os.path.getsize(p) / 1024 / 1024
        print(f"\n[DONE] Saved to {p} ({fs:.1f} MB)")
        self._summary(data)
        return p

    def _summary(self, data):
        print(f"\n{'='*50}")
        print(f"  GRAPH SUMMARY")
        print(f"{'='*50}")
        print(f"  Nodes:       {data['num_nodes']:,}")
        print(f"  Edges:       {data['num_edges']:,}")
        print(f"  Avg degree:  {data['num_edges']/data['num_nodes']:.1f}")
        print(f"  Features:    {data['num_node_features']}")
        print(f"  Pos/neg:     {data['labels'].sum():,} / {len(data['labels'])-data['labels'].sum():,}")
        print(f"  Imbalance:   {data['labels'].sum()/len(data['labels'])*100:.1f}%")
        if data['edge_weight'] is not None and len(data['edge_weight']) > 0:
            print(f"  Edge weight: min={data['edge_weight'].min():.3f}, "
                  f"max={data['edge_weight'].max():.3f}, "
                  f"mean={data['edge_weight'].mean():.3f}")
        if data['tax_adj'] is not None:
            print(f"  Tax items:   {data['tax_adj'].shape[1]}")
            print(f"  Tax edges:   {int(data['tax_adj'].sum()):,}")
        print(f"{'='*50}")

    def run(self, with_text=True):
        self.load_data()
        self.build_node_features()
        self.build_change_graph()
        self.build_tax_bipartite()
        if with_text:
            self.build_text_embeddings()
        return self.save()


if __name__ == "__main__":
    builder = StarSchemaGraphBuilder()
    builder.run()
