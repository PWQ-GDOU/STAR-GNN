"""
graph_builder.py — 星型模式 → 异构图网络

核心创新: 将 8 张星型表建模为异构图
- 同构图: 基于企业间共享变更代码的 Jaccard 相似度
- 二分图: 企业-税目关联
- 文本节点: 新闻嵌入拼接到企业特征

输出: DGL/PyG 图对象 + 节点特征矩阵
"""
import numpy as np, pandas as pd
import os, sys, pickle, warnings
from collections import defaultdict
from scipy.sparse import coo_matrix, csr_matrix
from sklearn.preprocessing import StandardScaler
from sklearn.feature_extraction.text import TfidfVectorizer
import torch

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from config import *

warnings.filterwarnings('ignore')


class StarSchemaGraphBuilder:
    """星型模式图构建器"""

    def __init__(self, data_dir: str = RAW_DATA_DIR):
        self.data_dir = data_dir
        self.dfs = {}
        self.enterprise_ids = None          # 企业节点 ID 映射
        self.id_to_idx = {}                  # enterprise_id → node index
        self.idx_to_id = {}                  # node index → enterprise_id

        # 图数据
        self.adj_matrix = None               # 邻接矩阵 (N x N, 同构图)
        self.node_features = None            # 节点特征矩阵 (N x F)
        self.edge_index = None               # PyTorch Geometric 格式 [2, E]
        self.edge_weight = None              # 边权重 [E]
        self.labels = None                   # 标签 [N]

        # 二分图 (企业-税目)
        self.tax_adj = None                  # [N, T]
        self.tax_items = None                # 税目名称列表
        self.tax_features = None             # 税目特征 [T, F_tax]

        # 文本特征
        self.text_embeddings = None          # 新闻嵌入 [N, D_text]

    # ================================================================
    #  Step 1: 加载数据
    # ================================================================

    def load_data(self) -> dict:
        """加载 8 张星型表"""
        print("[GraphBuilder] 加载数据...")
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
                print(f"  ✓ {name}: {self.dfs[name].shape}")
            else:
                print(f"  ✗ {name}: 文件不存在 ({path})")

        # 建立企业节点索引
        self.enterprise_ids = sorted(self.dfs['entprise_info']['id'].unique())
        self.id_to_idx = {eid: i for i, eid in enumerate(self.enterprise_ids)}
        self.idx_to_id = {i: eid for eid, i in self.id_to_idx.items()}
        N = len(self.enterprise_ids)
        print(f"  → 企业节点数: {N}")

        # 标签
        self.labels = np.zeros(N, dtype=np.int64)
        label_map = dict(zip(
            self.dfs['entprise_info']['id'],
            self.dfs['entprise_info']['label']
        ))
        for i, eid in enumerate(self.enterprise_ids):
            if eid in label_map:
                self.labels[i] = int(label_map[eid])
        n_pos = self.labels.sum()
        print(f"  → 风险企业: {n_pos} ({n_pos/N*100:.1f}%), 正常: {N-n_pos}")
        return self.dfs

    # ================================================================
    #  Step 2: 构建同构图（企业变更网络）
    # ================================================================

    def build_change_graph(self,
                           similarity_threshold: float = None,
                           max_edges_per_node: int = None,
                           min_co_changes: int = None) -> "StarSchemaGraphBuilder":
        """
        基于 change_info 构建企业-企业图
        边权重 = Jaccard(企业A的bgxmdm集, 企业B的bgxmdm集)
        """
        threshold = similarity_threshold or GRAPH_CONFIG['change_similarity_threshold']
        max_edges = max_edges_per_node or GRAPH_CONFIG['change_max_edges_per_node']
        min_co = min_co_changes or GRAPH_CONFIG['change_min_co_changes']

        print(f"\n[GraphBuilder] 构建企业变更网络图")
        print(f"  Jaccard阈值={threshold}, 每节点最大边数={max_edges}, 最少共享变更={min_co}")

        df = self.dfs.get('change_info')
        if df is None:
            print("  ⚠ change_info 不存在，跳过")
            return self

        # 每个企业 → bgxmdm 集合
        eid_to_changes = defaultdict(set)
        for _, row in df.iterrows():
            eid = row.get('id')
            bgxmdm = str(row.get('bgxmdm', '')).strip()
            if eid in self.id_to_idx and bgxmdm:
                eid_to_changes[eid].add(bgxmdm)

        N = len(self.enterprise_ids)
        change_sets = [eid_to_changes.get(eid, set()) for eid in self.enterprise_ids]

        # 构建稀疏边 (KNN 思路: 每节点保留 top-k 相似邻居)
        sources, targets, weights = [], [], []

        for i in range(N):
            if i % 2000 == 0 and i > 0:
                print(f"    处理中... {i}/{N}")

            set_i = change_sets[i]
            if len(set_i) < min_co:
                continue

            # 计算与所有节点的 Jaccard (可优化为 LSH, 但 15k 还行)
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

            # 保留 top-k
            sims.sort(key=lambda x: x[1], reverse=True)
            for j, w in sims[:max_edges]:
                sources.append(i)
                targets.append(j)
                weights.append(w)

        num_edges = len(sources)
        print(f"  → 边数: {num_edges}")
        print(f"  → 图密度: {num_edges / (N*N):.6f}")
        print(f"  → 平均度: {num_edges / N:.1f}")

        # 转为 PyTorch Geometric 格式
        self.edge_index = torch.tensor([sources, targets], dtype=torch.long)
        self.edge_weight = torch.tensor(weights, dtype=torch.float32)

        # 同时保存稀疏邻接矩阵（备用）
        data = np.array(weights, dtype=np.float32)
        self.adj_matrix = csr_matrix((data, (sources, targets)), shape=(N, N))

        return self

    # ================================================================
    #  Step 3: 构建二分图（企业-税目）
    # ================================================================

    def build_tax_bipartite(self, min_tax_count: int = None) -> "StarSchemaGraphBuilder":
        """企业-税目二分图"""
        min_count = min_tax_count or GRAPH_CONFIG['tax_min_count']
        print(f"\n[GraphBuilder] 构建企业-税目二分图 (min_count={min_count})")

        df = self.dfs.get('tax_info')
        if df is None:
            print("  ⚠ tax_info 不存在，跳过")
            return self

        # 获取税目频率
        tax_col = None
        for c in ['TAX_ITEMS', 'tax_items', 'TAXITEM']:
            if c in df.columns:
                tax_col = c
                break
        if tax_col is None:
            print("  ⚠ 找不到税目列名，跳过")
            return self

        tax_freq = df[tax_col].value_counts()
        self.tax_items = tax_freq[tax_freq >= min_count].index.tolist()
        T = len(self.tax_items)
        tax_to_idx = {t: i for i, t in enumerate(self.tax_items)}
        N = len(self.enterprise_ids)

        # 构建二分邻接矩阵 [N, T]
        self.tax_adj = np.zeros((N, T), dtype=np.float32)
        for _, row in df.iterrows():
            eid = row.get('id')
            tax = row.get(tax_col)
            if eid in self.id_to_idx and tax in tax_to_idx:
                self.tax_adj[self.id_to_idx[eid], tax_to_idx[tax]] = 1.0

        # 税目节点特征: 该税目关联的企业的风险比例
        self.tax_features = np.zeros((T, 4), dtype=np.float32)
        for t_idx in range(T):
            enterprises_with_tax = self.tax_adj[:, t_idx] > 0
            if enterprises_with_tax.sum() > 0:
                risk_ratio = self.labels[enterprises_with_tax].mean()
                self.tax_features[t_idx, 0] = risk_ratio
                self.tax_features[t_idx, 1] = enterprises_with_tax.sum()
                self.tax_features[t_idx, 2] = enterprises_with_tax.sum() / N
                self.tax_features[t_idx, 3] = np.log1p(enterprises_with_tax.sum())

        print(f"  → 税目数: {T}, 边数: {int(self.tax_adj.sum())}")
        return self

    # ================================================================
    #  Step 4: 构建节点特征（表格属性）
    # ================================================================

    def build_node_features(self) -> "StarSchemaGraphBuilder":
        """从 base_info + annual_report_info 构建企业节点特征"""
        print(f"\n[GraphBuilder] 构建节点特征...")
        N = len(self.enterprise_ids)
        features = []

        # ---- base_info 数值特征 ----
        base = self.dfs.get('base_info')
        if base is not None:
            base = base.set_index('id', drop=False)
            feat_cols = []
            for col in ['regcap', 'reccap', 'empnum']:
                if col in base.columns:
                    val = pd.to_numeric(base[col], errors='coerce').fillna(0)
                    feat_cols.append(col)
            if feat_cols:
                for eid in self.enterprise_ids:
                    if eid in base.index:
                        row = base.loc[eid]
                        f = [np.log1p(pd.to_numeric(row[c], errors='coerce') or 0) for c in feat_cols]
                        features.append(f)
                    else:
                        features.append([0.0] * len(feat_cols))
                print(f"  → base_info: {len(feat_cols)} 维")

        # ---- annual_report_info 聚合特征 ----
        annual = self.dfs.get('annual_report_info')
        annual_feats = []
        if annual is not None and 'id' in annual.columns:
            for eid in self.enterprise_ids:
                sub = annual[annual['id'] == eid]
                if len(sub) > 0:
                    # 年均营收/利润, 报告数量
                    for num_col in ['business_scope', 'total_equity', 'total_liability',
                                   'total_assets', 'total_profit', 'prime_operating_revenue']:
                        if num_col in sub.columns:
                            vals = pd.to_numeric(sub[num_col], errors='coerce')
                            annual_feats.append(np.nanmean(vals) if len(vals.dropna()) > 0 else 0.0)
                    annual_feats.append(len(sub))
                else:
                    annual_feats.extend([0.0] * 7)  # 6个数值列 + 1计数
            # 只取实际添加的维度
            n_annual = len(annual_feats) // N if len(annual_feats) >= N else 0

        # ---- 拼接 ----
        if features:
            base_arr = np.array(features, dtype=np.float32)
        else:
            base_arr = np.zeros((N, 0), dtype=np.float32)

        if annual_feats:
            annual_arr = np.array(annual_feats, dtype=np.float32).reshape(N, -1)
            self.node_features = np.hstack([base_arr, annual_arr])
        else:
            self.node_features = base_arr

        # 标准化
        scaler = StandardScaler()
        self.node_features = scaler.fit_transform(self.node_features)

        # 转为 tensor
        self.node_features = torch.tensor(self.node_features, dtype=torch.float32)
        print(f"  → 节点特征维度: {self.node_features.shape[1]}")
        return self

    # ================================================================
    #  Step 5: 文本嵌入（可选，耗时长）
    # ================================================================

    def build_text_embeddings(self, use_cache: bool = True) -> "StarSchemaGraphBuilder":
        """新闻文本 → TF-IDF / BERT 嵌入"""
        cache_path = os.path.join(OUTPUT_DIR, "text_embeddings.npy")
        if use_cache and os.path.exists(cache_path):
            self.text_embeddings = np.load(cache_path)
            print(f"  → 从缓存加载文本嵌入: {self.text_embeddings.shape}")
            return self

        news = self.dfs.get('news_info')
        if news is None:
            print("  ⚠ news_info 不存在，跳过")
            return self

        print(f"\n[GraphBuilder] 构建文本嵌入...")
        # 聚合每个企业的所有新闻文本
        text_col = None
        for c in ['content', 'text', 'NEWS_CONTENT']:
            if c in news.columns:
                text_col = c
                break
        if text_col is None:
            print("  ⚠ 找不到新闻文本列，使用 TF-IDF...")
            # 使用 TF-IDF 作为轻量嵌入
            eid_to_text = defaultdict(list)
            for _, row in news.iterrows():
                eid = row.get('id')
                txt = str(row.iloc[1]) if len(row) > 1 else ""
                if eid in self.id_to_idx and txt:
                    eid_to_text[eid].append(txt)

            N = len(self.enterprise_ids)
            corpus = [' '.join(eid_to_text.get(eid, [''])) for eid in self.enterprise_ids]
            tfidf = TfidfVectorizer(max_features=128, stop_words=None)
            self.text_embeddings = tfidf.fit_transform(corpus).toarray().astype(np.float32)
            print(f"  → TF-IDF 嵌入维度: {self.text_embeddings.shape[1]}")

        np.save(cache_path, self.text_embeddings)
        return self

    # ================================================================
    #  Step 6: 保存图数据
    # ================================================================

    def save(self, path: str = None):
        """保存图数据到文件"""
        p = path or os.path.join(OUTPUT_DIR, "graph_data.pkl")
        data = {
            'num_nodes': len(self.enterprise_ids),
            'num_edges': self.edge_index.shape[1] if self.edge_index is not None else 0,
            'num_node_features': self.node_features.shape[1] if self.node_features is not None else 0,
            'num_classes': 2,
            'enterprise_ids': self.enterprise_ids,
            'id_to_idx': self.id_to_idx,
            'idx_to_id': self.idx_to_id,
            'edge_index': self.edge_index,
            'edge_weight': self.edge_weight,
            'node_features': self.node_features,
            'labels': torch.tensor(self.labels, dtype=torch.long),
            'adj_matrix': self.adj_matrix,
            'tax_adj': self.tax_adj,
            'tax_items': self.tax_items,
            'tax_features': self.tax_features,
            'text_embeddings': self.text_embeddings,
        }
        torch.save(data, p)
        print(f"\n[GraphBuilder] 图数据已保存: {p}")
        print(f"  节点: {data['num_nodes']}, 边: {data['num_edges']}, 特征: {data['num_node_features']}")
        return p

    def run(self, data_dir: str = None) -> str:
        """一键运行: 加载→建图→特征→保存"""
        if data_dir:
            self.data_dir = data_dir
        self.load_data()
        self.build_node_features()
        self.build_change_graph()
        self.build_tax_bipartite()
        self.build_text_embeddings(use_cache=False)
        return self.save()


# ================================================================
#  便捷函数
# ================================================================

def build_and_save_graph(data_dir: str = RAW_DATA_DIR, output_path: str = None):
    """一键构建并保存图"""
    builder = StarSchemaGraphBuilder(data_dir)
    return builder.run(data_dir)


def load_graph(path: str = None):
    """加载已保存的图数据"""
    p = path or os.path.join(OUTPUT_DIR, "graph_data.pkl")
    if os.path.exists(p):
        data = torch.load(p, weights_only=False)
        print(f"[GraphBuilder] 加载图数据: {p}")
        print(f"  节点: {data['num_nodes']}, 边: {data['num_edges']}")
        return data
    else:
        print(f"  ⚠ 图数据不存在 ({p}), 请先运行 build_and_save_graph()")
        return None


if __name__ == "__main__":
    build_and_save_graph()
