"""
gnn_models.py — 图神经网络模型 (GCN / GAT / GraphSAGE / RGCN)

用于企业变更网络的节点分类: 风险(=1) vs 正常(=0)
"""
import torch, torch.nn as nn, torch.nn.functional as F
from torch_geometric.nn import GCNConv, GATConv, SAGEConv, RGCNConv
from torch_geometric.nn import global_mean_pool, global_max_pool, global_add_pool
import os, sys

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from config import *


# ================================================================
#  GCN (Graph Convolutional Network)
# ================================================================

class GCN(nn.Module):
    """两层 GCN 用于节点分类"""

    def __init__(self, in_dim: int, hidden_dim: int = None,
                 out_dim: int = 2, num_layers: int = None,
                 dropout: float = None):
        super().__init__()
        h_dim = hidden_dim or GRAPH_CONFIG['gnn_hidden_dim']
        n_layers = num_layers or GRAPH_CONFIG['gnn_num_layers']
        drop = dropout or GRAPH_CONFIG['gnn_dropout']

        self.convs = nn.ModuleList()
        self.bns = nn.ModuleList()
        self.dropout = drop

        # 第一层
        self.convs.append(GCNConv(in_dim, h_dim))
        self.bns.append(nn.BatchNorm1d(h_dim))

        # 中间层
        for _ in range(n_layers - 2):
            self.convs.append(GCNConv(h_dim, h_dim))
            self.bns.append(nn.BatchNorm1d(h_dim))

        # 分类头
        self.classifier = nn.Sequential(
            nn.Linear(h_dim, h_dim // 2),
            nn.ReLU(),
            nn.Dropout(drop),
            nn.Linear(h_dim // 2, out_dim),
        )

    def forward(self, x, edge_index, edge_weight=None, return_embed=False):
        for conv, bn in zip(self.convs, self.bns):
            x = conv(x, edge_index, edge_weight)
            x = bn(x)
            x = F.relu(x)
            x = F.dropout(x, p=self.dropout, training=self.training)

        if return_embed:
            return x
        return self.classifier(x)

    def get_embedding(self, x, edge_index, edge_weight=None):
        return self.forward(x, edge_index, edge_weight, return_embed=True)


# ================================================================
#  GAT (Graph Attention Network)
# ================================================================

class GAT(nn.Module):
    """两层 GAT 用于节点分类"""

    def __init__(self, in_dim: int, hidden_dim: int = None,
                 out_dim: int = 2, num_layers: int = None,
                 dropout: float = None, heads: int = 4):
        super().__init__()
        h_dim = hidden_dim or GRAPH_CONFIG['gnn_hidden_dim']
        n_layers = num_layers or GRAPH_CONFIG['gnn_num_layers']
        drop = dropout or GRAPH_CONFIG['gnn_dropout']
        self.dropout = drop

        self.convs = nn.ModuleList()
        self.bns = nn.ModuleList()

        # 第一层: multi-head
        self.convs.append(GATConv(in_dim, h_dim // heads, heads=heads, dropout=drop))
        self.bns.append(nn.BatchNorm1d(h_dim))

        # 第二层: single-head (或保持 multi-head)
        if n_layers >= 2:
            self.convs.append(GATConv(h_dim, h_dim // heads, heads=heads, dropout=drop))
            self.bns.append(nn.BatchNorm1d(h_dim))

        # 中间层
        for _ in range(n_layers - 2):
            self.convs.append(GATConv(h_dim, h_dim // heads, heads=heads, dropout=drop))
            self.bns.append(nn.BatchNorm1d(h_dim))

        # 分类头
        self.classifier = nn.Sequential(
            nn.Linear(h_dim, h_dim // 2),
            nn.ReLU(),
            nn.Dropout(drop),
            nn.Linear(h_dim // 2, out_dim),
        )

    def forward(self, x, edge_index, return_embed=False):
        for conv, bn in zip(self.convs, self.bns):
            x = conv(x, edge_index)
            x = bn(x)
            x = F.elu(x)
            x = F.dropout(x, p=self.dropout, training=self.training)

        if return_embed:
            return x
        return self.classifier(x)


# ================================================================
#  GraphSAGE
# ================================================================

class GraphSAGE(nn.Module):
    """两层 GraphSAGE 用于节点分类"""

    def __init__(self, in_dim: int, hidden_dim: int = None,
                 out_dim: int = 2, num_layers: int = None,
                 dropout: float = None):
        super().__init__()
        h_dim = hidden_dim or GRAPH_CONFIG['gnn_hidden_dim']
        n_layers = num_layers or GRAPH_CONFIG['gnn_num_layers']
        drop = dropout or GRAPH_CONFIG['gnn_dropout']

        self.convs = nn.ModuleList()
        self.bns = nn.ModuleList()
        self.dropout = drop

        for i in range(n_layers):
            in_d = in_dim if i == 0 else h_dim
            self.convs.append(SAGEConv(in_d, h_dim))
            self.bns.append(nn.BatchNorm1d(h_dim))

        self.classifier = nn.Sequential(
            nn.Linear(h_dim, h_dim // 2),
            nn.ReLU(),
            nn.Dropout(drop),
            nn.Linear(h_dim // 2, out_dim),
        )

    def forward(self, x, edge_index, return_embed=False):
        for conv, bn in zip(self.convs, self.bns):
            x = conv(x, edge_index)
            x = bn(x)
            x = F.relu(x)
            x = F.dropout(x, p=self.dropout, training=self.training)

        if return_embed:
            return x
        return self.classifier(x)


# ================================================================
#  下游分类器: GNN 嵌入 + XGBoost / MLP
# ================================================================

class GNNEmbeddingClassifier(nn.Module):
    """GNN 嵌入提取 → 下游分类器 (MLP)"""

    def __init__(self, gnn_model: nn.Module, embed_dim: int,
                 out_dim: int = 2, dropout: float = 0.3):
        super().__init__()
        self.gnn = gnn_model
        self.classifier = nn.Sequential(
            nn.Linear(embed_dim, embed_dim // 2),
            nn.BatchNorm1d(embed_dim // 2),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(embed_dim // 2, embed_dim // 4),
            nn.BatchNorm1d(embed_dim // 4),
            nn.ReLU(),
            nn.Linear(embed_dim // 4, out_dim),
        )

    def forward(self, x, edge_index, edge_weight=None):
        if isinstance(self.gnn, GCN):
            embed = self.gnn(x, edge_index, edge_weight, return_embed=True)
        else:
            embed = self.gnn(x, edge_index, return_embed=True)
        return self.classifier(embed)


# ================================================================
#  模型工厂
# ================================================================

def get_gnn_model(model_name: str, in_dim: int, **kwargs):
    """获取 GNN 模型实例"""
    models = {
        'GCN': GCN,
        'GAT': GAT,
        'GraphSAGE': GraphSAGE,
    }
    if model_name not in models:
        raise ValueError(f"Unknown GNN model: {model_name}. Choose from {list(models.keys())}")
    return models[model_name](in_dim=in_dim, **kwargs)


def get_emb_classifier(model_name: str, in_dim: int, **kwargs):
    """获取 GNN嵌入 + MLP 的联合模型"""
    gnn = get_gnn_model(model_name, in_dim, **kwargs)
    embed_dim = kwargs.get('hidden_dim', GRAPH_CONFIG['gnn_hidden_dim'])
    return GNNEmbeddingClassifier(gnn, embed_dim=embed_dim)
