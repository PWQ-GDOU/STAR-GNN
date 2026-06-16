"""
config.py — SCI 研究版全局配置
"""
import os, json

# ======== 路径 ========
BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
DATA_DIR = os.path.join(os.path.dirname(BASE_DIR), "code", "output")  # 复用原数据
# 如果没有 preprocessed 则回退到原始 CSV
RAW_DATA_DIR = os.path.join(os.path.dirname(os.path.dirname(BASE_DIR)), "data")  # 数据CSV目录
OUTPUT_DIR = os.path.join(BASE_DIR, "results")
FIGURE_DIR = os.path.join(BASE_DIR, "figures")
CHECKPOINT_DIR = os.path.join(OUTPUT_DIR, "checkpoints")
RESULTS_DIR = os.path.join(OUTPUT_DIR, "logs")

os.makedirs(OUTPUT_DIR, exist_ok=True)
os.makedirs(FIGURE_DIR, exist_ok=True)
os.makedirs(CHECKPOINT_DIR, exist_ok=True)
os.makedirs(RESULTS_DIR, exist_ok=True)

# ======== 数据表 ========
TABLES = [
    "base_info", "annual_report_info", "change_info",
    "news_info", "tax_info", "other_info",
    "entprise_info", "entprise_evaluate",
]

# ======== 图构建超参数 ========
GRAPH_CONFIG = {
    # 同构图 (企业变更网络)
    "change_similarity_threshold": 0.3,   # Jaccard 相似度 > 此值才建边
    "change_max_edges_per_node": 50,      # 每节点最大出边数（防稠密图）
    "change_min_co_changes": 2,           # 至少共享 2 次变更才建边

    # 二分图 (企业-税目)
    "tax_min_count": 5,                   # 税目出现 ≥ 此值才纳入图

    # 文本嵌入
    "text_model": "bert-base-chinese",    # 或 paraphrase-multilingual-MiniLM-L12-v2
    "text_embedding_dim": 768,
    "text_max_length": 128,

    # GNN
    "gnn_hidden_dim": 128,
    "gnn_num_layers": 2,
    "gnn_dropout": 0.3,
    "gnn_epochs": 200,
    "gnn_lr": 0.001,
    "gnn_patience": 20,                  # Early stopping
}

# ======== 评估配置（严格模式）========
EVAL_CONFIG = {
    "n_folds": 5,                         # 交叉验证折数
    "n_bootstrap": 1000,                  # Bootstrap 采样次数
    "ci_alpha": 0.05,                     # 95% 置信区间
    "noise_levels": [0, 0.05, 0.10, 0.20], # 测试集特征噪声比例
    "random_state": 42,
    # 时间序列切分（如按年报年份）
    "temporal_split": False,              # 如果有年份字段则启用
    "temporal_ratio": 0.8,                # 训练:测试 = 前80%:后20%
}

# ======== 模型配置 ========
MODEL_CONFIG = {
    # Baseline models (从 code\ 继承)
    "baselines": ["LR", "RF", "XGBoost", "LightGBM", "GNB", "SVM"],

    # GNN models
    "gnn_models": ["GCN", "GAT", "GraphSAGE", "RGCN"],

    # Downstream classifiers
    "classifiers": ["XGBoost", "MLP"],
}

# ======== GPU ========
DEVICE = "cuda"  # XGBoost device (3.x API)
LGB_DEVICE = "gpu"
TORCH_DEVICE = "cuda"

# ======== 实验矩阵 ========
EXPERIMENTS = {
    "exp01_graph_construction": {
        "description": "图构建超参数搜索",
        "grid": {
            "change_similarity_threshold": [0.1, 0.2, 0.3, 0.5],
            "change_max_edges_per_node": [20, 50, 100],
        },
        "metrics": ["f1", "auc", "recall", "precision"],
    },
    "exp02_gnn_architecture": {
        "description": "GNN 架构对比",
        "models": ["GCN", "GAT", "GraphSAGE", "RGCN", "HGT"],
        "layers": [1, 2, 3],
        "hidden_dims": [64, 128, 256],
    },
    "exp03_fusion_strategy": {
        "description": "融合策略消融",
        "strategies": [
            "concat_all",       # 全部拼接
            "gnn_only",          # 只用 GNN 嵌入
            "tab_only",          # 只用表格特征
            "gnn_plus_tab",      # GNN + 表格
            "attention_fusion",  # 注意力融合
        ],
    },
    "exp04_noise_robustness": {
        "description": "噪声鲁棒性测试",
    },
    "exp05_ablation": {
        "description": "消融实验（数据源贡献）",
    },
}

# ======== 45 种 bgxmdm 变更代码映射（从 code\config.py 继承）========
BGXMDM_MAP = {
    "0101": "经营范围变更", "0102": "章程备案", "0103": "住所变更",
    "0104": "法定代表人变更", "0105": "注册资本变更", "0106": "营业期限变更",
    "0107": "名称变更", "0108": "经营场所变更", "0109": "投资人变更",
    "0110": "董事备案", "0111": "监事备案", "0112": "经理备案",
    "0113": "高级管理人员备案", "0114": "分公司备案", "0115": "清算组备案",
    "0116": "负责人变更", "0117": "出资额变更", "0118": "出资方式变更",
    "0119": "出资时间变更", "0120": "出资比例变更", "0121": "营业期限变更",
    "0122": "投资人出资额变更", "0123": "实收资本变更", "0124": "企业类型变更",
    "0125": "经营期限变更", "0126": "分支机构变更", "0127": "联络员备案",
    "0128": "财务负责人备案", "0129": "撤销分支机构", "0130": "增补证照",
    "0131": "补发证照", "0132": "换发证照", "0133": "撤销变更登记",
    "0134": "注销备案", "0135": "迁移变更", "0136": "股权变更",
    "0137": "股东变更", "0138": "股东出资变更", "0139": "外资变更",
    "0140": "中外合资变更", "0141": "外资转内资", "0142": "内资转外资",
    "0143": "改制变更", "0144": "合并变更", "0145": "分立变更",
}

HIGH_RISK_CHANGES = [
    "0109", "0110", "0111", "0112", "0136", "0137", "0138",
    "0143", "0144", "0145", "0104", "0105", "0113",
]
