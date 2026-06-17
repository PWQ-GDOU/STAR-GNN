# STAR-GNN 实验结果与发现

> 运行环境: Linux, 4×RTX 2080Ti (11GB), PyTorch 2.12+cu130  
> 运行时间: 2026-06-17, ~40 min (3 datasets × 3 seeds × 5 folds × nested HP)

---

## 1. 主实验结果（Table 2: 3 Public Datasets）

### F1 汇总 (mean ± std across 3 seeds × 5 folds)

| Dataset | Feats | Sparsity | GCN | GraphSAGE | XGB | LGB | RF | LR | MLP |
|---------|-------|----------|-----|-----------|-----|-----|-----|-----|-----|
| CreditCard | 29 | sparse (.17) | 0.147 | 0.894 | 0.905 | **0.916** | 0.913 | 0.839 | 0.900 |
| Adult | 14 | medium (.43) | 0.657 | 0.811 | **0.837** | 0.834 | 0.828 | 0.768 | 0.767 |
| Covtype | 54 | medium (.33) | 0.662 | 0.887 | 0.929 | **0.933** | 0.921 | 0.878 | 0.921 |

### AUC 汇总

| Dataset | GraphSAGE | XGB | LGB |
|---------|-----------|-----|-----|
| CreditCard | 0.962 | 0.979 | **0.980** |
| Adult | 0.887 | **0.920** | 0.918 |
| Covtype | 0.948 | 0.980 | **0.982** |

### GNN vs Tabular Delta

| Dataset | Tab F1 | GNN F1 | ΔF1 |
|---------|--------|--------|------|
| CreditCard | 0.916 (LGB) | 0.894 (SAGE) | **-0.022** |
| Adult | 0.837 (XGB) | 0.811 (SAGE) | **-0.026** |
| Covtype | 0.933 (LGB) | 0.887 (SAGE) | **-0.046** |

> 结论: 在所有 3 个公开数据集上，tree-based models (LGB/XGB) 一致优于 GraphSAGE。GNN 不带来性能提升。

---

## 2. 🔑 关键发现：Graph Permutation Test 揭示 k-NN 图 circularity

Graph Permutation Test: 随机重连图的边（保留 degree distribution），重新训练 GNN。

| Dataset | 真实图 F1 | **打乱图 F1** | 差异 |
|---------|----------|-------------|------|
| CreditCard | 0.894 | **0.901 ± 0.015** | +0.007 |
| Adult | 0.811 | **0.812 ± 0.011** | +0.001 |
| Covtype | 0.887 | **0.889 ± 0.007** | +0.002 |

**发现**: 打乱图边后 GraphSAGE 的 F1 几乎不变。这说明 k-NN 图的拓扑结构对预测没有额外贡献——GNN 完全是从 node features 学习的，graph message passing 只是在做特征平滑而非提供新的结构信号。

**论文意义**: 这是 k-NN 图 circularity 问题的直接实验证据。当图是从预测特征构建时（如 k-NN），graph topology 不包含任何超出 node features 的信息。这也解释了为什么 GNN 无法超越 tree models——tree models 直接使用这些特征，而 GNN 通过图结构间接使用，且图结构本身不增加新信号。

---

## 3. 排列检验结果

### Label Permutation (XGB)
| Dataset | Real F1 | Shuffled F1 | p-value |
|---------|---------|-------------|---------|
| CreditCard | 0.905 | 0.080 ± 0.022 | <0.001 |
| Adult | 0.837 | 0.503 ± 0.003 | <0.001 |
| Covtype | 0.929 | 0.505 ± 0.003 | <0.001 |

### Label Permutation (GNN)
| Dataset | Real F1 | Shuffled F1 | p-value |
|---------|---------|-------------|---------|
| CreditCard | 0.894 | 0.127 ± 0.007 | <0.001 |
| Adult | 0.811 | 0.618 ± 0.023 | <0.001 |
| Covtype | 0.887 | 0.615 ± 0.009 | <0.001 |

> 所有排列检验 p<0.001，结果非随机。

---

## 4. GCN Oversmoothing 现象

GCN 在所有 k-NN 图数据集上表现极差:
- CreditCard: F1=0.147, AUC=0.497 → 等价于随机猜测
- Adult: F1=0.657, AUC=0.500 → 预测多数类
- Covtype: F1=0.662, AUC=0.499 → 预测多数类

原因是 2 层 GCN 在密集 k-NN 图上出现了严重的 oversmoothing——所有节点特征被平均到无法区分的状态。而 GraphSAGE 通过 concat(self, neighbors) 机制避免了此问题。

---

## 5. 特征稀疏度分析

| Dataset | d | PCA50 | PCA95 | Effective Rank | Category |
|---------|---|-------|-------|----------------|----------|
| CreditCard | 29 | 5 | 22 | 0.172 | sparse |
| Adult | 14 | 6 | 13 | 0.429 | medium |
| Covtype | 54 | 18 | 42 | 0.333 | medium |

CreditCard 虽然只有 29 维，但有效秩仅 0.17——仅需 5 个 PCA 成分即可解释 50% 方差。这说明其信息高度集中，tree models 容易捕获。

---

## 6. 已修复的 Bug

本次实验中发现并修复了 3 个 bug:

1. **`random` 模块未导入** (`benchmark_clean.py:13`): `nested_hp_search` 的 GNN 分支使用 `random.sample()` 但 `random` 只在内部 `_search` 函数局部导入
2. **返回值数量不匹配** (`benchmark_clean.py:323`): `graph_permutation_test` 返回 3 值，但调用处 unpack 4 值
3. **输出缓冲** (非代码bug): 为后台运行添加了 `PYTHONUNBUFFERED=1`

---

## 7. 运行复现指南

```bash
git clone https://github.com/PWQ-GDOU/STAR-GNN.git
cd STAR-GNN

# 解压数据（仓库中为 .gz 格式）
gunzip -k data/creditcard.csv.gz

# 安装依赖
pip install torch numpy pandas scikit-learn xgboost lightgbm matplotlib imbalanced-learn

# 运行主实验（约 40 min on RTX 2080Ti）
export STAR_GNN_HOME=$(pwd)
python src/benchmark_clean.py
```

> 注意: Enterprise 数据集（graph_data_v2.pkl）为私密数据，需单独提供。
