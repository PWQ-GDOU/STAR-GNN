# STAR-GNN 实验运行说明

## 环境要求

```bash
git clone https://github.com/PWQ-GDOU/STAR-GNN.git
cd STAR-GNN
pip install -r requirements.txt
# 额外需要: torch>=2.0 (GPU版), pyarrow (用于读取pickle)
```

需要的数据文件（放在 `data/` 目录下，不在仓库中）：
- `creditcard.csv` (Kaggle Credit Card Fraud Detection)
- `adult.data` + `adult.test` (UCI Adult/Census Income)  
- `covtype.data.gz` (UCI Covertype)
- `graph_data_v2.pkl` (Enterprise 图数据，通过 graph_builder 生成)

## 步骤 0：重建图（如未提供 graph_data_v2.pkl）

```bash
python src/graph_builder_v2.py
# 输出: results/graph_data_v2.pkl (~26MB)
# 节点: 14,865, 边: 154,280, 特征: 57 维 (leakage-mitigated)
```

## 实验 1：主实验 — 4 数据集基准对比（论文 Table 1 + Fig 1-3）

```bash
python src/benchmark_clean.py
```

**运行时间**: ~45-60 分钟 (RTX 8GB+ GPU)

**输出文件**:
```
results/benchmark_clean/
├── benchmark_f1.png          # Fig 1: 4数据集×7模型 F1 柱状图
├── results.json               # Table 1: 完整数值结果 (JSON)
```

**论文用途**:
- `benchmark_f1.png` → 论文 Fig. 2 (Model Comparison Across Datasets)
- `results.json` → 论文 Table 2 (Main Results)

## 实验 2：特征消融 + 推理效率（论文 Fig 4 + Fig 5）

```bash
python src/ablation_infer.py
```

**运行时间**: ~15 分钟

**输出文件**:
```
results/benchmark/
├── ablation_crossover.png     # Fig 4: 特征维度 vs F1 趋势图
└── ablation_results.json      # 数值结果
```

**论文用途**:
- `ablation_crossover.png` → 论文 Fig. 4 (Feature Ablation: XGB vs GraphSAGE)
- 推理时间数据 → 论文 Section 4.4 (Inference Efficiency)

## 实验 3：GPU 详细配置（前两个脚本会自动使用CUDA）

所有脚本会自动检测 `torch.cuda.is_available()` 并使用 GPU。
手动指定:
```bash
export CUDA_VISIBLE_DEVICES=0
```

---

## 论文需要的全部图表清单

| 图号 | 内容 | 来源 | 脚本 |
|------|------|------|------|
| Fig. 1 | STAR-GNN 框架架构图 | 手动绘制 | — |
| Fig. 2 | 4数据集 × 7模型 F1 对比 | `benchmark_f1.png` | benchmark_clean.py |
| Fig. 3 | 特征稀疏度 vs GNN-Tab Delta | `sparsity_vs_delta.png` | benchmark_clean.py |
| Fig. 4 | 特征消融趋势 (维度 vs F1) | `ablation_crossover.png` | ablation_infer.py |
| Fig. 5 | 推理效率对比 (XGB vs GNN) | `inference_time.png` | ablation_infer.py |
| Fig. 6 | 排列检验 (Real vs Shuffled) | 数据来自 results.json | benchmark_clean.py |

## 论文需要的全部表格

| 表号 | 内容 | 来源 |
|------|------|------|
| Table 1 | 数据集统计 | 手动整理 |
| Table 2 | 主实验结果 (4数据集×7模型) | `results.json` → `results.Enterprise/LGB.f1_mean` 等 |
| Table 3 | 消融实验结果 | `ablation_results.json` |
| Table 4 | 超参数搜索空间 | 见 `benchmark_clean.py` HP_* 变量 |

---

## 额外：快速验证（确认环境正常）

```bash
python src/quick_verify.py
# 预期输出: 4行 F1/AUC 数值（GCN/SAGE/XGB/LGB）
```
