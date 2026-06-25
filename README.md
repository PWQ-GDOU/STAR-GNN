# STAR-GNN: From Star-Schema Tables to Graph Networks for Enterprise Default Detection

### When Does the Graph Help?

[![Python 3.10+](https://img.shields.io/badge/python-3.10+-blue.svg)](https://www.python.org/downloads/)
[![PyTorch](https://img.shields.io/badge/PyTorch-%E2%89%A52.0-red.svg)](https://pytorch.org/)
[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](https://opensource.org/licenses/MIT)
[![Status](https://img.shields.io/badge/status-submitted%20to%20ESWA-brightgreen.svg)]()

**Authors**: Wenquan Peng, Wenjing Zhen, Yilin Zhen, Junwen Xie, Zhiyu Zhang (advisor)  
**Affiliation**: College of Computer Science & Engineering, Guangdong Ocean University  
**Paper**: Submitted to *Expert Systems with Applications* (Elsevier, JCR Q1, IF 8.3)



## Overview

**Can graph neural networks improve enterprise default prediction when rich tabular features are available?**

STAR-GNN provides the **first systematic empirical answer** — by converting relational star-schema data into k-NN feature-similarity graphs and rigorously benchmarking GNNs against tree-based models across four real-world datasets, we discover the **Feature Ceiling Effect**: tabular features reach near-perfect performance (AUC > 0.99), leaving virtually no headroom for graph-based methods.

### Key Contributions

1. **Feature Ceiling Effect** — quantified the precise point where tabular features saturate
2. **Dual verification** — permutation tests (p < 0.001) + edge rewiring (degree-preserving XSwap)
3. **Cross-dataset benchmark** — 4 datasets × 7 model architectures × Nested CV
4. **Ablation from 57D → 3D** — identify the "ceiling height" where GNN becomes competitive
5. **Open-source release** — full pipeline from raw star-schema tables to final evaluation

### Core Finding at a Glance

| Dataset | Samples | Features | Tab F1 | GNN F1 | ΔF1 | p-value |
|---------|---------|----------|--------|--------|------|---------|
| Enterprise | 14,865 | 57 | .818 | .777 | **−0.041** | <0.001 |
| CreditCard | 6,000 | 29 | .918 | .902 | −0.016 | <0.001 |
| Adult/Census | 6,000 | 14 | .839 | .822 | −0.016 | <0.001 |
| Covtype | 6,000 | 54 | .936 | .921 | −0.015 | <0.001 |

> **GNN ≠ always better.** In star-schema enterprise data, tabular models dominate. The graph provides a genuine but supplementary signal (F1 ≈ 0.50 standalone), systematically overpowered by the 57 engineered tabular features. Use the right tool for the right data.



## Architecture

```
                      ┌─────────────────────────────┐
                      │     Star Schema (8 tables)   │
                      │  base | annual | change |    │
                      │  news | tax | other | eval   │
                      └─────────────┬───────────────┘
                                    │ label-leakage removal (79D → 57D)
                                    ▼
                      ┌─────────────────────────────┐
                      │   Feature Matrix (N × 57)   │
                      │  StandardScaler → k-NN (k=10)│
                      └─────────────┬───────────────┘
                                    │ cosine similarity graph (14,865 nodes, ~154K edges)
                                    ▼
              ┌─────────────────────┴─────────────────────┐
              ▼                                           ▼
    ┌──────────────────┐                     ┌──────────────────┐
    │  Tabular Models   │                     │    GNN Models     │
    │  XGBoost / LGBM   │                     │  GCN / GraphSAGE  │
    │  RF / LR / MLP     │                     │  128D embeddings  │
    └────────┬───────────┘                     └────────┬─────────┘
             │                                          │
             └──────────────┬───────────────────────────┘
                            │
                            ▼
              ┌─────────────────────────────┐
              │     Triple Verification     │
              │  ┌───────────────────────┐  │
              │  │ Ablation: 57D → 3D   │  │
              │  │ Permutation: 2×20     │  │
              │  │ Edge Rewiring: XSwap  │  │
              │  └───────────────────────┘  │
              └─────────────────────────────┘
```

> **IMPORTANT**: The Jaccard change-code similarity graph (early experiments) was **abandoned** — it implicitly encodes label signal: high-risk enterprises naturally share more high-risk change codes (e.g., investor changes, deregistration filings). Final experiments use **cosine-similarity k-NN graphs** over the cleaned 57-dimensional feature matrix.



## Project Structure

```
STAR-GNN/
├── src/
│   ├── benchmark_clean.py       ★ Main entry: Nested CV + HP search + ablation + perm test + rewiring
│   ├── graph_builder_v2.py       Star-schema → 57D features + k-NN graph construction
│   ├── graph_builder_np.py       NumPy-only version (no PyTorch needed)
│   ├── config.py                 BGXMDM mappings, GNN/GCN hyperparameters
│   ├── ablation_infer.py         Ablation experiment standalone runner
│   ├── gnn_experiment.py         Initial GNN vs baseline experiments (reproduced in benchmark_clean)
│   ├── gnn_embed_xgb.py          GNN embedding → XGBoost augmentation experiments
│   ├── graph_builder.py          Earlier graph builder (Jaccard-based, deprecated)
│   └── legacy/                   ⚠ DO NOT USE — contains bugs (wrong KNN normalization, HP search leakage)
│
├── results/                      ★ Output directory (created on first run)
├── figures/                      ★ Generated plots
├── data/                         Public datasets (see Quick Start below)
├── requirements.txt              Python dependencies
├── README.md                     This file
└── .gitignore
```

### Which Script Should I Use?

| Scenario | Script |
|----------|--------|
| **Reproduce the paper** | `src/benchmark_clean.py` (one command, all experiments) |
| Build graph only (no training) | `src/graph_builder_v2.py` |
| Ablation experiments standalone | `src/ablation_infer.py` |
| GNN embedding + XGBoost augmentation | `src/gnn_embed_xgb.py` |
| Any other `benchmark_*.py` | **Deprecated** — do not use; results differ from paper |



## Quick Start

### 1. Clone & Install

```bash
git clone https://github.com/PWQ-GDOU/STAR-GNN.git
cd STAR-GNN
pip install -r requirements.txt
```

<details>
<summary>Manual install (if pip fails)</summary>

```bash
pip install torch numpy pandas scikit-learn scipy xgboost lightgbm matplotlib imbalanced-learn
```
</details>

### 2. Prepare Data

**Public datasets (3 of 4)** — downloaded automatically on first run via `sklearn.datasets`:
- **CreditCard**: [`sklearn.datasets.fetch_openml('credit-g')`](https://www.openml.org/d/31)
- **Adult/Census**: [`sklearn.datasets.fetch_openml('adult')`](https://www.openml.org/d/1590)
- **Covtype**: [`sklearn.datasets.fetch_covtype()`](https://archive.ics.uci.edu/ml/datasets/Covertype)

**Enterprise dataset (private)** — the 4th dataset is proprietary regulatory data and **cannot be publicly shared**. The paper's results are fully reproducible for the 3 public datasets. To run with the private dataset:

```bash
# Place your star-schema CSV files in data/enterprise/
# Expected tables:
#   entprise_info.csv, base_info.csv, annual_report_info.csv,
#   change_info.csv, news_info.csv, tax_info.csv, other_info.csv

export STAR_GNN_DATA="./data"          # Linux/Mac
set STAR_GNN_DATA=.\data               # Windows
```

### 3. Run Experiments

```bash
# Full benchmark (4 datasets, ~30 min on GPU, ~2h on CPU)
python src/benchmark_clean.py

# With custom output directory
mkdir -p results/my_run
export STAR_GNN_RESULTS="./results/my_run"
python src/benchmark_clean.py
```

**What `benchmark_clean.py` does (in order):**

| Step | Description | Output |
|------|-------------|--------|
| 1 | Load 4 datasets + build k-NN graphs | Console log |
| 2 | Nested 5×2 CV: 7 tabular models (XGB/LGB/RF/LR/MLP) | `tabular_*.json` |
| 3 | Train GCN + GraphSAGE with early stopping | `gnn_*.json` |
| 4 | Ablation: 57→50→30→20→15→10→7→5→3 dimensions | `ablation_*.json` |
| 5 | Dual permutation tests (XGB + GNN, 20 trials each) | `perm_*.json` |
| 6 | Edge rewiring (degree-preserving XSwap, 10 trials) | `rewire_*.json` |
| 7 | Generate summary table + diagnostic plots | `results/*.png` |

### 4. Expected Output

```
results/benchmark_clean/
├── summary.json              # All metrics in one file
├── tabular_enterprise.json   # Tabular model results per dataset
├── gnn_enterprise.json       # GNN model results per dataset
├── ablation_enterprise.png   # Ablation trajectory plot
├── perm_enterprise.png       # Permutation test histogram
├── rewire_enterprise.png     # Edge rewiring analysis
├── ... (×4 datasets)
└── comparison_table.png      # Final cross-dataset comparison
```



## Reproducibility Checklist

To exactly reproduce the paper's results:

- [ ] Python 3.10+ with PyTorch ≥ 2.0
- [ ] `pip install -r requirements.txt` (locks all dependency versions)
- [ ] Random seed fixed at `42` (`RANDOM_SEED` in `benchmark_clean.py`)
- [ ] Nested CV: outer 5-fold × inner 2-fold (`N_OUTER_FOLDS=5`, `N_INNER_FOLDS=2`)
- [ ] Subsampled to 6,000 samples for public datasets (`N_SUBSAMPLE=6000`)
- [ ] k-NN graph: `k=10`, cosine metric, row-normalized (out-degree)
- [ ] GNN: 2-layer, 128D hidden, 200 epochs, patience=30, lr=0.001
- [ ] Permutation: 20 trials each for XGBoost and GNN
- [ ] Edge rewiring: 10 XSwap iterations, degree-preserving
- [ ] Enterprise data: 57-dimensional clean version (22 label-leakage features removed)

> **GPU recommended** but not required. CPU mode works for all experiments (expect 2-4× longer runtime).



## Key Design Decisions

### Why 57 Features (not 79)?

The original 79-dimensional feature set included 22 columns from `change_info` and `tax_info` tables that directly correlate with the label. For example:
- `n_high_risk_changes` — enterprises flagged as risky naturally have more high-risk change codes
- `n_tax_items` — risky enterprises trigger more tax audits → more tax records

These 22 features were **removed** to eliminate data leakage. The remaining 57 features represent genuine predictive signal.

### Why k-NN Cosine Similarity (not Jaccard)?

Early experiments used Jaccard similarity over change-code co-occurrence vectors. This was **abandoned** because:
1. High-risk enterprises inherently share more high-risk change codes (e.g., investor changes #111, director changes #110)
2. This encodes the label signal directly into the graph topology
3. Correct in-practice interpretation: "similar risky enterprises cluster together" — but this doesn't help *predict* risk

The k-NN cosine graph over the full 57-dim feature space provides a label-agnostic similarity measure.

### Why Nested CV?

Standard CV (train/val split + test holdout) causes **HP search information leakage** when the same fold is used for both hyperparameter tuning and evaluation. Nested CV isolates HP search to an inner loop, ensuring unbiased performance estimates.



## Citation

```bibtex
@article{peng2026stargnn,
  title     = {{STAR-GNN}: {From} Star-Schema Tables to Graph Networks
               for Enterprise Default Detection — When Does the Graph Help?},
  author    = {Peng, Wenquan and Zhen, Wenjing and Zhen, Yilin
               and Xie, Junwen and Zhang, Zhiyu},
  journal   = {Expert Systems with Applications},
  year      = {2026},
  note      = {Under review}
}
```



## Troubleshooting

<details>
<summary><b>CUDA Out of Memory</b> on GPU</summary>

The dense adjacency matrix for 14,865 nodes requires ~3.3 GB. If your GPU has <6 GB VRAM:
```bash
export CUDA_VISIBLE_DEVICES=""         # Linux/Mac
set CUDA_VISIBLE_DEVICES=              # Windows
```
Or set `device = torch.device("cpu")` in `benchmark_clean.py`.
</details>

<details>
<summary><b>"FileNotFoundError: graph_data_v2.pkl"</b></summary>

Run `graph_builder_v2.py` first, or set the correct `GRAPH_PATH`:
```bash
export STAR_GNN_HOME="/path/to/your/STAR-GNN"
```
</details>

<details>
<summary><b>ImportError: No module named 'lightgbm'</b></summary>

```bash
pip uninstall lightgbm
pip install lightgbm
```
Then disable `device='gpu'` in the LGBM config for CPU-only.
</details>

<details>
<summary><b>Results differ slightly from paper</b></summary>

Minor variance (±0.002 F1) is expected due to floating-point behavior across PyTorch/XGBoost versions and OS-dependent `StratifiedKFold` ordering. Set `RANDOM_SEED=42` and run `benchmark_clean.py` as-is for maximum reproducibility.
</details>



## License

MIT License.

---

For questions, please open a [GitHub Issue](https://github.com/PWQ-GDOU/STAR-GNN/issues) or contact the corresponding author.
