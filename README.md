# STAR-GNN: Star-Schema to Heterogeneous Graph for Enterprise Risk Assessment

[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](https://opensource.org/licenses/MIT)
[![Python 3.10+](https://img.shields.io/badge/python-3.10+-blue.svg)](https://www.python.org/downloads/)

> **When do graph neural networks help enterprise financial risk assessment — and when do they not?**

STAR-GNN converts relational star-schema data (8 tables: base info, annual reports, registration changes, tax records, news sentiment, etc.) into a heterogeneous graph network and systematically benchmarks GNNs (GCN, GraphSAGE) against tree-based models (XGBoost, LightGBM) across four real-world datasets.

## Key Findings

- **Feature Ceiling Effect**: Tabular features (79-dim engineered) drive tree models to AUC ~0.99, leaving no headroom for GNN embeddings.
- **Graph Structure Has Signal**: Pure graph structure achieves F1=0.50 — not random, but insufficient to surpass rich tabular features.
- **Cross-Dataset Consistency**: GNN ≤ Tabular across all 4 datasets (Enterprise, CreditCard, Adult, Covtype), with ΔF1 from -0.012 to -0.060.
- **Rigorously Verified**: Permutation tests (p<.001) confirm results are not artifacts of data leakage.

## Architecture

```
Star Schema (8 tables)
    │
    ├── change_info → Jaccard shared-registration-change graph (154K edges)
    ├── tax_info    → Enterprise-taxitem bipartite graph (205 tax items)
    ├── news_info   → TF-IDF text embeddings (128-dim)
    └── base/annual  → 79-dim engineered tabular features
                    ↓
            Heterogeneous Graph
                    ↓
        ┌─────── GNN Encoder ───────┐
        │ GCN  /  GraphSAGE         │
        │ + Feature Concatenation   │
        └───────────┬───────────────┘
                    ↓
         MLP / XGBoost Classifier
                    ↓
         Risk Assessment + Explanation
```

## Dataset Coverage

| Dataset | Samples | Features | Domain | Tab F1 | GNN F1 | ΔF1 |
|---------|---------|----------|--------|--------|--------|------|
| Enterprise (private) | 14,865 | 57 | Regulatory | .818 | .777 | -0.041 |
| CreditCard (public) | 6,000 | 29 | Financial fraud | .918 | .902 | -0.016 |
| Adult/Census (public) | 6,000 | 14 | Income prediction | .839 | .822 | -0.016 |
| Covtype (public) | 6,000 | 54 | Ecology | .936 | .921 | -0.015 |

> **Rigor**: Nested CV with full HP grid search (8 GNN + 8 XGB + 8 LGB combos per fold).
> Dual permutation tests (XGB + GNN label shuffling, 20 trials each) all pass p<0.001.
> Label leakage mitigated: Enterprise features 79→57 by excluding change/tax columns.

## Project Structure

```
code_sci/
├── src/
│   ├── graph_builder_np.py    # Star-schema → graph construction (numpy)
│   ├── graph_builder_v2.py    # 79-dim feature engineering + graph
│   ├── gnn_experiment.py      # GNN vs Baseline CV experiments
│   ├── gnn_embed_xgb.py       # GNN embedding augmentation
│   ├── benchmark_4ds.py       # 4-dataset unified benchmark
│   └── config.py              # BGXMDM mapping, graph hyperparams
├── results/
│   └── graph_data_v2.pkl      # Built graph (14,865 nodes, 154K edges)
├── data/                      # Public datasets (download separately)
└── figures/                   # Output visualizations
```

## Quick Start

```bash
# Clone
git clone git@github.com:YOUR_USERNAME/STAR-GNN.git
cd STAR-GNN

# Install dependencies
pip install torch numpy pandas scikit-learn xgboost lightgbm matplotlib

# Build graph from star-schema data
python src/graph_builder_v2.py

# Run benchmark
python src/benchmark_4ds.py
```

## Requirements

- Python 3.10+
- PyTorch ≥ 2.0 (CUDA optional)
- NumPy, Pandas, Scikit-learn
- XGBoost, LightGBM
- Matplotlib

## Citation

```bibtex
@article{star-gnn2026,
  title={STAR-GNN: Star-Schema to Heterogeneous Graph Networks for Enterprise Risk Assessment},
  author={},
  journal={arXiv preprint},
  year={2026}
}
```

## License

MIT License. See [LICENSE](LICENSE) for details.
