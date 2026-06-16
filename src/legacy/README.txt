LEGACY FILES — DO NOT USE
These files contain bugs fixed in benchmark_clean.py:
- Wrong KNN normalization direction (sum(adj,0) instead of sum(adj,1))
- HP search information leakage (Fold1 used for both HP search and CV)
- Missing GNN permutation test
- Missing label leakage mitigation
- KDDCup synthetic data mislabeled as real

Use benchmark_clean.py as the single authoritative experiment script.
