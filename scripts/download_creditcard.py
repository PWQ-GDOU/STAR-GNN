"""
download_creditcard.py — 下载 Credit Card Fraud Detection 数据集
来源: Kaggle (https://www.kaggle.com/datasets/mlg-ulb/creditcardfraud)
License: Database Contents License (DbCL) v1.0
"""
import urllib.request, zipfile, io, os, sys

URL = "https://www.kaggle.com/api/v1/datasets/mlg-ulb/creditcardfraud/download"
OUT = os.path.join(os.path.dirname(__file__), "..", "data", "creditcard.csv")

# Try kagglehub first (no auth needed for some versions)
try:
    import kagglehub
    path = kagglehub.dataset_download("mlg-ulb/creditcardfraud")
    import shutil
    for f in os.listdir(path):
        if f.endswith('.csv'):
            shutil.copy(os.path.join(path, f), OUT)
            print("Downloaded via kagglehub: %s (%d MB)" % (OUT, os.path.getsize(OUT)//1024//1024))
            sys.exit(0)
except ImportError:
    pass

print("Please download manually from:")
print("  https://www.kaggle.com/datasets/mlg-ulb/creditcardfraud")
print("Place creditcard.csv in data/ directory.")
