"""Task 1: load, check, split and convert the data to tensors."""
import os
import numpy as np
import torch
import matplotlib
matplotlib.use("Agg")
from load_data import load_raw
from plot_beats import plot_one_per_class

def prepare(data_dir="data"):
    raw = load_raw(data_dir)
    train_df, test_df = raw["train"], raw["test"]

    # 1.2 signal = columns 0..186, label = column 187 (stored as float -> int)
    X_all = train_df.iloc[:, :187].values
    y_all = train_df.iloc[:, 187].values.astype(np.int64)
    X_test = test_df.iloc[:, :187].values
    y_test = test_df.iloc[:, 187].values.astype(np.int64)

    # 1.6 exact split given in the assignment
    rng = np.random.default_rng(42)
    idx = rng.permutation(len(train_df))
    n_val = int(0.1 * len(train_df))
    val_idx, train_idx = idx[:n_val], idx[n_val:]

    f, l = torch.float32, torch.long      # 1.7 tensors
    data = {
        "X_train": torch.tensor(X_all[train_idx], dtype=f),
        "y_train": torch.tensor(y_all[train_idx], dtype=l),
        "X_val":   torch.tensor(X_all[val_idx], dtype=f),
        "y_val":   torch.tensor(y_all[val_idx], dtype=l),
        "X_test":  torch.tensor(X_test, dtype=f),
        "y_test":  torch.tensor(y_test, dtype=l),
    }
    return data, raw

if __name__ == "__main__":
    os.makedirs("results", exist_ok=True)
    data, raw = prepare()
    train_df, test_df = raw["train"], raw["test"]
    lines = []
    def out(s):
        print(s); lines.append(s)

    out(f"1.1 train shape: {train_df.shape}, test shape: {test_df.shape}")
    out(f"1.2 label dtype in file: {train_df[187].dtype} -> converted to int64")
    sig = train_df.iloc[:, :187].values
    out(f"1.3 signal min = {sig.min():.4f}, max = {sig.max():.4f}")
    ctr = np.bincount(train_df[187].astype(int), minlength=5)
    cte = np.bincount(test_df[187].astype(int), minlength=5)
    out(f"1.4 rows per class (N,S,V,F,Q) train: {ctr.tolist()}  (expected 72471,2223,5788,641,6431)")
    out(f"1.4 rows per class (N,S,V,F,Q) test : {cte.tolist()}  (expected 18118,556,1448,162,1608)")
    out(f"1.6 rows -> train {len(data['y_train'])}, val {len(data['y_val'])}, test {len(data['y_test'])}")
    out(f"1.7 dtypes: X {data['X_train'].dtype}, y {data['y_train'].dtype}, X shape {tuple(data['X_train'].shape)}")
    plot_one_per_class(train_df)
    with open("results/table1_data.txt", "w") as fh:
        fh.write("\n".join(lines))
