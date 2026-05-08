"""Full-batch logistic regression on scikit-learn Digits.

The implementation uses row-vector inputs x_i and weights W in R^{k x d}.
Cov-MuonW right-preconditions the k x d momentum by the d x d input
covariance inverse. The script saves raw multi-seed results and seed-42 curves.
"""
import os
os.environ.setdefault("OPENBLAS_NUM_THREADS", "1")
os.environ.setdefault("OMP_NUM_THREADS", "1")
os.environ.setdefault("MKL_NUM_THREADS", "1")

import csv
from pathlib import Path
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from sklearn.datasets import load_digits
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler

ROOT = Path(__file__).resolve().parents[1]
RESULT_DIR = ROOT / "results"
FIG_DIR = ROOT / "figures"
RESULT_DIR.mkdir(exist_ok=True)
FIG_DIR.mkdir(exist_ok=True)


def softmax(z):
    z = z - np.max(z, axis=1, keepdims=True)
    e = np.exp(z)
    return e / e.sum(axis=1, keepdims=True)


def msign(M):
    U, _, Vt = np.linalg.svd(M, full_matrices=False)
    return U @ Vt


def accuracy(W, X, y):
    pred = np.argmax(X @ W.T, axis=1)
    return np.mean(pred == y)


def ce_loss(W, X, Y):
    P = softmax(X @ W.T)
    return -np.mean(np.sum(Y * np.log(P + 1e-12), axis=1))


def load_data(seed):
    X, y = load_digits(return_X_y=True)
    Xtr_raw, Xte_raw, ytr, yte = train_test_split(
        X, y, test_size=0.2, random_state=seed, stratify=y
    )
    scaler = StandardScaler().fit(Xtr_raw)
    Xtr = scaler.transform(Xtr_raw).astype(np.float64)
    Xte = scaler.transform(Xte_raw).astype(np.float64)
    return Xtr, Xte, ytr.astype(int), yte.astype(int)


def run(method, seed=42, record=False, steps=100):
    Xtr, Xte, ytr, yte = load_data(seed)
    n, d = Xtr.shape
    k = 10
    Y = np.eye(k)[ytr]
    Cinv = np.linalg.inv((Xtr.T @ Xtr) / n + 1e-4 * np.eye(d))
    W = np.zeros((k, d), dtype=np.float64)
    M = np.zeros((k, d), dtype=np.float64)
    adam_m = np.zeros((k, d), dtype=np.float64)
    adam_v = np.zeros((k, d), dtype=np.float64)
    history = {"loss": [], "train_acc": [], "test_acc": []}

    for t in range(1, steps + 1):
        P = softmax(Xtr @ W.T)
        G = (P - Y).T @ Xtr / n
        if method == "gd":
            W -= 0.1 * G
        elif method in ["adam", "adamw"]:
            if method == "adamw":
                W *= (1 - 0.01 * 0.01)
            adam_m = 0.9 * adam_m + 0.1 * G
            adam_v = 0.999 * adam_v + 0.001 * (G * G)
            mh = adam_m / (1 - 0.9 ** t)
            vh = adam_v / (1 - 0.999 ** t)
            W -= 0.01 * mh / (np.sqrt(vh) + 1e-8)
        elif method in ["muon", "muonw", "rms_muonw"]:
            if method in ["muonw", "rms_muonw"]:
                W *= (1 - 0.1 * 0.01)
            M = 0.9 * M + 0.1 * G
            U = msign(M)
            if method == "rms_muonw":
                U = np.sqrt(max(U.shape)) * U
                lr = 0.01
                s = 1.0
            else:
                lr = 0.1
                s = np.linalg.norm(G) / (np.linalg.norm(U) + 1e-8)
            W -= lr * s * U
        elif method == "cov_muonw":
            W *= (1 - 0.1 * 0.01)
            M = 0.9 * M + 0.1 * G
            U = msign(M @ Cinv)
            s = np.linalg.norm(G) / (np.linalg.norm(U) + 1e-8)
            W -= 0.1 * s * U
        else:
            raise ValueError(method)
        history["loss"].append(ce_loss(W, Xtr, Y))
        history["train_acc"].append(accuracy(W, Xtr, ytr))
        history["test_acc"].append(accuracy(W, Xte, yte))
    return history if record else history["test_acc"][-1]


def main():
    methods = ["gd", "adam", "adamw", "muon", "muonw", "rms_muonw", "cov_muonw"]
    seeds = list(range(5))
    rows = []
    for method in methods:
        vals = [run(method, seed=s, record=False) for s in seeds]
        rows.append({
            "method": method,
            "mean": float(np.mean(vals)),
            "std": float(np.std(vals)),
            "values": [float(v) for v in vals],
        })

    csv_path = RESULT_DIR / "digits_logreg_results.csv"
    with csv_path.open("w", newline="") as f:
        writer = csv.writer(f)
        writer.writerow(["method", "mean", "std", "seed_values"])
        for row in rows:
            writer.writerow([row["method"], row["mean"], row["std"], row["values"]])

    # Seed-42 curves for figures.
    histories = {m: run(m, seed=42, record=True) for m in methods}
    x = np.arange(1, 101)
    plt.figure(figsize=(5, 3.4))
    for m, hist in histories.items():
        plt.plot(x, hist["loss"], label=m)
    plt.xlabel("Iteration")
    plt.ylabel("Training loss")
    plt.legend(fontsize=8)
    plt.tight_layout()
    plt.savefig(FIG_DIR / "training_loss.png", dpi=200)
    plt.close()

    plt.figure(figsize=(5, 3.4))
    for m, hist in histories.items():
        plt.plot(x, hist["train_acc"], label=m)
    plt.xlabel("Iteration")
    plt.ylabel("Training accuracy")
    plt.legend(fontsize=8)
    plt.tight_layout()
    plt.savefig(FIG_DIR / "training_accuracy.png", dpi=200)
    plt.close()

    for row in rows:
        print(row)


if __name__ == "__main__":
    main()
