"""Finite Newton--Schulz MuonW training diagnostic on scikit-learn Digits.

This script places the finite Newton--Schulz polar approximation inside the
training loop.  It compares K in {1, 3, 5, 8} for a fixed full-batch MLP
protocol.  The experiment is diagnostic rather than a tuned benchmark: it uses
MuonW-style decoupled decay, momentum, Frobenius-norm scaling of the approximate
polar direction, and fixed-step gradient updates for biases.
"""
import os
os.environ.setdefault("OPENBLAS_NUM_THREADS", "1")
os.environ.setdefault("OMP_NUM_THREADS", "1")
os.environ.setdefault("MKL_NUM_THREADS", "1")

import csv
import time
from pathlib import Path

import numpy as np
from sklearn.datasets import load_digits
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler

ROOT = Path(__file__).resolve().parents[1]
RESULT_DIR = ROOT / "results"
TABLE_DIR = ROOT / "tables"
RESULT_DIR.mkdir(exist_ok=True)
TABLE_DIR.mkdir(exist_ok=True)

K_VALUES = [1, 3, 5, 8]


def load_data(seed):
    X, y = load_digits(return_X_y=True)
    Xtr, Xte, ytr, yte = train_test_split(
        X, y, test_size=0.2, random_state=seed, stratify=y
    )
    scaler = StandardScaler().fit(Xtr)
    return scaler.transform(Xtr).astype(np.float64), scaler.transform(Xte).astype(np.float64), ytr.astype(int), yte.astype(int)


def softmax(Z):
    Z = Z - Z.max(axis=1, keepdims=True)
    E = np.exp(Z)
    return E / E.sum(axis=1, keepdims=True)


def init_params(seed, hidden=32):
    rng = np.random.default_rng(seed)
    return {
        "W1": rng.standard_normal((hidden, 64)) / np.sqrt(64),
        "b1": np.zeros(hidden),
        "W2": rng.standard_normal((10, hidden)) / np.sqrt(hidden),
        "b2": np.zeros(10),
    }


def forward(params, X):
    Z1 = X @ params["W1"].T + params["b1"]
    H = np.maximum(Z1, 0.0)
    logits = H @ params["W2"].T + params["b2"]
    return Z1, H, logits


def accuracy(params, X, y):
    return float(np.mean(forward(params, X)[2].argmax(axis=1) == y))


def gradients(params, X, y):
    n = X.shape[0]
    Z1, H, logits = forward(params, X)
    P = softmax(logits)
    Y = np.eye(10)[y]
    dlogits = (P - Y) / n
    G2 = dlogits.T @ H
    gb2 = dlogits.sum(axis=0)
    dH = dlogits @ params["W2"]
    dZ1 = dH * (Z1 > 0)
    G1 = dZ1.T @ X
    gb1 = dZ1.sum(axis=0)
    loss = -np.mean(np.sum(Y * np.log(P + 1e-12), axis=1))
    return {"W1": G1, "b1": gb1, "W2": G2, "b2": gb2}, float(loss)


def ns_polar(M, steps):
    """Rectangular cubic Newton--Schulz approximation of the polar factor.

    The initial Frobenius normalization avoids an SVD inside the approximation.
    If M is m x n with m <= n, the target has approximately orthonormal rows;
    if m >= n, the target has approximately orthonormal columns.
    """
    norm = np.linalg.norm(M, ord="fro")
    if norm <= 1e-12:
        return np.zeros_like(M)
    X = M / norm
    for _ in range(steps):
        X = 1.5 * X - 0.5 * (X @ X.T @ X)
    return X


def orthogonality_error(X):
    m, n = X.shape
    if min(m, n) == 0:
        return 0.0
    if m <= n:
        E = X @ X.T - np.eye(m)
        denom = np.sqrt(m)
    else:
        E = X.T @ X - np.eye(n)
        denom = np.sqrt(n)
    return float(np.linalg.norm(E, ord="fro") / denom)


def run(seed, ns_steps, epochs=30, lr=1.0, weight_decay=0.01, momentum=0.9, bias_lr=0.1):
    Xtr, Xte, ytr, yte = load_data(seed)
    params = init_params(seed)
    state = {"M_W1": np.zeros_like(params["W1"]), "M_W2": np.zeros_like(params["W2"])}
    orth_errors = []
    t0 = time.perf_counter()
    train_loss = np.nan
    for _ in range(epochs):
        grads, train_loss = gradients(params, Xtr, ytr)
        for name in ["W1", "W2"]:
            params[name] *= 1.0 - lr * weight_decay
            state["M_" + name] = momentum * state["M_" + name] + (1.0 - momentum) * grads[name]
            D = ns_polar(state["M_" + name], ns_steps)
            orth_errors.append(orthogonality_error(D))
            scale = np.linalg.norm(grads[name], ord="fro") / (np.linalg.norm(D, ord="fro") + 1e-8)
            params[name] -= lr * scale * D
        # Biases remain plain-GD to match the Muon-family MLP diagnostic.
        params["b1"] -= bias_lr * grads["b1"]
        params["b2"] -= bias_lr * grads["b2"]
    runtime = time.perf_counter() - t0
    final_grads, final_loss = gradients(params, Xtr, ytr)
    return {
        "test_acc": accuracy(params, Xte, yte),
        "train_loss": float(final_loss),
        "orth_error": float(np.mean(orth_errors)),
        "runtime_sec": float(runtime),
        "W1_spec": float(np.linalg.norm(params["W1"], 2)),
        "W2_spec": float(np.linalg.norm(params["W2"], 2)),
    }


def summarize(rows):
    out = []
    for k in K_VALUES:
        group = [r for r in rows if int(r["ns_steps"]) == k]
        summary = {"ns_steps": k}
        for key in ["test_acc", "train_loss", "orth_error", "runtime_sec", "W1_spec", "W2_spec"]:
            vals = np.array([float(r[key]) for r in group], dtype=float)
            summary[key + "_mean"] = float(vals.mean())
            summary[key + "_std"] = float(vals.std())
        out.append(summary)
    return out


def write_table(summary):
    path = TABLE_DIR / "finite_ns_training_table.tex"
    # best accuracy: max; best loss/error/runtime: min
    best_acc = max(r["test_acc_mean"] for r in summary)
    best_loss = min(r["train_loss_mean"] for r in summary)
    best_orth = min(r["orth_error_mean"] for r in summary)
    best_time = min(r["runtime_sec_mean"] for r in summary)
    def bold(val, best, fmt, higher=False):
        txt = fmt.format(val)
        ok = abs(val - best) <= 1e-12
        return f"\\textbf{{{txt}}}" if ok else txt
    with path.open("w") as f:
        f.write("\\footnotesize\n")
        f.write("\\begin{tabular}{rrrrr}\n")
        f.write("\\toprule\n")
        f.write("$K$ & Test acc. (\\%) & Train loss & Orth. error & Runtime (s) \\\\\n")
        f.write("\\midrule\n")
        for r in summary:
            f.write(
                f"{r['ns_steps']} & "
                f"{bold(100*r['test_acc_mean'], 100*best_acc, '{:.2f}')} $\\pm$ {100*r['test_acc_std']:.2f} & "
                f"{bold(r['train_loss_mean'], best_loss, '{:.4f}')} & "
                f"{bold(r['orth_error_mean'], best_orth, '{:.4f}')} & "
                f"{bold(r['runtime_sec_mean'], best_time, '{:.4f}')} \\\\\n"
            )
        f.write("\\bottomrule\n")
        f.write("\\end{tabular}\n")


def main():
    seeds = list(range(5))
    rows = []
    for k in K_VALUES:
        for seed in seeds:
            print(f"Running NS-MuonW K={k}, seed={seed}...", flush=True)
            out = run(seed, k)
            rows.append({"ns_steps": k, "seed": seed, **out})
    with (RESULT_DIR / "finite_ns_training_raw.csv").open("w", newline="") as f:
        fieldnames = list(rows[0].keys())
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader(); writer.writerows(rows)
    summary = summarize(rows)
    with (RESULT_DIR / "finite_ns_training_summary.csv").open("w", newline="") as f:
        fieldnames = list(summary[0].keys())
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader(); writer.writerows(summary)
    write_table(summary)
    for row in summary:
        print(row)


if __name__ == "__main__":
    main()
