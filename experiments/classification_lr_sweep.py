"""Validation-selected classification diagnostics on scikit-learn Digits.

This script complements the fixed-hyperparameter diagnostics.  It uses
train/validation/test splits, fits the scaler on the training split only, and
selects a learning rate for each optimizer from a small fixed grid using
validation accuracy.  The goal is not to tune for state of the art; it is to
check whether the qualitative optimizer conclusions survive a transparent
validation protocol.
"""
import os

os.environ.setdefault("OPENBLAS_NUM_THREADS", "1")
os.environ.setdefault("OMP_NUM_THREADS", "1")
os.environ.setdefault("MKL_NUM_THREADS", "1")

import csv
from collections import Counter
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


METHODS = ["sgd", "adam", "adamw", "muon", "muonw", "rms_muonw", "cov_muonw"]
LABELS = {
    "sgd": "GD/SGD",
    "adam": "Adam",
    "adamw": "AdamW",
    "muon": "Muon",
    "muonw": "MuonW",
    "rms_muonw": "RMS-MuonW",
    "cov_muonw": "Cov-MuonW",
}
LR_GRIDS = {
    "sgd": [0.02, 0.05, 0.1, 0.2],
    "adam": [0.003, 0.01, 0.03],
    "adamw": [0.003, 0.01, 0.03],
    "muon": [0.3, 1.0, 3.0],
    "muonw": [0.3, 1.0, 3.0],
    "rms_muonw": [0.005, 0.01, 0.02, 0.05],
    "cov_muonw": [0.3, 1.0, 3.0],
}


def load_split(seed: int):
    X, y = load_digits(return_X_y=True)
    X_trainval, X_test, y_trainval, y_test = train_test_split(
        X, y, test_size=0.2, random_state=seed, stratify=y
    )
    X_train, X_val, y_train, y_val = train_test_split(
        X_trainval, y_trainval, test_size=0.25, random_state=seed, stratify=y_trainval
    )
    scaler = StandardScaler().fit(X_train)
    return (
        scaler.transform(X_train).astype(np.float64),
        scaler.transform(X_val).astype(np.float64),
        scaler.transform(X_test).astype(np.float64),
        y_train.astype(int),
        y_val.astype(int),
        y_test.astype(int),
    )


def softmax(Z):
    Z = Z - Z.max(axis=1, keepdims=True)
    E = np.exp(Z)
    return E / E.sum(axis=1, keepdims=True)


def msign(M):
    U, s, Vt = np.linalg.svd(M, full_matrices=False)
    r = int(np.sum(s > 1e-12))
    if r == 0:
        return np.zeros_like(M)
    return U[:, :r] @ Vt[:r, :]


def covariance_inverse(Z, damping=1e-4):
    d = Z.shape[1]
    return np.linalg.inv((Z.T @ Z) / Z.shape[0] + damping * np.eye(d))


def logreg_accuracy(W, X, y):
    return float(np.mean((X @ W.T).argmax(axis=1) == y))


def run_logreg(method: str, seed: int, lr: float, steps: int = 100):
    Xtr, Xval, Xte, ytr, yval, yte = load_split(seed)
    n, d = Xtr.shape
    k = 10
    Y = np.eye(k)[ytr]
    Cinv = covariance_inverse(Xtr)
    W = np.zeros((k, d), dtype=np.float64)
    M = np.zeros_like(W)
    adam_m = np.zeros_like(W)
    adam_v = np.zeros_like(W)
    for t in range(1, steps + 1):
        P = softmax(Xtr @ W.T)
        G = (P - Y).T @ Xtr / n
        if method == "sgd":
            W -= lr * G
        elif method in ["adam", "adamw"]:
            if method == "adamw":
                W *= 1.0 - lr * 0.01
            adam_m = 0.9 * adam_m + 0.1 * G
            adam_v = 0.999 * adam_v + 0.001 * (G * G)
            mh = adam_m / (1 - 0.9**t)
            vh = adam_v / (1 - 0.999**t)
            W -= lr * mh / (np.sqrt(vh) + 1e-8)
        elif method in ["muon", "muonw", "rms_muonw"]:
            if method in ["muonw", "rms_muonw"]:
                W *= 1.0 - lr * 0.01
            M = 0.9 * M + 0.1 * G
            D = msign(M)
            if method == "rms_muonw":
                D = np.sqrt(max(D.shape)) * D
                scale = 1.0
            else:
                scale = np.linalg.norm(G) / (np.linalg.norm(D) + 1e-8)
            W -= lr * scale * D
        elif method == "cov_muonw":
            W *= 1.0 - lr * 0.01
            M = 0.9 * M + 0.1 * G
            D = msign(M @ Cinv)
            scale = np.linalg.norm(G) / (np.linalg.norm(D) + 1e-8)
            W -= lr * scale * D
        else:
            raise ValueError(method)
    return {
        "val_acc": logreg_accuracy(W, Xval, yval),
        "test_acc": logreg_accuracy(W, Xte, yte),
        "max_logit": float(np.max(np.abs(Xtr @ W.T))),
        "W1_spec": float(np.linalg.norm(W, 2)),
        "W2_spec": 0.0,
    }


def init_mlp(seed: int, hidden: int = 32):
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


def mlp_accuracy(params, X, y):
    return float(np.mean(forward(params, X)[2].argmax(axis=1) == y))


def mlp_gradients(params, X, y):
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
    return {"W1": G1, "b1": gb1, "W2": G2, "b2": gb2}, H


def adam_mlp_update(params, grads, state, lr, t, weight_decay):
    for name in params:
        if weight_decay and name.startswith("W"):
            params[name] *= 1.0 - lr * weight_decay
        state.setdefault("m_" + name, np.zeros_like(params[name]))
        state.setdefault("v_" + name, np.zeros_like(params[name]))
        state["m_" + name] = 0.9 * state["m_" + name] + 0.1 * grads[name]
        state["v_" + name] = 0.999 * state["v_" + name] + 0.001 * (grads[name] * grads[name])
        mh = state["m_" + name] / (1 - 0.9**t)
        vh = state["v_" + name] / (1 - 0.999**t)
        params[name] -= lr * mh / (np.sqrt(vh) + 1e-8)


def muon_mlp_update(params, grads, state, Xtr, H, method, lr, weight_decay):
    Cx_inv = state.setdefault("Cx_inv", covariance_inverse(Xtr))
    Ch_inv = covariance_inverse(H)
    for name in ["W1", "W2"]:
        if method in ["muonw", "rms_muonw", "cov_muonw"]:
            params[name] *= 1.0 - lr * weight_decay
        state.setdefault("M_" + name, np.zeros_like(params[name]))
        state["M_" + name] = 0.9 * state["M_" + name] + 0.1 * grads[name]
        M = state["M_" + name]
        if method == "cov_muonw":
            Cinv = Cx_inv if name == "W1" else Ch_inv
            D = msign(M @ Cinv)
            scale = np.linalg.norm(grads[name]) / (np.linalg.norm(D) + 1e-8)
        elif method == "rms_muonw":
            D = np.sqrt(max(M.shape)) * msign(M)
            scale = 1.0
        else:
            D = msign(M)
            scale = np.linalg.norm(grads[name]) / (np.linalg.norm(D) + 1e-8)
        params[name] -= lr * scale * D
    params["b1"] -= 0.1 * grads["b1"]
    params["b2"] -= 0.1 * grads["b2"]


def run_mlp(method: str, seed: int, lr: float, epochs: int = 30):
    Xtr, Xval, Xte, ytr, yval, yte = load_split(seed)
    params = init_mlp(seed)
    state = {}
    for epoch in range(1, epochs + 1):
        grads, H = mlp_gradients(params, Xtr, ytr)
        if method == "sgd":
            for name in params:
                params[name] -= lr * grads[name]
        elif method in ["adam", "adamw"]:
            wd = 0.01 if method == "adamw" else 0.0
            adam_mlp_update(params, grads, state, lr=lr, t=epoch, weight_decay=wd)
        elif method in ["muon", "muonw", "rms_muonw", "cov_muonw"]:
            wd = 0.0 if method == "muon" else 0.01
            muon_mlp_update(params, grads, state, Xtr, H, method, lr=lr, weight_decay=wd)
        else:
            raise ValueError(method)
    return {
        "val_acc": mlp_accuracy(params, Xval, yval),
        "test_acc": mlp_accuracy(params, Xte, yte),
        "max_logit": float(np.max(np.abs(forward(params, Xtr)[2]))),
        "W1_spec": float(np.linalg.norm(params["W1"], 2)),
        "W2_spec": float(np.linalg.norm(params["W2"], 2)),
    }


def select_and_evaluate(task: str, method: str, seed: int):
    runner = run_logreg if task == "logreg" else run_mlp
    candidates = []
    for lr in LR_GRIDS[method]:
        out = runner(method, seed, lr)
        candidates.append((out["val_acc"], -lr, lr, out))
    _, _, best_lr, best = max(candidates)
    return {
        "task": task,
        "method": method,
        "seed": seed,
        "selected_lr": best_lr,
        **best,
    }


def summarize(rows):
    summary = []
    for task in ["logreg", "mlp"]:
        task_rows = [r for r in rows if r["task"] == task]
        adamw_by_seed = {
            r["seed"]: r["test_acc"] for r in task_rows if r["method"] == "adamw"
        }
        for method in METHODS:
            group = [r for r in task_rows if r["method"] == method]
            tests = np.array([r["test_acc"] for r in group])
            vals = np.array([r["val_acc"] for r in group])
            deltas = np.array([100.0 * (r["test_acc"] - adamw_by_seed[r["seed"]]) for r in group])
            lr_counts = Counter(r["selected_lr"] for r in group)
            mode_lr = sorted(lr_counts.items(), key=lambda x: (-x[1], x[0]))[0][0]
            summary.append({
                "task": task,
                "method": method,
                "val_mean": float(vals.mean()),
                "val_std": float(vals.std()),
                "test_mean": float(tests.mean()),
                "test_std": float(tests.std()),
                "delta_vs_adamw_pp": float(deltas.mean()),
                "selected_lr_mode": mode_lr,
                "max_logit_mean": float(np.mean([r["max_logit"] for r in group])),
                "W1_spec_mean": float(np.mean([r["W1_spec"] for r in group])),
                "W2_spec_mean": float(np.mean([r["W2_spec"] for r in group])),
            })
    return summary


def write_table(summary):
    path = TABLE_DIR / "classification_sweep_table.tex"
    with path.open("w") as f:
        f.write("\\footnotesize\n")
        f.write("\\begin{tabular}{llrrrr}\n")
        f.write("\\toprule\n")
        f.write("Task & Method & Val. acc. & Test acc. & $\\Delta$ AdamW & LR \\\\\n")
        f.write("\\midrule\n")
        for task in ["logreg", "mlp"]:
            for row in [r for r in summary if r["task"] == task]:
                task_name = "LogReg" if task == "logreg" else "MLP"
                f.write(
                    f"{task_name} & {LABELS[row['method']]} & "
                    f"${100*row['val_mean']:.2f} \\pm {100*row['val_std']:.2f}$ & "
                    f"${100*row['test_mean']:.2f} \\pm {100*row['test_std']:.2f}$ & "
                    f"${row['delta_vs_adamw_pp']:+.2f}$ & "
                    f"{row['selected_lr_mode']:.3g} \\\\\n"
                )
            if task == "logreg":
                f.write("\\midrule\n")
        f.write("\\bottomrule\n")
        f.write("\\end{tabular}\n")


def main():
    seeds = list(range(5))
    rows = []
    for task in ["logreg", "mlp"]:
        for method in METHODS:
            print(f"Running {task} {method}...", flush=True)
            for seed in seeds:
                rows.append(select_and_evaluate(task, method, seed))

    raw_path = RESULT_DIR / "classification_lr_sweep_raw.csv"
    with raw_path.open("w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=list(rows[0].keys()))
        writer.writeheader()
        writer.writerows(rows)

    summary = summarize(rows)
    summary_path = RESULT_DIR / "classification_lr_sweep_summary.csv"
    with summary_path.open("w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=list(summary[0].keys()))
        writer.writeheader()
        writer.writerows(summary)

    write_table(summary)
    for row in summary:
        print(row)


if __name__ == "__main__":
    main()
