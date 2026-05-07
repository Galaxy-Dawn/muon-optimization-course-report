"""Small full-batch MLP optimizer ablation on scikit-learn Digits.

This diagnostic compares Adam/AdamW with several Muon-style variants.  Inputs
are row vectors and weight matrices are stored as W in R^{out x in}.  Matrix
methods are applied only to weight matrices; bias vectors use plain gradient
steps for Muon variants.  The Newton--Muon variants use oracle full-batch
covariances, so they are diagnostic rather than scalable implementations.
"""
import os
os.environ.setdefault("OPENBLAS_NUM_THREADS", "1")
os.environ.setdefault("OMP_NUM_THREADS", "1")
os.environ.setdefault("MKL_NUM_THREADS", "1")

import csv
import sys
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


def softmax(Z):
    Z = Z - Z.max(axis=1, keepdims=True)
    E = np.exp(Z)
    return E / E.sum(axis=1, keepdims=True)


def msign(M):
    U, _, Vt = np.linalg.svd(M, full_matrices=False)
    return U @ Vt


def covariance_inverse(Z, damping=1e-4):
    d = Z.shape[1]
    return np.linalg.inv((Z.T @ Z) / Z.shape[0] + damping * np.eye(d))


def load_data(seed):
    X, y = load_digits(return_X_y=True)
    Xtr_raw, Xte_raw, ytr, yte = train_test_split(
        X, y, test_size=0.2, random_state=seed, stratify=y
    )
    scaler = StandardScaler().fit(Xtr_raw)
    Xtr = scaler.transform(Xtr_raw).astype(np.float64)
    Xte = scaler.transform(Xte_raw).astype(np.float64)
    return Xtr, Xte, ytr.astype(int), yte.astype(int)


def init_params(seed, hidden=32):
    rng = np.random.default_rng(seed)
    W1 = rng.standard_normal((hidden, 64)) / np.sqrt(64)
    b1 = np.zeros(hidden)
    W2 = rng.standard_normal((10, hidden)) / np.sqrt(hidden)
    b2 = np.zeros(10)
    return {"W1": W1, "b1": b1, "W2": W2, "b2": b2}


def forward(params, X):
    Z1 = X @ params["W1"].T + params["b1"]
    H = np.maximum(Z1, 0.0)
    logits = H @ params["W2"].T + params["b2"]
    return Z1, H, logits


def accuracy(params, X, y):
    _, _, logits = forward(params, X)
    return np.mean(logits.argmax(axis=1) == y)


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
    return {"W1": G1, "b1": gb1, "W2": G2, "b2": gb2}, H, loss


def adam_update(params, grads, state, lr=0.01, t=1, weight_decay=0.0):
    for k in params:
        g = grads[k]
        if weight_decay and k.startswith("W"):
            params[k] *= (1 - lr * weight_decay)
        state.setdefault("m_"+k, np.zeros_like(params[k]))
        state.setdefault("v_"+k, np.zeros_like(params[k]))
        state["m_"+k] = 0.9 * state["m_"+k] + 0.1 * g
        state["v_"+k] = 0.999 * state["v_"+k] + 0.001 * (g*g)
        mh = state["m_"+k] / (1 - 0.9**t)
        vh = state["v_"+k] / (1 - 0.999**t)
        params[k] -= lr * mh / (np.sqrt(vh) + 1e-8)


def muon_update(params, grads, state, Xtr, H, method, lr=1.0, weight_decay=0.01):
    Cx_inv = state.setdefault("Cx_inv", covariance_inverse(Xtr))
    Ch_inv = covariance_inverse(H)
    for name in ["W1", "W2"]:
        if method in ["muonw", "rms_muonw", "newton_muonw"]:
            params[name] *= (1 - lr * weight_decay)
        state.setdefault("M_"+name, np.zeros_like(params[name]))
        state["M_"+name] = 0.9 * state["M_"+name] + 0.1 * grads[name]
        M = state["M_"+name]
        if method == "newton_muonw":
            Cinv = Cx_inv if name == "W1" else Ch_inv
            D = msign(M @ Cinv)
            scale = np.linalg.norm(grads[name]) / (np.linalg.norm(D) + 1e-8)
        elif method == "rms_muonw":
            D = np.sqrt(max(M.shape)) * msign(M)
            scale = 1.0
            # RMS-scaled direction can be large; use smaller base lr.
        else:
            D = msign(M)
            scale = np.linalg.norm(grads[name]) / (np.linalg.norm(D) + 1e-8)
        params[name] -= lr * scale * D
    # Biases use plain gradient steps for the Muon family.
    params["b1"] -= 0.1 * grads["b1"]
    params["b2"] -= 0.1 * grads["b2"]


def run(method, seed=0, epochs=30, record=False):
    Xtr, Xte, ytr, yte = load_data(seed)
    params = init_params(seed)
    state = {}
    hist = {"train_acc": [], "test_acc": [], "loss": [], "max_logit": []}
    lr_map = {
        "sgd": 0.05,
        "adam": 0.01,
        "adamw": 0.01,
        "muon": 1.0,
        "muonw": 1.0,
        "rms_muonw": 0.02,
        "newton_muonw": 1.0,
    }
    for epoch in range(1, epochs+1):
        grads, H, loss = gradients(params, Xtr, ytr)
        if method == "sgd":
            for k in params:
                params[k] -= lr_map[method] * grads[k]
        elif method == "adam":
            adam_update(params, grads, state, lr=lr_map[method], t=epoch, weight_decay=0.0)
        elif method == "adamw":
            adam_update(params, grads, state, lr=lr_map[method], t=epoch, weight_decay=0.01)
        elif method in ["muon", "muonw", "rms_muonw", "newton_muonw"]:
            wd = 0.0 if method == "muon" else 0.01
            muon_update(params, grads, state, Xtr, H, method, lr=lr_map[method], weight_decay=wd)
        else:
            raise ValueError(method)
        hist["loss"].append(float(loss))
        hist["train_acc"].append(float(accuracy(params, Xtr, ytr)))
        hist["test_acc"].append(float(accuracy(params, Xte, yte)))
        _, _, logits = forward(params, Xtr)
        hist["max_logit"].append(float(np.max(np.abs(logits))))
    if record:
        return hist
    return {
        "test_acc": hist["test_acc"][-1],
        "max_logit": hist["max_logit"][-1],
        "W1_spec": float(np.linalg.norm(params["W1"], 2)),
        "W2_spec": float(np.linalg.norm(params["W2"], 2)),
    }


def main():
    methods = ["sgd", "adam", "adamw", "muon", "muonw", "rms_muonw", "newton_muonw"]
    seeds = list(range(5))
    rows = []
    stability_rows = []
    for method in methods:
        print(f"Running {method}...", flush=True)
        summaries = [run(method, seed=s, record=False) for s in seeds]
        vals = [s["test_acc"] for s in summaries]
        rows.append({"method": method, "mean": float(np.mean(vals)), "std": float(np.std(vals)), "values": [float(v) for v in vals]})
        stability_rows.append({
            "method": method,
            "max_logit_mean": float(np.mean([s["max_logit"] for s in summaries])),
            "max_logit_std": float(np.std([s["max_logit"] for s in summaries])),
            "W1_spec_mean": float(np.mean([s["W1_spec"] for s in summaries])),
            "W1_spec_std": float(np.std([s["W1_spec"] for s in summaries])),
            "W2_spec_mean": float(np.mean([s["W2_spec"] for s in summaries])),
            "W2_spec_std": float(np.std([s["W2_spec"] for s in summaries])),
        })
    with (RESULT_DIR / "digits_mlp_results.csv").open("w", newline="") as f:
        writer = csv.writer(f)
        writer.writerow(["method", "mean", "std", "seed_values"])
        for row in rows:
            writer.writerow([row["method"], row["mean"], row["std"], row["values"]])
    with (RESULT_DIR / "digits_mlp_stability.csv").open("w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=list(stability_rows[0].keys()))
        writer.writeheader(); writer.writerows(stability_rows)

    plot_methods = ["sgd", "adam", "adamw", "muon", "muonw", "rms_muonw", "newton_muonw"]
    histories = {m: run(m, seed=42, record=True) for m in plot_methods}
    x = np.arange(1, 31)
    plt.figure(figsize=(5, 3.4))
    for m, hist in histories.items():
        plt.plot(x, hist["train_acc"], label=m)
    plt.xlabel("Epoch"); plt.ylabel("Training accuracy"); plt.legend(fontsize=6)
    plt.tight_layout(); plt.savefig(FIG_DIR / "mlp_train_acc.png", dpi=200); plt.close()

    plt.figure(figsize=(5, 3.4))
    for m, hist in histories.items():
        plt.plot(x, hist["test_acc"], label=m)
    plt.xlabel("Epoch"); plt.ylabel("Test accuracy"); plt.legend(fontsize=6)
    plt.tight_layout(); plt.savefig(FIG_DIR / "mlp_test_acc.png", dpi=200); plt.close()

    plt.figure(figsize=(5, 3.4))
    for m, hist in histories.items():
        plt.plot(x, hist["max_logit"], label=m)
    plt.xlabel("Epoch"); plt.ylabel("Max absolute logit"); plt.legend(fontsize=6)
    plt.tight_layout(); plt.savefig(FIG_DIR / "mlp_max_logit.png", dpi=200); plt.close()

    for row in rows:
        print(row)
    print("Stability summaries:")
    for row in stability_rows:
        print(row)

if __name__ == "__main__":
    main()
    plt.close("all")
    sys.exit(0)
