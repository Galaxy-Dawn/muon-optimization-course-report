"""Newton-Schulz approximation diagnostics for the matrix sign."""
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

ROOT = Path(__file__).resolve().parents[1]
RESULT_DIR = ROOT / "results"
FIG_DIR = ROOT / "figures"
RESULT_DIR.mkdir(exist_ok=True)
FIG_DIR.mkdir(exist_ok=True)


def make_matrix(n, kappa, seed=0):
    rng = np.random.default_rng(seed)
    Q1, _ = np.linalg.qr(rng.standard_normal((n,n)))
    Q2, _ = np.linalg.qr(rng.standard_normal((n,n)))
    s = np.geomspace(1.0, 1.0/kappa, n)
    return Q1 @ np.diag(s) @ Q2.T


def msign(M):
    U, _, Vt = np.linalg.svd(M, full_matrices=False)
    return U @ Vt


def ns_iter(M, steps):
    # Cubic Newton-Schulz for polar factor; normalize by spectral norm.
    X = M / (np.linalg.norm(M, 2) + 1e-12)
    for _ in range(steps):
        X = 1.5 * X - 0.5 * X @ X.T @ X
    return X


def metrics(X, P):
    n = X.shape[1]
    align = np.sum(X*P) / (np.linalg.norm(X, 'fro') * np.linalg.norm(P, 'fro') + 1e-12)
    orth = np.linalg.norm(X.T @ X - np.eye(n), 'fro') / np.sqrt(n)
    s = np.linalg.svd(X, compute_uv=False)
    spread = float(np.max(s) - np.min(s))
    return float(align), float(orth), spread


def run():
    rows=[]
    for kappa in [1, 10, 100, 1000]:
        M = make_matrix(32, kappa, seed=123)
        P = msign(M)
        for steps in [1, 3, 5, 8]:
            X = ns_iter(M, steps)
            align, orth, spread = metrics(X, P)
            rows.append({"kappa": kappa, "steps": steps, "alignment": align, "orth_error": orth, "sv_spread": spread})
    return rows


def main():
    rows=run()
    path=RESULT_DIR/"ns_approx_quality.csv"
    with path.open('w', newline='') as f:
        writer=csv.DictWriter(f, fieldnames=list(rows[0].keys()))
        writer.writeheader(); writer.writerows(rows)
    for row in rows:
        print(row)
    kappas = sorted({r["kappa"] for r in rows})
    plt.figure(figsize=(6,3.6))
    for kappa in kappas:
        g = [r for r in rows if r["kappa"] == kappa]
        plt.plot([r["steps"] for r in g], [r["alignment"] for r in g], marker="o", label=f"kappa={kappa}")
    plt.xlabel("Newton-Schulz steps")
    plt.ylabel("Alignment with exact polar factor")
    plt.ylim(0.55, 1.02)
    plt.title("Newton-Schulz approximation alignment")
    plt.legend()
    plt.tight_layout()
    plt.savefig(FIG_DIR/"ns_alignment.png", dpi=200)
    plt.close()
    plt.figure(figsize=(6,3.6))
    for kappa in kappas:
        g = [r for r in rows if r["kappa"] == kappa]
        plt.semilogy([r["steps"] for r in g], [r["orth_error"] for r in g], marker="o", label=f"kappa={kappa}")
    plt.xlabel("Newton-Schulz steps")
    plt.ylabel("Orthogonality error")
    plt.title("Newton-Schulz orthogonality error")
    plt.legend()
    plt.tight_layout()
    plt.savefig(FIG_DIR/"ns_orth_error.png", dpi=200)
    plt.close("all")

if __name__ == "__main__":
    main()
