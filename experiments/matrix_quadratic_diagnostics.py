"""Matrix-quadratic diagnostics for preconditioning and exact msign.

Objective: f(W)=1/2 || A W B - C ||_F^2 with C=A W* B.
The script compares GD, oracle diagonal preconditioning, ideal left--right
preconditioning, and a scaled exact matrix-sign update. It saves iteration
counts to reach a fixed tolerance. Runs marked with budget+1 did not converge
within the budget or diverged numerically.
"""
import os
os.environ.setdefault("OPENBLAS_NUM_THREADS", "1")
os.environ.setdefault("OMP_NUM_THREADS", "1")
os.environ.setdefault("MKL_NUM_THREADS", "1")

import csv
from pathlib import Path
import numpy as np

ROOT = Path(__file__).resolve().parents[1]
RESULT_DIR = ROOT / "results"
RESULT_DIR.mkdir(exist_ok=True)


def orthogonal(n, rng):
    q, _ = np.linalg.qr(rng.standard_normal((n, n)))
    return q


def make_factor(n, kappa, rotated, rng):
    s = np.geomspace(1.0, float(kappa), n)
    factor = np.diag(s)
    if rotated:
        return orthogonal(n, rng) @ factor @ orthogonal(n, rng).T
    return factor


def msign(M):
    U, s, Vt = np.linalg.svd(M, full_matrices=False)
    r = int(np.sum(s > 1e-12))
    if r == 0:
        return np.zeros_like(M)
    return U[:, :r] @ Vt[:r, :]


def run_one(kappa, rotated, method, n=5, tol=1e-8, max_steps=5000, seed=0):
    rng = np.random.default_rng(seed)
    A = make_factor(n, kappa, rotated, rng)
    B = make_factor(n, kappa, rotated, rng)
    W_star = rng.standard_normal((n, n))
    C = A @ W_star @ B
    W = np.zeros_like(W_star)
    AtA = A.T @ A
    BBt = B @ B.T
    H = np.kron(BBt, AtA)

    if method == "gd":
        eig = np.linalg.eigvalsh(H)
        eta = 2.0 / (eig.min() + eig.max())
    elif method == "diag":
        # Column-major convention to match vec(L X R)=(R^T \otimes L)vec(X).
        diag_h = np.diag(H)
        Dinv = 1.0 / diag_h
        S = np.diag(1.0 / np.sqrt(diag_h)) @ H @ np.diag(1.0 / np.sqrt(diag_h))
        eig = np.linalg.eigvalsh(S)
        eta = 2.0 / (eig.min() + eig.max())
    elif method == "lr":
        eta = 1.0
    elif method == "scaled_msign":
        eta = 0.02
    else:
        raise ValueError(method)

    def objective(W_):
        with np.errstate(over="ignore", invalid="ignore"):
            R = A @ W_ @ B - C
            val = 0.5 * np.linalg.norm(R, ord="fro") ** 2
        return val

    for step in range(max_steps + 1):
        val = objective(W)
        if val < tol:
            return step
        if (not np.isfinite(val)) or np.linalg.norm(W, ord="fro") > 1e100:
            return max_steps + 1
        with np.errstate(over="ignore", invalid="ignore"):
            G = AtA @ (W - W_star) @ BBt
            if not np.all(np.isfinite(G)):
                return max_steps + 1
            if method == "gd":
                W = W - eta * G
            elif method == "diag":
                vec = W.reshape(-1, order="F") - eta * (Dinv * G.reshape(-1, order="F"))
                W = vec.reshape(n, n, order="F")
            elif method == "lr":
                W = W - eta * np.linalg.solve(AtA, G) @ np.linalg.inv(BBt)
            elif method == "scaled_msign":
                U = msign(G)
                scale = np.linalg.norm(G, ord="fro") / (np.linalg.norm(U, ord="fro") + 1e-12)
                W = W - eta * scale * U
    return max_steps + 1


def main():
    max_steps = 5000
    rows = []
    for rotated in [False, True]:
        for kappa in [1, 3, 10]:
            row = {
                "rotated": rotated,
                "kappa_A_B": kappa,
                "kappa_H": int(kappa ** 4),
                "budget": max_steps,
            }
            for method in ["gd", "diag", "lr", "scaled_msign"]:
                row[method] = run_one(kappa, rotated, method, max_steps=max_steps)
            rows.append(row)

    csv_path = RESULT_DIR / "matrix_quadratic_diagnostics.csv"
    with csv_path.open("w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=list(rows[0].keys()))
        writer.writeheader()
        writer.writerows(rows)

    for row in rows:
        print(row)


if __name__ == "__main__":
    main()
