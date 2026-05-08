"""Matrix-quadratic diagnostics for preconditioning and exact msign.

Objective: f(W)=1/2 || A W B - C ||_F^2 with C=A W* B.  The
script compares GD, oracle diagonal preconditioning, ideal left--right
preconditioning, and a scaled exact matrix-sign update.  It now uses a
relative objective gap criterion f(W_t)/f(W_0) < tol and records final
relative objective gap and parameter error for each method.
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
TABLE_DIR = ROOT / "tables"
RESULT_DIR.mkdir(exist_ok=True)
TABLE_DIR.mkdir(exist_ok=True)

METHODS = ["gd", "diag", "lr", "scaled_msign"]
METHOD_LABELS = {
    "gd": "GD",
    "diag": "Diag.",
    "lr": "LR",
    "scaled_msign": "s-\\texttt{msign}",
}


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


def run_one(kappa, rotated, method, n=5, rel_tol=1e-8, max_steps=5000, seed=0):
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
            return 0.5 * np.linalg.norm(R, ord="fro") ** 2

    f0 = objective(W)
    denom = max(f0, np.finfo(float).tiny)
    wstar_norm = max(np.linalg.norm(W_star, ord="fro"), np.finfo(float).tiny)
    final_step = max_steps + 1
    final_rel_gap = np.inf
    final_param_err = np.inf

    for step in range(max_steps + 1):
        val = objective(W)
        rel_gap = val / denom
        param_err = np.linalg.norm(W - W_star, ord="fro") / wstar_norm
        final_step = step
        final_rel_gap = rel_gap
        final_param_err = param_err
        if rel_gap < rel_tol:
            return {
                "iterations": step,
                "final_rel_gap": float(rel_gap),
                "final_param_rel_error": float(param_err),
            }
        if (not np.isfinite(val)) or np.linalg.norm(W, ord="fro") > 1e100:
            return {
                "iterations": max_steps + 1,
                "final_rel_gap": float(rel_gap) if np.isfinite(rel_gap) else np.inf,
                "final_param_rel_error": float(param_err) if np.isfinite(param_err) else np.inf,
            }
        with np.errstate(over="ignore", invalid="ignore"):
            G = AtA @ (W - W_star) @ BBt
            if not np.all(np.isfinite(G)):
                return {
                    "iterations": max_steps + 1,
                    "final_rel_gap": float(final_rel_gap),
                    "final_param_rel_error": float(final_param_err),
                }
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

    return {
        "iterations": max_steps + 1,
        "final_rel_gap": float(final_rel_gap),
        "final_param_rel_error": float(final_param_err),
    }


def fmt_iters(value, budget=5000):
    value = int(value)
    return "$>5000$" if value > budget else str(value)


def fmt_sci(value):
    value = float(value)
    if not np.isfinite(value):
        return "$\\infty$"
    if value == 0:
        return "$0$"
    exponent = int(np.floor(np.log10(abs(value))))
    mantissa = value / (10 ** exponent)
    return f"${mantissa:.1f}\\times10^{{{exponent}}}$"


def write_tables(wide_rows, long_rows, max_steps):
    # Main compact iteration table.
    with (TABLE_DIR / "matrix_diagnostics_table.tex").open("w") as f:
        f.write("\\footnotesize\n")
        f.write("\\begin{tabular}{llrrrrr}\n")
        f.write("\\toprule\n")
        f.write("Rot. & $\\kappa$ & $\\kappa_H$ & GD & Diag. & LR & s-\\texttt{msign} \\\\\n")
        f.write("\\midrule\n")
        for row in wide_rows:
            rot = "Yes" if row["rotated"] else "No"
            vals = [int(row[m]) for m in METHODS]
            min_conv = min([v for v in vals if v <= max_steps] or vals)
            cells = []
            for method, val in zip(METHODS, vals):
                cell = fmt_iters(val, max_steps)
                if val == min_conv and val <= max_steps:
                    cell = f"\\textbf{{{cell}}}"
                cells.append(cell)
            f.write(
                f"{rot} & {row['kappa_A_B']} & {row['kappa_H']} & "
                + " & ".join(cells)
                + " \\\\\n"
            )
        f.write("\\bottomrule\n")
        f.write("\\end{tabular}\n")

    # Appendix table with final relative residuals and parameter errors.
    with (TABLE_DIR / "matrix_final_diagnostics_table.tex").open("w") as f:
        f.write("\\scriptsize\n")
        f.write("\\setlength{\\tabcolsep}{3pt}\n")
        f.write("\\begin{tabular}{llrrrr}\n")
        f.write("\\toprule\n")
        f.write("Rot. & Method & $\\kappa$ & Iter. & $e_f$ & $e_W$ \\\\\n")
        f.write("\\midrule\n")
        for r in long_rows:
            rot = "Yes" if r["rotated"] else "No"
            f.write(
                f"{rot} & {METHOD_LABELS[r['method']]} & {r['kappa_A_B']} & "
                f"{fmt_iters(r['iterations'], max_steps)} & {fmt_sci(r['final_rel_gap'])} & "
                f"{fmt_sci(r['final_param_rel_error'])} \\\\\n"
            )
        f.write("\\bottomrule\n")
        f.write("\\end{tabular}\n")


def main():
    max_steps = 5000
    rel_tol = 1e-8
    wide_rows = []
    long_rows = []
    for rotated in [False, True]:
        for kappa in [1, 3, 10]:
            wide = {
                "rotated": rotated,
                "kappa_A_B": kappa,
                "kappa_H": int(kappa ** 4),
                "rel_tol": rel_tol,
                "budget": max_steps,
            }
            for method in METHODS:
                out = run_one(kappa, rotated, method, rel_tol=rel_tol, max_steps=max_steps)
                wide[method] = out["iterations"]
                wide[f"{method}_rel_gap"] = out["final_rel_gap"]
                wide[f"{method}_param_rel_error"] = out["final_param_rel_error"]
                long_rows.append({
                    "rotated": rotated,
                    "kappa_A_B": kappa,
                    "kappa_H": int(kappa ** 4),
                    "rel_tol": rel_tol,
                    "budget": max_steps,
                    "method": method,
                    **out,
                })
            wide_rows.append(wide)

    wide_path = RESULT_DIR / "matrix_quadratic_diagnostics.csv"
    with wide_path.open("w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=list(wide_rows[0].keys()))
        writer.writeheader()
        writer.writerows(wide_rows)

    long_path = RESULT_DIR / "matrix_quadratic_final_diagnostics.csv"
    with long_path.open("w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=list(long_rows[0].keys()))
        writer.writeheader()
        writer.writerows(long_rows)

    write_tables(wide_rows, long_rows, max_steps)
    for row in wide_rows:
        print(row)


if __name__ == "__main__":
    main()
