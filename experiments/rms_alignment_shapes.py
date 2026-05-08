"""Numerical SVD check of RMS alignment for matrix-sign updates.

The algebraic identity is RMS(msign(M)) = sqrt(rank(M)/(m n)).  Unlike the
previous table-only check, this script forms Gaussian matrices, computes the
exact SVD polar factor, and compares the observed RMS to the theoretical value.
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

ROOT = Path(__file__).resolve().parents[1]
RESULT_DIR = ROOT / "results"
FIG_DIR = ROOT / "figures"
TABLE_DIR = ROOT / "tables"
RESULT_DIR.mkdir(exist_ok=True)
FIG_DIR.mkdir(exist_ok=True)
TABLE_DIR.mkdir(exist_ok=True)


def msign(M, tol=1e-12):
    U, s, Vt = np.linalg.svd(M, full_matrices=False)
    r = int(np.sum(s > tol))
    if r == 0:
        return np.zeros_like(M), r
    return U[:, :r] @ Vt[:r, :], r


def run(num_seeds=1):
    shapes = [(16, 16), (16, 64), (64, 16), (32, 128), (128, 32), (64, 256)]
    rows = []
    for m, n in shapes:
        raw_values = []
        scaled_values = []
        theory_values = []
        ranks = []
        for seed in range(num_seeds):
            rng = np.random.default_rng(10_000 * m + 100 * n + seed)
            M = rng.standard_normal((m, n))
            Phi, r = msign(M)
            raw = float(np.linalg.norm(Phi, ord="fro") / np.sqrt(m * n))
            theory = float(np.sqrt(r / (m * n)))
            scaled = float(np.sqrt(max(m, n)) * raw)
            raw_values.append(raw)
            scaled_values.append(scaled)
            theory_values.append(theory)
            ranks.append(r)
        rows.append({
            "shape": f"{m}x{n}",
            "rank_mean": float(np.mean(ranks)),
            "raw_mean": float(np.mean(raw_values)),
            "raw_std": float(np.std(raw_values)),
            "theory": float(np.mean(theory_values)),
            "max_abs_error": float(np.max(np.abs(np.array(raw_values) - np.array(theory_values)))),
            "scaled_mean": float(np.mean(scaled_values)),
            "scaled_std": float(np.std(scaled_values)),
            "num_seeds": num_seeds,
        })
    return rows


def write_table(rows):
    with (TABLE_DIR / "rms_alignment_table.tex").open("w") as f:
        f.write("\\footnotesize\n")
        f.write("\\begin{tabular}{lcccc}\n")
        f.write("\\toprule\n")
        f.write("Shape & Obs. RMS & Theory & Max error & Scaled RMS \\\\\n")
        f.write("\\midrule\n")
        for r in rows:
            f.write(
                f"{r['shape']} & {r['raw_mean']:.4f} & {r['theory']:.4f} & "
                f"{r['max_abs_error']:.1e} & \\textbf{{{r['scaled_mean']:.2f}}} \\\\\n"
            )
        f.write("\\bottomrule\n")
        f.write("\\end{tabular}\n")


def main():
    rows = run()
    path = RESULT_DIR / "rms_alignment_shapes.csv"
    with path.open("w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=list(rows[0].keys()))
        writer.writeheader()
        writer.writerows(rows)
    write_table(rows)
    for row in rows:
        print(row)
    shapes = [r["shape"] for r in rows]
    x = list(range(len(rows)))
    plt.figure(figsize=(6, 3.6))
    plt.plot(x, [r["raw_mean"] for r in rows], marker="o", label="observed SVD msign RMS")
    plt.plot(x, [r["theory"] for r in rows], marker="x", linestyle="--", label="theory")
    plt.plot(x, [r["scaled_mean"] for r in rows], marker="s", label="shape-scaled RMS")
    plt.xticks(x, shapes, rotation=35, ha="right")
    plt.ylabel("Update RMS")
    plt.xlabel("Matrix shape")
    plt.title("Numerical SVD check of RMS alignment")
    plt.legend()
    plt.tight_layout()
    plt.savefig(FIG_DIR / "rms_alignment_shapes.png", dpi=200)
    plt.close("all")


if __name__ == "__main__":
    main()
