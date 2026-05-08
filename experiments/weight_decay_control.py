"""Weight decay control diagnostic for bounded spectral-norm updates.

We compare random polar updates with adversarially aligned polar updates.  The
adversarial case is included to show that the theorem is a worst-case norm
control statement; random updates are usually far below the bound.
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
RESULT_DIR.mkdir(exist_ok=True); FIG_DIR.mkdir(exist_ok=True)


def random_polar(m, n, rng):
    Z = rng.standard_normal((m,n))
    U, _, Vt = np.linalg.svd(Z, full_matrices=False)
    return U @ Vt


def msign(M):
    U, s, Vt = np.linalg.svd(M, full_matrices=False)
    r = int(np.sum(s > 1e-12))
    if r == 0:
        return np.zeros_like(M)
    return U[:, :r] @ Vt[:r, :]


def run(T=800, eta=0.05, S=1.0):
    rng = np.random.default_rng(7)
    m = n = 32
    W0 = 0.2 * random_polar(m,n,rng)
    rows=[]; curves={}
    for lam in [0.0, 0.01, 0.05, 0.1, 0.2]:
        mode_results={}
        for mode in ["random", "adversarial"]:
            W = W0.copy()
            norms=[]
            for _ in range(T):
                if mode == "random":
                    Phi = S * random_polar(m,n,rng)
                else:
                    # Push in the current polar direction to approach the worst-case radius.
                    Phi = -S * msign(W)
                W = (1 - eta*lam) * W - eta * Phi
                norms.append(float(np.linalg.norm(W,2)))
            mode_results[mode] = norms
            curves[f"{mode}, lambda={lam}"] = norms
        bound = np.inf if lam == 0 else max(float(np.linalg.norm(W0,2)), S/lam)
        rows.append({
            "lambda": lam,
            "random_max_norm": max(mode_results["random"]),
            "random_final_norm": mode_results["random"][-1],
            "adversarial_max_norm": max(mode_results["adversarial"]),
            "adversarial_final_norm": mode_results["adversarial"][-1],
            "theory_bound": bound,
        })
    return rows, curves


def main():
    rows, curves = run()
    path = RESULT_DIR / "weight_decay_control.csv"
    with path.open('w', newline='') as f:
        writer=csv.DictWriter(f, fieldnames=list(rows[0].keys()))
        writer.writeheader(); writer.writerows(rows)
    plt.figure(figsize=(6,3.6))
    x=np.arange(1, len(next(iter(curves.values())))+1)
    for key,norms in curves.items():
        if key.startswith("adversarial") and ("0.05" in key or "0.1" in key or "0.2" in key):
            plt.plot(x,norms,label=key)
        elif key.startswith("random") and ("0.05" in key or "0.1" in key or "0.2" in key):
            plt.plot(x,norms,linestyle="--",label=key)
    plt.xlabel("Iteration"); plt.ylabel("Spectral norm of W")
    plt.legend(fontsize=6); plt.tight_layout()
    plt.savefig(FIG_DIR/"weight_decay_control.png", dpi=200)
    plt.close("all")
    for row in rows:
        print(row)

if __name__ == "__main__":
    main()
