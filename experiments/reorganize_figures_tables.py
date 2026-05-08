"""Reorganize paper figures and tables into publication-ready composites.

The script keeps raw CSV files untouched and writes new figure/table assets for
the main paper. Figures are built with the installed `pubfig` package; tables
are written as LaTeX fragments with booktabs and light cell highlighting.
"""
import csv
import importlib.util
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.patches import FancyArrowPatch, FancyBboxPatch
import numpy as np
import pubfig as pf

ROOT = Path(__file__).resolve().parents[1]
RESULT_DIR = ROOT / "results"
FIG_DIR = ROOT / "figures"
TABLE_DIR = ROOT / "tables"

METHOD_LABELS = {
    "gd": "GD",
    "sgd": "SGD",
    "adam": "Adam",
    "adamw": "AdamW",
    "muon": "Muon",
    "muonw": "MuonW",
    "rms_muonw": "RMS-MuonW",
    "cov_muonw": "Cov-MuonW",
}
METHOD_ORDER = ["sgd", "adam", "adamw", "muon", "muonw", "rms_muonw", "cov_muonw"]
LOGREG_ORDER = ["sgd", "adam", "adamw", "muon", "muonw", "rms_muonw", "cov_muonw"]
PALETTE = {
    "baseline": "#4C78A8",
    "precond": "#59A14F",
    "muon": "#B279A2",
    "warn": "#E15759",
    "neutral": "#6B7280",
    "grid": "#E5E7EB",
    "best": "#DFF2E1",
    "caution": "#FCE5E1",
    "family": "#EEF2FF",
}


def read_rows(name):
    with (RESULT_DIR / name).open() as f:
        return list(csv.DictReader(f))


def load_module(name):
    path = ROOT / "experiments" / f"{name}.py"
    spec = importlib.util.spec_from_file_location(name, path)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def save_pubfig(fig, base, width="double", aspect_ratio=None):
    pdf_path = pf.save_figure(
        fig,
        FIG_DIR / f"{base}.pdf",
        spec="nature",
        width=width,
        aspect_ratio=aspect_ratio,
        raster_dpi=300,
        trim=True,
    )
    png_path = pf.save_figure(
        fig,
        FIG_DIR / f"{base}.png",
        spec="nature",
        width=width,
        aspect_ratio=aspect_ratio,
        raster_dpi=300,
        trim=True,
    )
    plt.close(fig)
    return [*pdf_path, *png_path]


def add_panel_labels(fig, axes, labels, pad_pt=1.5):
    # Place labels on row/column-aligned anchors computed from tight bboxes
    # (including axis/ticks labels): A/B share y, A/C share x, etc.
    fig.canvas.draw()
    renderer = fig.canvas.get_renderer()
    inv = fig.transFigure.inverted()
    if hasattr(axes, "shape") and len(axes.shape) == 2:
        axes2d = axes
        nrows, ncols = axes2d.shape
        if len(labels) != nrows * ncols:
            raise ValueError("Number of labels must match subplot count.")
        bboxes = [[axes2d[r, c].get_tightbbox(renderer) for c in range(ncols)] for r in range(nrows)]
        col_lefts = [min(bboxes[r][c].x0 for r in range(nrows)) + pad_pt for c in range(ncols)]
        row_tops = [max(bboxes[r][c].y1 for c in range(ncols)) - pad_pt for r in range(nrows)]
        idx = 0
        for r in range(nrows):
            for c in range(ncols):
                x_fig, y_fig = inv.transform((col_lefts[c], row_tops[r]))
                fig.text(
                    x_fig,
                    y_fig,
                    labels[idx],
                    fontsize=9.0,
                    fontweight="bold",
                    ha="left",
                    va="top",
                )
                idx += 1
        return

    for ax, label in zip(axes, labels):
        tight = ax.get_tightbbox(renderer)
        x_fig, y_fig = inv.transform((tight.x0 + pad_pt, tight.y1 - pad_pt))
        fig.text(x_fig, y_fig, label, fontsize=9.0, fontweight="bold", ha="left", va="top")


def style_axis(ax):
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)
    ax.grid(axis="y", color=PALETTE["grid"], linewidth=0.6)
    ax.set_axisbelow(True)
    ax.tick_params(axis="both", labelsize=6.8)
    ax.xaxis.label.set_size(7.2)
    ax.yaxis.label.set_size(7.2)


def make_mechanism_figure():
    fig, ax = plt.subplots(figsize=(7.2, 2.15))
    ax.set_xlim(0, 1)
    ax.set_ylim(0, 1)
    ax.axis("off")

    boxes = [
        (0.04, 0.52, 0.22, 0.28, "Objective spectrum", "$H_W=BB^\\top\\!\\otimes A^\\top A$", "#E8F1FB"),
        (0.39, 0.52, 0.22, 0.28, "Update spectrum", "$\\operatorname{msign}(M)$", "#F4EAF3"),
        (0.74, 0.52, 0.22, 0.28, "Usable step", "$\\Phi_t=s_t\\operatorname{msign}(M_t)$", "#E9F6EE"),
        (0.04, 0.14, 0.22, 0.22, "Left--right preconditioner", "$P_LGP_R\\Rightarrow H_{\\rm eff}=I$", "#F7F7F7"),
        (0.39, 0.14, 0.22, 0.22, "Spectral-norm direction", "flattens update singular values", "#F7F7F7"),
        (0.74, 0.14, 0.22, 0.22, "Stability mechanisms", "decay, RMS scale, NS quality", "#F7F7F7"),
    ]
    for x, y, w, h, title, body, color in boxes:
        ax.add_patch(
            FancyBboxPatch(
                (x, y),
                w,
                h,
                boxstyle="round,pad=0.015,rounding_size=0.025",
                linewidth=0.8,
                edgecolor="#2F3A45",
                facecolor=color,
            )
        )
        ax.text(x + 0.02, y + h - 0.07, title, fontsize=8.5, fontweight="bold", va="top")
        ax.text(x + 0.02, y + 0.07, body, fontsize=7.6, va="bottom")

    arrows = [
        ((0.27, 0.66), (0.38, 0.66), "different object"),
        ((0.62, 0.66), (0.73, 0.66), "scale + approx."),
        ((0.15, 0.52), (0.15, 0.37), ""),
        ((0.50, 0.52), (0.50, 0.37), ""),
        ((0.85, 0.52), (0.85, 0.37), ""),
    ]
    for start, end, label in arrows:
        ax.add_patch(FancyArrowPatch(start, end, arrowstyle="-|>", mutation_scale=10, lw=0.8, color="#374151"))
        if label:
            mx = (start[0] + end[0]) / 2
            my = (start[1] + end[1]) / 2 + 0.045
            ax.text(mx, my, label, ha="center", va="center", fontsize=7, color="#374151")

    ax.text(
        0.5,
        0.93,
        "Muon-style orthogonalization is an update-spectrum operation, not a Hessian inverse.",
        ha="center",
        fontsize=9.2,
        fontweight="bold",
    )
    save_pubfig(fig, "main_mechanism_reorganized", width="double", aspect_ratio=0.32)


def make_synthetic_figure(base_name="synthetic_diagnostics_composite", add_labels=False):
    matrix_rows = read_rows("matrix_quadratic_diagnostics.csv")
    decay_rows = read_rows("weight_decay_control.csv")
    rms_rows = read_rows("rms_alignment_shapes.csv")
    ns_rows = read_rows("ns_approx_quality.csv")

    fig, axes = plt.subplots(2, 2, figsize=(7.0, 4.65))

    ax = axes[0, 0]
    heat = np.array([[min(float(r[k]), 5500.0) for r in matrix_rows] for k in ["gd", "diag", "lr", "scaled_msign"]])
    pf.heatmap(
        heat,
        x_label="Condition",
        y_label="Method",
        annotate=True,
        annotate_fmt=".0f",
        ax=ax,
    )
    ax.set_xticks(np.arange(len(matrix_rows)))
    ax.set_xticklabels(
        [f"{'rot' if r['rotated'] == 'True' else 'axis'}\n$\\kappa_H$={r['kappa_H']}" for r in matrix_rows],
        rotation=0,
        ha="center",
    )
    ax.set_yticks(np.arange(4))
    ax.set_yticklabels(["GD", "Diag.", "LR", "s-msign"])

    ax = axes[0, 1]
    d = [r for r in decay_rows if float(r["lambda"]) > 0]
    lambdas = np.array([float(r["lambda"]) for r in d])
    random_final = np.array([float(r["random_final_norm"]) for r in d])
    adv_final = np.array([float(r["adversarial_final_norm"]) for r in d])
    bound = np.array([float(r["theory_bound"]) for r in d])
    pf.line(
        np.column_stack([bound, adv_final, random_final]),
        x=lambdas,
        series_names=["Bound $S/\\lambda$", "Adversarial final", "Random final"],
        x_label="$\\lambda$",
        y_label="$\\Vert W_T\\Vert_2$",
        ax=ax,
    )

    ax = axes[1, 0]
    shapes = [r["shape"] for r in rms_rows]
    raw = np.array([float(r["raw_mean"]) for r in rms_rows])
    scaled = np.array([float(r["scaled_mean"]) for r in rms_rows])
    x = np.arange(len(shapes))
    pf.line(
        np.column_stack([raw, scaled]),
        x=x,
        series_names=["Raw", "$\\sqrt{\\max(m,n)}$ scaled"],
        x_label="Shape index",
        y_label="Update RMS",
        ax=ax,
    )

    ax = axes[1, 1]
    step_x = np.array([1, 3, 5, 8])
    ns_data = []
    ns_names = []
    for kappa in [10, 100, 1000]:
        group = [r for r in ns_rows if int(r["kappa"]) == kappa]
        ns_data.append([float(r["orth_error"]) for r in group])
        ns_names.append(f"$\\kappa$={kappa}")
    pf.line(
        np.array(ns_data).T,
        x=step_x,
        series_names=ns_names,
        x_label="Newton--Schulz steps",
        y_label="Orthogonality error",
        ax=ax,
    )

    # User-requested tweak: slightly larger axis-title fonts for Figure 2.
    for ax in axes.ravel():
        ax.xaxis.label.set_size(7.4)
        ax.yaxis.label.set_size(7.4)

    fig.tight_layout(pad=0.6, w_pad=1.0, h_pad=0.8)
    if add_labels:
        add_panel_labels(fig, axes, ["A", "B", "C", "D"])
    save_pubfig(fig, base_name, width="double", aspect_ratio=0.68)


def make_classification_figure(base_name="classification_diagnostics_composite", add_labels=False):
    logreg_module = load_module("digits_logreg")
    mlp_module = load_module("digits_mlp")
    sweep_rows = read_rows("classification_lr_sweep_summary.csv")
    raw_sweep = read_rows("classification_lr_sweep_raw.csv")

    fig, axes = plt.subplots(2, 2, figsize=(7.0, 4.65))

    ax = axes[0, 0]
    methods = LOGREG_ORDER
    logreg_map = {r["method"]: r for r in sweep_rows if r["task"] == "logreg"}
    mlp_map = {r["method"]: r for r in sweep_rows if r["task"] == "mlp"}
    pf.bar(
        np.column_stack([
            [100 * float(logreg_map[m]["test_mean"]) for m in methods],
            [100 * float(mlp_map[m]["test_mean"]) for m in methods],
        ]),
        category_names=[METHOD_LABELS[m].replace("-MuonW", "-M") for m in methods],
        series_names=["LogReg", "MLP"],
        x_label="Method",
        y_label="Val-selected test acc. (%)",
        ax=ax,
    )
    ax.set_ylim(bottom=80)

    ax = axes[0, 1]
    mlp_rows = {r["method"]: r for r in sweep_rows if r["task"] == "mlp"}
    methods = METHOD_ORDER
    delta = np.array([float(mlp_rows[m]["delta_vs_adamw_pp"]) for m in methods])
    max_logit = {}
    for r in raw_sweep:
        if r["task"] == "mlp":
            max_logit.setdefault(r["method"], []).append(float(r["max_logit"]))
    logits = np.array([np.mean(max_logit[m]) for m in methods])
    pf.scatter(
        delta,
        logits,
        labels=np.array([METHOD_LABELS[m].replace("-MuonW", "-M") for m in methods]),
        x_label="$\\Delta$ test acc. vs AdamW (pp)",
        y_label="MLP max logit",
        scatter_size=7.0,
        legend_show=True,
        ax=ax,
    )

    ax = axes[1, 0]
    log_methods = ["gd", "adamw", "muon", "muonw", "rms_muonw", "cov_muonw"]
    histories = {m: logreg_module.run(m, seed=42, record=True) for m in log_methods}
    pf.line(
        np.column_stack([histories[m]["loss"] for m in log_methods]),
        x=np.arange(1, len(next(iter(histories.values()))["loss"]) + 1),
        series_names=[METHOD_LABELS[m] for m in log_methods],
        x_label="Iteration",
        y_label="LogReg training loss",
        marker=None,
        ax=ax,
    )

    ax = axes[1, 1]
    mlp_methods = ["sgd", "adamw", "muonw", "rms_muonw", "cov_muonw"]
    histories = {m: mlp_module.run(m, seed=42, record=True) for m in mlp_methods}
    pf.line(
        np.column_stack([histories[m]["max_logit"] for m in mlp_methods]),
        x=np.arange(1, len(next(iter(histories.values()))["max_logit"]) + 1),
        series_names=[METHOD_LABELS[m] for m in mlp_methods],
        x_label="Epoch",
        y_label="MLP max logit",
        marker=None,
        ax=ax,
    )

    # User-requested tweak: slightly larger axis-title fonts for Figure 3.
    for ax in axes.ravel():
        ax.xaxis.label.set_size(7.4)
        ax.yaxis.label.set_size(7.4)

    fig.tight_layout(pad=0.6, w_pad=1.1, h_pad=0.8)
    if add_labels:
        add_panel_labels(fig, axes, ["A", "B", "C", "D"])
    save_pubfig(fig, base_name, width="double", aspect_ratio=0.68)


def fmt_count(value, budget=5000):
    value = int(float(value))
    if value > budget:
        return "\\warncell{$>$5000}"
    if value == 1:
        return "\\bestcell{\\textbf{1}}"
    return str(value)


def fmt_acc(mean, std, bold=False, shade=False):
    text = f"${100 * float(mean):.2f} \\pm {100 * float(std):.2f}$"
    if bold:
        text = f"\\textbf{{{text}}}"
    if shade:
        text = f"\\bestcell{{{text}}}"
    return text


def write_matrix_table():
    rows = read_rows("matrix_quadratic_diagnostics.csv")
    with (TABLE_DIR / "matrix_diagnostics_reorganized_table.tex").open("w") as f:
        f.write("\\footnotesize\n")
        f.write("\\setlength{\\tabcolsep}{3.1pt}\n")
        f.write("\\begin{tabular}{llrrrr}\n")
        f.write("\\toprule\n")
        f.write("Rot. & $\\kappa_H$ & GD & Diag. & LR & s-\\texttt{msign} \\\\\n")
        f.write("\\midrule\n")
        for row in rows:
            rot = "Yes" if row["rotated"] == "True" else "No"
            f.write(
                f"{rot} & {row['kappa_H']} & "
                f"{fmt_count(row['gd'])} & {fmt_count(row['diag'])} & "
                f"{fmt_count(row['lr'])} & {fmt_count(row['scaled_msign'])} \\\\\n"
            )
        f.write("\\bottomrule\n")
        f.write("\\end{tabular}\n")


def write_scale_approx_table():
    decay = {float(r["lambda"]): r for r in read_rows("weight_decay_control.csv")}
    rms = read_rows("rms_alignment_shapes.csv")
    ns = read_rows("ns_approx_quality.csv")
    ns_lookup = {(int(r["kappa"]), int(r["steps"])): r for r in ns}
    with (TABLE_DIR / "scale_approx_reorganized_table.tex").open("w") as f:
        f.write("\\footnotesize\n")
        f.write("\\setlength{\\tabcolsep}{3.0pt}\n")
        f.write("\\begin{tabular}{llrrl}\n")
        f.write("\\toprule\n")
        f.write("Diagnostic & Setting & Value 1 & Value 2 & Takeaway \\\\\n")
        f.write("\\midrule\n")
        for lam in [0.05, 0.10, 0.20]:
            r = decay[lam]
            f.write(
                f"Decay & $\\lambda={lam:.2f}$ & "
                f"{float(r['adversarial_final_norm']):.2f} & {float(r['theory_bound']):.1f} & "
                "\\bestcell{below bound} \\\\\n"
            )
        f.write("\\midrule\n")
        for r in [rms[0], rms[1], rms[3], rms[-1]]:
            f.write(
                f"RMS & {r['shape']} & {float(r['raw_mean']):.4f} & {float(r['scaled_mean']):.2f} & "
                "\\bestcell{RMS=1} \\\\\n"
            )
        f.write("\\midrule\n")
        for kappa in [10, 100, 1000]:
            r = ns_lookup[(kappa, 8)]
            takeaway = "\\bestcell{near polar}" if kappa == 10 else "\\warncell{approx.}"
            f.write(
                f"NS & $\\kappa={kappa}$, 8 steps & {float(r['alignment']):.3f} & {float(r['orth_error']):.3f} & "
                f"{takeaway} \\\\\n"
            )
        f.write("\\bottomrule\n")
        f.write("\\end{tabular}\n")


def write_classification_table():
    fixed_logreg = {r["method"]: r for r in read_rows("digits_logreg_results.csv")}
    fixed_mlp = {r["method"]: r for r in read_rows("digits_mlp_results.csv")}
    sweep = read_rows("classification_lr_sweep_summary.csv")
    stability = {r["method"]: r for r in read_rows("digits_mlp_stability.csv")}
    by_task = {}
    for row in sweep:
        by_task[(row["task"], row["method"])] = row

    with (TABLE_DIR / "classification_reorganized_table.tex").open("w") as f:
        f.write("\\footnotesize\n")
        f.write("\\setlength{\\tabcolsep}{2.7pt}\n")
        f.write("\\begin{tabular}{llrrrrr}\n")
        f.write("\\toprule\n")
        f.write("Task & Method & Fixed test & Val test & $\\Delta$ & LR & Stability \\\\\n")
        f.write("\\midrule\n")
        for task, order, fixed in [
            ("logreg", LOGREG_ORDER, fixed_logreg),
            ("mlp", METHOD_ORDER, fixed_mlp),
        ]:
            best_fixed = max(float(fixed["gd" if m == "sgd" and task == "logreg" else m]["mean"]) for m in order)
            best_val = max(float(by_task[(task, m)]["test_mean"]) for m in order)
            for method in order:
                fixed_key = "gd" if method == "sgd" and task == "logreg" else method
                fixed_row = fixed[fixed_key]
                val_row = by_task[(task, method)]
                fixed_cell = fmt_acc(
                    fixed_row["mean"],
                    fixed_row["std"],
                    bold=abs(float(fixed_row["mean"]) - best_fixed) < 1e-12,
                    shade=abs(float(fixed_row["mean"]) - best_fixed) < 1e-12,
                )
                val_cell = fmt_acc(
                    val_row["test_mean"],
                    val_row["test_std"],
                    bold=abs(float(val_row["test_mean"]) - best_val) < 1e-12,
                    shade=abs(float(val_row["test_mean"]) - best_val) < 1e-12,
                )
                delta = f"${float(val_row['delta_vs_adamw_pp']):+.2f}$"
                lr = f"{float(val_row['selected_lr_mode']):.3g}"
                if task == "mlp":
                    max_logit = float(stability[method]["max_logit_mean"])
                    stab = f"{max_logit:.1f}"
                    if method == "rms_muonw":
                        stab = f"\\warncell{{{stab}}}"
                    elif method in ["muonw", "cov_muonw"]:
                        stab = f"\\bestcell{{{stab}}}"
                else:
                    stab = "--"
                task_name = "LogReg" if task == "logreg" else "MLP"
                method_name = METHOD_LABELS[fixed_key] if task == "logreg" and method == "sgd" else METHOD_LABELS[method]
                f.write(f"{task_name} & {method_name} & {fixed_cell} & {val_cell} & {delta} & {lr} & {stab} \\\\\n")
            if task == "logreg":
                f.write("\\midrule\n")
        f.write("\\bottomrule\n")
        f.write("\\end{tabular}\n")


def main():
    pf.set_default_theme("nature")
    make_mechanism_figure()
    make_synthetic_figure(add_labels=True)
    make_classification_figure(add_labels=True)
    write_matrix_table()
    write_scale_approx_table()
    write_classification_table()
    print("Wrote reorganized figures and tables.")


if __name__ == "__main__":
    main()
