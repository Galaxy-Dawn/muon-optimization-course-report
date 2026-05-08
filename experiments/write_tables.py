"""Regenerate LaTeX tables from CSV outputs."""
import ast
import csv
from pathlib import Path

import numpy as np


ROOT = Path(__file__).resolve().parents[1]
RESULT_DIR = ROOT / "results"
TABLE_DIR = ROOT / "tables"


LABELS = {
    "gd": "GD",
    "sgd": "SGD",
    "adam": "Adam",
    "adamw": "AdamW",
    "muon": "Muon",
    "muonw": "MuonW",
    "rms_muonw": "RMS-MuonW",
    "cov_muonw": "Cov-MuonW",
}


def read_rows(name):
    with (RESULT_DIR / name).open() as f:
        return list(csv.DictReader(f))


def fmt_acc(mean, std):
    return f"${100 * float(mean):.2f} \\pm {100 * float(std):.2f}$"


def write_accuracy_table(csv_name, tex_name, method_order):
    rows_by_method = {r["method"]: r for r in read_rows(csv_name)}
    with (TABLE_DIR / tex_name).open("w") as f:
        f.write("\\begin{tabular}{lc}\n")
        f.write("\\toprule\n")
        f.write("Method & Test accuracy (\\%) \\\\\n")
        f.write("\\midrule\n")
        for method in method_order:
            row = rows_by_method[method]
            f.write(f"{LABELS[method]} & {fmt_acc(row['mean'], row['std'])} \\\\\n")
        f.write("\\bottomrule\n")
        f.write("\\end{tabular}\n")


def write_stability_table():
    rows = read_rows("digits_mlp_stability.csv")
    with (TABLE_DIR / "mlp_stability_table.tex").open("w") as f:
        f.write("\\footnotesize\n")
        f.write("\\begin{tabular}{lrrr}\n")
        f.write("\\toprule\n")
        f.write("Method & Max logit & $\\|W_1\\|_2$ & $\\|W_2\\|_2$ \\\\\n")
        f.write("\\midrule\n")
        for row in rows:
            f.write(
                f"{LABELS[row['method']]} & "
                f"{float(row['max_logit_mean']):.2f} & "
                f"{float(row['W1_spec_mean']):.2f} & "
                f"{float(row['W2_spec_mean']):.2f} \\\\\n"
            )
        f.write("\\bottomrule\n")
        f.write("\\end{tabular}\n")


def write_sweep_table():
    rows = read_rows("classification_lr_sweep_summary.csv")
    with (TABLE_DIR / "classification_sweep_table.tex").open("w") as f:
        f.write("\\footnotesize\n")
        f.write("\\begin{tabular}{llrrrr}\n")
        f.write("\\toprule\n")
        f.write("Task & Method & Val. acc. & Test acc. & $\\Delta$ AdamW & LR \\\\\n")
        f.write("\\midrule\n")
        for task in ["logreg", "mlp"]:
            for row in [r for r in rows if r["task"] == task]:
                task_name = "LogReg" if task == "logreg" else "MLP"
                f.write(
                    f"{task_name} & {LABELS[row['method']]} & "
                    f"{fmt_acc(row['val_mean'], row['val_std'])} & "
                    f"{fmt_acc(row['test_mean'], row['test_std'])} & "
                    f"${float(row['delta_vs_adamw_pp']):+.2f}$ & "
                    f"{float(row['selected_lr_mode']):.3g} \\\\\n"
                )
            if task == "logreg":
                f.write("\\midrule\n")
        f.write("\\bottomrule\n")
        f.write("\\end{tabular}\n")


def main():
    write_accuracy_table(
        "digits_logreg_results.csv",
        "digits_acc_table.tex",
        ["gd", "adam", "adamw", "muon", "muonw", "rms_muonw", "cov_muonw"],
    )
    write_accuracy_table(
        "digits_mlp_results.csv",
        "mlp_acc_table.tex",
        ["sgd", "adam", "adamw", "muon", "muonw", "rms_muonw", "cov_muonw"],
    )
    write_stability_table()
    write_sweep_table()


if __name__ == "__main__":
    main()
