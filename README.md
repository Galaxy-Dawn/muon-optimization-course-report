# Muon Optimization Course Report

This repository contains the code, data outputs, figures, tables, and LaTeX source for a graduate course report on Muon-style matrix optimization.

Report title:

**Spectral-Norm Steepest Descent and Matrix Preconditioning: A Mechanistic Study Inspired by Muon Optimizers**

Author:

**Gaorui Zhang**  
Zhejiang University  
gaoruizhang@zju.edu.cn

## What this report studies

Muon-style optimizers and matrix preconditioners both transform matrix-valued gradients. The report asks whether these transformations act on the same mathematical object.

The short answer is no. The report separates:

- **Objective-spectrum preconditioning**: left-right matrix preconditioners can change the Hessian spectrum of a quadratic objective.
- **Update-spectrum normalization**: the matrix-sign or polar step changes the singular values of the update direction.
- **Step-stability mechanisms**: decoupled weight decay, update-RMS alignment, and Newton-Schulz approximation quality determine whether the spectral direction becomes a usable optimizer step.

The experiments are controlled diagnostics for a course report. They are not large-scale optimizer benchmarks.

## Repository contents

```text
.
├── main.tex                         # Course report source
├── main.pdf                         # Compiled report
├── references.bib                   # Bibliography
├── requirements.txt                 # Python dependencies
├── experiments/                     # Reproducibility scripts
│   ├── README.md                    # Script-level run guide
│   ├── matrix_quadratic_diagnostics.py
│   ├── weight_decay_control.py
│   ├── rms_alignment_shapes.py
│   ├── ns_approx_quality.py
│   ├── finite_ns_training.py
│   ├── digits_logreg.py
│   ├── digits_mlp.py
│   ├── classification_lr_sweep.py
│   ├── write_tables.py
│   └── reorganize_figures_tables.py
├── results/                         # Raw CSV outputs
├── tables/                          # LaTeX table fragments
├── figures/                         # Final paper figures
├── acmart.cls                       # Local ACM class copy
└── ACM-Reference-Format.bst         # ACM bibliography style
```

The course submission code archive contains only `README.md`, `requirements.txt`, and `experiments/`. The full GitHub repository also keeps generated CSV files, LaTeX tables, figures, and the compiled report.

## Experiment map

| Script | Purpose | Main outputs |
|---|---|---|
| `matrix_quadratic_diagnostics.py` | Tests GD, diagonal preconditioning, ideal left-right preconditioning, and scaled matrix-sign updates on realizable matrix quadratics. | `results/matrix_quadratic_diagnostics.csv`, `results/matrix_quadratic_final_diagnostics.csv`, matrix diagnostic tables |
| `weight_decay_control.py` | Checks the spectral-norm bound induced by decoupled weight decay under bounded polar updates. | `results/weight_decay_control.csv`, weight-decay plot |
| `rms_alignment_shapes.py` | Verifies the exact-SVD identity for matrix-sign RMS across rectangular shapes. | `results/rms_alignment_shapes.csv`, RMS table and plot |
| `ns_approx_quality.py` | Measures finite Newton-Schulz approximation quality against exact SVD polar factors. | `results/ns_approx_quality.csv`, Newton-Schulz plot |
| `finite_ns_training.py` | Inserts finite Newton-Schulz polar approximation into a fixed full-batch Digits MLP training loop. | `results/finite_ns_training_raw.csv`, `results/finite_ns_training_summary.csv`, finite-NS table |
| `digits_logreg.py` | Runs full-batch logistic-regression diagnostics on scikit-learn Digits. | `results/digits_logreg_results.csv`, loss and accuracy curves |
| `digits_mlp.py` | Runs full-batch MLP diagnostics comparing Adam/AdamW with Muon-style variants. | `results/digits_mlp_results.csv`, `results/digits_mlp_stability.csv` |
| `classification_lr_sweep.py` | Runs a small validation-selected learning-rate sweep for classification diagnostics. | `results/classification_lr_sweep_raw.csv`, `results/classification_lr_sweep_summary.csv` |
| `write_tables.py` | Regenerates LaTeX tables from CSV outputs. | `tables/*.tex` |
| `reorganize_figures_tables.py` | Builds composite paper figures with `pubfig` and rewrites selected table fragments. | `figures/*composite*.pdf`, `figures/*composite*.png` |

## Setup

Use Python 3.10 or newer.

```bash
python -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
```

Dependencies:

- `numpy`
- `matplotlib`
- `scikit-learn`
- `pubfig`

The scripts create `results/`, `tables/`, and `figures/` if these directories do not already exist.

## Reproduce the experiments

Run from the repository root.

```bash
export OMP_NUM_THREADS=1
export OPENBLAS_NUM_THREADS=1
export MKL_NUM_THREADS=1

python experiments/matrix_quadratic_diagnostics.py
python experiments/weight_decay_control.py
python experiments/rms_alignment_shapes.py
python experiments/ns_approx_quality.py
python experiments/finite_ns_training.py
python experiments/digits_logreg.py
python experiments/digits_mlp.py
python experiments/classification_lr_sweep.py
python experiments/write_tables.py
python experiments/reorganize_figures_tables.py
```

The BLAS thread variables keep the small SVD-heavy experiments from oversubscribing CPU threads. They also make local runtime more predictable.

## Build the report

The report uses an ACM-style LaTeX layout. Compile with pdfLaTeX:

```bash
pdflatex -interaction=nonstopmode -halt-on-error main.tex
bibtex main
pdflatex -interaction=nonstopmode -halt-on-error main.tex
pdflatex -interaction=nonstopmode -halt-on-error main.tex
```

The compiled file is `main.pdf`.

## Scope and limitations

The synthetic experiments directly test the algebraic mechanisms in the report. The Digits experiments are small full-batch sanity checks. They do not establish optimizer rankings for large neural-network training, minibatch stochasticity, distributed training, language models, or wall-clock efficiency.

`Cov-MuonW` is an oracle diagnostic. It uses full-batch covariance information and exact SVD polar factors, so it should not be read as a scalable optimizer implementation.

Muon-family MLP runs update bias vectors with fixed-step gradient descent. Adam and AdamW update biases adaptively. The report treats this as an implementation-level comparison confound.

## Troubleshooting

| Symptom | Likely cause | Fix |
|---|---|---|
| `ModuleNotFoundError: pubfig` | `pubfig` is not installed in the active environment. | Run `pip install -r requirements.txt`. |
| Very slow SVD-heavy scripts | BLAS uses too many threads for small matrices. | Set `OMP_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 MKL_NUM_THREADS=1`. |
| `reorganize_figures_tables.py` cannot find CSV files | The raw experiment scripts were not run first. | Run all experiment scripts before generating composite figures. |
| LaTeX cannot find `acmart.cls` or `ACM-Reference-Format.bst` | The local template files were not copied. | Use the full repository, or keep `acmart.cls` and `ACM-Reference-Format.bst` beside `main.tex`. |

## License

Code is released under the MIT License. The report source, figures, and generated outputs are included so the diagnostics can be checked and rerun.
