# Experiment Reproduction Guide

This directory contains the Python scripts used for the diagnostics in the course report. The scripts are intentionally small and CPU-friendly. They are designed to expose the mechanism being tested, not to tune optimizers for best possible accuracy.

Run all commands from the repository root.

## Environment

Create an environment and install dependencies:

```bash
python -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
```

Set BLAS thread counts before running the scripts:

```bash
export OMP_NUM_THREADS=1
export OPENBLAS_NUM_THREADS=1
export MKL_NUM_THREADS=1
```

The scripts also set these variables internally before importing NumPy. Setting them in the shell makes the behavior explicit and avoids slow CPU oversubscription in small SVD-heavy diagnostics.

## Recommended run order

```bash
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

Run `write_tables.py` after the CSV-producing scripts. Run `reorganize_figures_tables.py` last because it reads CSV files and writes the composite figures used in the paper.

## Scripts

| Script | What it checks | Notes |
|---|---|---|
| `matrix_quadratic_diagnostics.py` | Whether objective-spectrum preconditioning and update-spectrum normalization behave differently on a realizable matrix quadratic. | Uses $f(W)=\frac12\|AWB-C\|_F^2$ with known optimum. Reports iteration counts, final relative objective gaps, and parameter errors. |
| `weight_decay_control.py` | Whether decoupled weight decay controls the spectral norm of weights under bounded spectral updates. | Compares random polar directions with adversarially aligned directions. |
| `rms_alignment_shapes.py` | Whether exact matrix-sign RMS follows the algebraic shape law. | Computes exact SVD polar factors for several rectangular matrix shapes. |
| `ns_approx_quality.py` | How finite Newton-Schulz steps approximate the exact polar factor. | Reports alignment, orthogonality error, and singular-value spread for condition-number blocks. |
| `finite_ns_training.py` | What happens when finite Newton-Schulz approximation is placed inside the MLP training loop. | Uses the same fixed full-batch Digits MLP protocol as the MuonW diagnostic and varies the number of Newton-Schulz steps. |
| `digits_logreg.py` | Whether the mechanisms leave traces in full-batch logistic regression. | Uses scikit-learn Digits, train-only feature standardization, and five stratified splits. |
| `digits_mlp.py` | Whether the same mechanisms appear in a small MLP. | Compares Adam, AdamW, Muon, MuonW, RMS-MuonW, and Cov-MuonW under a fixed protocol. |
| `classification_lr_sweep.py` | Whether classification conclusions are sensitive to a small validation-selected learning-rate grid. | Uses train/validation/test splits. The sweep is a limited sensitivity check, not a full hyperparameter search. |
| `write_tables.py` | Converts CSV outputs into LaTeX table fragments. | Writes files under `tables/`. |
| `reorganize_figures_tables.py` | Builds the final composite figures and selected table fragments. | Uses `pubfig`; run after CSV outputs exist. |

## Output files

| Output path | Produced by | Description |
|---|---|---|
| `results/matrix_quadratic_diagnostics.csv` | `matrix_quadratic_diagnostics.py` | Iterations to tolerance for matrix-quadratic diagnostics. |
| `results/matrix_quadratic_final_diagnostics.csv` | `matrix_quadratic_diagnostics.py` | Final relative objective gaps and parameter errors. |
| `results/weight_decay_control.csv` | `weight_decay_control.py` | Spectral-norm statistics for random and adversarial polar updates. |
| `results/rms_alignment_shapes.csv` | `rms_alignment_shapes.py` | Observed RMS, theoretical RMS, and shape-scaled RMS. |
| `results/ns_approx_quality.csv` | `ns_approx_quality.py` | Newton-Schulz approximation metrics. |
| `results/finite_ns_training_raw.csv` | `finite_ns_training.py` | Per-split finite Newton-Schulz MLP results. |
| `results/finite_ns_training_summary.csv` | `finite_ns_training.py` | Aggregated finite Newton-Schulz MLP summary. |
| `results/digits_logreg_results.csv` | `digits_logreg.py` | Logistic-regression accuracy and curve summaries. |
| `results/digits_mlp_results.csv` | `digits_mlp.py` | MLP accuracy summaries. |
| `results/digits_mlp_stability.csv` | `digits_mlp.py` | MLP maximum logits and final spectral norms. |
| `results/classification_lr_sweep_raw.csv` | `classification_lr_sweep.py` | Per-split validation-selected classification results. |
| `results/classification_lr_sweep_summary.csv` | `classification_lr_sweep.py` | Aggregated validation-selected classification table input. |
| `tables/*.tex` | `write_tables.py`, selected experiment scripts | LaTeX table fragments used by `main.tex`. |
| `figures/` | Plotting scripts and `reorganize_figures_tables.py` | Figure assets generated for the report. The repository keeps only the final paper figures after cleanup. |

## Protocol details

The classification scripts use `sklearn.datasets.load_digits`.

- Feature standardization is fitted on the training split only.
- Fixed-setting classification diagnostics use five stratified train/test splits.
- Validation-selected diagnostics use train/validation/test splits and choose learning rate by validation accuracy.
- Logistic regression uses full-batch softmax cross-entropy.
- The MLP uses a `64 -> 32 -> 10` architecture with ReLU activation.
- Matrix-specific Muon-family updates are applied to weight matrices only.
- Bias vectors in Muon-family MLP runs use fixed-step gradient descent.

`Cov-MuonW` uses oracle covariance information and exact polar factors. It is included to probe the relation between covariance preconditioning and matrix-sign normalization. It is not a scalable optimizer baseline.

## Figure regeneration

`reorganize_figures_tables.py` uses the installed `pubfig` package to rebuild the composite figures:

- synthetic mechanism diagnostics;
- Digits classification diagnostics.

The script expects the CSV files in `results/` to already exist. If the code-only archive is used, run the CSV-producing scripts first.

## Minimal smoke check

To check that the environment works before running every experiment:

```bash
python experiments/matrix_quadratic_diagnostics.py
python experiments/ns_approx_quality.py
python experiments/write_tables.py
```

This creates CSV outputs and table fragments without requiring the full classification sweep.

## Known boundaries

These scripts do not evaluate language models, minibatch stochasticity, distributed training, wall-clock scaling, or attention-specific stabilization mechanisms. They are mechanism diagnostics for a course report.
