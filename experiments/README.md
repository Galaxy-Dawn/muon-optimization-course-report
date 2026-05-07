# Experiments

Run commands from the project root. The scripts set BLAS thread counts to one before importing NumPy, which avoids slow CPU thread oversubscription in small SVD-heavy diagnostics. You can also set these variables explicitly:

```bash
OMP_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 MKL_NUM_THREADS=1 python experiments/matrix_quadratic_diagnostics.py
OMP_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 MKL_NUM_THREADS=1 python experiments/weight_decay_control.py
OMP_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 MKL_NUM_THREADS=1 python experiments/rms_alignment_shapes.py
OMP_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 MKL_NUM_THREADS=1 python experiments/ns_approx_quality.py
OMP_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 MKL_NUM_THREADS=1 python experiments/digits_logreg.py
OMP_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 MKL_NUM_THREADS=1 python experiments/digits_mlp.py
OMP_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 MKL_NUM_THREADS=1 python experiments/classification_lr_sweep.py
python experiments/write_tables.py
```

Outputs are saved to `results/`; figures are saved to `figures/`; LaTeX tables are saved to `tables/`. The fixed-hyperparameter classification experiments are full-batch diagnostics for the course report, not tuned benchmarks. The validation-selected sweep uses train/validation/test splits and a small fixed learning-rate grid for each optimizer. The MLP script also writes `results/digits_mlp_stability.csv`, which records maximum logits and final weight spectral norms.
