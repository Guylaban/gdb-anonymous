# Reproducibility scripts

Six small scripts that reproduce the headline numbers in the paper from
the data in this repository. All paths are resolved relative to the
repo root via `_paths.py`; no absolute paths are hardcoded.

## Requirements

```
pip install -r ../requirements.txt
```

The scripts depend only on `pandas`, `numpy`, `scipy`, `scikit-learn`,
`statsmodels`, and `matplotlib`. No vendor SDKs or network access are
needed; the scripts only analyze the CSV / JSONL data shipped in the
repository.

## What each script reproduces

| Script | Reproduces |
|---|---|
| `headline_table.py`   | Table 4 (per-model F0 / F1 / F2 / F3 / Trap / All), plus the tier rollup of response-pooled trap rates. |
| `reliability.py`      | Table 3 (author IRR + external rater kappas, with a diff against the precomputed `kappa_results.json`). |
| `tier_test.py`        | Tier chi-squared on the per-response binary outcome and the LRT for tier in a logistic GLM controlling for trap family; falls back to cluster-robust SEs (cluster on `scenario_id`) when the binomial GLMM is unavailable. |
| `arena_correlation.py`| Figure 2 numbers (Spearman / Pearson between trap displacement and Chatbot Arena Elo, plus leave-one-out range). |
| `probe1_stages.py`    | Section 7.2 / Appendix P (per-checkpoint F0 / F1 / F2 / F3 / Trap rates from the post-training probe, averaged across the GPT-4.1 and Kimi-K2 judges). |
| `discriminant_2x2.py` | Section 7.1 / Appendix O (per-cell DS>=2, sycophancy-positive, and within-cell Spearman rho between DS and max-sycophancy). |

## Usage

```
python headline_table.py
python reliability.py
python tier_test.py
python arena_correlation.py
python probe1_stages.py
python discriminant_2x2.py
```

Each script prints to stdout only. There are no file outputs. Scripts
that need shared utilities import them from `headline_table.py` or
`_paths.py`.

## Notes on numerical reproducibility

Small numerical differences from the printed paper values (typically
within 0.1 pp) are expected and arise from floating-point aggregation
order or judge non-determinism at temperature 0. The Arena Elo values
in `arena_correlation.py` are paper-time approximations; users may
update the dict with current values without changing the rest of the
pipeline.

The mixed-effects model in `tier_test.py` will use a binomial GLMM if
`statsmodels.genmod.bayes_mixed_glm.BinomialBayesMixedGLM` is available
and converges; otherwise it falls back to a logistic regression with
cluster-robust standard errors clustered on `scenario_id` and prints a
notice. Both paths report a chi-squared statistic, df, and p-value.
