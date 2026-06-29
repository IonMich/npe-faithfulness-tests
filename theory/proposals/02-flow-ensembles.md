# Proposal 02: Train And Evaluate Flow Ensembles

## Claim

Add ensemble training for custom NPE flows. Single flow estimators can be
overconfident or seed-sensitive; equal-weight ensembles often improve
calibration and robustness.

## Literature Signal

- Hermans et al. report that ensembling posterior surrogates can mitigate
  overconfident SBI posteriors: https://arxiv.org/abs/2110.06581
- The practical SBI workflow recommends ensembles when individual estimators
  are overconfident or unstable: https://arxiv.org/abs/2508.12939
- `sbi` exposes an `EnsemblePosterior`, which reflects the same practical
  direction in the toolkit:
  https://sbi.readthedocs.io/en/latest/api_reference/_autosummary/sbi.inference.EnsemblePosterior.html

## Current Code Touchpoints

- `scripts/npe_flow_stress_tests.py::train_flow` trains one flow.
- `scripts/npe_flow_stress_tests.py::sample_npe` samples one trained flow.
- `scripts/npe_flow_stress_tests.py::run_npe` controls one train/validation
  split, one seed, and one posterior sample set.

## Implementation Sketch

Add CLI option:

```text
--ensemble-seeds 20260701,20260702,20260703
```

Behavior:

- collect local training data once per run;
- reuse the same accepted simulations and train/validation indices, or support
  an option to resplit per seed;
- for each ensemble seed:
  - set `torch.manual_seed(seed)`;
  - train a separate flow;
  - save `case_npe_model_seed_<seed>.pt`;
  - sample `npe_samples / n_models` samples;
- pool samples equally across models;
- report per-member and ensemble agreement metrics.

Optional later improvement:

- use validation NLL weights or stacking weights, but start with equal weights
  to avoid overfitting to NLL.

## Acceptance Criteria

- Existing single-model behavior remains unchanged when `--ensemble-seeds` is
  omitted.
- Ensemble summary contains per-member validation NLL and per-member reference
  agreement.
- Ensemble posterior samples are saved and plotted as the main NPE output.
- Summary reports whether the ensemble improves over the median single member.
- Calibration diagnostics can run on pooled ensemble samples.

## First Target

Run a 3-model ensemble on the current best two-exponential residual
configuration and on one known passing stress case. If the ensemble improves
two-exponential Wasserstein but not enough to pass, it still helps identify
seed variance versus systematic conditional bias.
