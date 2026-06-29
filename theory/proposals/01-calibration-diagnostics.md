# Proposal 01: Add NPE Calibration Diagnostics

## Claim

Add a reusable calibration script for trained NPE posteriors. Reference
agreement and corner plots are not enough; flow-based NPE can be visually close
while still overconfident or biased.

## Literature Signal

- Hermans et al. show that SBI posteriors, including NPE/SNPE, can be
  computationally unfaithful and overconfident:
  https://arxiv.org/abs/2110.06581
- SBC is a standard posterior-sampler validation tool:
  https://arxiv.org/abs/1804.06788
- TARP tests posterior estimators from samples:
  https://arxiv.org/abs/2302.03026
- `sbi` recommends expected coverage, SBC, TARP, L-C2ST, and model
  misspecification checks:
  https://sbi.readthedocs.io/en/latest/how_to_guide/14_choose_diagnostic_tool.html

## Current Code Touchpoints

- Custom NPE flow artifacts are saved in
  `scripts/npe_flow_stress_tests.py::run_npe`.
- Pairwise reference agreement is computed in
  `scripts/npe_flow_stress_tests.py::pairwise_agreement`.
- Posterior predictive overlays already exist in
  `scripts/npe_flow_stress_tests.py::plot_predictive_overlay`.
- `sbi` posteriors in `scripts/sbi_two_exp_ordered.py` can use `sbi.diagnostics`
  directly.

## Implementation Sketch

Create:

```text
scripts/check_npe_calibration.py
```

Core behavior:

- load a trained posterior artifact or rerun a lightweight posterior builder;
- generate held-out prior-predictive pairs `(theta_i, x_i)`;
- infer posterior samples for each `x_i`;
- compute SBC ranks for each target dimension;
- compute expected coverage on joint log-probability or distance statistics;
- compute TARP when using posterior samples only;
- save JSON summaries and figures.

For custom Zuko flows, implement minimal SBC manually:

```text
rank_j(theta_true, samples) = count(samples[:, j] < theta_true[j])
```

For `sbi` posteriors, use:

```text
sbi.diagnostics.run_sbc
sbi.diagnostics.run_tarp
sbi.analysis.sbc_rank_plot
sbi.analysis.plot_tarp
```

## Acceptance Criteria

- A calibration run writes `calibration_summary.json`.
- Summary includes per-dimension SBC histogram counts and p-value or distance
  from uniformity.
- Summary includes expected coverage curve data if log probabilities are
  available.
- Summary includes TARP ATC and KS p-value where applicable.
- Figures are written for SBC and coverage/TARP.
- Existing run READMEs can link to the calibration artifact.

## First Target

Run this first on the best passing stress cases and then on the two-exponential
case. The goal is to determine whether the two-exponential failure is mainly
overconfidence, location bias, correlation error, or local misspecification.
