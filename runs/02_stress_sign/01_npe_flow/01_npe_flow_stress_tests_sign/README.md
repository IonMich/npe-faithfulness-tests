# 01_npe_flow / 01_npe_flow_stress_tests_sign

Status: `legacy_pairwise_pass`
Reason: passed the inherited MCMC/HMC/NPE pairwise agreement target, but failed the sign-specific exact-grid target.
Metric: pairwise max diagnostic Wasserstein = `0.0269104`; NPE-to-grid diagnostic W = `0.0326068`
Target: inherited pairwise target `0.034`; calibrated sign target `0.023314`

Artifacts:
- `results/` - result files and summary JSON for this run
- `figures/` - plots for this run, when available
- script: `scripts/npe_flow_stress_tests.py`
- note: `notes/npe-flow-model-results.md`
