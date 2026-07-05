# Stress Test: Linear 6D

Runs are grouped by method folder inside this model.

| Status | Run | Metric | Target | Reason |
| --- | --- | --- | --- | --- |
| `near_floor` | [03_population_npe / 01_flow2_residual_full_prior_512k_ensemble4](03_population_npe/01_flow2_residual_full_prior_512k_ensemble4) | full-prior z-NLL: `-10.77984 +/- 0.00353` | entropy floor: `-10.78631 +/- 0.00353` | Population-trained Linear6 NPE is close to the full-prior NLL floor; gap is `0.00647` in z units |
| `legacy_pairwise_pass` | [01_npe_flow / 13_npe_flow_stress_tests_linear6_q008](01_npe_flow/13_npe_flow_stress_tests_linear6_q008) | pairwise max diagnostic Wasserstein: 0.03301 | 0.034 inherited | Model-specific calibration pending |
| `fail` | [01_npe_flow / 12_npe_flow_stress_tests_linear6](01_npe_flow/12_npe_flow_stress_tests_linear6) | max diagnostic Wasserstein: 0.04737 | 0.034 | MCMC, HMC, and NPE agreement target |
