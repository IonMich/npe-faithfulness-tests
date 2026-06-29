# Stress Test: Linear 6D

Runs are grouped by method folder inside this model.

| Status | Run | Metric | Target | Reason |
| --- | --- | --- | --- | --- |
| `legacy_pairwise_pass` | [01_npe_flow / 13_npe_flow_stress_tests_linear6_q008](01_npe_flow/13_npe_flow_stress_tests_linear6_q008) | pairwise max diagnostic Wasserstein: 0.03301 | 0.034 inherited | Model-specific calibration pending |
| `fail` | [01_npe_flow / 12_npe_flow_stress_tests_linear6](01_npe_flow/12_npe_flow_stress_tests_linear6) | max diagnostic Wasserstein: 0.04737 | 0.034 | MCMC, HMC, and NPE agreement target |
