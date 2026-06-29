# Stress Test: Linear 6D / 01_npe_flow

| Status | Run | Metric | Target | Reason |
| --- | --- | --- | --- | --- |
| `legacy_pairwise_pass` | [13_npe_flow_stress_tests_linear6_q008](13_npe_flow_stress_tests_linear6_q008) | pairwise max diagnostic Wasserstein: 0.03301 | 0.034 inherited | Model-specific calibration pending |
| `fail` | [12_npe_flow_stress_tests_linear6](12_npe_flow_stress_tests_linear6) | max diagnostic Wasserstein: 0.04737 | 0.034 | MCMC, HMC, and NPE agreement target |
