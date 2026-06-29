# Stress Test: Banana

Runs are grouped by method folder inside this model.

| Status | Run | Metric | Target | Reason |
| --- | --- | --- | --- | --- |
| `legacy_pairwise_pass` | [01_npe_flow / 03_npe_flow_stress_tests_banana_q008](01_npe_flow/03_npe_flow_stress_tests_banana_q008) | pairwise max diagnostic Wasserstein: 0.01844 | 0.034 inherited | Model-specific calibration pending |
| `fail` | [01_npe_flow / 01_npe_flow_stress_tests_banana](01_npe_flow/01_npe_flow_stress_tests_banana) | max diagnostic Wasserstein: 0.0476 | 0.034 | MCMC, HMC, and NPE agreement target |
| `fail` | [01_npe_flow / 02_npe_flow_stress_tests_banana_nolinear](01_npe_flow/02_npe_flow_stress_tests_banana_nolinear) | max diagnostic Wasserstein: 0.07106 | 0.034 | MCMC, HMC, and NPE agreement target |
| `fail` | [01_npe_flow / 04_npe_flow_stress_tests_banana_q10](01_npe_flow/04_npe_flow_stress_tests_banana_q10) | max diagnostic Wasserstein: 0.07013 | 0.034 | MCMC, HMC, and NPE agreement target |
| `fail` | [01_npe_flow / 16_npe_flow_stress_tests_smoke_banana](01_npe_flow/16_npe_flow_stress_tests_smoke_banana) | max diagnostic Wasserstein: 0.2251 | 0.034 | MCMC, HMC, and NPE agreement target |
