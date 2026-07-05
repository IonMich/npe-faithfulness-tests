# Stress Test: Banana

Runs are grouped by method folder inside this model.

| Status | Run | Metric | Target | Reason |
| --- | --- | --- | --- | --- |
| `floor_pass` | [03_population_npe / 01_flow2_residual_full_prior_512k_ensemble4](03_population_npe/01_flow2_residual_full_prior_512k_ensemble4) | full-prior raw NLL: `-0.52753 +/- 0.00100` | entropy floor: `-0.52826 +/- 0.00100` | Gap is `0.00073`, or `0.52` combined SE, on 1M full-prior validation examples. |
| `legacy_pairwise_pass` | [01_npe_flow / 03_npe_flow_stress_tests_banana_q008](01_npe_flow/03_npe_flow_stress_tests_banana_q008) | pairwise max diagnostic Wasserstein: 0.01844 | 0.034 inherited | Model-specific calibration pending |
| `fail` | [01_npe_flow / 01_npe_flow_stress_tests_banana](01_npe_flow/01_npe_flow_stress_tests_banana) | max diagnostic Wasserstein: 0.0476 | 0.034 | MCMC, HMC, and NPE agreement target |
| `fail` | [01_npe_flow / 02_npe_flow_stress_tests_banana_nolinear](01_npe_flow/02_npe_flow_stress_tests_banana_nolinear) | max diagnostic Wasserstein: 0.07106 | 0.034 | MCMC, HMC, and NPE agreement target |
| `fail` | [01_npe_flow / 04_npe_flow_stress_tests_banana_q10](01_npe_flow/04_npe_flow_stress_tests_banana_q10) | max diagnostic Wasserstein: 0.07013 | 0.034 | MCMC, HMC, and NPE agreement target |
| `fail` | [01_npe_flow / 16_npe_flow_stress_tests_smoke_banana](01_npe_flow/16_npe_flow_stress_tests_smoke_banana) | max diagnostic Wasserstein: 0.2251 | 0.034 | MCMC, HMC, and NPE agreement target |
