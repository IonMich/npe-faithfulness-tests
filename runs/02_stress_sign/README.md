# Stress Test: Sign

Runs are grouped by method folder inside this model.

| Status | Run | Metric | Target | Reason |
| --- | --- | --- | --- | --- |
| `grid-faithful` | [01_npe_flow / 21_npe_flow_stress_tests_sign_absfold_q008_linear](01_npe_flow/21_npe_flow_stress_tests_sign_absfold_q008_linear) | NPE-to-grid diagnostic W: 0.02326 | 0.02331 | Exact-grid calibrated MCMC/HMC faithfulness target |
| `legacy_pairwise_pass` | [01_npe_flow / 01_npe_flow_stress_tests_sign](01_npe_flow/01_npe_flow_stress_tests_sign) | pairwise max diagnostic Wasserstein: 0.02691 | 0.034 inherited | Superseded: NPE-to-grid diagnostic W was 0.03261 against calibrated target 0.02331 |
| `legacy_pairwise_pass` | [01_npe_flow / 15_npe_flow_stress_tests_sign_nolinear](01_npe_flow/15_npe_flow_stress_tests_sign_nolinear) | pairwise max diagnostic Wasserstein: 0.02691 | 0.034 inherited | Superseded by grid-faithful run 21 |
| `near` | [01_npe_flow / 14_npe_flow_stress_tests_medium](01_npe_flow/14_npe_flow_stress_tests_medium) | max diagnostic Wasserstein: 0.04143 | 0.034 | MCMC, HMC, and NPE agreement target |
| `fail` | [01_npe_flow / 16_npe_flow_stress_tests_smoke_sign](01_npe_flow/16_npe_flow_stress_tests_smoke_sign) | max diagnostic Wasserstein: 0.3692 | 0.034 | MCMC, HMC, and NPE agreement target |
