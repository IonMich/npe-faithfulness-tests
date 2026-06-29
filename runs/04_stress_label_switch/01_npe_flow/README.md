# Stress Test: Label Switching / 01_npe_flow

| Status | Run | Metric | Target | Reason |
| --- | --- | --- | --- | --- |
| `legacy_pairwise_pass` | [05_npe_flow_stress_tests_label_em](05_npe_flow_stress_tests_label_em) | pairwise max diagnostic Wasserstein: 0.02868 | 0.034 inherited | Model-specific calibration pending |
| `fail` | [06_npe_flow_stress_tests_label_enhanced](06_npe_flow_stress_tests_label_enhanced) | max diagnostic Wasserstein: 0.242 | 0.034 | MCMC, HMC, and NPE agreement target |
| `fail` | [07_npe_flow_stress_tests_label_medium](07_npe_flow_stress_tests_label_medium) | max diagnostic Wasserstein: 0.3486 | 0.034 | MCMC, HMC, and NPE agreement target |
| `fail` | [08_npe_flow_stress_tests_label_ordered](08_npe_flow_stress_tests_label_ordered) | max diagnostic Wasserstein: 0.1942 | 0.034 | MCMC, HMC, and NPE agreement target |
| `fail` | [09_npe_flow_stress_tests_label_ordered_nolinear](09_npe_flow_stress_tests_label_ordered_nolinear) | max diagnostic Wasserstein: 0.2432 | 0.034 | MCMC, HMC, and NPE agreement target |
| `fail` | [10_npe_flow_stress_tests_label_ordered_q008](10_npe_flow_stress_tests_label_ordered_q008) | max diagnostic Wasserstein: 0.1666 | 0.034 | MCMC, HMC, and NPE agreement target |
| `fail` | [11_npe_flow_stress_tests_label_q008](11_npe_flow_stress_tests_label_q008) | max diagnostic Wasserstein: 0.2668 | 0.034 | MCMC, HMC, and NPE agreement target |
