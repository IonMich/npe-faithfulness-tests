# Stress Test: Label Switching

Runs are grouped by method folder inside this model.

| Status | Run | Metric | Target | Reason |
| --- | --- | --- | --- | --- |
| `legacy_pairwise_pass` | [01_npe_flow / 05_npe_flow_stress_tests_label_em](01_npe_flow/05_npe_flow_stress_tests_label_em) | pairwise max diagnostic Wasserstein: 0.02868 | 0.034 inherited | Model-specific calibration pending |
| `fail` | [01_npe_flow / 06_npe_flow_stress_tests_label_enhanced](01_npe_flow/06_npe_flow_stress_tests_label_enhanced) | max diagnostic Wasserstein: 0.242 | 0.034 | MCMC, HMC, and NPE agreement target |
| `fail` | [01_npe_flow / 07_npe_flow_stress_tests_label_medium](01_npe_flow/07_npe_flow_stress_tests_label_medium) | max diagnostic Wasserstein: 0.3486 | 0.034 | MCMC, HMC, and NPE agreement target |
| `fail` | [01_npe_flow / 08_npe_flow_stress_tests_label_ordered](01_npe_flow/08_npe_flow_stress_tests_label_ordered) | max diagnostic Wasserstein: 0.1942 | 0.034 | MCMC, HMC, and NPE agreement target |
| `fail` | [01_npe_flow / 09_npe_flow_stress_tests_label_ordered_nolinear](01_npe_flow/09_npe_flow_stress_tests_label_ordered_nolinear) | max diagnostic Wasserstein: 0.2432 | 0.034 | MCMC, HMC, and NPE agreement target |
| `fail` | [01_npe_flow / 10_npe_flow_stress_tests_label_ordered_q008](01_npe_flow/10_npe_flow_stress_tests_label_ordered_q008) | max diagnostic Wasserstein: 0.1666 | 0.034 | MCMC, HMC, and NPE agreement target |
| `fail` | [01_npe_flow / 11_npe_flow_stress_tests_label_q008](01_npe_flow/11_npe_flow_stress_tests_label_q008) | max diagnostic Wasserstein: 0.2668 | 0.034 | MCMC, HMC, and NPE agreement target |
