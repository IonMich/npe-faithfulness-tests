# Calibrated Successful And Reference Runs

Use this page when you want the shortest path to runs that either passed a calibrated faithfulness target or establish a reference baseline.

Runs that only passed the inherited `0.034` pairwise agreement threshold for a different model are no longer listed as successful here.

## Calibrated Target-Passing Runs

| Group | Run | Metric | Reason |
| --- | --- | --- | --- |
| `01_exponential_decay` | [03_npe_flow_search / 11_npe_flow_local_q0005_linear_150k_t8_seed20260706](../01_exponential_decay/03_npe_flow_search/11_npe_flow_local_q0005_linear_150k_t8_seed20260706) | mean normalized Wasserstein: 0.0331 / target 0.03348 | NPE target pass flag |
| `01_exponential_decay` | [05_abc_faithfulness / 02_abc_faithfulness_scaled2m_snpe_diag_refined](../01_exponential_decay/05_abc_faithfulness/02_abc_faithfulness_scaled2m_snpe_diag_refined) | best mean normalized Wasserstein: 0.03204 / target 0.03348 | ABC best-result faithfulness flag |
| `01_exponential_decay` | [05_abc_faithfulness / 07_abc_faithfulness_validation_snpe_diag_refined](../01_exponential_decay/05_abc_faithfulness/07_abc_faithfulness_validation_snpe_diag_refined) | best mean normalized Wasserstein: 0.03129 / target 0.03348 | ABC best-result faithfulness flag |
| `02_stress_sign` | [01_npe_flow / 21_npe_flow_stress_tests_sign_absfold_q008_linear](../02_stress_sign/01_npe_flow/21_npe_flow_stress_tests_sign_absfold_q008_linear) | NPE-to-grid diagnostic W: 0.02326 / calibrated target 0.02331 | exact-grid calibrated MCMC/HMC faithfulness target |

## Reference Or Diagnostic-Passing Runs

| Group | Run | Metric | Reason |
| --- | --- | --- | --- |
| `01_exponential_decay` | [01_mcmc_hmc_reference / 00_root_decay_sampler_results](../01_exponential_decay/01_mcmc_hmc_reference/00_root_decay_sampler_results) | best mean normalized Wasserstein: 0.03162 | diagnostic/reference metric |
| `01_exponential_decay` | [01_mcmc_hmc_reference / 01_hmc_mps](../01_exponential_decay/01_mcmc_hmc_reference/01_hmc_mps) | convergence:  | sampler convergence reference |
| `01_exponential_decay` | [01_mcmc_hmc_reference / 02_mcmc_mps](../01_exponential_decay/01_mcmc_hmc_reference/02_mcmc_mps) | convergence:  | sampler convergence reference |
| `01_exponential_decay` | [02_npe_stage1_local_summary / 01_npe_focused](../01_exponential_decay/02_npe_stage1_local_summary/01_npe_focused) | best mean normalized Wasserstein: 0.03061 | diagnostic/reference metric |
| `01_exponential_decay` | [06_oracle_target_checks / 01_faithfulness_target_check](../01_exponential_decay/06_oracle_target_checks/01_faithfulness_target_check) | best mean normalized Wasserstein: 0.03162 | diagnostic/reference metric |

## Near Misses

These runs did not meet the target but are close enough to be useful for comparison.

| Group | Run | Metric | Reason |
| --- | --- | --- | --- |
| `01_exponential_decay` | [05_abc_faithfulness / 04_abc_faithfulness_scaled_snpe_diag_refined](../01_exponential_decay/05_abc_faithfulness/04_abc_faithfulness_scaled_snpe_diag_refined) | best mean normalized Wasserstein: 0.03411 / target 0.03348 | ABC best-result faithfulness flag |
| `01_exponential_decay` | [03_npe_flow_search / 09_npe_flow_local_q0005_linear_100k_t8](../01_exponential_decay/03_npe_flow_search/09_npe_flow_local_q0005_linear_100k_t8) | mean normalized Wasserstein: 0.03472 / target 0.03348 | NPE target pass flag |
| `01_exponential_decay` | [06_oracle_target_checks / 02_oracle_posterior_fit](../01_exponential_decay/06_oracle_target_checks/02_oracle_posterior_fit) | best discovered target metric: 0.03481 / target 0.03348 | nested target flags were false |
| `01_exponential_decay` | [05_abc_faithfulness / 01_abc_faithfulness](../01_exponential_decay/05_abc_faithfulness/01_abc_faithfulness) | best mean normalized Wasserstein: 0.03557 / target 0.03348 | ABC best-result faithfulness flag |
| `01_exponential_decay` | [03_npe_flow_search / 08_npe_flow_local_q0005_linear_100k_t4_seed20260701](../01_exponential_decay/03_npe_flow_search/08_npe_flow_local_q0005_linear_100k_t4_seed20260701) | mean normalized Wasserstein: 0.03663 / target 0.03348 | NPE target pass flag |
| `01_exponential_decay` | [05_abc_faithfulness / 03_abc_faithfulness_scaled_snpe_diag](../01_exponential_decay/05_abc_faithfulness/03_abc_faithfulness_scaled_snpe_diag) | best mean normalized Wasserstein: 0.03828 / target 0.03348 | ABC best-result faithfulness flag |
| `01_exponential_decay` | [03_npe_flow_search / 10_npe_flow_local_q0005_linear_100k_t8_seed20260703](../01_exponential_decay/03_npe_flow_search/10_npe_flow_local_q0005_linear_100k_t8_seed20260703) | mean normalized Wasserstein: 0.03832 / target 0.03348 | NPE target pass flag |
| `01_exponential_decay` | [03_npe_flow_search / 12_npe_flow_local_q0005_linear_40k_t8](../01_exponential_decay/03_npe_flow_search/12_npe_flow_local_q0005_linear_40k_t8) | mean normalized Wasserstein: 0.03868 / target 0.03348 | NPE target pass flag |
| `02_stress_sign` | [01_npe_flow / 14_npe_flow_stress_tests_medium](../02_stress_sign/01_npe_flow/14_npe_flow_stress_tests_medium) | max diagnostic Wasserstein: 0.04143 / target 0.034 | MCMC, HMC, and NPE agreement target |

## Legacy Pairwise Passes

These runs passed the inherited `0.034` pairwise agreement threshold, but they
are not categorized as successful until their model has a calibrated truth or
reference target.

| Group | Run | Metric | Status |
| --- | --- | --- | --- |
| `02_stress_sign` | [01_npe_flow / 01_npe_flow_stress_tests_sign](../02_stress_sign/01_npe_flow/01_npe_flow_stress_tests_sign) | pairwise max diagnostic Wasserstein: 0.02691 / inherited target 0.034 | superseded by grid-faithful run 21 |
| `02_stress_sign` | [01_npe_flow / 15_npe_flow_stress_tests_sign_nolinear](../02_stress_sign/01_npe_flow/15_npe_flow_stress_tests_sign_nolinear) | pairwise max diagnostic Wasserstein: 0.02691 / inherited target 0.034 | superseded by grid-faithful run 21 |
| `03_stress_banana` | [01_npe_flow / 03_npe_flow_stress_tests_banana_q008](../03_stress_banana/01_npe_flow/03_npe_flow_stress_tests_banana_q008) | pairwise max diagnostic Wasserstein: 0.01844 / inherited target 0.034 | model-specific calibration pending |
| `04_stress_label_switch` | [01_npe_flow / 05_npe_flow_stress_tests_label_em](../04_stress_label_switch/01_npe_flow/05_npe_flow_stress_tests_label_em) | pairwise max diagnostic Wasserstein: 0.02868 / inherited target 0.034 | model-specific calibration pending |
| `05_stress_linear6` | [01_npe_flow / 13_npe_flow_stress_tests_linear6_q008](../05_stress_linear6/01_npe_flow/13_npe_flow_stress_tests_linear6_q008) | pairwise max diagnostic Wasserstein: 0.03301 / inherited target 0.034 | model-specific calibration pending |
