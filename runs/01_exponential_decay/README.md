# Exponential Decay

Runs are grouped by method folder inside this model.

| Status | Run | Metric | Target | Reason |
| --- | --- | --- | --- | --- |
| `pass` | [03_npe_flow_search / 11_npe_flow_local_q0005_linear_150k_t8_seed20260706](03_npe_flow_search/11_npe_flow_local_q0005_linear_150k_t8_seed20260706) | mean normalized Wasserstein: 0.0331 | 0.034 | NPE target pass flag |
| `pass` | [05_abc_faithfulness / 02_abc_faithfulness_scaled2m_snpe_diag_refined](05_abc_faithfulness/02_abc_faithfulness_scaled2m_snpe_diag_refined) | best mean normalized Wasserstein: 0.03204 | 0.034 | ABC best-result faithfulness flag |
| `pass` | [05_abc_faithfulness / 07_abc_faithfulness_validation_snpe_diag_refined](05_abc_faithfulness/07_abc_faithfulness_validation_snpe_diag_refined) | best mean normalized Wasserstein: 0.03129 | 0.034 | ABC best-result faithfulness flag |
| `diagnostic_pass` | [01_mcmc_hmc_reference / 00_root_decay_sampler_results](01_mcmc_hmc_reference/00_root_decay_sampler_results) | best mean normalized Wasserstein: 0.03162 | 0.034 | diagnostic/reference metric |
| `reference` | [01_mcmc_hmc_reference / 01_hmc_mps](01_mcmc_hmc_reference/01_hmc_mps) | convergence:  | 0.034 | sampler convergence reference |
| `reference` | [01_mcmc_hmc_reference / 02_mcmc_mps](01_mcmc_hmc_reference/02_mcmc_mps) | convergence:  | 0.034 | sampler convergence reference |
| `diagnostic_pass` | [02_npe_stage1_local_summary / 01_npe_focused](02_npe_stage1_local_summary/01_npe_focused) | best mean normalized Wasserstein: 0.03061 | 0.034 | diagnostic/reference metric |
| `fail` | [02_npe_stage1_local_summary / 02_npe_local_region_q0001_mdn_20k](02_npe_stage1_local_summary/02_npe_local_region_q0001_mdn_20k) | best discovered target metric: 0.2065 | 0.034 | nested target flags were false |
| `fail` | [02_npe_stage1_local_summary / 03_npe_local_region_q0005_flow_20k](02_npe_stage1_local_summary/03_npe_local_region_q0005_flow_20k) | best discovered target metric: 0.1512 | 0.034 | nested target flags were false |
| `fail` | [02_npe_stage1_local_summary / 04_npe_local_region_q0005_gaussians_20k](02_npe_stage1_local_summary/04_npe_local_region_q0005_gaussians_20k) | best discovered target metric: 0.08882 | 0.034 | nested target flags were false |
| `fail` | [02_npe_stage1_local_summary / 05_npe_local_region_q0005_mdn_20k](02_npe_stage1_local_summary/05_npe_local_region_q0005_mdn_20k) | best discovered target metric: 0.08666 | 0.034 | nested target flags were false |
| `fail` | [02_npe_stage1_local_summary / 06_npe_local_region_q0005_mdn_60k_h256_c16](02_npe_stage1_local_summary/06_npe_local_region_q0005_mdn_60k_h256_c16) | best discovered target metric: 0.1523 | 0.034 | nested target flags were false |
| `fail` | [02_npe_stage1_local_summary / 07_npe_local_region_q002_mdn_20k](02_npe_stage1_local_summary/07_npe_local_region_q002_mdn_20k) | best discovered target metric: 0.2316 | 0.034 | nested target flags were false |
| `fail` | [02_npe_stage1_local_summary / 08_npe_multi_x](02_npe_stage1_local_summary/08_npe_multi_x) | best discovered target metric: 0.1339 | 0.034 | nested target flags were false |
| `fail` | [02_npe_stage1_local_summary / 09_npe_multi_x_scaled](02_npe_stage1_local_summary/09_npe_multi_x_scaled) | best discovered target metric: 0.08339 | 0.034 | nested target flags were false |
| `fail` | [02_npe_stage1_local_summary / 10_npe_raw_kernel_q0005_mdn_200k](02_npe_stage1_local_summary/10_npe_raw_kernel_q0005_mdn_200k) | best discovered target metric: 0.2756 | 0.034 | nested target flags were false |
| `diagnostic` | [02_npe_stage1_local_summary / 11_npe_stage1](02_npe_stage1_local_summary/11_npe_stage1) | best mean normalized Wasserstein: 0.3291 | 0.034 | diagnostic/reference metric |
| `diagnostic` | [02_npe_stage1_local_summary / 12_npe_stage1_scaled](02_npe_stage1_local_summary/12_npe_stage1_scaled) | best mean normalized Wasserstein: 0.1564 | 0.034 | diagnostic/reference metric |
| `fail` | [02_npe_stage1_local_summary / 13_npe_summary_broad_mdn_100k](02_npe_stage1_local_summary/13_npe_summary_broad_mdn_100k) | best discovered target metric: 0.1149 | 0.034 | nested target flags were false |
| `fail` | [02_npe_stage1_local_summary / 14_npe_summary_kernel_q0005_mdn_200k](02_npe_stage1_local_summary/14_npe_summary_kernel_q0005_mdn_200k) | best discovered target metric: 0.1171 | 0.034 | nested target flags were false |
| `fail` | [02_npe_stage1_local_summary / 15_npe_summary_local_q0005_mdn_diag_20k](02_npe_stage1_local_summary/15_npe_summary_local_q0005_mdn_diag_20k) | best discovered target metric: 0.1038 | 0.034 | nested target flags were false |
| `fail` | [03_npe_flow_search / 01_npe_flow_enhanced_q0005_linear_100k_t8](03_npe_flow_search/01_npe_flow_enhanced_q0005_linear_100k_t8) | mean normalized Wasserstein: 0.04989 | 0.034 | NPE target pass flag |
| `fail` | [03_npe_flow_search / 02_npe_flow_enhanced_smoke](03_npe_flow_search/02_npe_flow_enhanced_smoke) | mean normalized Wasserstein: 0.1947 | 0.034 | NPE target pass flag |
| `fail` | [03_npe_flow_search / 03_npe_flow_local_linear_smoke](03_npe_flow_search/03_npe_flow_local_linear_smoke) | mean normalized Wasserstein: 0.1389 | 0.034 | NPE target pass flag |
| `fail` | [03_npe_flow_search / 04_npe_flow_local_q0001_40k_t8](03_npe_flow_search/04_npe_flow_local_q0001_40k_t8) | mean normalized Wasserstein: 0.05944 | 0.034 | NPE target pass flag |
| `fail` | [03_npe_flow_search / 05_npe_flow_local_q00025_linear_80k_t8_seed20260702](03_npe_flow_search/05_npe_flow_local_q00025_linear_80k_t8_seed20260702) | mean normalized Wasserstein: 0.04717 | 0.034 | NPE target pass flag |
| `fail` | [03_npe_flow_search / 06_npe_flow_local_q0005_40k_t8](03_npe_flow_search/06_npe_flow_local_q0005_40k_t8) | mean normalized Wasserstein: 0.04977 | 0.034 | NPE target pass flag |
| `fail` | [03_npe_flow_search / 07_npe_flow_local_q0005_kernel0025_linear_100k_t8](03_npe_flow_search/07_npe_flow_local_q0005_kernel0025_linear_100k_t8) | mean normalized Wasserstein: 0.04469 | 0.034 | NPE target pass flag |
| `near` | [03_npe_flow_search / 08_npe_flow_local_q0005_linear_100k_t4_seed20260701](03_npe_flow_search/08_npe_flow_local_q0005_linear_100k_t4_seed20260701) | mean normalized Wasserstein: 0.03663 | 0.034 | NPE target pass flag |
| `near` | [03_npe_flow_search / 09_npe_flow_local_q0005_linear_100k_t8](03_npe_flow_search/09_npe_flow_local_q0005_linear_100k_t8) | mean normalized Wasserstein: 0.03472 | 0.034 | NPE target pass flag |
| `near` | [03_npe_flow_search / 10_npe_flow_local_q0005_linear_100k_t8_seed20260703](03_npe_flow_search/10_npe_flow_local_q0005_linear_100k_t8_seed20260703) | mean normalized Wasserstein: 0.03832 | 0.034 | NPE target pass flag |
| `near` | [03_npe_flow_search / 12_npe_flow_local_q0005_linear_40k_t8](03_npe_flow_search/12_npe_flow_local_q0005_linear_40k_t8) | mean normalized Wasserstein: 0.03868 | 0.034 | NPE target pass flag |
| `fail` | [03_npe_flow_search / 13_npe_flow_local_q002_kernel005_80k_t8](03_npe_flow_search/13_npe_flow_local_q002_kernel005_80k_t8) | mean normalized Wasserstein: 0.05958 | 0.034 | NPE target pass flag |
| `fail` | [03_npe_flow_search / 14_npe_flow_local_smoke](03_npe_flow_search/14_npe_flow_local_smoke) | mean normalized Wasserstein: 2.625 | 0.034 | NPE target pass flag |
| `fail` | [03_npe_flow_search / 15_npe_flow_smoke](03_npe_flow_search/15_npe_flow_smoke) | mean normalized Wasserstein: 3.27 | 0.034 | NPE target pass flag |
| `fail` | [03_npe_flow_search / 16_npe_flow_smoke_mixture](03_npe_flow_search/16_npe_flow_smoke_mixture) | mean normalized Wasserstein: 12.03 | 0.034 | NPE target pass flag |
| `fail` | [03_npe_flow_search / 18_npe_flow_snpe_diag_mix005_150k_t8](03_npe_flow_search/18_npe_flow_snpe_diag_mix005_150k_t8) | mean normalized Wasserstein: 0.08698 | 0.034 | NPE target pass flag |
| `fail` | [03_npe_flow_search / 19_npe_flow_snpe_diag_mix02_100k_t6](03_npe_flow_search/19_npe_flow_snpe_diag_mix02_100k_t6) | mean normalized Wasserstein: 0.06442 | 0.034 | NPE target pass flag |
| `fail` | [04_snpe_sbi / 01_sbi_snle_maf_25k](04_snpe_sbi/01_sbi_snle_maf_25k) | best discovered target metric: 0.2534 | 0.034 | nested target flags were false |
| `fail` | [04_snpe_sbi / 02_sbi_snre_resnet_25k](04_snpe_sbi/02_sbi_snre_resnet_25k) | best discovered target metric: 1.337 | 0.034 | nested target flags were false |
| `fail` | [04_snpe_sbi / 03_snpe_sbi_maf_r4_n25k](04_snpe_sbi/03_snpe_sbi_maf_r4_n25k) | best discovered target metric: 0.4241 | 0.034 | nested target flags were false |
| `fail` | [04_snpe_sbi / 05_snpe_sbi_summary_smoke](04_snpe_sbi/05_snpe_sbi_summary_smoke) | best discovered target metric: 13.28 | 0.034 | nested target flags were false |
| `fail` | [04_snpe_sbi / 06_snpe_sequential_gaussians_r4_n25k](04_snpe_sbi/06_snpe_sequential_gaussians_r4_n25k) | best discovered target metric: 0.08479 | 0.034 | nested target flags were false |
| `fail` | [04_snpe_sbi / 07_snpe_sequential_mdn_r4_n25k](04_snpe_sbi/07_snpe_sequential_mdn_r4_n25k) | best discovered target metric: 0.08782 | 0.034 | nested target flags were false |
| `fail` | [04_snpe_sbi / 08_snpe_sequential_mdn_r6_n25k_infl15](04_snpe_sbi/08_snpe_sequential_mdn_r6_n25k_infl15) | best discovered target metric: 0.115 | 0.034 | nested target flags were false |
| `near` | [05_abc_faithfulness / 01_abc_faithfulness](05_abc_faithfulness/01_abc_faithfulness) | best mean normalized Wasserstein: 0.03557 | 0.034 | ABC best-result faithfulness flag |
| `near` | [05_abc_faithfulness / 03_abc_faithfulness_scaled_snpe_diag](05_abc_faithfulness/03_abc_faithfulness_scaled_snpe_diag) | best mean normalized Wasserstein: 0.03828 | 0.034 | ABC best-result faithfulness flag |
| `near` | [05_abc_faithfulness / 04_abc_faithfulness_scaled_snpe_diag_refined](05_abc_faithfulness/04_abc_faithfulness_scaled_snpe_diag_refined) | best mean normalized Wasserstein: 0.03411 | 0.034 | ABC best-result faithfulness flag |
| `fail` | [05_abc_faithfulness / 05_abc_faithfulness_smoke](05_abc_faithfulness/05_abc_faithfulness_smoke) | best mean normalized Wasserstein: 0.0974 | 0.034 | ABC best-result faithfulness flag |
| `fail` | [05_abc_faithfulness / 06_abc_faithfulness_smoke_refined](05_abc_faithfulness/06_abc_faithfulness_smoke_refined) | best mean normalized Wasserstein: 0.09696 | 0.034 | ABC best-result faithfulness flag |
| `diagnostic_pass` | [06_oracle_target_checks / 01_faithfulness_target_check](06_oracle_target_checks/01_faithfulness_target_check) | best mean normalized Wasserstein: 0.03162 | 0.034 | diagnostic/reference metric |
| `near` | [06_oracle_target_checks / 02_oracle_posterior_fit](06_oracle_target_checks/02_oracle_posterior_fit) | best discovered target metric: 0.03481 | 0.034 | nested target flags were false |
