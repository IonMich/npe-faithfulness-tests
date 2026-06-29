# NPE Flow Stress-Test Results

This report compares exact-likelihood random-walk MCMC, exact-likelihood HMC, and conditional normalizing-flow NPE.

| case | MCMC diag ok | HMC diag ok | max mean diag W | target met | MCMC-HMC | MCMC-NPE | HMC-NPE |
|---|---:|---:|---:|---:|---:|---:|---:|
| label_switch | True | True | 0.2420 | False | 0.0147 | 0.2420 | 0.2377 |

## label_switch

- Summary JSON: `runs/04_stress_label_switch/01_npe_flow/06_npe_flow_stress_tests_label_enhanced/results/label_switch_summary.json`
- Corner overlay: `runs/04_stress_label_switch/01_npe_flow/06_npe_flow_stress_tests_label_enhanced/figures/label_switch_mcmc_hmc_npe_corner.png`
- Trace plot: `runs/04_stress_label_switch/01_npe_flow/06_npe_flow_stress_tests_label_enhanced/figures/label_switch_trace.png`
- Predictive plot: `runs/04_stress_label_switch/01_npe_flow/06_npe_flow_stress_tests_label_enhanced/figures/label_switch_predictive.png`
- Runtime seconds: MCMC 1.83, HMC 35.41, NPE train 471.51
- MCMC acceptance: 0.493
- HMC acceptance: 0.996

- Mode metrics: `{"mcmc": {"mu1_less_than_mu2_fraction": 0.49914, "mode_mass_error_vs_half": 0.0008600000000000274}, "hmc": {"mu1_less_than_mu2_fraction": 0.5, "mode_mass_error_vs_half": 0.0}, "npe": {"mu1_less_than_mu2_fraction": 0.49178, "mode_mass_error_vs_half": 0.008220000000000005}}`
