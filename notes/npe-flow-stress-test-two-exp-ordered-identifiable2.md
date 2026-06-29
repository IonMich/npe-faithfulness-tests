# NPE Flow Stress-Test Results

This report compares exact-likelihood random-walk MCMC, exact-likelihood HMC, and conditional normalizing-flow NPE.

| case | MCMC diag ok | HMC diag ok | max mean diag W | target met | MCMC-HMC | MCMC-NPE | HMC-NPE |
|---|---:|---:|---:|---:|---:|---:|---:|
| two_exp_ordered | True | True | 0.2067 | False | 0.0285 | 0.2067 | 0.1844 |

## two_exp_ordered

- Summary JSON: `runs/06_two_exponential/01_npe_flow/03_npe_flow_stress_tests_two_exp_ordered_identifiable2/results/two_exp_ordered_summary.json`
- Corner overlay: `runs/06_two_exponential/01_npe_flow/03_npe_flow_stress_tests_two_exp_ordered_identifiable2/figures/two_exp_ordered_mcmc_hmc_npe_corner.png`
- Trace plot: `runs/06_two_exponential/01_npe_flow/03_npe_flow_stress_tests_two_exp_ordered_identifiable2/figures/two_exp_ordered_trace.png`
- Predictive plot: `runs/06_two_exponential/01_npe_flow/03_npe_flow_stress_tests_two_exp_ordered_identifiable2/figures/two_exp_ordered_predictive.png`
- Runtime seconds: MCMC 8.97, HMC 51.68, NPE train 274.26
- MCMC acceptance: 0.116
- HMC acceptance: 0.939
