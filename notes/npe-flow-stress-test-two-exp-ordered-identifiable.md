# NPE Flow Stress-Test Results

This report compares exact-likelihood random-walk MCMC, exact-likelihood HMC, and conditional normalizing-flow NPE.

| case | MCMC diag ok | HMC diag ok | max mean diag W | target met | MCMC-HMC | MCMC-NPE | HMC-NPE |
|---|---:|---:|---:|---:|---:|---:|---:|
| two_exp_ordered | True | True | 0.1639 | False | 0.0356 | 0.1639 | 0.1327 |

## two_exp_ordered

- Summary JSON: `runs/06_two_exponential/01_npe_flow/02_npe_flow_stress_tests_two_exp_ordered_identifiable/results/two_exp_ordered_summary.json`
- Corner overlay: `runs/06_two_exponential/01_npe_flow/02_npe_flow_stress_tests_two_exp_ordered_identifiable/figures/two_exp_ordered_mcmc_hmc_npe_corner.png`
- Trace plot: `runs/06_two_exponential/01_npe_flow/02_npe_flow_stress_tests_two_exp_ordered_identifiable/figures/two_exp_ordered_trace.png`
- Predictive plot: `runs/06_two_exponential/01_npe_flow/02_npe_flow_stress_tests_two_exp_ordered_identifiable/figures/two_exp_ordered_predictive.png`
- Runtime seconds: MCMC 10.35, HMC 52.22, NPE train 187.34
- MCMC acceptance: 0.308
- HMC acceptance: 0.989
