# NPE Flow Stress-Test Results

This report compares exact-likelihood random-walk MCMC, exact-likelihood HMC, and conditional normalizing-flow NPE.

| case | MCMC diag ok | HMC diag ok | max mean diag W | target met | MCMC-HMC | MCMC-NPE | HMC-NPE |
|---|---:|---:|---:|---:|---:|---:|---:|
| two_exp_ordered | True | True | 0.2604 | False | 0.0193 | 0.2517 | 0.2604 |

## two_exp_ordered

- Summary JSON: `runs/06_two_exponential/01_npe_flow/10_npe_flow_stress_tests_two_exp_ordered_proposal/results/two_exp_ordered_summary.json`
- Corner overlay: `runs/06_two_exponential/01_npe_flow/10_npe_flow_stress_tests_two_exp_ordered_proposal/figures/two_exp_ordered_mcmc_hmc_npe_corner.png`
- Trace plot: `runs/06_two_exponential/01_npe_flow/10_npe_flow_stress_tests_two_exp_ordered_proposal/figures/two_exp_ordered_trace.png`
- Predictive plot: `runs/06_two_exponential/01_npe_flow/10_npe_flow_stress_tests_two_exp_ordered_proposal/figures/two_exp_ordered_predictive.png`
- Runtime seconds: MCMC 15.16, HMC 49.83, NPE train 321.17
- MCMC acceptance: 0.342
- HMC acceptance: 0.992
