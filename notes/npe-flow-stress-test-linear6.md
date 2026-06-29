# NPE Flow Stress-Test Results

This report compares exact-likelihood random-walk MCMC, exact-likelihood HMC, and conditional normalizing-flow NPE.

| case | MCMC diag ok | HMC diag ok | max mean diag W | target met | MCMC-HMC | MCMC-NPE | HMC-NPE |
|---|---:|---:|---:|---:|---:|---:|---:|
| linear6 | True | True | 0.0474 | False | 0.0289 | 0.0474 | 0.0355 |

## linear6

- Summary JSON: `runs/05_stress_linear6/01_npe_flow/12_npe_flow_stress_tests_linear6/results/linear6_summary.json`
- Corner overlay: `runs/05_stress_linear6/01_npe_flow/12_npe_flow_stress_tests_linear6/figures/linear6_mcmc_hmc_npe_corner.png`
- Trace plot: `runs/05_stress_linear6/01_npe_flow/12_npe_flow_stress_tests_linear6/figures/linear6_trace.png`
- Predictive plot: `runs/05_stress_linear6/01_npe_flow/12_npe_flow_stress_tests_linear6/figures/linear6_predictive.png`
- Runtime seconds: MCMC 2.25, HMC 29.88, NPE train 269.82
- MCMC acceptance: 0.115
- HMC acceptance: 0.930
