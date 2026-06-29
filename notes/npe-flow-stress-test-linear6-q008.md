# NPE Flow Stress-Test Results

This report compares exact-likelihood random-walk MCMC, exact-likelihood HMC, and conditional normalizing-flow NPE.

| case | MCMC diag ok | HMC diag ok | max mean diag W | target met | MCMC-HMC | MCMC-NPE | HMC-NPE |
|---|---:|---:|---:|---:|---:|---:|---:|
| linear6 | True | True | 0.0330 | True | 0.0155 | 0.0330 | 0.0292 |

## linear6

- Summary JSON: `runs/05_stress_linear6/01_npe_flow/13_npe_flow_stress_tests_linear6_q008/results/linear6_summary.json`
- Corner overlay: `runs/05_stress_linear6/01_npe_flow/13_npe_flow_stress_tests_linear6_q008/figures/linear6_mcmc_hmc_npe_corner.png`
- Trace plot: `runs/05_stress_linear6/01_npe_flow/13_npe_flow_stress_tests_linear6_q008/figures/linear6_trace.png`
- Predictive plot: `runs/05_stress_linear6/01_npe_flow/13_npe_flow_stress_tests_linear6_q008/figures/linear6_predictive.png`
- Runtime seconds: MCMC 5.86, HMC 24.07, NPE train 331.69
- MCMC acceptance: 0.155
- HMC acceptance: 0.930
