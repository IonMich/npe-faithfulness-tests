# NPE Flow Stress-Test Results

This report compares exact-likelihood random-walk MCMC, exact-likelihood HMC, and conditional normalizing-flow NPE.

| case | MCMC diag ok | HMC diag ok | max mean diag W | target met | MCMC-HMC | MCMC-NPE | HMC-NPE |
|---|---:|---:|---:|---:|---:|---:|---:|
| banana | True | True | 0.0184 | True | 0.0120 | 0.0184 | 0.0166 |

## banana

- Summary JSON: `runs/03_stress_banana/01_npe_flow/03_npe_flow_stress_tests_banana_q008/results/banana_summary.json`
- Corner overlay: `runs/03_stress_banana/01_npe_flow/03_npe_flow_stress_tests_banana_q008/figures/banana_mcmc_hmc_npe_corner.png`
- Trace plot: `runs/03_stress_banana/01_npe_flow/03_npe_flow_stress_tests_banana_q008/figures/banana_trace.png`
- Predictive plot: `runs/03_stress_banana/01_npe_flow/03_npe_flow_stress_tests_banana_q008/figures/banana_predictive.png`
- Runtime seconds: MCMC 1.50, HMC 14.78, NPE train 128.99
- MCMC acceptance: 0.500
- HMC acceptance: 0.981
