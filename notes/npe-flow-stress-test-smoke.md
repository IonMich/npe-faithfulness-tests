# NPE Flow Stress-Test Results

This report compares exact-likelihood random-walk MCMC, exact-likelihood HMC, and conditional normalizing-flow NPE.

| case | MCMC diag ok | HMC diag ok | max mean diag W | target met | MCMC-HMC | MCMC-NPE | HMC-NPE |
|---|---:|---:|---:|---:|---:|---:|---:|
| banana | False | True | 0.2251 | False | 0.1372 | 0.2251 | 0.1611 |

## banana

- Summary JSON: `runs/03_stress_banana/01_npe_flow/16_npe_flow_stress_tests_smoke_banana/results/banana_summary.json`
- Corner overlay: `runs/03_stress_banana/01_npe_flow/16_npe_flow_stress_tests_smoke_banana/figures/banana_mcmc_hmc_npe_corner.png`
- Trace plot: `runs/03_stress_banana/01_npe_flow/16_npe_flow_stress_tests_smoke_banana/figures/banana_trace.png`
- Predictive plot: `runs/03_stress_banana/01_npe_flow/16_npe_flow_stress_tests_smoke_banana/figures/banana_predictive.png`
- Runtime seconds: MCMC 0.05, HMC 0.98, NPE train 0.11
- MCMC acceptance: 0.662
- HMC acceptance: 0.993
