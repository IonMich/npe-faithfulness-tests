# NPE Flow Stress-Test Results

This report compares exact-likelihood random-walk MCMC, exact-likelihood HMC, and conditional normalizing-flow NPE.

| case | MCMC diag ok | HMC diag ok | max mean diag W | target met | MCMC-HMC | MCMC-NPE | HMC-NPE |
|---|---:|---:|---:|---:|---:|---:|---:|
| banana | True | True | 0.0711 | False | 0.0120 | 0.0711 | 0.0648 |

## banana

- Summary JSON: `runs/03_stress_banana/01_npe_flow/02_npe_flow_stress_tests_banana_nolinear/results/banana_summary.json`
- Corner overlay: `runs/03_stress_banana/01_npe_flow/02_npe_flow_stress_tests_banana_nolinear/figures/banana_mcmc_hmc_npe_corner.png`
- Trace plot: `runs/03_stress_banana/01_npe_flow/02_npe_flow_stress_tests_banana_nolinear/figures/banana_trace.png`
- Predictive plot: `runs/03_stress_banana/01_npe_flow/02_npe_flow_stress_tests_banana_nolinear/figures/banana_predictive.png`
- Runtime seconds: MCMC 4.26, HMC 22.53, NPE train 130.91
- MCMC acceptance: 0.500
- HMC acceptance: 0.981
