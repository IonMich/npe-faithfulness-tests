# NPE Flow Stress-Test Results

Status after the original stress-test investigation: MCMC, HMC, and normalizing-flow NPE had pairwise agreement for four stress tests. Later calibration showed that the inherited `0.034` target is not universal across models, so these pairwise passes are not all categorized as calibrated successes.

Agreement metric is mean marginal normalized Wasserstein on the diagnostic parameterization. Historical pairwise target: `<= 0.034`.

| case | MCMC diag ok | HMC diag ok | MCMC-HMC | MCMC-NPE | HMC-NPE | max mean | target met |
|---|---:|---:|---:|---:|---:|---:|---:|
| sign multimodal | true | true | 0.0186 | 0.0202 | 0.0269 | 0.0269 | true |
| banana | true | true | 0.0120 | 0.0184 | 0.0166 | 0.0184 | true |
| label-switching mixture | true | true | 0.0147 | 0.0277 | 0.0287 | 0.0287 | true |
| linear6 | true | true | 0.0155 | 0.0330 | 0.0292 | 0.0330 | true |
| two-exp ordered best so far | true | true | 0.0193 | 0.0429 | 0.0532 | 0.0532 | false |

## Pairwise-Passing Configurations

### Sign multimodality

- Grid-faithful summary: `runs/02_stress_sign/01_npe_flow/21_npe_flow_stress_tests_sign_absfold_q008_linear/results/sign_summary.json`
- Grid-faithful corner: `runs/02_stress_sign/01_npe_flow/21_npe_flow_stress_tests_sign_absfold_q008_linear/figures/sign_mcmc_hmc_npe_corner.png`
- Exact-grid calibration: `runs/02_stress_sign/02_reference_calibration/06_sign_absfold_q008_linear/results/sign_target_calibration_summary.json`
- Calibrated result: NPE-to-grid diagnostic W `0.02326` against target `0.02331`.
- Fixes needed: train the flow on `(|theta_1|, theta_2)`, restore sign symmetry after sampling, tighten the local region to `q=0.008`, and use local linear adjustment.

### Banana

- Summary: `runs/03_stress_banana/01_npe_flow/03_npe_flow_stress_tests_banana_q008/results/banana_summary.json`
- Corner: `runs/03_stress_banana/01_npe_flow/03_npe_flow_stress_tests_banana_q008/figures/banana_mcmc_hmc_npe_corner.png`
- Current status: legacy pairwise pass; model-specific calibration pending.
- Fixes needed: longer MCMC, tighter local NPE region (`q=0.008`), keep linear target adjustment.

### Label-switching mixture

- Summary: `runs/04_stress_label_switch/01_npe_flow/05_npe_flow_stress_tests_label_em/results/label_switch_summary.json`
- Corner: `runs/04_stress_label_switch/01_npe_flow/05_npe_flow_stress_tests_label_em/figures/label_switch_mcmc_hmc_npe_corner.png`
- Current status: legacy pairwise pass; model-specific calibration pending.
- Fixes needed: ordered flow parameterization, random label restoration, and EM-based context summaries.
- Raw-label R-hat is intentionally bad because chains occupy different symmetric label modes; sorted diagnostic R-hat/ESS pass.

### Linear6

- Summary: `runs/05_stress_linear6/01_npe_flow/13_npe_flow_stress_tests_linear6_q008/results/linear6_summary.json`
- Corner: `runs/05_stress_linear6/01_npe_flow/13_npe_flow_stress_tests_linear6_q008/figures/linear6_mcmc_hmc_npe_corner.png`
- Current status: legacy pairwise pass; model-specific calibration pending.
- Fixes needed: tune random-walk MCMC proposal scale and use tighter local NPE region (`q=0.008`).

## Two-Exponential Status

Best posterior-agreement run so far:

- Summary: `runs/06_two_exponential/01_npe_flow/12_npe_flow_stress_tests_two_exp_ordered_residual/results/two_exp_ordered_summary.json`
- Corner: `runs/06_two_exponential/01_npe_flow/12_npe_flow_stress_tests_two_exp_ordered_residual/figures/two_exp_ordered_mcmc_hmc_npe_corner.png`
- MCMC-HMC: `0.0193`
- MCMC-NPE: `0.0429`
- HMC-NPE: `0.0532`

This best artifact was produced before the later experimental tweak to the two-exponential observation design (`n_obs=60`, `sigma_true=0.10`). The later identifiable-design attempt did not improve NPE agreement (`max mean = 0.2067`), so it is not the current best artifact.

What improved it:

- exact MCMC/HMC tuning until they agreed
- profile two-rate least-squares context
- profile-only context rather than raw/downsampled curve context
- context-centered residual NPE target
- disabling linear target adjustment after residual centering

What did not solve it:

- broader local regions
- tighter local regions below `q=0.008`
- proposal NPE from inflated HMC Gaussian with prior/proposal weighting
- full affine target whitening
- restoring linear target adjustment after residual centering
- making the synthetic rates more separated and lowering observation noise
- ridge coordinates: \(\log(A_1+A_2), \log(A_1/A_2), \log k_1, \log(k_2-k_1), \log\sigma\)
- conditioning the custom flow on the full raw curve in addition to the profile summary
- `sbi` SNPE-C on the raw curve from the broad prior
- `sbi` SNPE-C on the raw curve from an HMC-fitted Gaussian proposal

Current interpretation: the ordered two-exponential posterior still exposes residual density-estimation bias in this finite-budget conditional flow setup. The exact MC baselines are reliable, but the NPE posterior is not yet faithful at the `0.034` target.

Additional artifacts from continuation:

- Ridge-coordinate custom NPE, moderate-noise case: `runs/06_two_exponential/01_npe_flow/16_npe_flow_stress_tests_two_exp_ordered_ridgecoords/results/two_exp_ordered_summary.json`
- Ridge-coordinate plus raw-curve context, moderate-noise case: `runs/06_two_exponential/01_npe_flow/04_npe_flow_stress_tests_two_exp_ordered_moderate_ridge_rawctx/results/two_exp_ordered_summary.json`
- `sbi` SNPE-C raw curve, broad-prior rounds: `runs/06_two_exponential/02_sbi/05_sbi_two_exp_ordered_raw_nsf_r4/results/sbi_two_exp_ordered_summary.json`
- `sbi` SNPE-C raw curve, HMC-Gaussian initial proposal: `runs/06_two_exponential/02_sbi/04_sbi_two_exp_ordered_raw_hmcprop_nsf_r3/results/sbi_two_exp_ordered_summary.json`

Next focused avenues:

- isolate whether the two-exponential blocker is the unknown-noise dimension by running a known-\(\sigma\) ordered two-exponential variant
- try a lower-dimensional learned embedding for the raw curve instead of direct 45-dimensional conditioning
- try a spline flow trained on exact posterior samples as an oracle density-estimation diagnostic, explicitly marked as not NPE
- add simulation-based calibration diagnostics over several two-exponential observations before trying the unordered variant
