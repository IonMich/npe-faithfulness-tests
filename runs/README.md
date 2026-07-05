# Runs

This directory is the canonical artifact store for the project.
Top-level numbered folders are statistical models. Method folders sit inside each model, and each run folder contains real `results/` and `figures/` directories plus a README with status, metric, script, and note references.
The only symlinked convenience layer is `00_successful_runs/`, which points to canonical run folders that passed a calibrated target or serve as references.

Model folders:
- `01_exponential_decay`: original exponential-decay likelihood and all decay-focused methods.
- `02_stress_sign`: sign-symmetry stress likelihood.
- `03_stress_banana`: banana-shaped posterior stress likelihood.
- `04_stress_label_switch`: label-switching stress likelihood.
- `05_stress_linear6`: higher-dimensional linear-Gaussian stress likelihood.
- `06_two_exponential`: ordered two-exponential likelihood and SBI variants.

Status labels:
- `grid-faithful`: NPE matched a model-specific exact/reference posterior target.
- `pass`: calibrated target metric was met by the run summary.
- `reference`: sampler/reference run with convergence or baseline agreement.
- `diagnostic_pass`: diagnostic/reference metric met the target, but it is not a direct NPE success claim.
- `near_floor`: full-prior population NLL is close to the model-specific entropy floor, but the remaining gap is still resolved or otherwise not a strict floor hit.
- `legacy_pairwise_pass`: passed an inherited pairwise agreement target, but has not been calibrated against a model-specific truth target.
- `near`: missed the target but stayed within 25% of it.
- `fail`: explicit target metric was not met.
- `diagnostic`: no direct target metric was found.

Start here:
- [successful and reference runs](00_successful_runs/README.md)

Best run by model:

| Model | Best status | Run | Metric |
| --- | --- | --- | --- |
| `01_exponential_decay` | `pass` | [05_abc_faithfulness / 07_abc_faithfulness_validation_snpe_diag_refined](01_exponential_decay/05_abc_faithfulness/07_abc_faithfulness_validation_snpe_diag_refined) | best mean normalized Wasserstein: 0.03129 |
| `02_stress_sign` | `grid-faithful` | [01_npe_flow / 21_npe_flow_stress_tests_sign_absfold_q008_linear](02_stress_sign/01_npe_flow/21_npe_flow_stress_tests_sign_absfold_q008_linear) | NPE-to-grid diagnostic W: 0.02326 / calibrated target 0.02331 |
| `03_stress_banana` | `legacy_pairwise_pass` | [01_npe_flow / 03_npe_flow_stress_tests_banana_q008](03_stress_banana/01_npe_flow/03_npe_flow_stress_tests_banana_q008) | pairwise max diagnostic Wasserstein: 0.01844 / inherited target 0.034 |
| `04_stress_label_switch` | `legacy_pairwise_pass` | [01_npe_flow / 05_npe_flow_stress_tests_label_em](04_stress_label_switch/01_npe_flow/05_npe_flow_stress_tests_label_em) | pairwise max diagnostic Wasserstein: 0.02868 / inherited target 0.034 |
| `05_stress_linear6` | `near_floor` | [03_population_npe / 01_flow2_residual_full_prior_512k_ensemble4](05_stress_linear6/03_population_npe/01_flow2_residual_full_prior_512k_ensemble4) | full-prior z-NLL: -10.77984 / entropy floor: -10.78631 / gap 0.00647 |
| `06_two_exponential` | `fail` | [01_npe_flow / 12_npe_flow_stress_tests_two_exp_ordered_residual](06_two_exponential/01_npe_flow/12_npe_flow_stress_tests_two_exp_ordered_residual) | max diagnostic Wasserstein: 0.05317 |
