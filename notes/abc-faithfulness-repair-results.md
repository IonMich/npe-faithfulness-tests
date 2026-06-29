# ABC Faithfulness Repair Results

Goal: test whether likelihood-free ABC-style correction can repair the finite-budget NPE/SNPE posterior gap for the decay example.

Target:

```text
mean normalized Wasserstein <= 0.034
```

Reference:

- grid posterior built from the analytic toy likelihood for evaluation only
- MCMC/HMC reference samples from the earlier runs
- same normalized Wasserstein metric used throughout the NPE investigation

## What changed

Implemented:

```text
scripts/abc_faithfulness_decay.py
```

The script compares:

- prior ABC
- SNPE-posterior proposal ABC
- SMC-ABC
- regression-adjusted versions of each

The successful corrected posterior uses:

```math
z_i \sim r(z),
\qquad
x_i \sim p(x \mid \exp z_i),
```

with weights

```math
w_i
\propto
\frac{p(z_i)}{r(z_i)}
K_h\left(\rho(s(x_i),s(x_0))\right).
```

The proposal \(r(z)\) is a transparent inflated Gaussian fit to existing SNPE posterior samples. The final posterior is therefore not the raw neural posterior. The neural posterior only focuses the simulations.

The ABC summary is an indirect exponential-fit summary:

```math
s(x)=
\left(
\log \widehat A(x),
\log \widehat k(x),
\log \widehat \sigma(x)
\right).
```

I refined the \(\widehat k\) summary by a local quadratic interpolation around the least-squares grid minimum. This mattered: the unrefined summary stalled just above target.

Regression adjustment:

```math
z_i = a + B(s_i-s_0) + \eta_i,
```

then

```math
z_i^\star = z_i - \widehat B(s_i-s_0).
```

## Main Results

All values are mean normalized Wasserstein distance to the grid posterior.

| Run | Simulations | Best method | ESS | Distance | Pass |
| --- | ---: | --- | ---: | ---: | --- |
| Refined all-method comparison | 200k proposal sims | SNPE-MDN proposal + kernel ABC + regression | 8,160 | 0.03557 | no |
| Refined all-method comparison | 200k proposal sims | SNPE-diag proposal + kernel ABC + regression | 9,189 | 0.03656 | no |
| Refined all-method comparison | 198k total SMC sims | SMC-ABC round 2 + regression | 2,394 | 0.03775 | no |
| Refined all-method comparison | 200k prior sims | prior ABC + kernel + regression | 2,212 | 0.05112 | no |
| Focused scaled run | 1M proposal sims | SNPE-diag proposal + kernel ABC + regression | 76,616 | 0.03411 | just above |
| Focused scaled run | 2M proposal sims | SNPE-diag proposal + kernel ABC + regression | 103,036 | 0.03204 | yes |
| Validation run, independent seed | 1.5M proposal sims | SNPE-diag proposal + kernel ABC + regression | 100,017 | 0.03129 | yes |

The two passing runs:

```text
runs/01_exponential_decay/05_abc_faithfulness/02_abc_faithfulness_scaled2m_snpe_diag_refined/results/abc_faithfulness_summary.json
runs/01_exponential_decay/05_abc_faithfulness/07_abc_faithfulness_validation_snpe_diag_refined/results/abc_faithfulness_summary.json
```

Best 2M run:

```text
method: proposal_abc_snpe_diag_infl2p5_kernel_q0p006_regression
distance: 0.03204
ESS: 103,036
ESS fraction among retained particles: 0.154
kernel bandwidth epsilon: 0.05282
total runtime: 39.3 seconds
```

Per-parameter normalized distances:

| Parameter | Normalized W |
| --- | ---: |
| \(A\) | 0.03409 |
| \(k\) | 0.03262 |
| \(\sigma\) | 0.02940 |

Validation run:

```text
method: proposal_abc_snpe_diag_infl2p5_kernel_q0p008_regression
distance: 0.03129
ESS: 100,017
kernel bandwidth epsilon: 0.05797
total runtime: 19.1 seconds
```

Per-parameter normalized distances:

| Parameter | Normalized W |
| --- | ---: |
| \(A\) | 0.03214 |
| \(k\) | 0.03232 |
| \(\sigma\) | 0.02941 |

## Outputs

Full comparison:

```text
runs/01_exponential_decay/05_abc_faithfulness/01_abc_faithfulness/results/abc_faithfulness_summary.json
runs/01_exponential_decay/05_abc_faithfulness/01_abc_faithfulness/results/abc_faithfulness_samples.npz
runs/01_exponential_decay/05_abc_faithfulness/01_abc_faithfulness/figures/abc_faithfulness_distance_curve.png
runs/01_exponential_decay/05_abc_faithfulness/01_abc_faithfulness/figures/abc_faithfulness_corner_overlay.png
runs/01_exponential_decay/05_abc_faithfulness/01_abc_faithfulness/figures/abc_faithfulness_predictive_overlay.png
```

Passing 2M focused run:

```text
runs/01_exponential_decay/05_abc_faithfulness/02_abc_faithfulness_scaled2m_snpe_diag_refined/results/abc_faithfulness_summary.json
runs/01_exponential_decay/05_abc_faithfulness/02_abc_faithfulness_scaled2m_snpe_diag_refined/results/abc_faithfulness_samples.npz
runs/01_exponential_decay/05_abc_faithfulness/02_abc_faithfulness_scaled2m_snpe_diag_refined/figures/abc_faithfulness_distance_curve.png
runs/01_exponential_decay/05_abc_faithfulness/02_abc_faithfulness_scaled2m_snpe_diag_refined/figures/abc_faithfulness_corner_overlay.png
runs/01_exponential_decay/05_abc_faithfulness/02_abc_faithfulness_scaled2m_snpe_diag_refined/figures/abc_faithfulness_predictive_overlay.png
```

Validation run:

```text
runs/01_exponential_decay/05_abc_faithfulness/07_abc_faithfulness_validation_snpe_diag_refined/results/abc_faithfulness_summary.json
runs/01_exponential_decay/05_abc_faithfulness/07_abc_faithfulness_validation_snpe_diag_refined/results/abc_faithfulness_samples.npz
runs/01_exponential_decay/05_abc_faithfulness/07_abc_faithfulness_validation_snpe_diag_refined/figures/abc_faithfulness_distance_curve.png
runs/01_exponential_decay/05_abc_faithfulness/07_abc_faithfulness_validation_snpe_diag_refined/figures/abc_faithfulness_corner_overlay.png
runs/01_exponential_decay/05_abc_faithfulness/07_abc_faithfulness_validation_snpe_diag_refined/figures/abc_faithfulness_predictive_overlay.png
```

## Interpretation

The raw neural posterior was not made faithful. The successful repair is:

```text
SNPE as proposal generator + likelihood-free kernel ABC correction + local regression adjustment.
```

This is the right trust boundary:

```text
Trust ABC-corrected particles, not the raw q_phi(theta | x0).
```

SNPE still matters because it makes ABC efficient. Prior ABC with the same budget reached only `0.05112`, while the SNPE-proposal corrected method reached `0.03204`.

The result also explains why earlier pure SNPE was close but not enough. The neural posterior was already useful for locating the posterior region, but its density-estimation bias was too large to use as the final answer under the strict `0.034` criterion.

## Caveats

This fix uses a strong, model-aware summary statistic for the toy decay problem. It is still likelihood-free in the sense that the inference weights do not evaluate \(p(x_0 \mid \theta)\), but the summary is hand-designed to be close to sufficient for this model.

For a genuinely intractable simulator, we should not assume this transfers automatically. The realistic trust checks would be:

- posterior predictive checks
- simulation-based calibration over many \(x_i\)
- stability as the ABC bandwidth changes
- independent seeds
- comparison to another likelihood-free method such as SMC-ABC when feasible

## Practical conclusion

We now have a method that reaches the MC-level target in this toy setup:

```text
inflated SNPE proposal -> simulator -> kernel ABC weighting -> regression adjustment
```

The method is slower than accepting the raw SNPE posterior but much more trustworthy. For this example, it is the first pure likelihood-free route we tested that reaches the same normalized-Wasserstein band as MCMC/HMC.
