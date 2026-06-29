# NPE Flow Target-Pass Results

Goal: make NPE with a conditional normalizing flow reach

```text
mean normalized Wasserstein <= 0.034
```

without using ABC correction as the final posterior.

## Passing Result

Script:

```bash
uv run scripts/npe_flow_decay.py \
  --training-mode local_prior \
  --linear-target-adjustment \
  --seed 20260706 \
  --train-simulations 150000 \
  --val-simulations 35000 \
  --local-pilot-simulations 400000 \
  --local-quantile 0.005 \
  --local-max-candidates 45000000 \
  --simulate-chunk-size 100000 \
  --summary-chunk-size 50000 \
  --epochs 220 \
  --patience 55 \
  --batch-size 4096 \
  --learning-rate 6e-4 \
  --transforms 8 \
  --hidden-features 192,192 \
  --bins 16 \
  --posterior-samples 180000 \
  --output-dir runs/01_exponential_decay/03_npe_flow_search/11_npe_flow_local_q0005_linear_150k_t8_seed20260706/results \
  --figure-dir runs/01_exponential_decay/03_npe_flow_search/11_npe_flow_local_q0005_linear_150k_t8_seed20260706/figures
```

Result:

```text
mean normalized Wasserstein: 0.03310
target: 0.034
pass: true
```

Per-parameter normalized Wasserstein:

| Parameter | Normalized W |
| --- | ---: |
| \(A\) | 0.03370 |
| \(k\) | 0.03117 |
| \(\sigma\) | 0.03444 |

Even though \(\sigma\) is slightly above `0.034` individually, the pre-declared metric was the mean normalized Wasserstein across parameters, and that passes.

## What This Method Is

This is NPE with a conditional normalizing flow.

The trained density is:

```math
q_\phi(z \mid s(x)),
\qquad z=\log\theta.
```

The flow is a conditional neural spline flow from `zuko`:

```text
8 spline transforms
16 bins
hidden layers 192,192
context = refined 3D indirect summary
```

The context summary is:

```math
s(x)=
\left(
\log \widehat A(x),
\log \widehat k(x),
\log \widehat\sigma(x)
\right),
```

where the hats come from a fast least-squares exponential-decay fit to each simulated curve.

Training data was sampled from the prior and filtered to a local region:

```math
z_i \sim p(z),
\qquad
x_i \sim p(x\mid z_i),
\qquad
s(x_i) \approx s(x_0).
```

The local filter depends only on \(x\), so the conditional target is unchanged:

```math
p(z \mid s,\, s \in R)
=
p(z \mid s)
```

for \(s\) inside the retained region. This is local amortization, not ABC correction.

## Linear Target Adjustment

The successful run also used a context-dependent affine preconditioner:

```math
u_i = z_i - \widehat B(s_i-s_0).
```

The spline flow is trained on:

```math
q_\phi(u \mid s).
```

At the observed context \(s=s_0\),

```math
u=z,
```

so samples from the final posterior are direct samples from the learned flow at \(s_0\). This is not a post-hoc ABC regression adjustment. It is an invertible target reparameterization inside the NPE density estimator, with unit Jacobian.

## Outputs

Summary:

```text
runs/01_exponential_decay/03_npe_flow_search/11_npe_flow_local_q0005_linear_150k_t8_seed20260706/results/npe_flow_decay_summary.json
```

Samples:

```text
runs/01_exponential_decay/03_npe_flow_search/11_npe_flow_local_q0005_linear_150k_t8_seed20260706/results/npe_flow_decay_samples.npz
```

Model:

```text
runs/01_exponential_decay/03_npe_flow_search/11_npe_flow_local_q0005_linear_150k_t8_seed20260706/results/npe_flow_decay_model.pt
```

Figures:

```text
runs/01_exponential_decay/03_npe_flow_search/11_npe_flow_local_q0005_linear_150k_t8_seed20260706/figures/npe_flow_decay_corner_overlay.png
runs/01_exponential_decay/03_npe_flow_search/11_npe_flow_local_q0005_linear_150k_t8_seed20260706/figures/npe_flow_decay_predictive_overlay.png
runs/01_exponential_decay/03_npe_flow_search/11_npe_flow_local_q0005_linear_150k_t8_seed20260706/figures/npe_flow_decay_training.png
```

## Experiments That Did Not Pass

Notable misses:

| Run | Distance |
| --- | ---: |
| Weighted proposal flow, 100k, mixture proposal | 0.06442 |
| Local prior q=0.005, no linear target adjustment | 0.04977 |
| Local prior q=0.001, no linear target adjustment | 0.05944 |
| Local q=0.005 + linear target adjustment, 40k | 0.03868 |
| Local q=0.005 + linear target adjustment, 100k | 0.03472 |
| Local q=0.005 + linear target adjustment, 150k | 0.03310 |
| Enhanced 8D summary + linear target adjustment | 0.04989 |

The key fixes were:

- local prior training near \(s(x_0)\)
- conditional neural spline flow rather than the earlier affine flow
- linear target preconditioning
- enough local simulations

## Interpretation

The earlier pure-flow failures were not because normalizing-flow NPE is impossible here. They were due to:

- weak affine flow architecture
- insufficient local training accuracy
- residual local-regression bias
- occasional flow tail leakage

The passing setup shows that NPE with normalizing flows can reach the same strict posterior-faithfulness band as MCMC/HMC on this toy problem, provided we localize the amortization problem and use a stronger conditional flow.
