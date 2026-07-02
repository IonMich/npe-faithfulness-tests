# Neural Posterior Estimation Faithfulness Tests

This repository studies neural posterior estimation (NPE) on small
simulation-based inference problems. The experiments train neural conditional
density estimators and compare their posteriors with independent posterior
references.

The basic simulator setup is:

```math
\theta \sim p(\theta), \qquad x \sim p(x \mid \theta).
```

For a requested observed signal $`x_\star`$, the Bayesian target is:

```math
p(\theta \mid x_\star)
=
\frac{p(x_\star \mid \theta)p(\theta)}
{\int p(x_\star \mid \vartheta)p(\vartheta)\,d\vartheta}.
```

NPE trains a conditional density estimator $`q_\phi(\theta \mid x)`$ from
simulated pairs:

```math
\max_\phi\;
\mathbb E_{\theta \sim p(\theta),\,x \sim p(x \mid \theta)}
\left[\log q_\phi(\theta \mid x)\right].
```

After training, the object under test is $`q_\phi(\theta \mid x_\star)`$.

## Evaluation

The reference posterior is computed with exact numerical grids where dimension
permits it, exact-likelihood random-walk Metropolis Markov chain Monte Carlo
(MCMC), and exact-likelihood Hamiltonian Monte Carlo (HMC). MCMC and HMC use
the same prior and likelihood as the simulator under test.

For a diagnostic parameterization $`g(\theta)`$, the main scalar comparison is
mean marginal normalized Wasserstein distance:

```math
D(q, p_{\mathrm{ref}})
=
\frac{1}{d}
\sum_{j=1}^{d}
\frac{
W_1\!\left(g_j(\theta_q), g_j(\theta_{\mathrm{ref}})\right)
}{
\mathrm{sd}_{p_{\mathrm{ref}}}\!\left(g_j(\theta)\right)
}.
```

The symbols in this diagnostic are:

- $`p_{\mathrm{ref}}(\theta \mid x_\star)`$: the reference posterior for the
  requested observed signal $`x_\star`$.
- $`q(\theta \mid x_\star)`$: the learned posterior being evaluated, usually an NPE
  flow posterior.
- $`\theta_q`$: samples drawn from $`q(\theta \mid x_\star)`$.
- $`\theta_{\mathrm{ref}}`$: samples drawn from, or weighted grid points
  representing, $`p_{\mathrm{ref}}(\theta \mid x_\star)`$.
- $`g(\theta)`$ is the diagnostic parameterization used for comparison; it may
  be the raw parameter vector, a physical-parameter transform, or a
  symmetry-aware transform. $`g_j(\theta)`$ is coordinate $`j`$ of that diagnostic
  vector, and $`d`$ is the number of diagnostic coordinates.
- $`W_1(a,b)`$: the one-dimensional Wasserstein-1 distance between two scalar
  distributions. For distributions with cumulative distribution functions
  $`F_a`$ and $`F_b`$, it is:

```math
W_1(a,b)
=
\int_{-\infty}^{\infty}
\left|F_a(t)-F_b(t)\right|\,dt.
```

  Equivalently, using quantile functions $`F_a^{-1}`$ and $`F_b^{-1}`$:

```math
W_1(a,b)
=
\int_0^1
\left|F_a^{-1}(u)-F_b^{-1}(u)\right|\,du.
```

  In the run summaries, $a$ and $b$ are empirical one-dimensional sample sets
  or weighted grid marginals for one diagnostic coordinate.
- $`\mathrm{sd}_{p_{\mathrm{ref}}}(g_j(\theta))`$: the posterior standard
  deviation of diagnostic coordinate $`j`$ under the reference posterior.

The division by the reference standard deviation turns each coordinate's
Wasserstein distance into a scale-free error. The final value averages those
coordinate errors, so parameters measured on different numerical scales can be
reported in one summary.

The diagnostic parameterization is usually the raw parameter vector. Symmetric
models use transformed coordinates such as
$`\left(|\theta_1|,\theta_2\right)`$ or
$`\left(\mu_{\mathrm{low}},\mu_{\mathrm{high}},\sigma\right)`$ so that sampler
diagnostics measure posterior shape independently of arbitrary label
assignments.

Each serious run also records acceptance rates, rank-normalized `Rhat`, bulk
and tail effective sample size, trace plots, posterior summaries, corner
overlays, and posterior predictive overlays when the simulator generates
curves.

## Starting A New Model

Begin by writing down the statistical problem before changing code. A new test
case should have a prior, simulator, observation rule, diagnostic coordinates,
and reference plan:

```math
\theta \sim p(\theta),
\qquad
x \sim p(x\mid\theta),
\qquad
x_\star = f(\theta_\star,\epsilon_\star).
```

The posterior target for the requested signal $`x_\star`$ is:

```math
p(\theta\mid x_\star)
\propto
p(x_\star\mid\theta)p(\theta).
```

Define the parameterization used for numerical comparison at the same time:

```math
g:\Theta\to\mathbb R^d.
```

For an identifiable model, $`g(\theta)`$ is often the raw parameter vector. For a
model with signs, labels, ridges, or ordered components, choose $`g`$ so that the
comparison measures the statistical posterior rather than an arbitrary
coordinate convention.

Then implement the smallest exact-likelihood test loop that can answer whether
NPE is faithful:

1. Add the simulator, prior sampler, likelihood, context summary, display
   transform, and diagnostic transform. Simple stress tests usually belong in
   `scripts/npe_flow_stress_tests.py`; decay-style models with specialized
   references can use a dedicated script.
2. Pick a truth $`\theta_\star`$ and generate one observed signal $`x_\star`$. Keep
   this signal fixed while comparing methods, otherwise the reference target is
   changing between runs.
3. Build an independent reference posterior. Use a grid when $`d`$ is small
   enough for direct quadrature; otherwise use exact-likelihood MCMC or HMC and
   check trace behavior, acceptance, `Rhat`, and effective sample size.
4. Run a smoke NPE experiment first. It should verify that the simulator,
   context, neural posterior, sampling code, and plotting code all work before
   spending time on a larger run.
5. Run the serious NPE fit and compare posterior samples with the reference
   using the normalized Wasserstein diagnostic $`D(q,p_{\mathrm{ref}})`$ already
   defined above. Inspect marginal overlays, corner plots, posterior
   predictive overlays, and any model-specific mode or symmetry diagnostics.
6. Decide the run status from the reference comparison. A passing run should
   match the posterior target in the diagnostic coordinates and should not rely
   only on visually plausible predictive curves.
7. Record the run command, target, metric, plots, and conclusion in the run
   README, then update the root README only when the result changes the
   project-level understanding of that model.

Useful entry points are:

```sh
uv run scripts/npe_flow_stress_tests.py --help
uv run scripts/check_faithfulness_target.py
uv run scripts/build_runs_view.py
```

## Models And Progress

### Single-Exponential Decay

The base model is a noisy exponential decay curve:

```math
y_i = A\exp(-k t_i) + \epsilon_i,
\qquad
\epsilon_i \sim \mathcal N(0,\sigma^2).
```

The parameter vector is $\theta=(A,k,\sigma)$. The code samples and evaluates
the posterior in log coordinates:

```math
z=(\log A,\log k,\log\sigma),
\qquad
z \sim \mathcal N\!\left(
\log(4.0,0.50,0.40),
\mathrm{diag}(0.8^2,0.8^2,0.8^2)
\right).
```

The single-decay estimator is a population-trained neural posterior estimator
(NPE). It learns a conditional density for signals sampled from the same
population used by the simulator:

```math
p_{\mathrm{train}}(\theta,x)=p(\theta)p(x\mid\theta),
```

with objective

```math
\phi_{\mathrm{train}}
=
\arg\max_\phi
\mathbb E_{(\theta,x)\sim p_{\mathrm{train}}}
\left[
\log q_\phi(\theta\mid x)
\right].
```

#### Model Definitions

All reported negative log likelihoods (NLLs) in this section are in the
log-coordinate parameterization $z=(\log A,\log k,\log\sigma)$. Each neural
posterior estimator (NPE) receives a deterministic context vector
$c=f(x)$, then standardizes each context coordinate using training-set mean and
standard deviation before the neural density is evaluated.

The raw-curve context is the 40-dimensional signal itself:

```math
c_{\mathrm{raw}}(x)=(y_1,\ldots,y_{40}).
```

The decay-summary context adds 12 deterministic statistics: the intercept and
negative slope of a least-squares line through
$`\log\max(y_i,10^{-4})`$, the standard deviation of the log residuals, mean,
standard deviation, minimum, maximum, first value, last value, early mean, late
mean, and first-to-last log ratio. The fit-summary context is a coarse
least-squares exponential fit. For a 64-point grid over $`\log k`$ spanning
three prior standard deviations on either side of the prior mean,

```math
\phi_k(t_i)=\exp(-k t_i),
\qquad
\hat A(k)=
\frac{\sum_i y_i\phi_k(t_i)}{\sum_i \phi_k(t_i)^2},
```

and $`k_\star`$ is the grid point with the smallest residual sum of squares
$`\mathrm{SSE}(k)`$. The six fit features are

```math
\left(
\log \hat A(k_\star),
\log k_\star,
\log \hat\sigma,
\log(1+\mathrm{SSE}_\star),
\log(1+\mathrm{SSE}_{2}-\mathrm{SSE}_\star),
d_{\mathrm{edge}}
\right),
```

where $`\hat\sigma=\sqrt{\mathrm{SSE}_\star/40}`$,
$`\mathrm{SSE}_2`$ is the second-best grid SSE, and $`d_{\mathrm{edge}}`$
is the normalized distance of $`k_\star`$ from the grid edge. Thus
`raw_fit_summary` has 46 context features and `raw_decay_fit_summary` has
58 context features.

A mixture density network (MDN) models a finite Gaussian mixture with full
covariances:

```math
q_\phi(z\mid c)
=
\sum_{j=1}^{M}
\pi_j(c)\,
\mathcal N\!\left(z;\mu_j(c),L_j(c)L_j(c)^\top\right).
```

One earlier MDN baseline has $`M=5`$ components, raw-curve context, a
three-hidden-layer SiLU multilayer perceptron (MLP) with width 128, and 44,722
trainable parameters.

A neural spline flow (NSF) represents $`z`$ by an invertible conditional
transformation $`T_\phi(\cdot;c)`$ of a standard normal base variable:

```math
u=T_\phi^{-1}(z;c),
\qquad
q_\phi(z\mid c)
=
\mathcal N(u;0,I)
\left|\det\frac{\partial T_\phi^{-1}(z;c)}{\partial z}\right|.
```

Each NSF transform is a monotonic rational-quadratic spline with 8 bins. In the
names Flow2, Flow3, and Flow4, the number is the number of stacked NSF
transforms. These runs use fully autoregressive NSF transforms; they are not
coupling layers. In this code, coupling-style NSF transforms would correspond
to `flow_passes=2`, which is not used by the models listed here. A residual NSF
means the Zuko masked MLP conditioner inside each autoregressive spline uses
residual hidden blocks. This is separate from the repo's optional residual
target transform, which is not used by the models listed in this section.
Random permutations mean the feature order is randomly permuted between
successive NSF transforms.

An ensemble is a density mixture over trained NPEs:

```math
q_{\mathrm{ens}}(z\mid x)
=
\sum_{m=1}^{M} w_m q_m(z\mid x),
\qquad
w_m\ge 0,\quad \sum_m w_m=1.
```

The ensemble log density is evaluated with `logsumexp` over member log
densities plus $`\log w_m`$. The equal-weight 4-member ensemble has
$`w_m=1/4`$. The convex-weighted 16-member ensemble fits the weights by
minimizing validation NLL on 200,000 validation examples,

```math
\min_{w\in\Delta^{15}}
-
\frac{1}{n}
\sum_{i=1}^{n}
\log
\left(
\sum_{m=1}^{16} w_m q_m(z_i\mid x_i)
\right),
```

then reports NLL on the full 1M validation cache, including an 800,000-example
holdout not used for fitting the weights.

| Model or figure entry | Actual architecture |
| --- | --- |
| 4-member Flow2 residual NSF ensemble in the viewer | Four independently trained NSF members; each member uses `raw_decay_fit_summary` context, two NSF transforms, 8 spline bins, width-80 residual masked-MLP conditioners with two hidden layers, ReLU activation, random inter-transform permutations, and 72,938 trainable parameters. Each member is trained on 2.048M simulations for 15 epochs. |
| 16-member convex-weighted density ensemble in the viewer | Convex mixture of saved NSF checkpoints. The member weights, rounded to four decimals in saved-member order, are `(0.1541, 0.1386, 0.0458, 0.0065, 0.1088, 0.0530, 0.0950, 0.0745, 0.0138, 0.0008, 0.0012, 0.0923, 0.0548, 0.1088, 0.0130, 0.0391)`. The members are Flow2 or Flow3 NSF checkpoints using raw, `raw_fit_summary`, or `raw_decay_fit_summary` context; all use 8 spline bins and width-80, two-hidden-layer conditioners. All but one use residual masked-MLP conditioners. |
| Fixed parameter-count MDN diagnostic | Five-component full-covariance Gaussian MDN with raw-curve context and a width-128, three-hidden-layer SiLU MLP; 44,722 trainable parameters. |
| Fixed parameter-count spline-flow diagnostic | Flow4 NSF with raw-curve context, 8 spline bins, width-64, two-hidden-layer ReLU conditioners, no residual conditioners, no random permutations, and 45,844 trainable parameters. |
| Training-efficiency Flow4 single-model runs | Flow4 NSF single models with raw-curve context. The 90-epoch run uses width 64 and batch 512; the 74-epoch run uses width 80 and batch 1024. |
| Training-efficiency Flow3 single-model run | Flow3 NSF single model with raw-curve context, width 80, batch 1024, 8.192M simulations, and 27 epochs. |
| Earlier 4-member residual NSF ensemble curves | Equal-weight density mixtures of four Flow3 residual NSF members with width-80 conditioners and 8 spline bins; one curve uses raw-curve context, and one adds the six fit-summary features. |

For the 16-member convex-weighted density ensemble, the weight-vector indices
map to member architectures as follows: members 1, 2, 6, and 9 are Flow2
residual NSF models with random permutations and `raw_decay_fit_summary`
context; members 3 and 4 are the same Flow2 residual NSF architecture without
random permutations; members 5, 7, and 8 are Flow3 residual NSF models with
`raw_decay_fit_summary` context; members 10, 11, 15, and 16 are Flow3 residual
NSF models with `raw_fit_summary` context; members 12 and 13 are Flow3
residual NSF models with raw-curve context; member 14 is a Flow3 NSF with
raw-curve context and no residual conditioner. The Flow2 members have 72,938
trainable parameters; the Flow3 residual raw-context members have 105,087; the
Flow3 residual `raw_fit_summary` members have 106,527; the Flow3 residual
`raw_decay_fit_summary` members have 109,407; and the non-residual Flow3
raw-context member has 46,767.

The viewer includes two population-trained NPEs:

| Model | Description | Full validation negative log likelihood (NLL) |
| --- | --- | ---: |
| 4-member ensemble of Flow2 residual neural spline flows (NSFs) with random permutations, raw-curve, decay-summary, and fit-summary context features; 2.048M simulations per member, 15 epochs | Equal-weight density ensemble trained from initialization. | `-3.6306901328125` |
| 16-member convex-weighted density ensemble | Ensemble whose nonnegative weights are fitted on validation data; its measured cache NLL is slightly lower, but the difference is not statistically resolved. | `-3.63128073481036` |

The following prior-predictive signal was sampled from
$`p(\theta)p(x\mid\theta)`$. The plot compares both NPE estimates with an exact
numerical grid and a Markov chain Monte Carlo (MCMC) reference. The mean
normalized marginal Wasserstein distance $`D(q,p_{\mathrm{ref}})`$ is the
diagnostic defined in the Evaluation section. Smaller values mean the posterior
marginals are closer to the exact grid reference. For this signal, the distance
is `0.0597` for the 4-member Flow2 residual NSF ensemble, `0.0592` for the
convex-weighted density ensemble, and `0.0733` for MCMC.

![Single decay population posterior overlay](runs/00_shared_assets/readme_decay_posteriors/decay_population_posterior_corner.png)

[Single decay population signal predictive overlay](runs/00_shared_assets/readme_decay_posteriors/decay_population_posterior_signal.png)

The low-prior-density stress signal is harder. It was generated from a
parameter vector 4.33 prior standard deviations from the prior mean in
log-parameter space, with log prior density `9.375` below the prior mean. The
4-member Flow2 residual NSF ensemble has mean normalized marginal Wasserstein
distance `0.1868`, the convex-weighted density ensemble has `0.2379`, and
MCMC has `0.0761`. This is a useful counterexample to relying only on average
NLL.

![Single decay low-prior-density posterior overlay](runs/00_shared_assets/readme_decay_posteriors/decay_low_prior_stress_posterior_corner.png)

[Single decay low-prior-density signal predictive overlay](runs/00_shared_assets/readme_decay_posteriors/decay_low_prior_stress_posterior_signal.png)

The generated metadata for these diagnostic views is stored in
[decay_population_readme_posteriors_summary.json](runs/00_shared_assets/readme_decay_posteriors/decay_population_readme_posteriors_summary.json).

#### Flow2 Ensemble Data Scaling Diagnostic

The current single-decay data scaling diagnostic holds the deployed 4-member
Flow2 residual NSF ensemble recipe fixed and scales only the number of
simulated training pairs per member. The primary measurement is population
validation NLL. Because this is a continuous-density NLL in normalized `z` units
and can be negative, the log-scale companion panel plots excess NLL above the
estimated Bayes entropy floor `-3.64122 +/- 0.008`. Posterior Wasserstein is
kept as a separate faithfulness diagnostic, not as the main scaling-law loss.

![Single decay Flow2 ensemble data scaling](runs/00_shared_assets/readme_scaling/decay_flow2_ensemble_data_scaling_weng_style.png)

The equal-weight 4-member ensemble improves monotonically from `64k` to
`2.048M` simulations per member. Validation NLL improves from `-3.54993` to
`-3.63069`, reducing excess NLL above the entropy floor from `0.09129` to
`0.01053`. A raw-NLL asymptote fit gives `L_asym=-3.63610`, exponent `0.823`,
and raw `R2=0.999`. That fitted asymptote is only `0.00512` above the Bayes
entropy estimate, which is smaller than the current `+/-0.008` floor
uncertainty; it should not be read as a resolved residual training floor.

With the entropy estimate held fixed, a no-residual-floor fit to excess NLL
gives exponent `0.631`, but this exponent is floor-sensitive near the right
edge. Shifting the entropy estimate by its current uncertainty moves the
excess-loss exponent from about `0.490` to `0.997`.

The fixed-panel posterior diagnostic moves in the same direction: panel mean
normalized marginal Wasserstein decreases from `0.11046` to `0.03613`. Full
metadata and rows are stored in
[flow2_ensemble_data_scaling_summary.json](runs/00_shared_assets/readme_scaling/decay_flow2_ensemble_data_scaling_summary.json).

#### Population Entropy Floor

For the population validation NLL, the Bayes-optimal density is the exact
posterior $`p(\theta\mid x)`$. The irreducible loss is the conditional population
entropy:

```math
\mathcal L_\star
=
\mathbb E_{p(\theta,x)}
\left[
-\log p(\theta\mid x)
\right]
=
H(\theta\mid X).
```

The adaptive oracle estimate recorded in
[npe-next-2x-efficiency-decision-diary.md](notes/npe-next-2x-efficiency-decision-diary.md)
is approximately `-3.64122 +/- 0.008` in $`z`$ units. That is the estimated
population-NLL floor for validation on $`p(\theta)p(x\mid\theta)`$.

The reported model NLLs are measured on a finite 1M-example validation cache.
Per-example NLL standard-error estimates are about `0.00252`, or roughly
`+/-0.00495` for a 95% Monte Carlo half-width, for both NPEs listed above. The
oracle floor also has finite numerical uncertainty, reported above as
`+/-0.008`. The 4-member ensemble and convex-weighted density ensemble are
therefore close enough that the density ensemble's `0.00059` measured NLL
advantage should not be interpreted as a resolved population-level ordering
without a larger validation estimate. The validation-cache uncertainty
calculation is stored in
[decay_population_npe_validation_nll_uncertainty.json](runs/00_shared_assets/readme_scaling/decay_population_npe_validation_nll_uncertainty.json).

#### Training Efficiency

The wall-time plot below is an empirical training-efficiency comparison for
directly trained models, not a scaling-law plot. Curves show training NLL;
markers show exact full-cache validation NLL. The legend gives the model
description and final validation NLL for each line.

The plot includes representative trained models sorted by wall time:
`3140.0s -> 1569.2s` (`2.00x`) and `1569.2s -> 775.6s` (`2.02x`) for
single-model NSF runs, followed by ensemble runs at lower wall time. The
4-member Flow2 residual NSF ensemble reaches `-3.6307` in `246s`.

![Single decay NPE training efficiency curves](runs/00_shared_assets/readme_scaling/decay_population_npe_training_efficiency_curves.png)

Because panel means can hide rare failures, the comparison also looks at the
full distribution of per-signal panel marginal Wasserstein values. The
metric is the same coordinate-wise diagnostic defined in the evaluation
section: for each signal, exact grid posterior marginals over $`A`$, $`k`$, and
$`\sigma`$ are compared with NPE posterior samples using normalized 1D
Wasserstein distances, then averaged over coordinates. In this subsection,
$`D_{\mathrm{panel}}`$ denotes that per-signal mean normalized marginal
Wasserstein distance.

![Single decay NPE panel Wasserstein distribution](runs/00_shared_assets/readme_scaling/decay_panel_w_distribution_mdn512k_vs_spline4m_500.png)

Adding exact-likelihood random-walk MCMC to the same 500-signal panel gives
median $`D_{\mathrm{panel}}`$ values of 0.0273 for MCMC, 0.0308 for the
4-member Flow2 residual NSF ensemble, 0.115 for the 4.096M conditional
spline-flow checkpoint, and 0.161 for the 512k MDN. MCMC is lowest on 274 of
500 signals, the Flow2 ensemble is lowest on 223, the spline-flow checkpoint is
lowest on 3, and the MDN is never lowest. The MCMC curve is a useful
exact-likelihood sampler diagnostic, not a replacement for the exact grid
marginals used as the reference here: 287 of 500 MCMC fits pass the current
R-hat and effective-sample-size convergence flags. The remaining outliers show
where posterior-shape diagnostics can catch issues not visible from validation
NLL alone.

### Sign-Symmetry Stress Test

This model creates a two-mode posterior by observing a squared parameter:

```math
x =
\begin{bmatrix}
\theta_1^2 \\
\theta_2
\end{bmatrix}
+ \epsilon,
\qquad
\epsilon \sim \mathcal N\!\left(
0,
\mathrm{diag}(0.22^2,0.16^2)
\right).
```

The prior is:

```math
\theta \sim \mathcal N\!\left(
0,
\mathrm{diag}(1.8^2,1.8^2)
\right).
```

The benchmark observation is generated from:

```math
\theta_0=(0.85,-0.45).
```

The posterior is symmetric in the sign of $`\theta_1`$. The diagnostic
coordinates are:

```math
g(\theta)=(|\theta_1|,\theta_2).
```

Progress: the calibrated grid-faithful run trains the flow on
$`\left(|\theta_1|,\theta_2\right)`$, then restores sign symmetry by randomly assigning
the sign of $`\theta_1`$ after sampling. This run passes the exact-grid
diagnostic target and has good mode-mass behavior.

Best posterior:
[sign_absfold_q008_linear run](runs/02_stress_sign/01_npe_flow/21_npe_flow_stress_tests_sign_absfold_q008_linear/README.md).

![Sign-symmetry best posterior overlay](runs/00_shared_assets/readme_model_overlays/sign_symmetry_best_posterior_overlay.png)

### Banana Stress Test

This model bends an otherwise simple two-dimensional posterior:

```math
x =
\begin{bmatrix}
\theta_1 \\
\theta_2 + b(\theta_1^2-c)
\end{bmatrix}
+ \epsilon,
\qquad
b=0.65,\quad c=0.70.
```

The observation noise and prior are:

```math
\epsilon \sim \mathcal N\!\left(
0,
\mathrm{diag}(0.20^2,0.18^2)
\right),
\qquad
\theta \sim \mathcal N\!\left(
0,
\mathrm{diag}(1.8^2,1.8^2)
\right).
```

The benchmark observation is generated from:

```math
\theta_0=(0.90,-0.25).
```

The diagnostic coordinates are the raw coordinates:

```math
g(\theta)=(\theta_1,\theta_2).
```

Progress: the best run has MCMC, HMC, and NPE in close pairwise agreement and
uses a tighter proposal/training region with linear target adjustment. It is currently
a legacy pairwise pass. The remaining work is model-specific calibration
against a truth/reference target.

Best posterior:
[banana_q008 run](runs/03_stress_banana/01_npe_flow/03_npe_flow_stress_tests_banana_q008/README.md).

![Banana best posterior overlay](runs/00_shared_assets/readme_model_overlays/banana_best_posterior_overlay.png)

### Label-Switching Mixture

This model has exchangeable component labels:

```math
x_i \sim
\frac{1}{2}\mathcal N(\mu_1,\sigma^2)
+
\frac{1}{2}\mathcal N(\mu_2,\sigma^2),
\qquad
i=1,\ldots,80.
```

The code parameterizes noise in log coordinates:

```math
z=(\mu_1,\mu_2,\log\sigma),
\qquad
z \sim \mathcal N\!\left(
(0,0,\log 0.45),
\mathrm{diag}(2.2^2,2.2^2,0.55^2)
\right).
```

The benchmark observation is generated from:

```math
(\mu_1,\mu_2,\sigma)=(-1.25,1.15,0.34).
```

The raw posterior is invariant to swapping $`\mu_1`$ and $`\mu_2`$. The
diagnostic coordinates sort the component means:

```math
g(z)=(\mu_{\mathrm{low}},\mu_{\mathrm{high}},\sigma),
\qquad
\mu_{\mathrm{low}}=\min(\mu_1,\mu_2),
\quad
\mu_{\mathrm{high}}=\max(\mu_1,\mu_2).
```

Progress: the best run trains in ordered coordinates, restores random label
assignment after sampling, and uses EM-based context summaries. Sorted
diagnostics pass and pairwise agreement is strong. Final status remains a
legacy pairwise pass until model-specific calibration is added.

Best posterior:
[label_em run](runs/04_stress_label_switch/01_npe_flow/05_npe_flow_stress_tests_label_em/README.md).

![Label-switching best posterior overlay](runs/00_shared_assets/readme_model_overlays/label_switching_best_posterior_overlay.png)

### Linear6 Stress Test

This model tests smooth higher-dimensional inference. The simulator is:

```math
y_i =
\sum_{j=1}^{6} w_j \phi_j(t_i)
+ \epsilon_i,
\qquad
\epsilon_i \sim \mathcal N(0,\sigma^2),
\qquad
i=1,\ldots,32.
```

The basis functions are an orthonormalized version of:

```math
1,\quad
t-\frac{1}{2},\quad
\sin(2\pi t),\quad
\cos(2\pi t),\quad
\sin(4\pi t),\quad
\cos(4\pi t).
```

The parameterization and prior are:

```math
z=(w_1,\ldots,w_6,\log\sigma),
\qquad
w_j \sim \mathcal N(0,1.25^2),
\qquad
\log\sigma \sim \mathcal N(\log 0.25,0.50^2).
```

The benchmark observation is generated from:

```math
(w_1,\ldots,w_6,\sigma)
=
(0.70,-0.35,0.80,-0.20,0.35,0.12,0.20).
```

The diagnostic coordinates are:

```math
g(z)=(w_1,\ldots,w_6,\sigma).
```

Progress: the best run has converged MCMC/HMC references and close NPE
pairwise agreement after tuning the random-walk proposal and using a tighter
proposal/training region. It is a legacy pairwise pass pending model-specific
calibration.

Best posterior:
[linear6_q008 run](runs/05_stress_linear6/01_npe_flow/13_npe_flow_stress_tests_linear6_q008/README.md).

![Linear6 best posterior overlay](runs/00_shared_assets/readme_model_overlays/linear6_best_posterior_overlay.png)

### Ordered Two-Exponential Decay

This model is the current hard case:

```math
y_i =
A_1\exp(-k_1 t_i)
+
A_2\exp(-k_2 t_i)
+
\epsilon_i,
\qquad
\epsilon_i \sim \mathcal N(0,\sigma^2).
```

The ordered variant enforces $`k_2>k_1`$ through the code parameterization:

```math
z=(\log A_1,\log k_1,\log A_2,\log\Delta k,\log\sigma),
\qquad
k_2 = k_1 + \Delta k,
\qquad
\Delta k=\exp(\log\Delta k).
```

The current ordered prior is:

```math
z \sim \mathcal N\!\left(
(\log 2.5,\log 0.35,\log 1.4,\log 0.75,\log 0.25),
\mathrm{diag}(0.60^2,0.55^2,0.65^2,0.60^2,0.45^2)
\right).
```

The benchmark observation is generated from:

```math
(A_1,k_1,A_2,k_2,\sigma)
=
(2.7,0.32,1.35,1.22,0.18).
```

The diagnostic coordinates are the displayed physical parameters:

```math
g(z)=(A_1,k_1,A_2,k_2,\sigma).
```

Progress: MCMC and HMC agree well on the best current run. NPE remains outside
the reference agreement level. The best custom-flow result used a profiled
two-rate least-squares summary and a residual-centered NPE target. Further
attempts with broader and tighter proposal regions, proposal NPE, whitening,
ridge coordinates, raw-curve context, and `sbi` SNPE-C have left the gap
unresolved.

Best posterior:
[two_exp_ordered_residual run](runs/06_two_exponential/01_npe_flow/12_npe_flow_stress_tests_two_exp_ordered_residual/README.md).

![Ordered two-exponential best posterior overlay](runs/00_shared_assets/readme_model_overlays/two_exp_ordered_best_posterior_overlay.png)

## Main Reports

- [NPE faithfulness investigation report](notes/npe-faithfulness-investigation-report.md)
- [NPE flow stress-test results](notes/npe-flow-stress-test-results.md)
- [Sign target calibration](notes/sign-target-calibration.md)
- [ABC faithfulness repair results](notes/abc-faithfulness-repair-results.md)
- [Calibrated successful and reference runs](runs/00_successful_runs/README.md)
- [All run statuses](runs/README.md)

## Common Commands

Run Python scripts with `uv run`:

```sh
uv run scripts/check_faithfulness_target.py
uv run scripts/calibrate_sign_target.py
uv run scripts/npe_flow_stress_tests.py --help
uv run scripts/build_runs_view.py
```

## UI Summary

The interactive posterior viewer supports the single-exponential decay
diagnostics. It lets you draw signals, toggle the current population-trained NPE layers,
compare against grid and MCMC references, and inspect corner plots, predictive
plots, posterior quantiles, low-prior signal stress cases,
Wasserstein-to-grid distances, and runtime diagnostics.

To run the built viewer:

```sh
cd viewer-ui
npm install
npm run build
cd ..
uv run scripts/npe_posterior_viewer.py
```

To view the built UI from the Pixel, keep Tailscale enabled on the phone and
run:

```sh
scripts/start_posterior_viewer_phone.sh
```

The script builds `viewer-ui/dist`, binds the viewer to the MacBook Tailnet
address when available, and prints the phone URL. Override `HOST`, `PUBLIC_HOST`,
or `PORT` if you need a LAN address or alternate port.

For frontend development, run the backend and Vite dev server separately:

```sh
uv run scripts/npe_posterior_viewer.py
```

```sh
cd viewer-ui
npm run dev
```

For phone-based frontend development, keep the backend running locally and use
the Tailnet URL printed by Vite:

```sh
cd viewer-ui
npm run dev:phone
```
