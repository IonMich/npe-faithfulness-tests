# Decay UI Whole-Prior NPE Option Plan

Date: 2026-06-26

## Purpose

Add a UI option for the exponential-decay posterior viewer to sample from an NPE
trained on the whole decay-model prior predictive distribution, alongside the
current local `0.5% near x_0` decay-flow model.

This plan is only about the decay model:

```text
theta = (A, k, sigma)
x_j = A exp(-k t_j) + epsilon_j
epsilon_j ~ Normal(0, sigma^2)
```

It is not about the SBI tutorial Gaussian example.

The UI should make two choices independent:

```text
1. Which posterior estimator to use.
2. Which signal x to condition on.
```

This matters because the current posterior viewer is effectively a local-flow
viewer. It loads one conditional spline-flow checkpoint at server startup:

```text
runs/01_exponential_decay/03_npe_flow_search/
11_npe_flow_local_q0005_linear_150k_t8_seed20260706/
results/npe_flow_decay_model.pt
```

That checkpoint was trained with `training_mode = local_prior` and
`local_quantile = 0.005`, so it should not be presented as a globally
amortized estimator.

## Current State

Entry point:

```text
scripts/npe_posterior_viewer.py
```

Current behavior:

- The server loads exactly one model path through `--model`.
- The loaded model is assumed to be `ConditionalSplineFlow` from
  `scripts/npe_flow_decay.py`.
- The signal dropdown controls only the source of the observation:
  - new local-region signal;
  - original `x_0`;
  - prior-predictive stress signal.
- The UI reports local-region distance only because the loaded model has a
  saved local training region.

Terminology used in this plan:

- `broad prior-predictive` means training on simulations drawn from the full
  decay-model prior, without filtering to be near `x_0`.
- `MDN` means mixture density network: a conditional density model that outputs
  a mixture of Gaussian components for `z = log(theta)` given an observed curve
  `x`.
- `Stage 1` is repo shorthand for the first broad NPE training script,
  `scripts/npe_stage1_decay.py`. It is not a special Bayesian stage; it is just
  the name of an earlier experiment family.

Current loadable broad-prior candidates:

```text
runs/01_exponential_decay/02_npe_stage1_local_summary/
12_npe_stage1_scaled/results/mdn_model.pt
```

This is a broad prior-predictive MDN checkpoint trained on 100,000 simulations
by `scripts/npe_stage1_decay.py`. Its single-`x_0` mean normalized Wasserstein
was about `0.156`, so it is a useful comparison model but not a faithful
global-amortization claim.

Best broad-prior result found in the existing summaries:

```text
runs/01_exponential_decay/02_npe_stage1_local_summary/
13_npe_summary_broad_mdn_100k/results/npe_summary_context_summary.json
```

That summary-context broad MDN reached about `0.115` at `x_0`, but the run does
not appear to have saved a reusable model checkpoint. To expose it in the UI, we
must either reproduce it with checkpoint saving or add checkpoint persistence to
`scripts/npe_summary_context_decay.py` and rerun it.

## Target UX

Add a model selector, separate from the signal selector:

```text
Posterior estimator:
  - Local flow q=0.005, 150k simulations
  - Broad prior-predictive MDN, 100k simulations
  - Broad prior-predictive summary-context MDN, 100k simulations, if checkpointed

Signal source:
  - Original x_0 signal
  - New prior-predictive signal
  - New local-region signal, local models only or clearly labelled
```

For each rendered result, show:

- estimator label;
- training distribution: `local_prior q=0.005` or `whole_prior`;
- training simulations;
- checkpoint path;
- conditioning signal source;
- whether the selected signal is inside the local region, only when the selected
  estimator has a local region;
- warning text when using a local model on an out-of-local-region signal.

Avoid implying that the whole-prior model is already claim-grade amortized. The
label should say "whole-prior trained", not "globally amortized".

## Implementation Plan

### Step 1: Add a posterior-estimator adapter layer

Introduce a small internal interface in `scripts/npe_posterior_viewer.py`:

```text
PosteriorEstimatorAdapter
  label
  checkpoint_path
  training_metadata
  context_kind
  sample(x, n) -> (z_samples, theta_samples)
  local_distance(context) -> optional distance metadata
```

Then implement:

- `FlowDecayAdapter` for existing `ConditionalSplineFlow` checkpoints.
- `Stage1Adapter` for loadable Stage 1 models from
  `scripts/npe_stage1_decay.py`.
- Optional later: `SummaryContextAdapter` after saving a checkpoint for the
  summary-context broad model.

The existing rendering functions should consume adapter outputs instead of
directly calling `sample_flow_posterior`.

### Step 2: Build a model registry

Add CLI arguments:

```text
--models-config path/to/posterior_viewer_models.json
--default-model-id local_flow_q0005_150k
```

Suggested config schema:

```json
{
  "models": [
    {
      "id": "local_flow_q0005_150k",
      "label": "Local flow q=0.005, 150k",
      "kind": "flow_decay",
      "checkpoint": "runs/.../npe_flow_decay_model.pt",
      "summary": "runs/.../npe_flow_decay_summary.json"
    },
    {
      "id": "whole_prior_stage1_mdn_100k",
      "label": "Broad prior-predictive MDN, 100k",
      "kind": "stage1",
      "family": "mdn",
      "checkpoint": "runs/.../mdn_model.pt",
      "summary": "runs/.../npe_stage1_summary.json"
    }
  ]
}
```

Keep the current `--model` path working as a backwards-compatible shortcut for a
single `flow_decay` model.

### Step 3: Update the HTTP API

Current render request parameters include:

```text
mode
posterior_samples
include_grid
grid_size
```

Add:

```text
model_id
```

The server should cache loaded adapters by `model_id` so switching models does
not repeatedly reload checkpoint files.

### Step 4: Update the HTML controls

Add a `select` element for posterior estimator. Keep signal source as a separate
control.

When the selected model has no local-region metadata:

- hide `local_distance`, `local_radius`, and `inside_local_region`, or show
  them as "not applicable";
- do not display local-region warnings;
- keep grid comparison available.

When the selected model is local and the signal source is prior-predictive:

- keep the result allowed;
- display the local distance and inside/outside flag prominently;
- label it as stress-testing the local model outside its declared training
  region when applicable.

### Step 5: Optional checkpointing for the best broad summary-context model

If we want the numerically best existing whole-prior candidate in the UI,
modify `scripts/npe_summary_context_decay.py` to save:

```text
model_state_dict
config
x/context standardization
z standardization
family
context settings
```

Then rerun or reproduce:

```text
runs/01_exponential_decay/02_npe_stage1_local_summary/
13_npe_summary_broad_mdn_100k
```

Only after that should the UI expose `whole_prior_summary_context_mdn_100k`.

## Validation Plan

Smoke checks:

```text
uv run scripts/npe_posterior_viewer.py --help
```

Manual UI checks:

- Local flow plus original `x_0` still reproduces the current behavior.
- Local flow plus prior-predictive signal shows local-distance warning when
  outside the q=0.005 region.
- Broad prior-predictive MDN samples for original `x_0`.
- Broad prior-predictive MDN samples for new prior-predictive signals.
- Grid overlay works for both model kinds.

Quantitative checks:

- For `x_0`, UI-generated samples from the whole-prior MDN should reproduce the
  saved summary-level discrepancy approximately.
- For a fixed random seed and signal, repeated rendering with the same
  posterior sample count should give comparable posterior summaries.

## Deliverables

- Updated `scripts/npe_posterior_viewer.py`.
- Optional `configs/posterior_viewer_models.json`.
- Optional checkpoint-saving patch for `scripts/npe_summary_context_decay.py`.
- A short note recording which whole-prior model is selected as the default and
  why.

## Non-Goals

- Do not claim global amortization from adding this UI option.
- Do not remove the local q=0.005 model.
- Do not make the signal source implicitly choose the model.
- Do not use distance from `x_0` as a global model validity score.
