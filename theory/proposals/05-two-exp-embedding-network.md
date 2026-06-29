# Proposal 05: Add A Learned Embedding For Two-Exponential Raw Curves

## Claim

The unresolved two-exponential case should test a learned embedding network for
raw curves. Passing raw high-dimensional curves directly into a conditional
posterior flow increases sample complexity and may be hurting conditional
learning.

## Literature Signal

- Lueckmann et al. use learned features for time-series observations:
  https://arxiv.org/abs/1711.01861
- BayesFlow emphasizes learned summary networks with invertible posterior
  networks: https://arxiv.org/abs/2003.06281
- `sbi` recommends embedding nets for high-dimensional outputs:
  https://sbi.readthedocs.io/en/latest/advanced_tutorials/04_embedding_networks.html

## Current Code Touchpoints

In `scripts/sbi_two_exp_ordered.py`:

```text
posterior_nn(..., embedding_net=Identity())
```

implicitly uses the default identity embedding. The script supports
`context_kind=raw` and `context_kind=profile`, but not learned embeddings.

In `scripts/npe_flow_stress_tests.py`, custom contexts are hand-built per case.
There is no learned context embedding for the custom Zuko flow.

## Implementation Sketch

Start in `scripts/sbi_two_exp_ordered.py` because `sbi.posterior_nn` already
supports `embedding_net`.

Add CLI:

```text
--embedding none|mlp|conv1d
--embedding-dim 8
```

MLP option:

```text
nn.Sequential(
    nn.Linear(n_obs, 128),
    nn.ReLU(),
    nn.Linear(128, 64),
    nn.ReLU(),
    nn.Linear(64, embedding_dim),
)
```

Conv1D option:

```text
curve -> Conv1d blocks -> adaptive pooling -> MLP -> embedding_dim
```

Then:

```text
density_estimator = posterior_nn(
    model=args.density_estimator,
    embedding_net=embedding_net,
    ...
)
```

For `context_kind=profile`, keep identity or MLP over profile summaries.

## Acceptance Criteria

- `--context-kind raw --embedding mlp` runs end to end.
- Summary records embedding type, embedding dimension, and architecture.
- Compare raw identity versus raw MLP with the same simulation budget.
- If the embedding improves agreement, run calibration diagnostics before
  declaring success.

## First Target

Use the ordered two-exponential raw NSF setup with:

```text
--density-estimator nsf
--context-kind raw
--embedding mlp
--embedding-dim 8
```

The expected benefit is not guaranteed. The point is to distinguish "raw context
is too hard for the flow conditioner" from "the posterior geometry itself is the
blocker."
