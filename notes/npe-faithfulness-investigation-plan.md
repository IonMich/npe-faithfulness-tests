# NPE faithfulness investigation plan

Goal: explain why the current NPE/SNPE runs do not reach the strict MC-level target

```text
mean normalized Wasserstein <= 0.034
```

and identify practical fixes.

## Current evidence

Best results so far:

| Method | Best mean normalized W | Target ratio |
| --- | ---: | ---: |
| Broad scaled MDN at `x0` | 0.1564 | 4.60x |
| Local x-region MDN | 0.0867 | 2.55x |
| Sequential SNPE custom MDN | 0.0878 | 2.58x |
| Sequential SNPE custom diagonal Gaussian | 0.0848 | 2.49x |
| Exact-target correction using analytic likelihood | about 0.031 | passes |

The learned posteriors look visually close, but not MC-faithful by the strict target.

## Diagnostic avenues

### 1. Sanity-check the target

Question:

```text
Is 0.034 a fair numerical target, or is it below the natural MC/grid noise floor?
```

Tests:

- Split MCMC/HMC chains into independent halves and compare each half to the grid.
- Compare MCMC/HMC split halves to each other.
- Sample repeatedly from the grid posterior and compare grid-sample replicates to the exact weighted grid.

Interpretation:

- If independent MC splits are already near or above 0.034, the target is too aggressive.
- If split references are well below 0.034, then the target is fair and NPE is genuinely missing MC-level accuracy.

### 2. Oracle posterior density fit at `x0`

Question:

```text
Can the neural density family fit the true posterior at all if given posterior samples directly?
```

Test:

- Train the same density families on posterior samples from MCMC/HMC or grid at `x0`.
- Use a constant context, so the problem is only density estimation in `theta`, not learning `x -> posterior`.
- Compare generated samples to the grid posterior.

Interpretation:

- If the oracle fit fails, the density family/training is the bottleneck.
- If the oracle fit passes, the problem is the simulation-to-posterior learning objective, not the posterior family.

### 3. Lower-dimensional summaries

Question:

```text
Is mapping 40 raw observations to the posterior unnecessarily hard?
```

Test:

- Replace raw `x` with curve summaries: binned curve means, rough noise, rough scale, early-minus-late difference.
- Run broad and/or local NPE using those summaries as the context.
- Evaluate at `x0` against the same grid.

Interpretation:

- If summaries improve faithfulness, raw high-dimensional context is a major source of estimator bias.
- If summaries do not improve, the bottleneck is elsewhere.

### 4. Kernel-weighted local training

Question:

```text
Can we fix hard local accept/reject instability by using smooth local weights around x0?
```

Test:

- Simulate a pool from the prior.
- Compute summary distance to `x0`.
- Train with weights

```math
w(x) = \exp(-d(x,x_0)^2 / 2h^2)
```

instead of hard filtering.

Interpretation:

- If this improves over hard local filtering, the hard region boundary was causing bias/instability.
- If not, localizing alone is insufficient.

### 5. Alternate SBI variant

Question:

```text
Is NPE the wrong simulation-based inference objective for this problem?
```

Tests, as practical:

- Use `sbi` SNLE or NRE on this simulator.
- Evaluate the posterior samples against the same grid.

Interpretation:

- If SNLE/NRE passes while NPE fails, direct posterior regression is the problem.
- If they also fail at finite budget, the target may require either much larger budgets or likelihood-aware correction.

## Reporting

The final report should include:

- Which target sanity checks pass/fail.
- Whether oracle density fitting reaches `0.034`.
- Whether summaries improve over raw `x`.
- Whether kernel local weighting improves over hard local filtering.
- Whether any alternate SBI method reaches `0.034`.
- A concrete recommendation for the next reliable inference path.
