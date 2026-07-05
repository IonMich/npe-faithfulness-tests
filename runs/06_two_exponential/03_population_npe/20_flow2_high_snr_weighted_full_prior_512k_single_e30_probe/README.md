# Two-Exponential High-SNR Weighted Probe

This run tests whether the current full-prior two-exponential miss is repaired
by upweighting the high-SNR prior tail identified by the paired-gap diagnostic.

Recipe:

```text
target                       (log(A1 + A2), log(A1/A2), log k1, log Delta k, log sigma)
model                        Flow2 residual NSF, random permutations
train simulations            524288
ensemble members             1
epochs                       30
loss weighting               top 20% log-SNR draws, 4x tail weight
validation examples          10000
floor estimator              Gaussian-mixture importance, 8192 samples per signal
```

Result:

```text
validation NLL   -3.16299 +/- 0.02443
entropy floor    -3.27982 +/- 0.02425
paired gap        0.11683
common-floor gap  0.11851  (using README reference floor -3.28149)
paired gap SE     0.00526
gap z-score       22.21
weighted train    -4.16513
```

Conclusion: high-SNR tail weighting lowered the weighted training objective but
worsened the unweighted full-prior validation NLL. It is not a viable repair for
the global objective.
