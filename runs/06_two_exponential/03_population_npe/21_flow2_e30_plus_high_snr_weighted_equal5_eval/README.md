# Two-Exponential Equal-5 Ensemble Check

This evaluation tests whether the high-SNR weighted member is complementary to
the previous best 4-member Flow2 ridge-target ensemble.

Composition:

```text
members  4 x Flow2 ridge target, 512k simulations, 30 epochs
         1 x Flow2 high-SNR weighted ridge target, 512k simulations, 30 epochs
weights  equal
```

Result:

```text
validation NLL   -3.20086 +/- 0.02400
paired floor     -3.27982 +/- 0.02425
paired gap        0.07896
common-floor gap  0.08064  (using README reference floor -3.28149)
paired gap SE     0.00413
gap z-score       19.12
```

Conclusion: the high-SNR weighted member is mildly complementary, but the
equal-weight mixture remains far above the full-prior entropy floor.
