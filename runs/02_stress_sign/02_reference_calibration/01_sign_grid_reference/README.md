# 02_reference_calibration / 01_sign_grid_reference

Status: `diagnostic`
Reason: model-specific target calibration against a dense grid posterior

Artifacts:

- `results/sign_target_calibration_summary.json` - calibrated targets and sampler/NPE distances to grid
- `figures/sign_target_calibration.png` - diagnostic distance comparison
- script: `scripts/calibrate_sign_target.py`
- note: `notes/sign-target-calibration.md`

Key result:

- historical target: `0.034`
- calibrated diagnostic target: `0.023314`
- NPE diagnostic distance to grid: `0.032607`
- conclusion: the previous sign NPE run passes the inherited target but misses the model-calibrated diagnostic target.
