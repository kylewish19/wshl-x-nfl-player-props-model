# Model Card — v0.1.0 foundation

Status: foundation / challenger framework, not yet an official betting card.

Training population: NFL regular season player-weeks, 2021-current.

Initial validation: 2025 season holdout. Live refit may use completed 2026 games only after the algorithm family is chosen on the holdout.

Primary metrics:
- yardage: MAE, RMSE, residual calibration
- passing TD count: Poisson deviance, MAE
- sack yes/no: Brier score, log loss, ROC-AUC
- betting layer: W-L-P, hit rate by prop/side/edge bucket, closing-line comparison when available, ROI only when prices are recorded

Leakage rule: rolling player/team/opponent features are shifted by one game. Current-game yards, attempts, targets, sacks, or results cannot enter the same game's feature row.

Known v0.1 limitations to audit after the first live weeks:
- injury/active-status handling is loaded but not yet deeply encoded into the feature set
- offensive-line continuity and route participation may need richer features
- sack model should gain pressure/pass-rush win-rate features where reliable current-week data permits
- distribution models for yardage can later move from empirical residual bootstrap to calibrated quantile/conformal models if validation improves
