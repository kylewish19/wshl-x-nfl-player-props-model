# November 26 frozen-model diagnostic and reporting controls

Source: reports/2026_w12_frozen_diagnostic.json; GitHub Actions run 45840156575.
Both frozen artifacts were fitted through 2026 Week 2. No refit or promotion performed.
Selection: Weeks 4-7. Holdout: Weeks 8-11. Proxy thresholds are lagged rolling
averages (fixed 1.5 pass TDs and 0.5 sack threshold), not archived FanDuel lines.
Evaluation is conditional on recorded player participation, and feeds may contain
historical corrections. No betting W-L or ROI is inferred.

## Decisions
- QB pass yards: v0.5 improved holdout MAE (62.20 to 59.89), Brier (0.2198 to
  0.2094), log loss and directional accuracy on 116 rows. Priority candidate
  for prospective validation; retain v0.3 official.
- RB rushing and receiving: retain v0.3; v0.5 modest MAE gains did not improve
  probability scores.
- Probability shrinkage: reject global shrinkage. RB 50% shrink selected in
  Weeks 4-7 worsened holdout. Receiving 25% shrink improved Brier from
  0.220456 to 0.217957 and log loss from 0.635011 to 0.626235 on 545 rows;
  research-only candidate, not production.
- Sack model: 79.83% directional accuracy is approximately the always-no
  baseline (79.75%); do not interpret this as a sack-YES hit rate.
  Numeric feature mean coverage 40.38%; many missing offensive features are
  position-inapplicable rather than a failed data feed. Investigate position-
  specific feature selection and compare with a base-rate model.
- Defensive tackle source fields exist (def_tackles_solo,
  def_tackles_with_assist, def_tackle_assists); canonical def_tackles was not
  resolved. Verify field semantics and test a mapping in a challenger before
  any champion feature change.
- Snap feature block: any nonnull 97.475%; QB pregame numeric feature average
  98.32%; QB rush/RB rush/receiving baseline numeric coverage 100%.

## Reporting integrity
Added reporting.py and tests/test_reporting.py. The helper ranks all explicitly
valid supported rows by model probability, preserves raw BET signals unchanged,
and requires explicit context approval, valid offered odds, and positive
model-implied expected value for the official card. These are reporting
safeguards, not new trained-model parameters. Existing live workflows have
not yet been wired to call this helper; CI and integration must be verified.
Never modify original locked cards retrospectively.

## Next
Test conditional yardage residuals using strictly lagged workload and
opportunity stability, with a fresh evaluation window or historical
walk-forward splits. Weeks 8-11 have now been inspected and cannot be
called an untouched holdout for subsequent tuning. Refit fixed champion
specifications only on verified completed data, preserving artifact provenance.
