# Changelog

## 2026 Week 2 postgame — Giants at Rams
- First locked live player-prop card graded **5-5**.
- Added postgame context tags distinguishing sportsbook result from calibration eligibility.
- Jaxson Dart's two Under wins were retained in betting W-L but marked calibration-ineligible after his opening-drive knee injury.
- Preserved v0.3 as official; no coefficient/model promotion from one noisy game.
- Required v0.5 opportunity×efficiency shadow logging on the next live card after both Kyren Williams and Blake Corum beat their rushing-yard Under lines by large margins.
- Added target-vacancy preflight protection after Puka Nacua/Jordan Whittington absences materially redistributed Rams receiving volume.
- Receptions/TE-yardage add-on remains shadow-only after a 10-8 raw first card.
- Game-market family remains shadow-only after NYG +6.5 lost and prior historical testing failed to beat the closing market.
- Abdul Carter sack +198 shadow hit; sack probability threshold remains unchanged pending a larger sample.

## v0.2.0 — leakage correction + enriched challenger
- Invalidated v0.1.0 after discovering that raw same-game usage-share/team-volume columns could enter the model feature matrix.
- Restricted model inputs to explicitly lagged, rolling, or otherwise pregame-safe derived features.
- Added regression tests that perturb current-game targets/yards and verify the same game's model inputs do not change.
- Split model development chronologically: pre-2024 development, 2024 specification selection, untouched 2025 test.
- Added lagged snap-count, Next Gen Stats, and PFR advanced-stat enrichments.
- Competes baseline vs enriched feature sets per prop rather than automatically assuming more features are better.
- Final live refit can use all completed 2026 data only after the specification is frozen.

## v0.1.0 — foundation (INVALIDATED)
- Initial nflverse/nflreadpy ingestion and six-prop framework.
- This version must not be used for betting or benchmarking because leakage was later discovered in feature selection.
