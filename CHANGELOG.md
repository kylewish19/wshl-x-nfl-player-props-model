# Changelog

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
