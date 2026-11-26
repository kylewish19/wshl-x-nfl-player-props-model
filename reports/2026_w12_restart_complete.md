# Week 12 restart handoff — November 26, 2026

## Approved production baseline
- Official v0.3.0 artifact 15129935005; v0.5 shadow artifact 15129795747.
- Both fitted to audited data through 2026-11-26 14:27 UTC (164 completed games). No architecture promotion.
- Joint synthetic pregame validation passed: 37 rows across three games, all six official and three shadow models, correct identities and alternate-line consistency. See reports/2026_w12_matched_smoke.json. Report recovered from successful computation logs after Git publishing failed.
- Active manifest authorizes reviewed drafts, not automatic betting-card locks. Real FanDuel inputs, actual availability/context and end-to-end input review still required.
- Previous official cumulative ledger 14–14 through Week 4; Weeks 5–11 had no locked cards and must not be graded as betting selections.

## Research decisions
- QB passing v0.5 improved 2026 Weeks 8–11 proxy probability and point metrics; keep as shadow, no promotion.
- Conditional residual distributions: QB passing workload + stability and RB rushing workload groups are candidates for future prospective shadows. No blanket workload veto; receiving and QB rushing results mixed.
- Receiving 25% shrinkage is a research candidate; global reduction rejected.
- Defense + matchup marginally improved historical Brier/log loss but not high-probability sack-YES hit rate. Keep official sack model.
- Game-script and prior-QB continuity had marginal or mixed benefit. No blanket QB-change veto or new promoted features.
- Future matchup feature coverage was zero; fix committed and synthetic regression CI passed; real-data corrected preflight pending. Do not deploy new matchup challengers until verified and evaluated. Existing model specs exclude these added blocks.

## Live workflow
1. Collect complete FanDuel board and timestamp, identify unsupported markets separately.
2. Verify target games are still pregame and data cutoff has not advanced; refresh and audit when needed.
3. Use pinned artifacts and scripts/predict_reviewed_week12.py for draft predictions. Explicit context review, including confirmed player availability, starting QB, role, opportunity, and target vacancy. No fabricated context decisions.
4. Generate Top-10 Accuracy, Raw Model BET, Official Value via scripts/build_reviewed_cards.py. Preserve raw labels; official requires valid offered price, positive model-implied EV, established probability/edge requirements and v0.5 direction agreement where shared.
5. Lock timestamped cards before kickoff and grade actuals, separately by card and prop category. Preserve previous locks. Do not infer profitability from proxy tests.

## Outstanding
- Real FanDuel end-to-end slate validation and reviewed card.
- Real-data corrected matchup preflight (run 45844849982); future shadow integration.
- Historical reports and schedule closing lines are not verified original prediction-time snapshots; no historical proxy result is an actual betting record.
