# Reporting Protocol — Week 4 onward

Beginning with 2026 Week 4, every supported FanDuel player-prop slate will be tracked in three separate layers.

## 1. Top-10 Accuracy Card
Purpose: measure pure model ranking quality.

- Rank all supported props by the model's estimated probability of the selected side hitting.
- Select the 10 highest-probability props.
- Do not optimize for price or expected value.
- Preserve only minimal validity exclusions: inactive/non-playing player, unresolved player identity, unsupported market, or clearly invalid role state.
- Grade hit rate, calibration by probability bucket, and result by prop family.
- This card is diagnostic and is not automatically the official betting card.

## 2. Official Value Card
Purpose: identify playable bets.

- Start from model probability.
- Apply FanDuel price/edge requirements.
- Apply the established injury, role, QB-change, target-vacancy, and opportunity-validity gates.
- Grade W-L, hit rate, ROI when odds are recorded, and results by prop family/side.
- This remains the official betting card.

## 3. Raw Model BET Card
Purpose: measure whether context filters help or hurt.

- Preserve every model-qualified BET signal before manual/context filtering.
- Grade independently from the official card.
- Compare raw BET results against the official context-gated card each week.

## Tuesday grading
Each Tuesday, report:
- Top-10 Accuracy Card record
- Official Value Card record
- Raw Model BET Card record
- performance by prop family
- performance by probability bucket
- context-gate wins/losses: avoided losses vs filtered winners
- v0.3 vs v0.5 shadow comparison where both support the market

## Promotion rule
This reporting change does not alter or retrain the model.

No probability threshold, context gate, feature set, or challenger model is promoted based on one slate or one week. Candidate changes must be tested chronologically and must improve out-of-sample performance before affecting official Week 4+ selections.
