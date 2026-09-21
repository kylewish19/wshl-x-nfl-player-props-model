# WSHL_X NFL Player Props Model

A real, versioned NFL player-prop ML pipeline for:

- QB passing yards
- QB passing touchdowns
- QB rushing yards
- RB rushing yards
- WR/RB receiving yards
- Player to record 1+ sack

## Philosophy

The project uses 2021-current NFL data, with strict pregame feature shifting so a game's result can never appear in its own prediction features. Initial algorithm selection is performed against a 2025 holdout. After selection, the chosen algorithm is refit on all completed games, including completed 2026 games, for live predictions.

This is not a "change coefficients until last week's picks win" system. Each week is locked before kickoff, graded after the games, diagnosed, and only then are challenger changes tested out-of-sample. A challenger is promoted only when it improves the relevant validation/calibration metrics without creating obvious instability.

## Data

The ingestion layer uses `nflreadpy` / nflverse and attempts to load:

- player weekly stats
- team weekly stats
- play-by-play
- schedules/results
- weekly rosters
- snap counts
- injury reports
- Next Gen passing/rushing/receiving stats
- PFR advanced passing/rushing/receiving/defense stats

Optional sources may be late; the pipeline continues if an enrichment table is unavailable. Core player stats and schedules are required.

## Models

Each prop has its own champion-selection process.

- Yardage: Elastic Net vs Random Forest vs histogram gradient boosting; lowest holdout MAE wins.
- Passing TDs: Poisson GLM vs Poisson gradient boosting; lowest holdout Poisson deviance wins.
- Sack yes/no: logistic regression vs gradient boosting; lowest holdout Brier score wins.

Yardage side probabilities use an empirical holdout-residual bootstrap rather than assuming every prop is normally distributed. Passing-TD probabilities come from the predicted Poisson rate. Sack probabilities are direct binary probabilities.

## Weekly workflow

```bash
pip install -r requirements.txt
pip install -e .
python scripts/update_data.py
python scripts/train_models.py
python scripts/predict_props.py inputs/week3_lines.csv --out predictions/2026_week3_locked.csv
# after games and data refresh
python scripts/grade_week.py predictions/2026_week3_locked.csv
```

### Required input columns

`player_display_name,prop,line`

Optional columns: `team,opponent,week,over_odds,under_odds,yes_odds`.

Prop names:
`qb_pass_yds`, `qb_pass_tds`, `qb_rush_yds`, `rb_rush_yds`, `rec_yds`, `sack_yes`.

## Version discipline

1. Refresh data.
2. Train challenger models.
3. Review holdout metrics and leakage checks.
4. Lock weekly predictions before games.
5. Grade every locked pick, including passes if desired.
6. Diagnose misses by prop, side, position, edge bucket, usage surprise, injury/role change, game script and opponent matchup.
7. Implement only justified feature/model changes.
8. Backtest the challenger on prior seasons/weeks.
9. Promote a new version only if the evidence supports it.

The GitHub Action runs a Tuesday-morning data refresh/challenger build after Monday night games. It uploads the models/reports as an artifact; it does **not** silently overwrite the champion model.
