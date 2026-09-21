from pathlib import Path
import json,yaml
import numpy as np
import pandas as pd
from sklearn.metrics import mean_absolute_error,brier_score_loss

from wshlx_nfl_props.data import load_raw
from wshlx_nfl_props.game_features import build_game_frame,game_feature_columns
from wshlx_nfl_props.game_market_informed import add_market_priors,market_informed_feature_columns
from wshlx_nfl_props.game_models import train_game_model,save_game_model

cfg=yaml.safe_load(Path("config/game_model_config.yaml").read_text())
d=load_raw()
frame=build_game_frame(
    d["schedules"],d.get("team_stats"),d.get("pbp"),
    windows=tuple(cfg["rolling_windows"])
)
frame=frame[frame["season"].isin(cfg["seasons"])].copy()

# Reattach sportsbook fields deliberately for this SHADOW family only.
s=d["schedules"].copy()
if "game_type" in s.columns:
    s=s[s.game_type.astype(str).str.upper().isin(["REG","REGULAR"])]
keep=["season","week","home_team","away_team"]+[c for c in [
    "spread_line","total_line","home_moneyline","away_moneyline"
] if c in s.columns]
market=s[keep].drop_duplicates(["season","week","home_team","away_team"])
frame=frame.merge(market,on=["season","week","home_team","away_team"],how="left")
frame=add_market_priors(frame)

base_num,base_cat=game_feature_columns(frame)
numeric,categorical=market_informed_feature_columns(frame,base_num,base_cat)

specs={
    "margin_residual":{"target":"target_margin_residual","kind":"regression"},
    "total_residual":{"target":"target_total_residual","kind":"regression"},
    "home_win_market":{"target":"target_home_win","kind":"binary"},
}
root="models/game_markets_market_v02"
report={
    "version":"0.2.0-shadow",
    "status":"market_informed_shadow",
    "training_line_source":"nflverse historical closing markets",
    "models":{}
}
for name,spec in specs.items():
    m=train_game_model(
        frame,name,spec["target"],spec["kind"],numeric,categorical,
        selection_season=cfg["selection_season"],test_season=cfg["test_season"],
        seed=cfg["random_state"]
    )
    save_game_model(m,root=root)
    report["models"][name]={"algorithm":m.estimator_name,**m.validation_metrics}

# Convert residual test error to comparable final-market metrics.
# Residual MAE is exactly final prediction MAE relative to actual margin/total.
report["comparables"]={
    "2025_margin_mae":report["models"]["margin_residual"]["test_metrics"]["mae"],
    "2025_total_mae":report["models"]["total_residual"]["test_metrics"]["mae"],
    "2025_moneyline_brier":report["models"]["home_win_market"]["test_metrics"]["brier"],
}
Path("reports").mkdir(exist_ok=True)
Path("reports/game_markets_market_v0.2.0.json").write_text(json.dumps(report,indent=2))
print(json.dumps(report,indent=2))
