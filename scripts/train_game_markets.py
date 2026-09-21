from pathlib import Path
import json, yaml
from wshlx_nfl_props.data import load_raw
from wshlx_nfl_props.game_features import build_game_frame, game_feature_columns
from wshlx_nfl_props.game_models import train_game_model, save_game_model

cfg=yaml.safe_load(Path("config/game_model_config.yaml").read_text())
d=load_raw()
if "schedules" not in d or d["schedules"].empty:
    raise SystemExit("Run scripts/update_data.py --config config/game_model_config.yaml first")

frame=build_game_frame(
    d["schedules"],d.get("team_stats"),d.get("pbp"),
    windows=tuple(cfg["rolling_windows"])
)
Path("data/processed").mkdir(parents=True,exist_ok=True)
frame.to_parquet("data/processed/game_market_frame.parquet",index=False)
numeric,categorical=game_feature_columns(frame)

report={
    "version":cfg["version"],
    "market_informed_features":cfg["market_informed_features"],
    "numeric_feature_count":len(numeric),
    "categorical_features":categorical,
    "models":{}
}
for market,spec in cfg["markets"].items():
    print(f"[game-train] {market}")
    m=train_game_model(
        frame,market,spec["target"],spec["kind"],numeric,categorical,
        selection_season=cfg["selection_season"],test_season=cfg["test_season"],
        seed=cfg["random_state"]
    )
    save_game_model(m)
    report["models"][market]={
        "algorithm":m.estimator_name,
        **m.validation_metrics,
    }

Path("reports").mkdir(exist_ok=True)
Path("reports/game_markets_v0.1.0.json").write_text(json.dumps(report,indent=2))
print(json.dumps(report,indent=2))
