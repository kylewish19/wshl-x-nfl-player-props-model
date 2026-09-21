from pathlib import Path
import json, yaml
from wshlx_nfl_props.data import load_raw
from wshlx_nfl_props.features import build_feature_frame
from wshlx_nfl_props.models import train_prop_model, save_model

cfg = yaml.safe_load(Path("config/model_config.yaml").read_text())
d = load_raw()
if "player_stats" not in d or d["player_stats"].empty:
    raise SystemExit("Run scripts/update_data.py first")
frame = build_feature_frame(d["player_stats"], d.get("schedules"), d.get("pbp"), tuple(cfg["rolling_windows"]))
Path("data/processed").mkdir(parents=True, exist_ok=True)
frame.to_parquet("data/processed/model_frame.parquet", index=False)
report = {}
for prop, spec in cfg["models"].items():
    print(f"[train] {prop}")
    m = train_prop_model(frame, prop, spec["target"], spec["positions"], spec["kind"],
                         holdout_season=cfg["holdout_season"], min_history_games=cfg["min_history_games"],
                         seed=cfg["random_state"])
    save_model(m)
    report[prop] = {"algorithm": m.estimator_name, **m.validation_metrics}
Path("reports").mkdir(exist_ok=True)
Path("reports/model_validation.json").write_text(json.dumps(report, indent=2))
print(json.dumps(report, indent=2))
