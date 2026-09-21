from pathlib import Path
import json,yaml
from wshlx_nfl_props.data import load_raw
from wshlx_nfl_props.features import build_feature_frame
from wshlx_nfl_props.models import train_prop_model,save_model

cfg=yaml.safe_load(Path("config/model_config.yaml").read_text())
d=load_raw()
frame=build_feature_frame(
    d["player_stats"],d.get("schedules"),d.get("pbp"),
    tuple(cfg["rolling_windows"]),auxiliary=d
)

specs={
  "receptions":{"target":"receptions","positions":["WR","RB","FB","TE"],"kind":"regression"},
  "te_rec_yds":{"target":"receiving_yards","positions":["TE"],"kind":"regression"}
}
report={"version":"addon-v0.1.0","status":"first_live_validation","models":{}}
for prop,s in specs.items():
    m=train_prop_model(
      frame,prop,s["target"],s["positions"],s["kind"],
      selection_season=cfg["selection_season"],test_season=cfg["test_season"],
      min_history_games=cfg["min_history_games"],seed=cfg["random_state"],
      feature_sets=("baseline","enriched","pregame","full")
    )
    save_model(m,root="models/addon_v01")
    report["models"][prop]={"algorithm":m.estimator_name,"feature_set":m.feature_set,**m.validation_metrics}

Path("reports").mkdir(exist_ok=True)
Path("reports/addon_v0.1.0_validation.json").write_text(json.dumps(report,indent=2))
print(json.dumps(report,indent=2))
