from pathlib import Path
import json
import pandas as pd
import yaml

from wshlx_nfl_props.data import load_raw
from wshlx_nfl_props.features import build_feature_frame

cfg=yaml.safe_load(Path("config/model_config.yaml").read_text())
d=load_raw()
frame=build_feature_frame(
    d["player_stats"],d.get("schedules"),d.get("pbp"),
    tuple(cfg["rolling_windows"]),auxiliary=d
)

prefixes=["snap_","ngs_pass_","ngs_rush_","ngs_rec_","pfr_pass_","pfr_rush_","pfr_rec_","pfr_def_","teamctx_","oppctx_","pregame_"]
report={"rows":int(len(frame)),"by_prefix":{}}
for pref in prefixes:
    cols=[c for c in frame.columns if c.startswith(pref)]
    if not cols:
        continue
    derived=[c for c in cols if c.endswith("_lag1") or "_r3" in c or "_r5" in c or pref=="pregame_"]
    use=derived or cols
    nonnull=frame[use].notna().any(axis=1)
    report["by_prefix"][pref]={
        "columns":len(use),
        "row_coverage_pct":float(nonnull.mean()*100),
        "coverage_by_position_pct":{
            str(k):float(v*100) for k,v in nonnull.groupby(frame["position"]).mean().sort_values(ascending=False).head(20).items()
        }
    }

Path("reports").mkdir(exist_ok=True)
Path("reports/feature_coverage.json").write_text(json.dumps(report,indent=2))
print(json.dumps(report,indent=2))
