"""Refit the existing v0.5 artifact without selecting new specs or calibration."""
import json
from pathlib import Path
import joblib
import numpy as np
import pandas as pd
import yaml
from sklearn.base import clone
from wshlx_nfl_props.data import load_raw
from wshlx_nfl_props.features import build_feature_frame
from wshlx_nfl_props.models import load_model

cfg=yaml.safe_load(Path("config/model_config.yaml").read_text())
d=load_raw()
frame=build_feature_frame(d["player_stats"],d["schedules"],d.get("pbp"),tuple(cfg["rolling_windows"]),auxiliary=d)
root=Path("models/challenger_v05")
report={"version":"0.5.0-shadow","source_artifact_id":10666874411,"status":"fixed_spec_refit","calibration":"Original historical out-of-fold residuals preserved","props":{}}
for prop in ["qb_pass_yds","rb_rush_yds","rec_yds"]:
    m=load_model(prop,root=root)
    assert m.validation_metrics["latest_season"]==2026 and m.validation_metrics["latest_week"]==2
    assert len(m.residuals)>=20 and np.isfinite(m.residuals).all()
    spec=cfg["models"][prop]
    f=frame[frame.position.isin(spec["positions"]) & (frame.history_games>=cfg["min_history_games"])].copy()
    y=pd.to_numeric(f[m.yardage_target],errors="coerce")
    v=pd.to_numeric(f[m.volume_target],errors="coerce")
    f=f[y.notna() & v.notna()].copy()
    y=y.loc[f.index]; v=v.loc[f.index]
    positive=v>0
    e=(y[positive]/v[positive]).clip(*m.efficiency_clip)
    cols=m.numeric_features+m.categorical_features
    assert not (set(cols)-set(f.columns)),prop
    X=f[cols].copy()
    for c in m.categorical_features:
        q=X[c].astype(object)
        X[c]=q.where(q.notna(),"__MISSING__").astype(str)
    m.volume_pipeline=clone(m.volume_pipeline).fit(X,v)
    m.efficiency_pipeline=clone(m.efficiency_pipeline).fit(X.loc[positive],e)
    latest=int(f.season.max()); week=int(f.loc[f.season==latest,"week"].max())
    assert latest==2026 and 11<=week<=12
    m.validation_metrics.update({"live_refit_rows":len(f),"latest_season":latest,"latest_week":week,"source_artifact_id":10666874411,"refit_cutoff_utc":"2026-11-26T14:27:00Z"})
    joblib.dump(m,root/f"{prop}.joblib")
    meta={"prop":prop,"estimator":m.estimator_name,"feature_set":m.feature_set,"numeric_features":m.numeric_features,"categorical_features":m.categorical_features,"validation_metrics":m.validation_metrics,"volume_target":m.volume_target,"yardage_target":m.yardage_target}
    (root/f"{prop}.json").write_text(json.dumps(meta,indent=2))
    report["props"][prop]={"algorithm":m.estimator_name,"feature_set":m.feature_set,"training_rows":len(f),"efficiency_rows":int(positive.sum()),"latest_season":latest,"latest_week":week,"residual_count":len(m.residuals)}
Path("reports/2026_w12_v05_refit.json").write_text(json.dumps(report,indent=2))
print(json.dumps(report,indent=2))
