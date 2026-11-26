"""Freeze two selected workload shadow specs; 2025 residuals, all audited data point fit."""
import json
from pathlib import Path
import numpy as np
import pandas as pd
import yaml
from wshlx_nfl_props.data import load_raw
from wshlx_nfl_props.features import build_feature_frame, model_feature_columns
from wshlx_nfl_props.models import _make_pipeline, save_model
from wshlx_nfl_props.workload_shadow import WorkloadShadow, workload_keys

cfg=yaml.safe_load(Path("config/model_config.yaml").read_text())
specs=yaml.safe_load(Path("config/champion_specs.yaml").read_text())["champions"]
d=load_raw()
f=build_feature_frame(d["player_stats"],d["schedules"],d.get("pbp"),tuple(cfg["rolling_windows"]),auxiliary=d)
report={"version":"0.6.0-workload-shadow","status":"shadow_only","calibration_year":2025,"point_training_cutoff":"2026-11-26T14:27:00Z","limitations":["Candidate selection used historical data. No untouched evaluation claim.","Point fit restricted to research workload eligibility, unlike full champion sample; paired pooled baseline isolates distribution effect.","No official pick or context gate changes."],"props":{}}
for prop,vol,bands,stability,minimum in [("qb_pass_yds","passing_attempts",[20,30,40],True,10),("rb_rush_yds","rushing_attempts",[5,12,18],False,2)]:
    s=cfg["models"][prop]; spec=specs[prop]; target=s["target"]
    x=f[f.position.isin(s["positions"])&(f.history_games>=cfg["min_history_games"])].dropna(subset=[target,vol+"_r3",vol+"_lag1",target+"_r3"]).copy()
    x=x[x[vol+"_r3"]>=minimum].copy()
    num,cat=model_feature_columns(x,feature_set=spec["feature_set"])
    cols=num+cat
    for c in cat:
        q=x[c].astype(object); x[c]=q.where(q.notna(),"__MISSING__").astype(str)
    train=x[x.season<2025]; cal=x[x.season==2025].copy()
    assert len(train)>0 and len(cal)>=20
    pipe=_make_pipeline("regression",spec["algorithm"],num,cat,cfg["random_state"])
    pipe.fit(train[cols],train[target])
    residual=cal[target].to_numpy()-pipe.predict(cal[cols])
    groups={}
    for key,r in zip(workload_keys(cal,vol,bands,stability),residual):
        if key is not None: groups.setdefault(key,[]).append(float(r))
    final=_make_pipeline("regression",spec["algorithm"],num,cat,cfg["random_state"])
    final.fit(x[cols],x[target])
    week=int(x.loc[x.season==2026,"week"].max())
    assert 11<=week<=12 and int(x.season.max())==2026
    meta={"latest_season":2026,"latest_week":week,"live_refit_rows":len(x),"calibration_rows":len(cal),"minimum_prior_workload":minimum,"status":"shadow_only"}
    m=WorkloadShadow(prop=prop,kind="regression",estimator_name=spec["algorithm"],feature_set=spec["feature_set"],pipeline=final,numeric_features=num,categorical_features=cat,validation_metrics=meta,residuals=residual.tolist(),volume=vol,bands=bands,stability=stability,groups=groups)
    save_model(m,root="models/challenger_v06_workload")
    X=x.tail(20)
    p=m.probability_over(X,50.5)
    assert np.isfinite(p).all() and ((p>=0)&(p<=1)).all()
    assert (p>=m.probability_over(X,60.5)).all()
    report["props"][prop]={**meta,"volume":vol,"bands":bands,"stability":stability,"group_counts":{str(k):len(v) for k,v in groups.items()},"minimum_group":20,"shrink":100}
Path("reports/2026_w12_workload_shadow_refit.json").write_text(json.dumps(report,indent=2))
print(json.dumps(report,indent=2))
