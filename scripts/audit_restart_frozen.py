"""Read-only frozen-model restart audit. Proxy thresholds are NOT historical sportsbook lines."""
from pathlib import Path
import json
import warnings
import numpy as np
import pandas as pd
import yaml
from scipy.stats import poisson
from wshlx_nfl_props.data import load_raw
from wshlx_nfl_props.features import build_feature_frame
from wshlx_nfl_props.models import load_model
warnings.filterwarnings("ignore", category=pd.errors.PerformanceWarning)
cfg=yaml.safe_load(Path("config/model_config.yaml").read_text())
d=load_raw()
audit=json.loads(Path("reports/2026_w12_restart_data_audit.json").read_text())
assert audit["core_pass"], "Core data gate did not pass"
s=d["schedules"].copy()
# Exclude Week 12+ outcomes from the retrospective evaluation and any raw stat sources.
# Schedule and pregame fields are retained; no future outcome is a model feature.
for name,x in list(d.items()):
    if name not in {"schedules","weekly_rosters","injuries"} and {"season","week"}.issubset(x.columns):
        d[name]=x[(x.season<2026)|((x.season==2026)&(x.week<=11))].copy()
frame=build_feature_frame(d["player_stats"],s,d.get("pbp"),tuple(cfg["rolling_windows"]),auxiliary=d)
test=frame[(frame.season==2026)&frame.week.between(4,11)].copy()
assert len(test)>0
report={"purpose":"Frozen artifact retrospective diagnostic, not a betting backtest or live refit",
"artifact_ids":{"v03":10692302469,"v05":10666874411},"selection_weeks":[4,5,6,7],"test_weeks":[8,9,10,11],
"limitations":["Proxy lines use lagged player averages, not historical FanDuel prices.","Player-stat rows condition on recorded participation; no test of pregame participation prediction.","Current historical feed may include corrections unavailable at original kickoff.","No promotion on this diagnostic alone."],
"frame_rows":int(len(frame)),"evaluation_frame_rows":int(len(test)),"coverage":{},"props":{}}
for pref in ["snap_","ngs_pass_","ngs_rush_","ngs_rec_","pfr_pass_","pfr_rush_","pfr_rec_","pfr_def_","teamctx_","oppctx_","pregame_"]:
    cols=[c for c in test if c.startswith(pref) and (c.endswith("_lag1") or c.endswith("_r3") or c.endswith("_r5") or pref=="pregame_")]
    if cols:
        report["coverage"][pref]={"columns":len(cols),"any_nonnull_pct":float(test[cols].notna().any(axis=1).mean()*100),"by_position_pct":{str(k):float(v*100) for k,v in test[cols].notna().any(axis=1).groupby(test.position).mean().items()}}
report["defense_tackle_source_columns"]=[str(c) for c in d["player_stats"].columns if "tackle" in c.lower()]
def probs(m,point,lines):
    if m.kind=="binary": return point
    if m.kind=="count": return 1-poisson.cdf(np.floor(lines),np.clip(point,.01,None))
    residual=np.sort(np.asarray(m.residuals,dtype=float))
    assert len(residual)>=20 and np.isfinite(residual).all(),"Insufficient valid residuals"
    return 1-np.searchsorted(residual,lines-point,side="right")/len(residual)
def metric(y,p,point,actual):
    p=np.clip(p,1e-6,1-1e-6)
    return {"n":int(len(y)),"brier":float(np.mean((p-y)**2)),"log_loss":float(-np.mean(y*np.log(p)+(1-y)*np.log1p(-p))),"accuracy":float(np.mean((p>=.5)==y)),"mean_probability":float(np.mean(p)),"event_rate":float(np.mean(y)),"point_mae":float(np.mean(np.abs(point-actual)))}
for prop,spec in cfg["models"].items():
    model=load_model(prop,root="models/official")
    latest=model.validation_metrics.get("latest_season")
    week=model.validation_metrics.get("latest_week")
    assert latest is not None and (latest<2026 or (latest==2026 and week is not None and week<4)),f"Artifact training cutoff cannot be verified: {prop} {latest} {week}"
    f=test[test.position.isin(spec["positions"]) & (test.history_games>=cfg["min_history_games"])].copy()
    if prop.startswith("qb_"): f=f[f.passing_attempts_r3>=10]
    if prop=="rb_rush_yds": f=f[f.rushing_attempts_r3>=2]
    if prop=="rec_yds": f=f[f.targets_r3>=1.5]
    target=spec["target"]
    if model.kind=="binary": f["actual"]=(f[target]>0).astype(float); f["line"]=.5
    elif model.kind=="count": f["actual"]=f[target]; f["line"]=1.5
    else:
        f["actual"]=f[target]; f["line"]=np.maximum(.5,np.floor(f[target+"_r3"])+.5)
    f=f.dropna(subset=["actual","line"]).copy()
    y=(f.actual.to_numpy()>f.line.to_numpy()).astype(int)
    actual=f.actual.to_numpy()
    lines=f.line.to_numpy()
    sel=f.week.to_numpy()<=7; hold=f.week.to_numpy()>=8
    assert sel.any() and hold.any(),prop
    result={"training_latest_season":latest,"training_latest_week":week,"algorithm":model.estimator_name,"feature_set":model.feature_set,"rows":int(len(f))}
    def predict(m):
        cols=m.numeric_features+m.categorical_features
        missing=sorted(set(cols)-set(f.columns))
        assert not missing,f"{prop}: missing model features {missing}"
        x=f[cols].copy()
        for c in m.categorical_features:
            q=x[c].astype(object); x[c]=q.where(pd.notna(q),"__MISSING__").astype(str)
        coverage={c:float(x[c].notna().mean()*100) for c in m.numeric_features}
        point=np.asarray(m.predict_point(x),dtype=float)
        assert np.isfinite(point).all(),prop
        p=probs(m,point,lines)
        return point,p,coverage
    point,p,coverage=predict(model)
    result["numeric_feature_coverage"]={"columns":len(coverage),"all_missing":[c for c,v in coverage.items() if v==0],"mean_nonnull_pct":float(np.mean(list(coverage.values())))}
    candidates={str(a): .5+a*(p-.5) for a in (1.0,.75,.5)}
    selected=min(candidates,key=lambda a:metric(y[sel],candidates[a][sel],point[sel],actual[sel])["brier"])
    result["selected_shrink_factor"]=float(selected)
    result["calibration_selection"]={a:metric(y[sel],v[sel],point[sel],actual[sel]) for a,v in candidates.items()}
    result["holdout"]={"v03":metric(y[hold],p[hold],point[hold],actual[hold]),"selected_shrink":metric(y[hold],candidates[selected][hold],point[hold],actual[hold])}
    if prop in {"qb_pass_yds","rb_rush_yds","rec_yds"}:
        m5=load_model(prop,root="models/challenger_v05")
        v=m5.validation_metrics
        assert v.get("latest_season") is not None and (v["latest_season"]<2026 or (v["latest_season"]==2026 and v.get("latest_week",99)<4)),f"v05 cutoff {prop}"
        point5,p5,_=predict(m5)
        result["v05_training_latest_week"]=v.get("latest_week")
        result["holdout"]["v05"]=metric(y[hold],p5[hold],point5[hold],actual[hold])
    report["props"][prop]=result
Path("reports/2026_w12_frozen_diagnostic.json").write_text(json.dumps(report,indent=2,default=str))
print(json.dumps(report,indent=2,default=str))
