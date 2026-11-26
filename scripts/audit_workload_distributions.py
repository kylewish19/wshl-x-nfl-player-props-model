"""Chronological yardage probability research; never writes live model artifacts."""
from pathlib import Path
import json
import warnings
import numpy as np
import pandas as pd
import yaml
from wshlx_nfl_props.data import load_raw
from wshlx_nfl_props.features import build_feature_frame, model_feature_columns
from wshlx_nfl_props.models import _make_pipeline
warnings.filterwarnings("ignore", category=pd.errors.PerformanceWarning)

cfg=yaml.safe_load(Path("config/model_config.yaml").read_text())
specs=yaml.safe_load(Path("config/champion_specs.yaml").read_text())["champions"]
d=load_raw()
# Historical folds only. The frozen specifications were previously selected
# using historical seasons, so these are diagnostic, not fresh blind tests.
for k,x in list(d.items()):
    if "season" in x: d[k]=x[x.season<=2025].copy()
frame=build_feature_frame(d["player_stats"],d["schedules"],d.get("pbp"),tuple(cfg["rolling_windows"]),auxiliary=d)
report={"status":"research_only","folds":[{"train_through":2022,"calibration":2023,"evaluation":2024},{"train_through":2023,"calibration":2024,"evaluation":2025}],
"method":"Identical frozen point-model specification within each fold; fit through train year, use next year residuals, score following year. Conditional CDF blended toward pooled CDF with 100 pseudo-observations.",
"limitations":["Existing champion specs were selected using historical seasons; these are not untouched holdouts for the project.","Proxy half-yard lines based on prior 3-game averages, not FanDuel odds.","Player-game participation conditioned; no inactive/void inference.","Historical reports and closing game lines may differ from original prediction-time snapshots.","No model promotion or live fit."],"props":{}}
# Predetermined, football-interpretable workload bands. Stability is lag1 vs prior three-game average.
usage={"qb_pass_yds":("passing_attempts",[20,30,40]),"qb_rush_yds":("rushing_attempts",[2,5,8]),"rb_rush_yds":("rushing_attempts",[5,12,18]),"rec_yds":("targets",[3,6,9])}

def tail(residuals,threshold):
    r=np.sort(np.asarray(residuals,dtype=float))
    return 1-np.searchsorted(r,threshold,side="right")/len(r)

def conditional(cal,te,threshold,group_cols,pooled):
    result=pooled.copy()
    weights=np.zeros(len(te))
    for key,sub in cal.groupby(group_cols,observed=True):
        key=key if isinstance(key,tuple) else (key,)
        mask=np.ones(len(te),dtype=bool)
        for c,v in zip(group_cols,key): mask &= te[c].to_numpy()==v
        n=len(sub)
        if n<20 or not mask.any(): continue
        w=n/(n+100)
        result[mask]=w*tail(sub.residual,threshold[mask])+(1-w)*pooled[mask]
        weights[mask]=w
    return result,weights

def metric(y,p):
    q=np.clip(p,1e-6,1-1e-6)
    side=p>=.5
    over80=p>=.8
    confident=np.maximum(p,1-p)>=.7
    return {"n":int(len(y)),"brier":float(np.mean((p-y)**2)),"log_loss":float(-np.mean(y*np.log(q)+(1-y)*np.log1p(-q))),"accuracy":float(np.mean(side==y)),"high_confidence_n":int(confident.sum()),"high_confidence_accuracy":float(np.mean(side[confident]==y[confident])) if confident.any() else None,"high_confidence_mean_p":float(np.mean(np.maximum(p,1-p)[confident])) if confident.any() else None,"over80_n":int(over80.sum()),"over80_hit_rate":float(y[over80].mean()) if over80.any() else None}

for prop,(vol,bands) in usage.items():
    s=cfg["models"][prop]
    spec=specs[prop]
    target=s["target"]
    f=frame[frame.position.isin(s["positions"]) & (frame.history_games>=cfg["min_history_games"])].copy()
    f=f.dropna(subset=[target,target+"_r3",vol+"_r3",vol+"_lag1"])
    if prop.startswith("qb_"): f=f[f.passing_attempts_r3>=10].copy()
    elif prop=="rb_rush_yds": f=f[f.rushing_attempts_r3>=2].copy()
    else: f=f[f.targets_r3>=1.5].copy()
    f["proxy_line"]=np.maximum(.5,np.floor(f[target+"_r3"])+.5)
    f["workload_band"]=np.digitize(f[vol+"_r3"],bands)
    f["unstable"]=((f[vol+"_lag1"]-f[vol+"_r3"]).abs()/np.maximum(f[vol+"_r3"],1)>=.5).astype(int)
    numeric,cat=model_feature_columns(f,feature_set=spec["feature_set"])
    cols=numeric+cat
    for c in cat:
        q=f[c].astype(object); f[c]=q.where(pd.notna(q),"__MISSING__").astype(str)
    results=[]
    aggregate={k:[] for k in ["pooled","workload","workload_stability"]}
    ys=[]
    for train_year in [2022,2023]:
        tr=f[f.season<=train_year]; ca=f[f.season==train_year+1].copy(); te=f[f.season==train_year+2].copy()
        assert min(len(tr),len(ca),len(te))>0,prop
        pipe=_make_pipeline("regression",spec["algorithm"],numeric,cat,cfg["random_state"])
        pipe.fit(tr[cols],tr[target])
        ca["residual"]=ca[target].to_numpy()-pipe.predict(ca[cols])
        point=pipe.predict(te[cols])
        actual=te[target].to_numpy(); threshold=te.proxy_line.to_numpy()-point
        y=(actual>te.proxy_line.to_numpy()).astype(int)
        p0=tail(ca.residual,threshold)
        p1,w1=conditional(ca,te,threshold,["workload_band"],p0)
        p2,w2=conditional(ca,te,threshold,["workload_band","unstable"],p0)
        ps={"pooled":p0,"workload":p1,"workload_stability":p2}
        out={"train_through":train_year,"calibration_year":train_year+1,"evaluation_year":train_year+2,"train_n":len(tr),"calibration_n":len(ca),"point_mae":float(np.mean(np.abs(point-actual))),"mean_local_weight":{"workload":float(w1.mean()),"workload_stability":float(w2.mean())},"scores":{k:metric(y,p) for k,p in ps.items()},"opportunity_diagnostic":{}}
        # Compare fixed pregame instability groups; no tuning and no claimed betting ROI.
        for unstable in (0,1):
            mask=(te.unstable.to_numpy()==unstable)&(p0>=.57)
            out["opportunity_diagnostic"][str(unstable)]={"over_signals":int(mask.sum()),"over_hit_rate":float(y[mask].mean()) if mask.any() else None,"over_mean_p":float(p0[mask].mean()) if mask.any() else None,"actual_volume_below_half_prior_average":float(np.mean(te[vol].to_numpy()[mask]<.5*te[vol+"_r3"].to_numpy()[mask])) if mask.any() else None}
        for k,p in ps.items(): aggregate[k].append(p)
        ys.append(y); results.append(out)
    report["props"][prop]={"workload_feature":vol+"_r3","band_edges":bands,"point_spec":spec,"folds":results,"combined":{k:metric(np.concatenate(ys),np.concatenate(v)) for k,v in aggregate.items()}}
Path("reports/2026_w12_workload_diagnostic.json").write_text(json.dumps(report,indent=2))
print(json.dumps(report,indent=2))
