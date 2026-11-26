"""Historical game-script and prior-QB continuity diagnostic, no live model changes."""
from pathlib import Path
import json
import warnings
import numpy as np
import pandas as pd
import yaml
from wshlx_nfl_props.data import load_raw
from wshlx_nfl_props.features import build_feature_frame, model_feature_columns, normalize_player_stats
from wshlx_nfl_props.enrichment import _clean_team
from wshlx_nfl_props.models import _make_pipeline
warnings.filterwarnings("ignore",category=pd.errors.PerformanceWarning)
cfg=yaml.safe_load(Path("config/model_config.yaml").read_text())
d=load_raw()
for k,x in list(d.items()):
    if "season" in x: d[k]=x[x.season<=2025].copy()
frame=build_feature_frame(d["player_stats"],d["schedules"],d.get("pbp"),tuple(cfg["rolling_windows"]),auxiliary=d)
# Prior primary passer is determined by recorded attempts in prior completed
# games, never by the current game's QB statistics.
q=normalize_player_stats(d["player_stats"])
q=q[(q.position=="QB") & (q.passing_attempts>0)].copy()
q["team"]=q.team.map(_clean_team)
q=q.sort_values(["season","week","team","passing_attempts","player_id"],ascending=[True,True,True,False,True]).drop_duplicates(["season","week","team"])
q=q.sort_values(["team","season","week"])
g=q.groupby(["team","season"])
q["prev_qb"]=g.player_id.shift(1)
q["prev2_qb"]=g.player_id.shift(2)
q["prior_qb_changed"]=np.where(q.prev_qb.notna()&q.prev2_qb.notna(),(q.prev_qb!=q.prev2_qb).astype(float),np.nan)
q=q[["team","season","week","prior_qb_changed"]]
frame=frame.merge(q,on=["team","season","week"],how="left",validate="many_to_one")
report={"status":"research_only","method":"Same Elastic Net estimator, historical expanding train/calibration/test. Game-script candidates use schedule pregame lines and lagged team/opponent PBP. Prior-QB continuity is known from preceding team games only.",
"limitations":["Historical closing schedule lines may differ from actual prediction-time FanDuel lines.","Prior-QB change is NOT a verified current-game starter change; no use of current-game QB identity in feature values.","Conditional on recorded player participation; proxy thresholds, no betting ROI.","Historical evaluation years previously used in model development. No official model changes.","Live matchup feature availability needs separate verification."],"props":{}}
def metric(y,p,point,actual):
    z=np.clip(p,1e-6,1-1e-6)
    return {"n":len(y),"mae":float(np.mean(np.abs(point-actual))),"brier":float(np.mean((p-y)**2)),"log_loss":float(-np.mean(y*np.log(z)+(1-y)*np.log1p(-z))),"accuracy":float(np.mean((p>=.5)==y))}
for prop,volume in [("rb_rush_yds","rushing_attempts"),("rec_yds","targets")]:
    s=cfg["models"][prop]; target=s["target"]
    f=frame[frame.position.isin(s["positions"])&(frame.history_games>=cfg["min_history_games"])].copy()
    f=f[(f[volume+"_r3"]>=(2 if prop=="rb_rush_yds" else 1.5))&f[target+"_r3"].notna()].copy()
    f["line"]=np.maximum(.5,np.floor(f[target+"_r3"])+.5)
    num,cat=model_feature_columns(f,feature_set="baseline")
    num=[c for c in num if c!="prior_qb_changed"]
    env=[c for c in ["pregame_team_spread","pregame_total_line","pregame_home","pregame_rest"] if c in f]
    ctx=[c for c in f if c.startswith(("teamctx_","oppctx_")) and c.endswith(("_r3","_r5"))]
    assert env and ctx
    for c in cat:
        v=f[c].astype(object); f[c]=v.where(v.notna(),"__MISSING__").astype(str)
    sets={"baseline":num,"game_script":num+env+ctx,"game_script_prior_qb":num+env+ctx+["prior_qb_changed"]}
    folds=[]
    for train_year in [2022,2023]:
        tr=f[f.season<=train_year]; ca=f[f.season==train_year+1]; te=f[f.season==train_year+2]
        actual=te[target].to_numpy(); lines=te.line.to_numpy(); y=(actual>lines).astype(int)
        result={"year":train_year+2,"scores":{},"prior_qb_change_diagnostic":{}}
        for name,cols in sets.items():
            cols=list(dict.fromkeys(cols))
            pipe=_make_pipeline("regression","elastic_net",cols,cat,cfg["random_state"])
            pipe.fit(tr[cols+cat],tr[target])
            residual=np.sort(ca[target].to_numpy()-pipe.predict(ca[cols+cat]))
            point=pipe.predict(te[cols+cat])
            p=1-np.searchsorted(residual,lines-point,side="right")/len(residual)
            result["scores"][name]=metric(y,p,point,actual)
            if name=="baseline":
                for changed in [0,1]:
                    mask=(te.prior_qb_changed.to_numpy()==changed)&(p>=.57)
                    result["prior_qb_change_diagnostic"][str(changed)]={"n":int(mask.sum()),"over_hit_rate":float(y[mask].mean()) if mask.any() else None,"mean_probability":float(p[mask].mean()) if mask.any() else None}
        folds.append(result)
    report["props"][prop]={"folds":folds,"game_script_features":env+ctx,"prior_qb_feature_coverage":float(f.prior_qb_changed.notna().mean())}
Path("reports/2026_w12_gamescript_diagnostic.json").write_text(json.dumps(report,indent=2))
print(json.dumps(report,indent=2))
