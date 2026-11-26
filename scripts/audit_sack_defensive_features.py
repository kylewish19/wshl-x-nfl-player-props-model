"""Position-specific sack probability diagnostic. No live model writes."""
from pathlib import Path
import json
import warnings
import numpy as np
import pandas as pd
import yaml
from wshlx_nfl_props.data import load_raw
from wshlx_nfl_props.features import build_feature_frame, model_feature_columns
from wshlx_nfl_props.models import _make_pipeline
from sklearn.metrics import brier_score_loss, log_loss, roc_auc_score
warnings.filterwarnings("ignore",category=pd.errors.PerformanceWarning)
cfg=yaml.safe_load(Path("config/model_config.yaml").read_text())
d=load_raw()
for k,x in list(d.items()):
    if "season" in x: d[k]=x[x.season<=2025].copy()
f=build_feature_frame(d["player_stats"],d["schedules"],d.get("pbp"),tuple(cfg["rolling_windows"]),auxiliary=d)
f=f[f.position.isin(cfg["models"]["sack_yes"]["positions"]) & (f.history_games>=cfg["min_history_games"])].copy()
f["y"]=(f.def_sacks>0).astype(int)
num,cat=model_feature_columns(f,feature_set="enriched")
# Existing features only: no changes to aliases, targets or champion.
defensive=[c for c in num if c in {"season","week","history_games"} or c.startswith(("def_sacks_","def_qb_hits_","def_tackles_","snap_defense_","pfr_def_","opp_dropbacks_faced_","defense_sacks_","defense_qb_hits_","opp_allowed_def_sacks_"))]
context=[c for c in f if c.startswith(("oppctx_team_dropbacks_","oppctx_sacks_allowed_","oppctx_qb_hits_allowed_","teamctx_opp_dropbacks_faced_","teamctx_defense_sacks_","teamctx_defense_qb_hits_")) and c.endswith(("_r3","_r5"))]
assert defensive and context
for c in cat:
    q=f[c].astype(object); f[c]=q.where(pd.notna(q),"__MISSING__").astype(str)
report={"status":"research_only","method":"Fixed historical expanding window comparison; evaluation seasons previously used for model development; not an untouched holdout.","limitations":["Recorded player participation only; not a sack-YES betting backtest.","No historical FanDuel prices.","No tackle alias correction yet.","Base rates and all fitting use only earlier seasons."],"features":{"champion":num,"defensive":defensive,"matchup":context},"folds":[]}
def metrics(y,p):
    p=np.clip(np.asarray(p),1e-6,1-1e-6)
    yes=p>=.57
    return {"n":len(y),"brier":float(brier_score_loss(y,p)),"log_loss":float(log_loss(y,p,labels=[0,1])),"auc":float(roc_auc_score(y,p)),"base_rate":float(np.mean(y)),"mean_probability":float(p.mean()),"yes_count":int(yes.sum()),"yes_hit_rate":float(np.mean(y[yes])) if yes.any() else None,"yes_mean_probability":float(p[yes].mean()) if yes.any() else None}
for year in [2024,2025]:
    tr=f[f.season<year]; te=f[f.season==year]
    y=te.y.to_numpy()
    base=tr.groupby("position_group_model").y.mean()
    bp=te.position_group_model.map(base).fillna(tr.y.mean()).to_numpy()
    out={"evaluation_year":year,"train_n":len(tr),"scores":{"position_base_rate":metrics(y,bp)}}
    for name,features,alg in [("champion_spec",num,"hist_gbc"),("defense_only",defensive,"hist_gbc"),("defense_matchup",defensive+context,"hist_gbc")]:
        features=list(dict.fromkeys(features))
        pipe=_make_pipeline("binary",alg,features,cat,cfg["random_state"])
        pipe.fit(tr[features+cat],tr.y)
        p=pipe.predict_proba(te[features+cat])[:,1]
        out["scores"][name]=metrics(y,p)
    report["folds"].append(out)
Path("reports/2026_w12_sack_diagnostic.json").write_text(json.dumps(report,indent=2))
print(json.dumps(report,indent=2))
