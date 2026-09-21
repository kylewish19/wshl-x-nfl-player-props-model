from pathlib import Path
import json,yaml,joblib
import numpy as np
import pandas as pd
from sklearn.metrics import mean_absolute_error,brier_score_loss,log_loss,roc_auc_score

from wshlx_nfl_props.data import load_raw
from wshlx_nfl_props.game_features import build_game_frame,game_feature_columns
from wshlx_nfl_props.game_market_informed import add_market_priors,market_informed_feature_columns
from wshlx_nfl_props.game_models import _make_pipeline

cfg=yaml.safe_load(Path("config/game_model_config.yaml").read_text())
d=load_raw()
frame=build_game_frame(d["schedules"],d.get("team_stats"),d.get("pbp"),windows=tuple(cfg["rolling_windows"]))
frame=frame[frame["season"].isin(cfg["seasons"])].copy()

s=d["schedules"].copy()
if "game_type" in s.columns:
    s=s[s.game_type.astype(str).str.upper().isin(["REG","REGULAR"])]
keep=["season","week","home_team","away_team"]+[c for c in ["spread_line","total_line","home_moneyline","away_moneyline"] if c in s.columns]
frame=frame.merge(s[keep].drop_duplicates(["season","week","home_team","away_team"]),
                  on=["season","week","home_team","away_team"],how="left")
frame=add_market_priors(frame)

base_num,base_cat=game_feature_columns(frame)
numeric,categorical=market_informed_feature_columns(frame,base_num,base_cat)
for c in categorical:
    z=frame[c].astype(object)
    frame[c]=z.where(pd.notna(z),"__MISSING__").astype(str)
cols=numeric+categorical

alphas=[0.0,0.25,0.5,0.75,1.0]
sel_year=cfg["selection_season"]
test_year=cfg["test_season"]
seed=cfg["random_state"]

report={
  "version":"0.3.0-shadow",
  "status":"shrunken_market_blend_shadow",
  "selection_year":sel_year,
  "test_year":test_year,
  "alphas":alphas,
  "markets":{}
}
root=Path("models/game_markets_blend_v03");root.mkdir(parents=True,exist_ok=True)

# Margin and total: market baseline + alpha * predicted residual
for market,target,resid_target,base_col,algorithm in [
    ("spread","target_margin","target_margin_residual","market_margin","random_forest"),
    ("total","target_total","target_total_residual","market_total","random_forest"),
]:
    f=frame[frame[target].notna() & frame[base_col].notna() & frame[resid_target].notna()].copy()
    dev=f[f.season<sel_year]; sel=f[f.season==sel_year]; test=f[f.season==test_year]

    sel_pipe=_make_pipeline("regression",algorithm,numeric,categorical,seed)
    sel_pipe.fit(dev[cols],dev[resid_target])
    sel_corr=sel_pipe.predict(sel[cols])
    sel_scores={}
    for a in alphas:
        pred=sel[base_col].to_numpy()+a*np.asarray(sel_corr)
        sel_scores[str(a)]=float(mean_absolute_error(sel[target],pred))
    alpha=min(alphas,key=lambda a:sel_scores[str(a)])

    pretest=f[f.season<=sel_year]
    test_pipe=_make_pipeline("regression",algorithm,numeric,categorical,seed)
    test_pipe.fit(pretest[cols],pretest[resid_target])
    test_corr=test_pipe.predict(test[cols])
    base_pred=test[base_col].to_numpy()
    blend_pred=base_pred+alpha*np.asarray(test_corr)
    test_base_mae=float(mean_absolute_error(test[target],base_pred))
    test_blend_mae=float(mean_absolute_error(test[target],blend_pred))

    final=_make_pipeline("regression",algorithm,numeric,categorical,seed)
    final.fit(f[cols],f[resid_target])
    joblib.dump({
      "market":market,"kind":"regression_blend","alpha":alpha,"algorithm":algorithm,
      "pipeline":final,"numeric_features":numeric,"categorical_features":categorical,
      "base_column":base_col,"residual_target":resid_target,
      "test_residuals":(test[target].to_numpy()-blend_pred).tolist(),
    },root/f"{market}.joblib")

    report["markets"][market]={
      "algorithm":algorithm,
      "selection_mae_by_alpha":sel_scores,
      "selected_alpha":alpha,
      "2025_market_baseline_mae":test_base_mae,
      "2025_blend_mae":test_blend_mae,
      "relative_improvement_pct":100*(test_base_mae-test_blend_mae)/test_base_mae
    }

# Moneyline: fair market probability blended with model probability
f=frame[frame["target_home_win"].notna() & frame["market_home_win_fair"].notna()].copy()
if "target_tie" in f.columns:
    f=f[f["target_tie"]==0].copy()
dev=f[f.season<sel_year]; sel=f[f.season==sel_year]; test=f[f.season==test_year]
algorithm="hist_gbc"

sel_pipe=_make_pipeline("binary",algorithm,numeric,categorical,seed)
sel_pipe.fit(dev[cols],dev["target_home_win"])
sel_model=sel_pipe.predict_proba(sel[cols])[:,1]
sel_market=sel["market_home_win_fair"].to_numpy()
sel_scores={}
for a in alphas:
    p=np.clip((1-a)*sel_market+a*sel_model,1e-6,1-1e-6)
    sel_scores[str(a)]={"brier":float(brier_score_loss(sel["target_home_win"],p)),
                        "log_loss":float(log_loss(sel["target_home_win"],p))}
alpha=min(alphas,key=lambda a:sel_scores[str(a)]["brier"])

pretest=f[f.season<=sel_year]
test_pipe=_make_pipeline("binary",algorithm,numeric,categorical,seed)
test_pipe.fit(pretest[cols],pretest["target_home_win"])
test_model=test_pipe.predict_proba(test[cols])[:,1]
test_market=test["market_home_win_fair"].to_numpy()
blend=np.clip((1-alpha)*test_market+alpha*test_model,1e-6,1-1e-6)
base=np.clip(test_market,1e-6,1-1e-6)

base_brier=float(brier_score_loss(test["target_home_win"],base))
blend_brier=float(brier_score_loss(test["target_home_win"],blend))
final=_make_pipeline("binary",algorithm,numeric,categorical,seed)
final.fit(f[cols],f["target_home_win"])
joblib.dump({
  "market":"moneyline","kind":"binary_blend","alpha":alpha,"algorithm":algorithm,
  "pipeline":final,"numeric_features":numeric,"categorical_features":categorical,
  "base_column":"market_home_win_fair"
},root/"moneyline.joblib")

report["markets"]["moneyline"]={
  "algorithm":algorithm,
  "selection_by_alpha":sel_scores,
  "selected_alpha":alpha,
  "2025_market_baseline_brier":base_brier,
  "2025_blend_brier":blend_brier,
  "2025_market_baseline_log_loss":float(log_loss(test["target_home_win"],base)),
  "2025_blend_log_loss":float(log_loss(test["target_home_win"],blend)),
  "2025_blend_auc":float(roc_auc_score(test["target_home_win"],blend)),
  "relative_brier_improvement_pct":100*(base_brier-blend_brier)/base_brier
}

Path("reports").mkdir(exist_ok=True)
Path("reports/game_markets_blend_v0.3.0.json").write_text(json.dumps(report,indent=2))
print(json.dumps(report,indent=2))
