from pathlib import Path
import json, yaml, joblib
import numpy as np
import pandas as pd

from wshlx_nfl_props.data import load_raw
from wshlx_nfl_props.features import build_feature_frame, model_feature_columns
from wshlx_nfl_props.models import _make_pipeline, _predict, _metrics
from wshlx_nfl_props.decomposed import DecomposedYardageModel

cfg=yaml.safe_load(Path("config/model_config.yaml").read_text())
official=yaml.safe_load(Path("config/champion_specs.yaml").read_text())
d=load_raw()
if "player_stats" not in d or d["player_stats"].empty:
    raise SystemExit("Run scripts/update_data.py first")

frame=build_feature_frame(
    d["player_stats"], d.get("schedules"), d.get("pbp"),
    tuple(cfg["rolling_windows"]), auxiliary=d
)

specs={
  "qb_pass_yds":{"yardage":"passing_yards","volume":"passing_attempts","positions":["QB"],"feature_set":"pregame","eff_clip":(0.0,15.0)},
  "rb_rush_yds":{"yardage":"rushing_yards","volume":"rushing_attempts","positions":["RB","FB"],"feature_set":"baseline","eff_clip":(-2.0,12.0)},
  "rec_yds":{"yardage":"receiving_yards","volume":"targets","positions":["WR","RB","FB"],"feature_set":"baseline","eff_clip":(0.0,20.0)},
}

algs=("elastic_net","random_forest","hist_gbr")
years=[2023,2024,2025]
root=Path("models/challenger_v05"); root.mkdir(parents=True,exist_ok=True)
report={"version":"0.5.0-shadow","status":"shadow_only","concept":"opportunity_x_efficiency","walk_forward_years":years,"props":{}}

for prop,s in specs.items():
    f=frame[frame["position"].isin(s["positions"])].copy()
    f["__yards"]=pd.to_numeric(f[s["yardage"]],errors="coerce")
    f["__volume"]=pd.to_numeric(f[s["volume"]],errors="coerce")
    denom=f["__volume"].replace(0,np.nan)
    f["__eff"]=(f["__yards"]/denom).replace([np.inf,-np.inf],np.nan)
    f["__eff"]=f["__eff"].clip(s["eff_clip"][0],s["eff_clip"][1])
    f=f[(f["history_games"]>=cfg["min_history_games"])&f["__yards"].notna()&f["__volume"].notna()].copy()
    for col in ["position_group_model","team","opponent_team","pregame_report_status","pregame_practice_status","pregame_roof","pregame_surface"]:
        if col in f.columns:
            z=f[col].astype(object); f[col]=z.where(pd.notna(z),"__MISSING__").astype(str)

    numeric,categorical=model_feature_columns(f,feature_set=s["feature_set"])
    cols=numeric+categorical

    # Official benchmark
    ospec=official["champions"][prop]
    onum,ocat=model_feature_columns(f,feature_set=ospec["feature_set"])
    ocols=onum+ocat
    official_folds=[]
    for year in years:
        tr=f[f.season<year]; te=f[f.season==year]
        p=_make_pipeline("regression",ospec["algorithm"],onum,ocat,cfg["random_state"])
        p.fit(tr[ocols],tr["__yards"])
        pred=p.predict(te[ocols])
        met=_metrics("regression",te["__yards"],pred)
        official_folds.append({"year":year,"rows":int(len(te)),**met})
    official_weighted=sum(z["mae"]*z["rows"] for z in official_folds)/sum(z["rows"] for z in official_folds)

    candidates={}
    best=None
    best_residuals=[]
    for va in algs:
      for ea in algs:
        folds=[]; residuals=[]
        valid_combo=True
        for year in years:
            tr=f[f.season<year]; te=f[f.season==year]
            # efficiency training only on actual opportunities
            tr_eff=tr[tr["__volume"]>0].copy()
            if len(tr_eff)<50:
                valid_combo=False; break
            vp=_make_pipeline("regression",va,numeric,categorical,cfg["random_state"])
            ep=_make_pipeline("regression",ea,numeric,categorical,cfg["random_state"])
            vp.fit(tr[cols],tr["__volume"])
            ep.fit(tr_eff[cols],tr_eff["__eff"])
            v=np.clip(vp.predict(te[cols]),0,None)
            e=np.clip(ep.predict(te[cols]),s["eff_clip"][0],s["eff_clip"][1])
            pred=v*e
            met=_metrics("regression",te["__yards"],pred)
            folds.append({"year":year,"rows":int(len(te)),**met})
            residuals.extend((te["__yards"].to_numpy()-pred).tolist())
        if not valid_combo:
            continue
        weighted=sum(z["mae"]*z["rows"] for z in folds)/sum(z["rows"] for z in folds)
        key=f"{va}+{ea}"
        candidates[key]={"volume_algorithm":va,"efficiency_algorithm":ea,"folds":folds,"weighted_mae":weighted}
        if best is None or weighted<best[0]:
            best=(weighted,va,ea); best_residuals=residuals

    if best is None:
        raise RuntimeError(f"No decomposed candidate trained for {prop}")
    weighted,va,ea=best
    improvement=100*(official_weighted-weighted)/official_weighted
    decision="SHADOW_TEST_NEXT_WEEK" if improvement<=1.0 else "HISTORICAL_PROMOTION_CANDIDATE"
    report["props"][prop]={
      "official":{"spec":f'{ospec["feature_set"]}:{ospec["algorithm"]}',"folds":official_folds,"weighted_mae":official_weighted},
      "challenger":{"spec":f'{s["feature_set"]}:decomposed[{va}+{ea}]',"weighted_mae":weighted,"folds":candidates[f"{va}+{ea}"]["folds"]},
      "all_decomposed_candidates":candidates,
      "historical_relative_improvement_pct":improvement,
      "decision":decision,
      "note":"Historical walk-forward is diagnostic. v0.3 remains official until prospective locked-card evidence supports promotion."
    }

    # Live refit through all completed rows
    vp=_make_pipeline("regression",va,numeric,categorical,cfg["random_state"])
    ep=_make_pipeline("regression",ea,numeric,categorical,cfg["random_state"])
    vp.fit(f[cols],f["__volume"])
    fe=f[f["__volume"]>0].copy()
    ep.fit(fe[cols],fe["__eff"])
    model=DecomposedYardageModel(
      prop=prop,kind="regression",estimator_name=f"decomposed_{va}_{ea}",
      feature_set=s["feature_set"],volume_pipeline=vp,efficiency_pipeline=ep,
      numeric_features=numeric,categorical_features=categorical,
      volume_target=s["volume"],yardage_target=s["yardage"],efficiency_clip=s["eff_clip"],
      residuals=best_residuals,
      validation_metrics={
        "shadow_version":"0.5.0-shadow",
        "historical_diagnostic":report["props"][prop],
        "live_refit_rows":int(len(f)),
        "latest_season":int(f.season.max()),
        "latest_week":int(f.loc[f.season==f.season.max(),"week"].max())
      }
    )
    joblib.dump(model,root/f"{prop}.joblib")
    (root/f"{prop}.json").write_text(json.dumps({
      "prop":prop,"feature_set":s["feature_set"],"estimator":model.estimator_name,
      "validation_metrics":model.validation_metrics,
      "numeric_features":numeric,"categorical_features":categorical,
      "volume_target":s["volume"],"yardage_target":s["yardage"]
    },indent=2))

Path("reports").mkdir(exist_ok=True)
Path("reports/v0.5.0_shadow_decomposed.json").write_text(json.dumps(report,indent=2))
print(json.dumps(report,indent=2))
