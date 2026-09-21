from pathlib import Path
import json, yaml
import numpy as np
import pandas as pd

from wshlx_nfl_props.data import load_raw
from wshlx_nfl_props.features import build_feature_frame, model_feature_columns
from wshlx_nfl_props.models import TrainedPropModel, _make_pipeline, _predict, _metrics, primary_score, save_model

cfg=yaml.safe_load(Path("config/model_config.yaml").read_text())
official=yaml.safe_load(Path("config/champion_specs.yaml").read_text())
chall=yaml.safe_load(Path("config/challenger_v04.yaml").read_text())
d=load_raw()
if "player_stats" not in d or d["player_stats"].empty:
    raise SystemExit("Run scripts/update_data.py first")

frame=build_feature_frame(
    d["player_stats"], d.get("schedules"), d.get("pbp"),
    tuple(cfg["rolling_windows"]), auxiliary=d
)
years=[2023,2024,2025]
root=Path("models/challenger_v04"); root.mkdir(parents=True,exist_ok=True)
report={"version":chall["version"],"status":"shadow_only","walk_forward_years":years,"props":{}}

for prop,pspec in cfg["models"].items():
    kind,target=pspec["kind"],pspec["target"]
    base_spec=official["champions"][prop]
    new_spec=chall["challengers"][prop]
    f=frame[frame["position"].isin(pspec["positions"])].copy()
    f["__target"]=(pd.to_numeric(f[target],errors="coerce").fillna(0)>0).astype(int) if kind=="binary" else pd.to_numeric(f[target],errors="coerce")
    f=f[(f["history_games"]>=cfg["min_history_games"])&f["__target"].notna()].copy()
    for col in ["position_group_model","team","opponent_team","pregame_report_status","pregame_practice_status","pregame_roof","pregame_surface"]:
        if col in f.columns:
            s=f[col].astype(object); f[col]=s.where(pd.notna(s),"__MISSING__").astype(str)

    comparisons={}
    challenger_residuals=[]
    for label,spec in [("official",base_spec),("challenger",new_spec)]:
        numeric,categorical=model_feature_columns(f,feature_set=spec["feature_set"])
        cols=numeric+categorical
        folds=[]
        residuals=[]
        for year in years:
            train=f[f.season<year]; test=f[f.season==year]
            pipe=_make_pipeline(kind,spec["algorithm"],numeric,categorical,cfg["random_state"])
            pipe.fit(train[cols],train["__target"])
            pred=_predict(pipe,kind,test[cols])
            met=_metrics(kind,test["__target"],pred)
            folds.append({"year":year,"rows":int(len(test)),**met})
            if kind=="regression":
                residuals.extend((test["__target"].to_numpy()-np.asarray(pred)).tolist())
        key="brier" if kind=="binary" else ("poisson_deviance" if kind=="count" else "mae")
        weighted=sum(z[key]*z["rows"] for z in folds)/sum(z["rows"] for z in folds)
        comparisons[label]={"spec":f'{spec["feature_set"]}:{spec["algorithm"]}',"folds":folds,"weighted_primary":weighted}
        if label=="challenger":
            challenger_residuals=residuals
            ch_num,ch_cat=numeric,categorical

    base_score=comparisons["official"]["weighted_primary"]
    new_score=comparisons["challenger"]["weighted_primary"]
    improvement=100*(base_score-new_score)/base_score
    report["props"][prop]={
        **comparisons,
        "historical_relative_improvement_pct":improvement,
        "decision":"SHADOW_TEST_NEXT_WEEK",
        "note":"Historical result is diagnostic only; do not replace v0.3 until prospective locked-card grading."
    }

    final=_make_pipeline(kind,new_spec["algorithm"],ch_num,ch_cat,cfg["random_state"])
    final.fit(f[ch_num+ch_cat],f["__target"])
    model=TrainedPropModel(
        prop=prop,kind=kind,estimator_name=new_spec["algorithm"],feature_set=new_spec["feature_set"],
        pipeline=final,numeric_features=ch_num,categorical_features=ch_cat,
        validation_metrics={
            "shadow_version":chall["version"],
            "historical_diagnostic":report["props"][prop],
            "live_refit_rows":int(len(f)),
            "latest_season":int(f.season.max()),
            "latest_week":int(f.loc[f.season==f.season.max(),"week"].max()),
        },
        residuals=challenger_residuals,
    )
    save_model(model,root=root)

Path("reports").mkdir(exist_ok=True)
Path("reports/v0.4.0_shadow_matchup.json").write_text(json.dumps(report,indent=2))
print(json.dumps(report,indent=2))
