from pathlib import Path
import json
import yaml
import numpy as np
import pandas as pd

from wshlx_nfl_props.data import load_raw
from wshlx_nfl_props.features import build_feature_frame, model_feature_columns
from wshlx_nfl_props.models import (
    TrainedPropModel, _make_pipeline, _predict, _metrics, save_model
)

cfg = yaml.safe_load(Path("config/model_config.yaml").read_text())
official = yaml.safe_load(Path("config/champion_specs.yaml").read_text())
d = load_raw()
if "player_stats" not in d or d["player_stats"].empty:
    raise SystemExit("Run scripts/update_data.py first")

frame = build_feature_frame(
    d["player_stats"], d.get("schedules"), d.get("pbp"),
    tuple(cfg["rolling_windows"]), auxiliary=d
)

CALIBRATION_YEARS = [2023, 2024, 2025]
root = Path("models/official")
root.mkdir(parents=True, exist_ok=True)
report = {"version": official["version"], "calibration_years": CALIBRATION_YEARS, "props": {}}

for prop, pspec in cfg["models"].items():
    frozen = official["champions"][prop]
    feature_set = frozen["feature_set"]
    algorithm = frozen["algorithm"]
    kind = pspec["kind"]
    target = pspec["target"]

    f = frame[frame["position"].isin(pspec["positions"])].copy()
    if kind == "binary":
        f["__target"] = (pd.to_numeric(f[target], errors="coerce").fillna(0) > 0).astype(int)
    else:
        f["__target"] = pd.to_numeric(f[target], errors="coerce")
    f = f[(f["history_games"] >= cfg["min_history_games"]) & f["__target"].notna()].copy()

    for c in [
        "position_group_model","team","opponent_team","pregame_report_status",
        "pregame_practice_status","pregame_roof","pregame_surface"
    ]:
        if c in f.columns:
            s = f[c].astype(object)
            f[c] = s.where(pd.notna(s), "__MISSING__").astype(str)

    numeric, categorical = model_feature_columns(f, feature_set=feature_set)
    cols = numeric + categorical

    folds = []
    residuals = []
    for year in CALIBRATION_YEARS:
        train = f[f.season < year]
        test = f[f.season == year]
        if train.empty or test.empty:
            continue
        pipe = _make_pipeline(kind, algorithm, numeric, categorical, cfg["random_state"])
        pipe.fit(train[cols], train["__target"])
        pred = _predict(pipe, kind, test[cols])
        folds.append({"year": year, "rows": int(len(test)), **_metrics(kind, test["__target"], pred)})
        if kind == "regression":
            residuals.extend((test["__target"].to_numpy() - np.asarray(pred)).tolist())

    final_pipe = _make_pipeline(kind, algorithm, numeric, categorical, cfg["random_state"])
    final_pipe.fit(f[cols], f["__target"])

    model = TrainedPropModel(
        prop=prop,
        kind=kind,
        estimator_name=algorithm,
        feature_set=feature_set,
        pipeline=final_pipe,
        numeric_features=numeric,
        categorical_features=categorical,
        validation_metrics={
            "official_version": official["version"],
            "frozen_spec": f"{feature_set}:{algorithm}",
            "walk_forward_folds": folds,
            "live_refit_rows": int(len(f)),
            "latest_season": int(f.season.max()),
            "latest_week": int(f.loc[f.season == f.season.max(), "week"].max()),
        },
        residuals=residuals,
    )
    save_model(model, root=root)
    report["props"][prop] = model.validation_metrics

Path("reports").mkdir(exist_ok=True)
Path("reports/official_refit.json").write_text(json.dumps(report, indent=2))
print(json.dumps(report, indent=2))
