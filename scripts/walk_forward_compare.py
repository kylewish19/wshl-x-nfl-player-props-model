from pathlib import Path
import json
import numpy as np
import pandas as pd
import yaml

from wshlx_nfl_props.data import load_raw
from wshlx_nfl_props.features import build_feature_frame, model_feature_columns
from wshlx_nfl_props.models import _make_pipeline, _predict, _metrics, primary_score

cfg = yaml.safe_load(Path("config/model_config.yaml").read_text())
d = load_raw()
frame = build_feature_frame(
    d["player_stats"], d.get("schedules"), d.get("pbp"),
    tuple(cfg["rolling_windows"]), auxiliary=d
)

SPECS = {
    "v0.2": {
        "qb_pass_yds": ("enriched", "random_forest"),
        "qb_pass_tds": ("baseline", "poisson_glm"),
        "qb_rush_yds": ("baseline", "elastic_net"),
        "rb_rush_yds": ("baseline", "elastic_net"),
        "rec_yds": ("baseline", "elastic_net"),
        "sack_yes": ("enriched", "hist_gbc"),
    },
    "v0.3": {
        "qb_pass_yds": ("pregame", "random_forest"),
        "qb_pass_tds": ("pregame", "poisson_glm"),
        "qb_rush_yds": ("baseline", "elastic_net"),
        "rb_rush_yds": ("pregame", "elastic_net"),
        "rec_yds": ("baseline", "elastic_net"),
        "sack_yes": ("full", "hist_gbc"),
    },
}

TEST_YEARS = [2023, 2024, 2025]
seed = cfg["random_state"]
min_hist = cfg["min_history_games"]
report = {"method": "fixed-spec expanding-window walk-forward", "test_years": TEST_YEARS, "props": {}}

for prop, pspec in cfg["models"].items():
    f = frame[frame["position"].isin(pspec["positions"])].copy()
    target = pspec["target"]
    kind = pspec["kind"]
    if kind == "binary":
        f["__target"] = (pd.to_numeric(f[target], errors="coerce").fillna(0) > 0).astype(int)
    else:
        f["__target"] = pd.to_numeric(f[target], errors="coerce")
    f = f[(f["history_games"] >= min_hist) & f["__target"].notna()].copy()

    for c in [
        "position_group_model","team","opponent_team","pregame_report_status",
        "pregame_practice_status","pregame_roof","pregame_surface"
    ]:
        if c in f.columns:
            s = f[c].astype(object)
            f[c] = s.where(pd.notna(s), "__MISSING__").astype(str)

    report["props"][prop] = {}
    for version, spec_map in SPECS.items():
        feature_set, algorithm = spec_map[prop]
        numeric, categorical = model_feature_columns(f, feature_set=feature_set)
        cols = numeric + categorical
        folds = []
        total_n = 0
        weighted_primary = 0.0
        for year in TEST_YEARS:
            train = f[f.season < year]
            test = f[f.season == year]
            if len(train) == 0 or len(test) == 0:
                continue
            pipe = _make_pipeline(kind, algorithm, numeric, categorical, seed)
            pipe.fit(train[cols], train["__target"])
            pred = _predict(pipe, kind, test[cols])
            metrics = _metrics(kind, test["__target"], pred)
            score = primary_score(kind, metrics)
            n = int(len(test))
            folds.append({"year": year, "rows": n, **metrics})
            weighted_primary += score * n
            total_n += n
        report["props"][prop][version] = {
            "feature_set": feature_set,
            "algorithm": algorithm,
            "folds": folds,
            "weighted_primary_metric": weighted_primary / total_n if total_n else None,
            "rows": total_n,
        }

    a = report["props"][prop]["v0.2"]["weighted_primary_metric"]
    b = report["props"][prop]["v0.3"]["weighted_primary_metric"]
    report["props"][prop]["comparison"] = {
        "v03_minus_v02": b - a if a is not None and b is not None else None,
        "v03_relative_change_pct": ((b / a) - 1) * 100 if a not in (None, 0) and b is not None else None,
        "lower_is_better": True,
    }

Path("reports").mkdir(exist_ok=True)
Path("reports/walk_forward_v02_vs_v03.json").write_text(json.dumps(report, indent=2))
print(json.dumps(report, indent=2))
