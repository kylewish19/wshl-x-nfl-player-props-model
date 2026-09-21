from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
import json
import joblib
import numpy as np
import pandas as pd
from scipy.stats import poisson
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler
from sklearn.impute import SimpleImputer
from sklearn.ensemble import RandomForestRegressor, HistGradientBoostingRegressor, HistGradientBoostingClassifier
from sklearn.linear_model import ElasticNet, LogisticRegression, PoissonRegressor
from sklearn.metrics import mean_absolute_error, mean_squared_error, mean_poisson_deviance, log_loss, brier_score_loss, roc_auc_score
from .features import model_feature_columns


@dataclass
class TrainedPropModel:
    prop: str
    kind: str
    estimator_name: str
    feature_set: str
    pipeline: object
    numeric_features: list[str]
    categorical_features: list[str]
    validation_metrics: dict
    residuals: list[float]

    def predict_point(self, X: pd.DataFrame) -> np.ndarray:
        if self.kind == "binary":
            return self.pipeline.predict_proba(X)[:, 1]
        return np.asarray(self.pipeline.predict(X), dtype=float)

    def probability_over(self, X: pd.DataFrame, line: float) -> np.ndarray:
        pred = self.predict_point(X)
        if self.kind == "binary":
            return pred
        if self.kind == "count":
            lam = np.clip(pred, 0.01, None)
            return 1 - poisson.cdf(np.floor(line), lam)
        residuals = np.asarray(self.residuals, dtype=float)
        if residuals.size < 20:
            residuals = np.array([0.0])
        return np.array([(p + residuals > line).mean() for p in pred])


def _preprocessor(numeric, categorical, scale=False):
    num_steps = [("impute", SimpleImputer(strategy="median"))]
    if scale:
        num_steps.append(("scale", StandardScaler()))
    num = Pipeline(num_steps)
    cat = Pipeline([
        ("impute", SimpleImputer(strategy="most_frequent")),
        ("onehot", OneHotEncoder(handle_unknown="ignore", sparse_output=False)),
    ])
    return ColumnTransformer([("num", num, numeric), ("cat", cat, categorical)], remainder="drop")


def _candidates(kind: str, seed: int):
    if kind == "regression":
        return {
            "elastic_net": (ElasticNet(alpha=0.08, l1_ratio=0.15, max_iter=5000), True),
            "random_forest": (
                RandomForestRegressor(
                    n_estimators=350, min_samples_leaf=7, max_features=0.75,
                    n_jobs=-1, random_state=seed
                ), False
            ),
            "hist_gbr": (
                HistGradientBoostingRegressor(
                    max_iter=300, learning_rate=0.045, max_leaf_nodes=31,
                    l2_regularization=1.0, random_state=seed
                ), False
            ),
        }
    if kind == "count":
        return {
            "poisson_glm": (PoissonRegressor(alpha=0.35, max_iter=2000), True),
            "poisson_hist_gbr": (
                HistGradientBoostingRegressor(
                    loss="poisson", max_iter=300, learning_rate=0.04,
                    l2_regularization=1.0, random_state=seed
                ), False
            ),
        }
    return {
        "logistic": (LogisticRegression(C=0.7, max_iter=3000, class_weight="balanced"), True),
        "hist_gbc": (
            HistGradientBoostingClassifier(
                max_iter=250, learning_rate=0.04, l2_regularization=1.0, random_state=seed
            ), False
        ),
    }


def _metrics(kind, y, pred):
    if kind == "binary":
        p = np.clip(pred, 1e-5, 1 - 1e-5)
        out = {"brier": float(brier_score_loss(y, p)), "log_loss": float(log_loss(y, p))}
        try:
            out["auc"] = float(roc_auc_score(y, p))
        except Exception:
            out["auc"] = None
        return out
    if kind == "count":
        p = np.clip(pred, 0.01, None)
        return {
            "poisson_deviance": float(mean_poisson_deviance(y, p)),
            "mae": float(mean_absolute_error(y, p)),
        }
    return {
        "mae": float(mean_absolute_error(y, pred)),
        "rmse": float(mean_squared_error(y, pred) ** 0.5),
    }


def primary_score(kind: str, metrics: dict) -> float:
    if kind == "binary":
        return metrics["brier"]
    if kind == "count":
        return metrics["poisson_deviance"]
    return metrics["mae"]


def _make_pipeline(kind: str, estimator_name: str, numeric: list[str], categorical: list[str], seed: int):
    est, scale = _candidates(kind, seed)[estimator_name]
    return Pipeline([
        ("pre", _preprocessor(numeric, categorical, scale=scale)),
        ("model", est),
    ])


def _predict(pipe, kind: str, X: pd.DataFrame):
    return pipe.predict_proba(X)[:, 1] if kind == "binary" else pipe.predict(X)


def train_prop_model(
    frame: pd.DataFrame,
    prop: str,
    target: str,
    positions: list[str],
    kind: str,
    selection_season: int = 2024,
    test_season: int = 2025,
    min_history_games: int = 3,
    seed: int = 42,
    feature_sets: tuple[str, ...] = ("baseline", "enriched"),
) -> TrainedPropModel:
    """Select feature set + algorithm on one season, evaluate once on the next.

    2021-2023 -> fit candidates
    2024      -> choose feature set and algorithm
    2025      -> untouched test
    2021-now  -> final live refit after specification is frozen
    """
    f = frame[frame["position"].isin(positions)].copy()
    if target not in f:
        raise ValueError(f"Target {target!r} not present")
    if kind == "binary":
        f["__target"] = (pd.to_numeric(f[target], errors="coerce").fillna(0) > 0).astype(int)
    else:
        f["__target"] = pd.to_numeric(f[target], errors="coerce")
    f = f[(f["history_games"] >= min_history_games) & f["__target"].notna()].copy()

    dev = f[f.season < selection_season]
    selection = f[f.season == selection_season]
    test = f[f.season == test_season]
    if min(len(dev), len(selection), len(test)) == 0:
        raise ValueError(
            f"Insufficient chronological split for {prop}: "
            f"dev={len(dev)}, selection={len(selection)}, test={len(test)}"
        )

    selection_trials = {}
    best = None
    feature_columns_by_set = {}

    for feature_set in feature_sets:
        numeric, categorical = model_feature_columns(f, feature_set=feature_set)
        if not numeric:
            continue
        cols = numeric + categorical
        feature_columns_by_set[feature_set] = (numeric, categorical)
        for name in _candidates(kind, seed):
            pipe = _make_pipeline(kind, name, numeric, categorical, seed)
            pipe.fit(dev[cols], dev["__target"])
            pred = _predict(pipe, kind, selection[cols])
            m = _metrics(kind, selection["__target"], pred)
            key = f"{feature_set}:{name}"
            selection_trials[key] = m
            score = primary_score(kind, m)
            if best is None or score < best[0]:
                best = (score, feature_set, name)

    if best is None:
        raise ValueError(f"No candidate models were trainable for {prop}")

    _, feature_set, best_name = best
    numeric, categorical = feature_columns_by_set[feature_set]
    cols = numeric + categorical

    # Freeze the chosen specification, retrain through selection season, then touch 2025 once.
    pretest = f[f.season <= selection_season]
    test_pipe = _make_pipeline(kind, best_name, numeric, categorical, seed)
    test_pipe.fit(pretest[cols], pretest["__target"])
    test_pred = _predict(test_pipe, kind, test[cols])
    test_metrics = _metrics(kind, test["__target"], test_pred)
    residuals = [] if kind in ("binary", "count") else (
        test["__target"].to_numpy() - np.asarray(test_pred)
    ).tolist()

    # Live refit uses every completed row after model/feature selection is frozen.
    final_pipe = _make_pipeline(kind, best_name, numeric, categorical, seed)
    final_pipe.fit(f[cols], f["__target"])

    selected_key = f"{feature_set}:{best_name}"
    return TrainedPropModel(
        prop=prop,
        kind=kind,
        estimator_name=best_name,
        feature_set=feature_set,
        pipeline=final_pipe,
        numeric_features=numeric,
        categorical_features=categorical,
        validation_metrics={
            "selection_season": selection_season,
            "test_season": test_season,
            "selected_spec": selected_key,
            "selection_metrics": selection_trials[selected_key],
            "test_metrics": test_metrics,
            "all_selection_candidates": selection_trials,
            "development_rows": int(len(dev)),
            "selection_rows": int(len(selection)),
            "test_rows": int(len(test)),
        },
        residuals=residuals,
    )


def save_model(model: TrainedPropModel, root="models"):
    root = Path(root)
    root.mkdir(parents=True, exist_ok=True)
    joblib.dump(model, root / f"{model.prop}.joblib")
    with open(root / f"{model.prop}.json", "w") as f:
        json.dump({
            "prop": model.prop,
            "kind": model.kind,
            "feature_set": model.feature_set,
            "estimator": model.estimator_name,
            "validation_metrics": model.validation_metrics,
            "numeric_features": model.numeric_features,
            "categorical_features": model.categorical_features,
        }, f, indent=2)


def load_model(prop: str, root="models") -> TrainedPropModel:
    return joblib.load(Path(root) / f"{prop}.joblib")
