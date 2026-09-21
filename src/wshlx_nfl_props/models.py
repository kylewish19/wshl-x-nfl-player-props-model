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
            "random_forest": (RandomForestRegressor(n_estimators=450, min_samples_leaf=7, max_features=0.75, n_jobs=-1, random_state=seed), False),
            "hist_gbr": (HistGradientBoostingRegressor(max_iter=300, learning_rate=0.045, max_leaf_nodes=31, l2_regularization=1.0, random_state=seed), False),
        }
    if kind == "count":
        return {
            "poisson_glm": (PoissonRegressor(alpha=0.35, max_iter=2000), True),
            "poisson_hist_gbr": (HistGradientBoostingRegressor(loss="poisson", max_iter=300, learning_rate=0.04, l2_regularization=1.0, random_state=seed), False),
        }
    return {
        "logistic": (LogisticRegression(C=0.7, max_iter=3000, class_weight="balanced"), True),
        "hist_gbc": (HistGradientBoostingClassifier(max_iter=250, learning_rate=0.04, l2_regularization=1.0, random_state=seed), False),
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
        return {"poisson_deviance": float(mean_poisson_deviance(y, p)), "mae": float(mean_absolute_error(y, p))}
    return {"mae": float(mean_absolute_error(y, pred)), "rmse": float(mean_squared_error(y, pred) ** 0.5)}


def _score(kind, m):
    if kind == "binary":
        return m["brier"]
    if kind == "count":
        return m["poisson_deviance"]
    return m["mae"]


def train_prop_model(frame: pd.DataFrame, prop: str, target: str, positions: list[str], kind: str,
                     holdout_season: int = 2025, min_history_games: int = 3, seed: int = 42) -> TrainedPropModel:
    f = frame[frame["position"].isin(positions)].copy()
    if target not in f:
        raise ValueError(f"Target {target!r} not present")
    if kind == "binary":
        f["__target"] = (pd.to_numeric(f[target], errors="coerce").fillna(0) > 0).astype(int)
    else:
        f["__target"] = pd.to_numeric(f[target], errors="coerce")
    f = f[(f["history_games"] >= min_history_games) & f["__target"].notna()].copy()
    numeric, categorical = model_feature_columns(f)
    if not numeric:
        raise ValueError("No numeric features were built")
    train = f[f.season < holdout_season]
    valid = f[f.season == holdout_season]
    if len(valid) < 25:
        seasons = sorted(f.season.dropna().unique())
        hs = seasons[-1]
        valid = f[f.season == hs]
        train = f[f.season < hs]
    Xtr, ytr = train[numeric + categorical], train["__target"]
    Xva, yva = valid[numeric + categorical], valid["__target"]

    trials = {}
    best = None
    for name, (est, scale) in _candidates(kind, seed).items():
        pipe = Pipeline([("pre", _preprocessor(numeric, categorical, scale=scale)), ("model", est)])
        pipe.fit(Xtr, ytr)
        pred = pipe.predict_proba(Xva)[:, 1] if kind == "binary" else pipe.predict(Xva)
        m = _metrics(kind, yva, pred)
        trials[name] = m
        if best is None or _score(kind, m) < best[0]:
            best = (_score(kind, m), name, pipe, pred)

    _, best_name, selected, valid_pred = best
    residuals = [] if kind in ("binary", "count") else (yva.to_numpy() - np.asarray(valid_pred)).tolist()

    est, scale = _candidates(kind, seed)[best_name]
    final_pipe = Pipeline([("pre", _preprocessor(numeric, categorical, scale=scale)), ("model", est)])
    final_pipe.fit(f[numeric + categorical], f["__target"])
    return TrainedPropModel(
        prop=prop, kind=kind, estimator_name=best_name, pipeline=final_pipe,
        numeric_features=numeric, categorical_features=categorical,
        validation_metrics={"selected": trials[best_name], "all_candidates": trials, "validation_rows": int(len(valid))},
        residuals=residuals,
    )


def save_model(model: TrainedPropModel, root="models"):
    root = Path(root); root.mkdir(parents=True, exist_ok=True)
    joblib.dump(model, root / f"{model.prop}.joblib")
    with open(root / f"{model.prop}.json", "w") as f:
        json.dump({
            "prop": model.prop, "kind": model.kind, "estimator": model.estimator_name,
            "validation_metrics": model.validation_metrics,
            "numeric_features": model.numeric_features, "categorical_features": model.categorical_features,
        }, f, indent=2)


def load_model(prop: str, root="models") -> TrainedPropModel:
    return joblib.load(Path(root) / f"{prop}.joblib")
