from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
import json
import joblib
import numpy as np
import pandas as pd
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler
from sklearn.impute import SimpleImputer
from sklearn.ensemble import RandomForestRegressor, HistGradientBoostingRegressor, HistGradientBoostingClassifier
from sklearn.linear_model import ElasticNet, LogisticRegression
from sklearn.metrics import mean_absolute_error, mean_squared_error, brier_score_loss, log_loss, roc_auc_score


@dataclass
class TrainedGameModel:
    market: str
    kind: str
    estimator_name: str
    pipeline: object
    numeric_features: list[str]
    categorical_features: list[str]
    validation_metrics: dict
    residuals: list[float]

    def predict(self, X: pd.DataFrame) -> np.ndarray:
        if self.kind == "binary":
            return self.pipeline.predict_proba(X)[:, 1]
        return np.asarray(self.pipeline.predict(X), dtype=float)

    def probability_over(self, X: pd.DataFrame, threshold: float) -> np.ndarray:
        point = self.predict(X)
        if self.kind == "binary":
            return point
        residuals = np.asarray(self.residuals, dtype=float)
        if residuals.size < 20:
            residuals = np.array([0.0])
        return np.array([(p + residuals > threshold).mean() for p in point])


def _preprocessor(numeric, categorical, scale=False):
    num_steps=[("impute",SimpleImputer(strategy="median"))]
    if scale:
        num_steps.append(("scale",StandardScaler()))
    num=Pipeline(num_steps)
    cat=Pipeline([
        ("impute",SimpleImputer(strategy="most_frequent")),
        ("onehot",OneHotEncoder(handle_unknown="ignore",sparse_output=False)),
    ])
    return ColumnTransformer([("num",num,numeric),("cat",cat,categorical)],remainder="drop")


def _candidates(kind: str, seed: int):
    if kind == "regression":
        return {
            "elastic_net":(ElasticNet(alpha=0.12,l1_ratio=0.15,max_iter=5000),True),
            "random_forest":(RandomForestRegressor(
                n_estimators=500,min_samples_leaf=5,max_features=0.75,n_jobs=-1,random_state=seed
            ),False),
            "hist_gbr":(HistGradientBoostingRegressor(
                max_iter=350,learning_rate=0.04,max_leaf_nodes=31,l2_regularization=1.0,random_state=seed
            ),False),
        }
    return {
        "logistic":(LogisticRegression(C=0.7,max_iter=3000),True),
        "hist_gbc":(HistGradientBoostingClassifier(
            max_iter=300,learning_rate=0.04,l2_regularization=1.0,random_state=seed
        ),False),
    }


def _make_pipeline(kind, name, numeric, categorical, seed):
    est,scale=_candidates(kind,seed)[name]
    return Pipeline([("pre",_preprocessor(numeric,categorical,scale)),("model",est)])


def _metrics(kind,y,pred):
    if kind=="binary":
        p=np.clip(pred,1e-5,1-1e-5)
        out={"brier":float(brier_score_loss(y,p)),"log_loss":float(log_loss(y,p))}
        try:
            out["auc"]=float(roc_auc_score(y,p))
        except Exception:
            out["auc"]=None
        return out
    return {
        "mae":float(mean_absolute_error(y,pred)),
        "rmse":float(mean_squared_error(y,pred)**0.5),
    }


def train_game_model(frame, market, target, kind, numeric, categorical,
                     selection_season=2024,test_season=2025,seed=42):
    f=frame[frame[target].notna()].copy()
    if kind=="binary" and "target_tie" in f:
        f=f[f["target_tie"]==0].copy()
    for c in categorical:
        s=f[c].astype(object)
        f[c]=s.where(pd.notna(s),"__MISSING__").astype(str)
    cols=numeric+categorical
    dev=f[f.season<selection_season]
    sel=f[f.season==selection_season]
    test=f[f.season==test_season]
    if min(len(dev),len(sel),len(test))==0:
        raise ValueError(f"Insufficient chronological split for {market}")

    trials={}
    best=None
    for name in _candidates(kind,seed):
        pipe=_make_pipeline(kind,name,numeric,categorical,seed)
        pipe.fit(dev[cols],dev[target])
        pred=pipe.predict_proba(sel[cols])[:,1] if kind=="binary" else pipe.predict(sel[cols])
        met=_metrics(kind,sel[target],pred)
        score=met["brier"] if kind=="binary" else met["mae"]
        trials[name]=met
        if best is None or score<best[0]:
            best=(score,name)

    name=best[1]
    pretest=f[f.season<=selection_season]
    test_pipe=_make_pipeline(kind,name,numeric,categorical,seed)
    test_pipe.fit(pretest[cols],pretest[target])
    test_pred=test_pipe.predict_proba(test[cols])[:,1] if kind=="binary" else test_pipe.predict(test[cols])
    test_metrics=_metrics(kind,test[target],test_pred)
    residuals=[] if kind=="binary" else (test[target].to_numpy()-np.asarray(test_pred)).tolist()

    final=_make_pipeline(kind,name,numeric,categorical,seed)
    final.fit(f[cols],f[target])
    return TrainedGameModel(
        market=market,kind=kind,estimator_name=name,pipeline=final,
        numeric_features=numeric,categorical_features=categorical,
        validation_metrics={
            "selection_season":selection_season,
            "test_season":test_season,
            "selection_candidates":trials,
            "test_metrics":test_metrics,
            "development_rows":int(len(dev)),
            "selection_rows":int(len(sel)),
            "test_rows":int(len(test)),
            "live_refit_rows":int(len(f)),
        },
        residuals=residuals,
    )


def save_game_model(model: TrainedGameModel, root="models/game_markets"):
    root=Path(root); root.mkdir(parents=True,exist_ok=True)
    joblib.dump(model,root/f"{model.market}.joblib")
    (root/f"{model.market}.json").write_text(json.dumps({
        "market":model.market,"kind":model.kind,"estimator":model.estimator_name,
        "validation_metrics":model.validation_metrics,
        "numeric_features":model.numeric_features,
        "categorical_features":model.categorical_features,
    },indent=2))


def load_game_model(market: str, root="models/game_markets") -> TrainedGameModel:
    return joblib.load(Path(root)/f"{market}.joblib")
