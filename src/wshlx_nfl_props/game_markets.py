from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
import joblib
import numpy as np
import pandas as pd

from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import OneHotEncoder, StandardScaler
from sklearn.linear_model import ElasticNet, LogisticRegression
from sklearn.ensemble import RandomForestRegressor, HistGradientBoostingRegressor, HistGradientBoostingClassifier


TEAM_ALIASES={"LA":"LAR","JAC":"JAX","WSH":"WAS","OAK":"LV","SD":"LAC","STL":"LAR"}


def clean_team(x):
    if pd.isna(x):
        return x
    v=str(x).upper()
    return TEAM_ALIASES.get(v,v)


def build_team_week_pbp(pbp: pd.DataFrame) -> pd.DataFrame:
    req={"season","week","posteam","defteam"}
    if pbp is None or pbp.empty or not req.issubset(pbp.columns):
        return pd.DataFrame()
    p=pbp.copy()
    if "season_type" in p:
        p=p[p.season_type.astype(str).str.upper().isin(["REG","REGULAR"])]
    for c in ["epa","success","pass_attempt","rush_attempt","sack","qb_hit","turnover","interception","fumble_lost","touchdown"]:
        if c not in p:
            p[c]=0.0
        p[c]=pd.to_numeric(p[c],errors="coerce").fillna(0.0)
    if "turnover" not in pbp.columns:
        p["turnover"]=(p["interception"]>0).astype(float)+(p["fumble_lost"]>0).astype(float)

    off=p.groupby(["season","week","posteam"],as_index=False).agg(
        plays=("epa","size"),
        off_epa=("epa","sum"),
        off_epa_per_play=("epa","mean"),
        off_success=("success","mean"),
        pass_plays=("pass_attempt","sum"),
        rush_plays=("rush_attempt","sum"),
        sacks_allowed=("sack","sum"),
        qb_hits_allowed=("qb_hit","sum"),
        turnovers=("turnover","sum"),
        touchdowns=("touchdown","sum"),
    ).rename(columns={"posteam":"team"})
    de=p.groupby(["season","week","defteam"],as_index=False).agg(
        def_epa_allowed=("epa","sum"),
        def_epa_per_play=("epa","mean"),
        def_success_allowed=("success","mean"),
        opp_pass_plays=("pass_attempt","sum"),
        opp_rush_plays=("rush_attempt","sum"),
        sacks_made=("sack","sum"),
        qb_hits_made=("qb_hit","sum"),
        takeaways=("turnover","sum"),
        touchdowns_allowed=("touchdown","sum"),
    ).rename(columns={"defteam":"team"})
    x=off.merge(de,on=["season","week","team"],how="outer")
    x["team"]=x.team.map(clean_team)
    return x


def add_score_context(team_week: pd.DataFrame, schedules: pd.DataFrame) -> pd.DataFrame:
    if team_week.empty or schedules is None or schedules.empty:
        return team_week
    s=schedules.copy()
    if "game_type" in s:
        s=s[s.game_type.astype(str).str.upper().isin(["REG","REGULAR"])]
    needed={"season","week","home_team","away_team","home_score","away_score"}
    if not needed.issubset(s.columns):
        return team_week
    h=s[["season","week","home_team","home_score","away_score"]].copy()
    h["team"]=h.home_team.map(clean_team); h["points_for"]=h.home_score; h["points_against"]=h.away_score
    a=s[["season","week","away_team","home_score","away_score"]].copy()
    a["team"]=a.away_team.map(clean_team); a["points_for"]=a.away_score; a["points_against"]=a.home_score
    sc=pd.concat([h[["season","week","team","points_for","points_against"]],a[["season","week","team","points_for","points_against"]]])
    for c in ["points_for","points_against"]:
        sc[c]=pd.to_numeric(sc[c],errors="coerce")
    return team_week.merge(sc,on=["season","week","team"],how="left")


def rolling_team_features(team_week: pd.DataFrame, windows=(3,5,8)) -> pd.DataFrame:
    x=team_week.sort_values(["team","season","week"]).reset_index(drop=True).copy()
    g=x.groupby("team",sort=False)
    ids={"season","week"}
    numeric=[c for c in x.columns if c not in {"season","week","team"} and pd.api.types.is_numeric_dtype(x[c])]
    out=x[["season","week","team"]].copy()
    for c in numeric:
        out[f"{c}_lag1"]=g[c].shift(1)
        for w in windows:
            out[f"{c}_r{w}"]=g[c].transform(lambda s,w=w:s.shift(1).rolling(w,min_periods=1).mean())
    return out


def build_game_frame(schedules: pd.DataFrame, pbp: pd.DataFrame, windows=(3,5,8)) -> pd.DataFrame:
    s=schedules.copy()
    if "game_type" in s:
        s=s[s.game_type.astype(str).str.upper().isin(["REG","REGULAR"])]
    for c in ["home_team","away_team"]:
        s[c]=s[c].map(clean_team)
    for c in ["home_score","away_score"]:
        s[c]=pd.to_numeric(s[c],errors="coerce")

    tw=rolling_team_features(add_score_context(build_team_week_pbp(pbp),s),windows)
    h=tw.rename(columns={"team":"home_team", **{c:f"home_{c}" for c in tw.columns if c not in {"season","week","team"}}})
    a=tw.rename(columns={"team":"away_team", **{c:f"away_{c}" for c in tw.columns if c not in {"season","week","team"}}})
    x=s.merge(h,on=["season","week","home_team"],how="left").merge(a,on=["season","week","away_team"],how="left")

    x["target_home_points"]=x["home_score"]
    x["target_away_points"]=x["away_score"]
    x["target_margin"]=x["home_score"]-x["away_score"]
    x["target_total"]=x["home_score"]+x["away_score"]
    x["target_home_win"]=(x["home_score"]>x["away_score"]).astype(float)
    x.loc[x["home_score"].isna()|x["away_score"].isna(),"target_home_win"]=np.nan

    # Safe pregame context only. Sportsbook lines are intentionally excluded.
    if "home_rest" in x: x["pregame_home_rest"]=pd.to_numeric(x.home_rest,errors="coerce")
    if "away_rest" in x: x["pregame_away_rest"]=pd.to_numeric(x.away_rest,errors="coerce")
    if "temp" in x: x["pregame_temp"]=pd.to_numeric(x.temp,errors="coerce")
    if "wind" in x: x["pregame_wind"]=pd.to_numeric(x.wind,errors="coerce")
    x["pregame_home_field"]=1.0
    return x


def feature_columns(frame):
    excluded={"home_score","away_score","spread_line","total_line","result","target_home_points","target_away_points","target_margin","target_total","target_home_win"}
    numeric=[
        c for c in frame.columns
        if c not in excluded and pd.api.types.is_numeric_dtype(frame[c])
        and (c.startswith("home_") or c.startswith("away_") or c.startswith("pregame_"))
    ]
    # season/week are useful for imputation/trend only if explicitly desired; excluded to reduce overfit.
    categorical=[c for c in ["home_team","away_team","roof","surface"] if c in frame.columns]
    return sorted(set(numeric)),categorical


def _pre(numeric,categorical,scale=False):
    ns=[("impute",SimpleImputer(strategy="median"))]
    if scale: ns.append(("scale",StandardScaler()))
    return ColumnTransformer([
        ("num",Pipeline(ns),numeric),
        ("cat",Pipeline([("impute",SimpleImputer(strategy="most_frequent")),("onehot",OneHotEncoder(handle_unknown="ignore",sparse_output=False))]),categorical)
    ])


def reg_candidates(seed=42):
    return {
        "elastic_net":(ElasticNet(alpha=.08,l1_ratio=.15,max_iter=5000),True),
        "random_forest":(RandomForestRegressor(n_estimators=400,min_samples_leaf=6,max_features=.75,n_jobs=-1,random_state=seed),False),
        "hist_gbr":(HistGradientBoostingRegressor(max_iter=350,learning_rate=.04,max_leaf_nodes=31,l2_regularization=1.0,random_state=seed),False),
    }


def clf_candidates(seed=42):
    return {
        "logistic":(LogisticRegression(C=.8,max_iter=3000),True),
        "hist_gbc":(HistGradientBoostingClassifier(max_iter=300,learning_rate=.04,l2_regularization=1.0,random_state=seed),False),
    }


def make_reg(name,numeric,categorical,seed=42):
    est,scale=reg_candidates(seed)[name]
    return Pipeline([("pre",_pre(numeric,categorical,scale)),("model",est)])


def make_clf(name,numeric,categorical,seed=42):
    est,scale=clf_candidates(seed)[name]
    return Pipeline([("pre",_pre(numeric,categorical,scale)),("model",est)])


@dataclass
class GameMarketModel:
    market:str
    kind:str
    estimator_name:str
    pipeline:object
    numeric_features:list[str]
    categorical_features:list[str]
    residuals:list[float]
    validation_metrics:dict

    def predict(self,X):
        if self.kind=="binary":
            return self.pipeline.predict_proba(X)[:,1]
        return np.asarray(self.pipeline.predict(X),dtype=float)

    def probability_over(self,X,line):
        pred=self.predict(X)
        if self.kind=="binary":
            return pred
        r=np.asarray(self.residuals,dtype=float)
        if len(r)<20:r=np.array([0.0])
        return np.array([(p+r>line).mean() for p in pred])


def save_game_model(model:GameMarketModel,root="models/game_markets"):
    p=Path(root);p.mkdir(parents=True,exist_ok=True)
    joblib.dump(model,p/f"{model.market}.joblib")
