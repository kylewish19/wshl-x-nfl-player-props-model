from __future__ import annotations

import numpy as np
import pandas as pd

from .game_predict import american_to_implied


def add_market_priors(frame: pd.DataFrame) -> pd.DataFrame:
    """Add pregame market baselines and residual targets.

    nflverse spread_line uses the home-team margin convention:
    +3 means the market expects the home team to win by 3.
    """
    x=frame.copy()
    if "spread_line" not in x.columns or "total_line" not in x.columns:
        raise ValueError("Schedule frame must contain spread_line and total_line for market-informed training")

    x["market_margin"]=pd.to_numeric(x["spread_line"],errors="coerce")
    x["market_total"]=pd.to_numeric(x["total_line"],errors="coerce")
    x["market_home_points"]=(x["market_total"]+x["market_margin"])/2.0
    x["market_away_points"]=(x["market_total"]-x["market_margin"])/2.0

    if {"home_moneyline","away_moneyline"}.issubset(x.columns):
        raw_h=x["home_moneyline"].map(american_to_implied)
        raw_a=x["away_moneyline"].map(american_to_implied)
        denom=raw_h+raw_a
        x["market_home_win_fair"]=np.where(denom>0,raw_h/denom,np.nan)
    else:
        x["market_home_win_fair"]=np.nan

    if "target_margin" in x:
        x["target_margin_residual"]=x["target_margin"]-x["market_margin"]
    if "target_total" in x:
        x["target_total_residual"]=x["target_total"]-x["market_total"]
    return x


def market_informed_feature_columns(frame: pd.DataFrame, base_numeric: list[str], base_categorical: list[str]):
    market_numeric=[
        c for c in [
            "market_margin","market_total","market_home_points","market_away_points","market_home_win_fair"
        ] if c in frame.columns
    ]
    return sorted(set(base_numeric+market_numeric)),list(base_categorical)
