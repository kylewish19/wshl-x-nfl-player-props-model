from __future__ import annotations

import numpy as np
import pandas as pd
from .enrichment import _clean_team


PBP_CTX = [
    "team_dropbacks", "team_rush_plays", "sacks_allowed", "qb_hits_allowed",
    "opp_dropbacks_faced", "opp_rush_plays_faced", "defense_sacks", "defense_qb_hits",
]


def build_team_week_context(pbp: pd.DataFrame | None) -> pd.DataFrame:
    if pbp is None or pbp.empty or not {"season","week","posteam","defteam"}.issubset(pbp.columns):
        return pd.DataFrame()
    p = pbp.copy()
    if "season_type" in p.columns:
        p = p[p["season_type"].astype(str).str.upper().isin(["REG","REGULAR"])]
    for c in ["posteam", "defteam"]:
        p[c] = p[c].map(_clean_team)
    for c in ["pass_attempt","rush_attempt","sack","qb_hit"]:
        if c not in p.columns:
            p[c] = 0
        p[c] = pd.to_numeric(p[c], errors="coerce").fillna(0.0)

    off = p.groupby(["season","week","posteam"], as_index=False).agg(
        team_dropbacks=("pass_attempt","sum"),
        team_rush_plays=("rush_attempt","sum"),
        sacks_allowed=("sack","sum"),
        qb_hits_allowed=("qb_hit","sum"),
    ).rename(columns={"posteam":"team"})
    deff = p.groupby(["season","week","defteam"], as_index=False).agg(
        opp_dropbacks_faced=("pass_attempt","sum"),
        opp_rush_plays_faced=("rush_attempt","sum"),
        defense_sacks=("sack","sum"),
        defense_qb_hits=("qb_hit","sum"),
    ).rename(columns={"defteam":"team"})
    return off.merge(deff,on=["season","week","team"],how="outer")


def add_matchup_context(base: pd.DataFrame, pbp: pd.DataFrame | None, windows=(3,5)) -> pd.DataFrame:
    """Attach the last completed team-game windows strictly before each target week.

    Future target rows need not exist in play-by-play. No target rows are added
    to the history; multiple future queries and bye weeks cannot consume a
    rolling-window slot. Preserve the caller's team labels and row order.
    Input PBP must already be restricted to the permitted observation cutoff.
    """
    x = base.copy()
    tw = build_team_week_context(pbp)
    if tw.empty:
        return x
    tw = tw.sort_values(["team", "season", "week"]).reset_index(drop=True)
    g = tw.groupby("team", sort=False)
    cols = []
    for c in PBP_CTX:
        for w in windows:
            col = f"{c}_r{w}"
            tw[col] = g[c].transform(lambda s, w=w: s.rolling(w, min_periods=1).mean())
            cols.append(col)
    # season/week order is monotonic across NFL seasons.
    tw["_order"] = tw.season.astype(int)*100 + tw.week.astype(int)
    query_order = x.season.astype(int).to_numpy()*100 + x.week.astype(int).to_numpy()
    histories = {str(team): part for team, part in tw.groupby("team", sort=False)}
    for key, prefix in [("team","teamctx_"),("opponent_team","oppctx_")]:
        if key not in x:
            continue
        labels = x[key].map(_clean_team).fillna("").astype(str).to_numpy()
        values = np.full((len(x), len(cols)), np.nan)
        for team, part in histories.items():
            idx = np.flatnonzero(labels == team)
            if not len(idx):
                continue
            # side=left excludes the same week, even if PBP already exists.
            prev = np.searchsorted(part["_order"].to_numpy(), query_order[idx], side="left") - 1
            valid = prev >= 0
            values[idx[valid], :] = part[cols].to_numpy(dtype=float)[prev[valid]]
        x = pd.concat([x, pd.DataFrame(values, index=x.index, columns=[prefix+c for c in cols])], axis=1)
    return x
