from __future__ import annotations

import pandas as pd


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
    """Add pregame team and opponent PBP tendencies.

    Every context feature is shifted one game at the TEAM level before it is
    attached to a player-week. This avoids both current-game leakage and the
    distortion caused by shifting duplicated team values inside player histories.
    """
    x = base.copy()
    tw = build_team_week_context(pbp)
    if tw.empty:
        return x
    tw = tw.sort_values(["team","season","week"]).reset_index(drop=True)
    g = tw.groupby("team", sort=False)
    derived = ["season","week","team"]
    for c in PBP_CTX:
        if c not in tw.columns:
            continue
        for w in windows:
            col=f"{c}_r{w}"
            tw[col]=g[c].transform(lambda s,w=w:s.shift(1).rolling(w,min_periods=1).mean())
            derived.append(col)

    own=tw[derived].copy().rename(columns={c:f"teamctx_{c}" for c in derived if c not in {"season","week","team"}})
    x=x.merge(own,on=["season","week","team"],how="left")

    if "opponent_team" in x.columns:
        opp=tw[derived].copy().rename(columns={"team":"opponent_team", **{c:f"oppctx_{c}" for c in derived if c not in {"season","week","team"}}})
        x=x.merge(opp,on=["season","week","opponent_team"],how="left")
    return x
