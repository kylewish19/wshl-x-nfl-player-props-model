from __future__ import annotations

import re
import numpy as np
import pandas as pd
from .enrichment import merge_auxiliary

PLAYER_ID_ALIASES = ["player_id", "gsis_id", "player_gsis_id"]
PLAYER_NAME_ALIASES = ["player_display_name", "player_name", "name"]
TEAM_ALIASES = ["recent_team", "team", "club"]
OPP_ALIASES = ["opponent_team", "opponent", "opp"]
POSITION_ALIASES = ["position", "pos"]

TARGET_ALIASES = {
    "passing_yards": ["passing_yards", "pass_yds"],
    "passing_tds": ["passing_tds", "passing_touchdowns", "pass_tds"],
    "rushing_yards": ["rushing_yards", "rush_yds"],
    "receiving_yards": ["receiving_yards", "rec_yds"],
    "def_sacks": ["def_sacks", "sacks", "sack"],
}

USAGE_ALIASES = {
    "passing_attempts": ["passing_attempts", "attempts", "pass_att"],
    "completions": ["completions", "passing_completions"],
    "rushing_attempts": ["carries", "rushing_attempts", "rush_att"],
    "targets": ["targets", "tgt"],
    "receptions": ["receptions", "rec"],
    "def_qb_hits": ["def_qb_hits", "qb_hits"],
    "def_tackles": ["def_tackles", "tackles_combined", "tackles"],
}

AUX_PREFIXES = ("snap_", "ngs_pass_", "ngs_rush_", "ngs_rec_", "pfr_pass_", "pfr_rush_", "pfr_rec_", "pfr_def_")


def first_present(df: pd.DataFrame, aliases: list[str]) -> str | None:
    for c in aliases:
        if c in df.columns:
            return c
    return None


def normalize_player_stats(df: pd.DataFrame) -> pd.DataFrame:
    x = df.copy()
    rename = {}
    for canonical, aliases in {
        "player_id": PLAYER_ID_ALIASES,
        "player_display_name": PLAYER_NAME_ALIASES,
        "team": TEAM_ALIASES,
        "opponent_team": OPP_ALIASES,
        "position": POSITION_ALIASES,
        **TARGET_ALIASES,
        **USAGE_ALIASES,
    }.items():
        c = first_present(x, aliases)
        if c is not None and c != canonical:
            rename[c] = canonical
    x = x.rename(columns=rename)
    for c in [*TARGET_ALIASES.keys(), *USAGE_ALIASES.keys()]:
        if c not in x:
            x[c] = 0.0
        x[c] = pd.to_numeric(x[c], errors="coerce").fillna(0.0)
    if "player_id" not in x:
        x["player_id"] = x.get("player_display_name", pd.Series(index=x.index, dtype=str)).astype(str)
    if "player_display_name" not in x:
        x["player_display_name"] = x["player_id"].astype(str)
    if "position" not in x:
        x["position"] = "UNK"
    if "team" not in x:
        x["team"] = "UNK"
    if "season_type" in x:
        x = x[x["season_type"].astype(str).str.upper().isin(["REG", "REGULAR"])]
    return x


def attach_opponents(stats: pd.DataFrame, schedules: pd.DataFrame | None) -> pd.DataFrame:
    x = stats.copy()
    if "opponent_team" in x and x["opponent_team"].notna().any():
        return x
    if schedules is None or schedules.empty:
        x["opponent_team"] = x.get("opponent_team", "UNK")
        return x
    s = schedules.copy()
    needed = {"season", "week", "home_team", "away_team"}
    if not needed.issubset(s.columns):
        x["opponent_team"] = x.get("opponent_team", "UNK")
        return x
    home = s[["season", "week", "home_team", "away_team"]].rename(
        columns={"home_team": "team", "away_team": "opponent_team"}
    )
    away = s[["season", "week", "home_team", "away_team"]].rename(
        columns={"away_team": "team", "home_team": "opponent_team"}
    )
    m = pd.concat([home, away], ignore_index=True).drop_duplicates(["season", "week", "team"])
    return x.drop(columns=["opponent_team"], errors="ignore").merge(m, on=["season", "week", "team"], how="left")


def add_pbp_team_context(stats: pd.DataFrame, pbp: pd.DataFrame | None) -> pd.DataFrame:
    x = stats.copy()
    if pbp is None or pbp.empty or not {"season", "week", "posteam", "defteam"}.issubset(pbp.columns):
        return x
    p = pbp.copy()
    for c in ["pass_attempt", "rush_attempt", "sack", "qb_hit"]:
        if c not in p:
            p[c] = 0
    off = p.groupby(["season", "week", "posteam"], as_index=False).agg(
        team_dropbacks=("pass_attempt", "sum"),
        team_rush_plays=("rush_attempt", "sum"),
        sacks_allowed=("sack", "sum"),
        qb_hits_allowed=("qb_hit", "sum"),
    ).rename(columns={"posteam": "team"})
    deff = p.groupby(["season", "week", "defteam"], as_index=False).agg(
        opp_dropbacks_faced=("pass_attempt", "sum"),
        opp_rush_plays_faced=("rush_attempt", "sum"),
        defense_sacks=("sack", "sum"),
        defense_qb_hits=("qb_hit", "sum"),
    ).rename(columns={"defteam": "team"})
    ctx = off.merge(deff, on=["season", "week", "team"], how="outer")
    return x.merge(ctx, on=["season", "week", "team"], how="left")


def build_feature_frame(player_stats: pd.DataFrame, schedules: pd.DataFrame | None = None,
                        pbp: pd.DataFrame | None = None, windows=(3, 5, 8),
                        auxiliary: dict[str, pd.DataFrame] | None = None) -> pd.DataFrame:
    x = normalize_player_stats(player_stats)
    x = attach_opponents(x, schedules)
    x = add_pbp_team_context(x, pbp)
    x = merge_auxiliary(x, auxiliary)
    x = x.sort_values(["player_id", "season", "week"]).reset_index(drop=True)

    base_numeric = [
        "passing_yards", "passing_tds", "rushing_yards", "receiving_yards", "def_sacks",
        "passing_attempts", "completions", "rushing_attempts", "targets", "receptions",
        "def_qb_hits", "def_tackles", "team_dropbacks", "team_rush_plays", "sacks_allowed",
        "qb_hits_allowed", "opp_dropbacks_faced", "opp_rush_plays_faced", "defense_sacks", "defense_qb_hits",
    ]
    base_numeric = [c for c in base_numeric if c in x.columns]
    aux_numeric = [
        c for c in x.columns
        if c.startswith(AUX_PREFIXES) and pd.api.types.is_numeric_dtype(x[c])
    ]

    g = x.groupby("player_id", sort=False)
    x["history_games"] = g.cumcount()

    for c in base_numeric:
        x[f"{c}_lag1"] = g[c].shift(1)
        for w in windows:
            x[f"{c}_r{w}"] = g[c].transform(lambda s, w=w: s.shift(1).rolling(w, min_periods=1).mean())

    # Auxiliary sources are all postgame measurements. They are never exposed raw:
    # only lagged/rolling values can become model features.
    aux_windows = tuple(w for w in windows if w <= 5) or (3, 5)
    for c in aux_numeric:
        x[f"{c}_lag1"] = g[c].shift(1)
        for w in aux_windows:
            x[f"{c}_r{w}"] = g[c].transform(lambda s, w=w: s.shift(1).rolling(w, min_periods=1).mean())

    for c in ["passing_yards", "rushing_yards", "receiving_yards", "def_sacks", "targets", "rushing_attempts", "passing_attempts"]:
        if c in x:
            x[f"{c}_ewm"] = g[c].transform(lambda s: s.shift(1).ewm(span=5, adjust=False, min_periods=1).mean())

    team_week = x.groupby(["season", "week", "team"], as_index=False).agg(
        tw_pass_att=("passing_attempts", "sum"), tw_rush_att=("rushing_attempts", "sum"), tw_targets=("targets", "sum")
    )
    x = x.merge(team_week, on=["season", "week", "team"], how="left")
    x["pass_attempt_share"] = np.where(x.tw_pass_att > 0, x.passing_attempts / x.tw_pass_att, 0)
    x["carry_share"] = np.where(x.tw_rush_att > 0, x.rushing_attempts / x.tw_rush_att, 0)
    x["target_share"] = np.where(x.tw_targets > 0, x.targets / x.tw_targets, 0)
    g = x.groupby("player_id", sort=False)
    for c in ["pass_attempt_share", "carry_share", "target_share"]:
        for w in windows:
            x[f"{c}_r{w}"] = g[c].transform(lambda s, w=w: s.shift(1).rolling(w, min_periods=1).mean())

    pos_grp = x["position"].replace({"FB": "RB", "EDGE": "DL", "DE": "DL", "DT": "DL", "OLB": "LB"})
    x["position_group_model"] = pos_grp
    if "opponent_team" in x:
        for target in ["passing_yards", "rushing_yards", "receiving_yards", "def_sacks"]:
            allowed = x.assign(def_team=x["opponent_team"]).groupby(
                ["season", "week", "def_team", "position_group_model"], as_index=False
            )[target].sum()
            allowed = allowed.sort_values(["def_team", "position_group_model", "season", "week"])
            ag = allowed.groupby(["def_team", "position_group_model"], sort=False)[target]
            allowed[f"opp_allowed_{target}_r5"] = ag.transform(lambda s: s.shift(1).rolling(5, min_periods=1).mean())
            x = x.merge(
                allowed[["season", "week", "def_team", "position_group_model", f"opp_allowed_{target}_r5"]],
                left_on=["season", "week", "opponent_team", "position_group_model"],
                right_on=["season", "week", "def_team", "position_group_model"], how="left"
            ).drop(columns=["def_team"], errors="ignore")
    return x


def _is_derived_feature(c: str) -> bool:
    return (
        c.endswith("_lag1")
        or c.endswith("_ewm")
        or re.search(r"_r\d+$", c) is not None
        or c in {"season", "week", "history_games"}
    )


def model_feature_columns(frame: pd.DataFrame, feature_set: str = "enriched") -> tuple[list[str], list[str]]:
    """Return only features known before kickoff.

    This intentionally rejects raw current-game usage, team-volume and auxiliary values.
    """
    exclude = {
        "passing_yards", "passing_tds", "rushing_yards", "receiving_yards", "def_sacks",
        "player_id", "player_display_name", "season_type", "game_id", "fantasy_points", "fantasy_points_ppr"
    }
    numeric = []
    for c in frame.columns:
        if c in exclude or not pd.api.types.is_numeric_dtype(frame[c]):
            continue
        if not _is_derived_feature(c):
            continue
        if feature_set == "baseline" and c.startswith(AUX_PREFIXES):
            continue
        numeric.append(c)
    categorical = [c for c in ["position_group_model", "team", "opponent_team"] if c in frame.columns]
    return sorted(set(numeric)), categorical
