from __future__ import annotations

import re
import numpy as np
import pandas as pd

TEAM_ALIASES = ["team", "recent_team", "posteam", "club"]
SAFE_SCHEDULE_NUMERIC = ["home_rest", "away_rest", "temp", "wind"]
SAFE_SCHEDULE_CATEGORICAL = ["roof", "surface"]


def _first(df: pd.DataFrame, choices: list[str]) -> str | None:
    for c in choices:
        if c in df.columns:
            return c
    return None


def _regular_schedule(schedules: pd.DataFrame) -> pd.DataFrame:
    s = schedules.copy()
    if "game_type" in s.columns:
        s = s[s["game_type"].astype(str).str.upper().isin(["REG", "REGULAR"])]
    return s


def _schedule_team_grid(schedules: pd.DataFrame) -> pd.DataFrame:
    s = _regular_schedule(schedules)
    home = s[["season", "week", "home_team"]].rename(columns={"home_team": "team"})
    away = s[["season", "week", "away_team"]].rename(columns={"away_team": "team"})
    return (
        pd.concat([home, away], ignore_index=True)
        .dropna(subset=["season", "week", "team"])
        .drop_duplicates(["season", "week", "team"])
        .sort_values(["team", "season", "week"])
        .reset_index(drop=True)
    )


def _normalize_team_stats(team_stats: pd.DataFrame | None) -> pd.DataFrame:
    if team_stats is None or team_stats.empty or not {"season", "week"}.issubset(team_stats.columns):
        return pd.DataFrame()
    t = team_stats.copy()
    if "season_type" in t.columns:
        t = t[t["season_type"].astype(str).str.upper().isin(["REG", "REGULAR"])]
    team_col = _first(t, TEAM_ALIASES)
    if team_col is None:
        return pd.DataFrame()
    t["team"] = t[team_col].astype(str)

    exclude = {
        "season", "week", team_col, "team", "game_id", "season_type",
        "fantasy_points", "fantasy_points_ppr"
    }
    numeric = [
        c for c in t.columns
        if c not in exclude and pd.api.types.is_numeric_dtype(t[c])
    ]
    # Bound the feature set so schema growth upstream cannot explode training.
    numeric = numeric[:90]
    if not numeric:
        return pd.DataFrame()
    out = t[["season", "week", "team"] + numeric].copy()
    return out.groupby(["season", "week", "team"], as_index=False)[numeric].mean()


def _pbp_team_week(pbp: pd.DataFrame | None) -> pd.DataFrame:
    if pbp is None or pbp.empty or not {"season", "week", "posteam", "defteam"}.issubset(pbp.columns):
        return pd.DataFrame()
    p = pbp.copy()
    if "season_type" in p.columns:
        p = p[p["season_type"].astype(str).str.upper().isin(["REG", "REGULAR"])]

    for c in ["epa", "success", "yards_gained", "pass_attempt", "rush_attempt", "sack",
              "interception", "fumble_lost", "touchdown"]:
        if c not in p.columns:
            p[c] = 0.0
        p[c] = pd.to_numeric(p[c], errors="coerce").fillna(0.0)

    p["turnover"] = ((p["interception"] > 0) | (p["fumble_lost"] > 0)).astype(float)
    p["explosive"] = (p["yards_gained"] >= 20).astype(float)

    off = p.groupby(["season", "week", "posteam"], as_index=False).agg(
        pbp_off_plays=("epa", "size"),
        pbp_off_epa=("epa", "mean"),
        pbp_off_success=("success", "mean"),
        pbp_off_ypp=("yards_gained", "mean"),
        pbp_off_pass_rate=("pass_attempt", "mean"),
        pbp_off_sack_rate=("sack", "mean"),
        pbp_off_turnovers=("turnover", "sum"),
        pbp_off_explosive_rate=("explosive", "mean"),
        pbp_off_tds=("touchdown", "sum"),
    ).rename(columns={"posteam": "team"})

    deff = p.groupby(["season", "week", "defteam"], as_index=False).agg(
        pbp_def_plays=("epa", "size"),
        pbp_def_epa_allowed=("epa", "mean"),
        pbp_def_success_allowed=("success", "mean"),
        pbp_def_ypp_allowed=("yards_gained", "mean"),
        pbp_def_sack_rate=("sack", "mean"),
        pbp_def_takeaways=("turnover", "sum"),
        pbp_def_explosive_allowed=("explosive", "mean"),
        pbp_def_tds_allowed=("touchdown", "sum"),
    ).rename(columns={"defteam": "team"})
    return off.merge(deff, on=["season", "week", "team"], how="outer")


def build_team_context(schedules: pd.DataFrame, team_stats: pd.DataFrame | None,
                       pbp: pd.DataFrame | None, windows=(3, 5, 8)) -> pd.DataFrame:
    grid = _schedule_team_grid(schedules)
    raw = _normalize_team_stats(team_stats)
    if not raw.empty:
        raw = raw.rename(columns={c: f"stat_{c}" for c in raw.columns if c not in {"season", "week", "team"}})
        grid = grid.merge(raw, on=["season", "week", "team"], how="left")
    pw = _pbp_team_week(pbp)
    if not pw.empty:
        grid = grid.merge(pw, on=["season", "week", "team"], how="left")

    raw_numeric = [
        c for c in grid.columns
        if c not in {"season", "week"} and c != "team" and pd.api.types.is_numeric_dtype(grid[c])
    ]
    grid = grid.sort_values(["team", "season", "week"]).reset_index(drop=True)
    g = grid.groupby("team", sort=False)

    derived = ["season", "week", "team"]
    for c in raw_numeric:
        grid[f"{c}_lag1"] = g[c].shift(1)
        derived.append(f"{c}_lag1")
        for w in windows:
            name = f"{c}_r{w}"
            grid[name] = g[c].transform(lambda s, w=w: s.shift(1).rolling(w, min_periods=1).mean())
            derived.append(name)
    return grid[derived]


def build_game_frame(schedules: pd.DataFrame, team_stats: pd.DataFrame | None = None,
                     pbp: pd.DataFrame | None = None, windows=(3, 5, 8)) -> pd.DataFrame:
    s = _regular_schedule(schedules)
    needed = {"season", "week", "home_team", "away_team"}
    if not needed.issubset(s.columns):
        raise ValueError(f"Schedules missing required columns: {sorted(needed - set(s.columns))}")

    keep = ["season", "week", "home_team", "away_team"]
    keep += [c for c in ["game_id", "home_score", "away_score"] if c in s.columns]
    keep += [c for c in SAFE_SCHEDULE_NUMERIC + SAFE_SCHEDULE_CATEGORICAL if c in s.columns]
    games = s[keep].drop_duplicates(["season", "week", "home_team", "away_team"]).copy()

    ctx = build_team_context(schedules, team_stats, pbp, windows=windows)
    home = ctx.rename(columns={"team": "home_team", **{
        c: f"home_{c}" for c in ctx.columns if c not in {"season", "week", "team"}
    }})
    away = ctx.rename(columns={"team": "away_team", **{
        c: f"away_{c}" for c in ctx.columns if c not in {"season", "week", "team"}
    }})
    games = games.merge(home, on=["season", "week", "home_team"], how="left")
    games = games.merge(away, on=["season", "week", "away_team"], how="left")

    if "home_rest" in games:
        games["pregame_home_rest"] = pd.to_numeric(games["home_rest"], errors="coerce")
    if "away_rest" in games:
        games["pregame_away_rest"] = pd.to_numeric(games["away_rest"], errors="coerce")
    if {"home_rest", "away_rest"}.issubset(games.columns):
        games["pregame_rest_diff"] = (
            pd.to_numeric(games["home_rest"], errors="coerce")
            - pd.to_numeric(games["away_rest"], errors="coerce")
        )
    for src, dst in [("temp", "pregame_temp"), ("wind", "pregame_wind")]:
        if src in games:
            games[dst] = pd.to_numeric(games[src], errors="coerce")
    if "roof" in games:
        games["pregame_roof"] = games["roof"].fillna("UNKNOWN").astype(str)
    if "surface" in games:
        games["pregame_surface"] = games["surface"].fillna("UNKNOWN").astype(str)

    # Pairwise matchup differences help linear models while trees retain both sides.
    suffixes = {}
    for c in games.columns:
        if c.startswith("home_") and (c.endswith("_lag1") or re.search(r"_r\d+$", c)):
            suffixes[c[5:]] = c
    for suffix, hc in suffixes.items():
        ac = f"away_{suffix}"
        if ac in games.columns and pd.api.types.is_numeric_dtype(games[hc]):
            games[f"diff_{suffix}"] = games[hc] - games[ac]

    if {"home_score", "away_score"}.issubset(games.columns):
        games["target_home_points"] = pd.to_numeric(games["home_score"], errors="coerce")
        games["target_away_points"] = pd.to_numeric(games["away_score"], errors="coerce")
        games["target_margin"] = games["target_home_points"] - games["target_away_points"]
        games["target_total"] = games["target_home_points"] + games["target_away_points"]
        games["target_home_win"] = np.where(
            games["target_home_points"].notna() & games["target_away_points"].notna(),
            (games["target_home_points"] > games["target_away_points"]).astype(float),
            np.nan,
        )
        games["target_tie"] = np.where(
            games["target_home_points"].notna() & games["target_away_points"].notna(),
            (games["target_home_points"] == games["target_away_points"]).astype(float),
            np.nan,
        )
    return games


def game_feature_columns(frame: pd.DataFrame) -> tuple[list[str], list[str]]:
    numeric = []
    for c in frame.columns:
        if not pd.api.types.is_numeric_dtype(frame[c]):
            continue
        if c.startswith(("home_stat_", "away_stat_", "home_pbp_", "away_pbp_", "diff_stat_", "diff_pbp_", "pregame_")):
            numeric.append(c)
    categorical = [c for c in ["pregame_roof", "pregame_surface"] if c in frame.columns]
    return sorted(set(numeric)), categorical
