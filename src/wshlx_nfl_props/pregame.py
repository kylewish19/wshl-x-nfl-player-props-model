from __future__ import annotations

import numpy as np
import pandas as pd
from .enrichment import _name_key, _clean_team, _first, NAME_ALIASES, TEAM_ALIASES


SAFE_SCHEDULE_NUMERIC = [
    "home_rest", "away_rest", "spread_line", "total_line", "temp", "wind",
]
SAFE_SCHEDULE_CATEGORICAL = ["roof", "surface"]


def add_schedule_pregame(base: pd.DataFrame, schedules: pd.DataFrame | None) -> pd.DataFrame:
    if schedules is None or schedules.empty:
        return base
    needed = {"season", "week", "home_team", "away_team"}
    if not needed.issubset(schedules.columns):
        return base
    s = schedules.copy()
    if "game_type" in s.columns:
        s = s[s["game_type"].astype(str).str.upper().isin(["REG", "REGULAR"])]
    keep = ["season", "week", "home_team", "away_team"]
    keep += [c for c in SAFE_SCHEDULE_NUMERIC + SAFE_SCHEDULE_CATEGORICAL if c in s.columns]
    s = s[keep].drop_duplicates(["season", "week", "home_team", "away_team"])

    home = s.copy()
    home["team"] = home["home_team"].map(_clean_team)
    home["pregame_home"] = 1.0
    if "home_rest" in home:
        home["pregame_rest"] = pd.to_numeric(home["home_rest"], errors="coerce")
    if "spread_line" in home:
        # nflverse spread_line is from the home-team perspective.
        home["pregame_team_spread"] = pd.to_numeric(home["spread_line"], errors="coerce")
    away = s.copy()
    away["team"] = away["away_team"].map(_clean_team)
    away["pregame_home"] = 0.0
    if "away_rest" in away:
        away["pregame_rest"] = pd.to_numeric(away["away_rest"], errors="coerce")
    if "spread_line" in away:
        away["pregame_team_spread"] = -pd.to_numeric(away["spread_line"], errors="coerce")

    frames = []
    for q in (home, away):
        out = q[["season", "week", "team", "pregame_home"]].copy()
        if "pregame_rest" in q:
            out["pregame_rest"] = q["pregame_rest"]
        if "pregame_team_spread" in q:
            out["pregame_team_spread"] = q["pregame_team_spread"]
        if "total_line" in q:
            out["pregame_total_line"] = pd.to_numeric(q["total_line"], errors="coerce")
        if "temp" in q:
            out["pregame_temp"] = pd.to_numeric(q["temp"], errors="coerce")
        if "wind" in q:
            out["pregame_wind"] = pd.to_numeric(q["wind"], errors="coerce")
        if "roof" in q:
            out["pregame_roof"] = q["roof"].astype("string")
        if "surface" in q:
            out["pregame_surface"] = q["surface"].astype("string")
        frames.append(out)
    ctx = pd.concat(frames, ignore_index=True).drop_duplicates(["season", "week", "team"])
    return base.merge(ctx, on=["season", "week", "team"], how="left")


def add_player_injury_pregame(base: pd.DataFrame, injuries: pd.DataFrame | None) -> pd.DataFrame:
    x = base.copy()
    x["__name_key"] = x["player_display_name"].map(_name_key)
    if injuries is None or injuries.empty or not {"season", "week"}.issubset(injuries.columns):
        x["pregame_injury_listed"] = 0.0
        x["pregame_report_status"] = "NONE"
        x["pregame_practice_status"] = "NONE"
        return x.drop(columns=["__name_key"], errors="ignore")

    inj = injuries.copy()
    name_col = _first(inj, NAME_ALIASES + ["full_name"])
    team_col = _first(inj, TEAM_ALIASES)
    if name_col is None or team_col is None:
        x["pregame_injury_listed"] = 0.0
        x["pregame_report_status"] = "NONE"
        x["pregame_practice_status"] = "NONE"
        return x.drop(columns=["__name_key"], errors="ignore")

    inj["__name_key"] = inj[name_col].map(_name_key)
    inj["team"] = inj[team_col].map(_clean_team)
    inj["season"] = pd.to_numeric(inj["season"], errors="coerce")
    inj["week"] = pd.to_numeric(inj["week"], errors="coerce")
    # Keep the latest report row available for that player-week when duplicates exist.
    sort_cols = [c for c in ["season", "week", "team", "__name_key", "date_modified"] if c in inj.columns]
    if sort_cols:
        inj = inj.sort_values(sort_cols)
    inj = inj.drop_duplicates(["season", "week", "team", "__name_key"], keep="last")

    rs = inj["report_status"].astype("string") if "report_status" in inj else pd.Series("NONE", index=inj.index, dtype="string")
    ps = inj["practice_status"].astype("string") if "practice_status" in inj else pd.Series("NONE", index=inj.index, dtype="string")
    z = inj[["season", "week", "team", "__name_key"]].copy()
    z["pregame_injury_listed"] = 1.0
    z["pregame_report_status"] = rs.fillna("NONE").str.upper()
    z["pregame_practice_status"] = ps.fillna("NONE").str.upper()
    z["pregame_questionable"] = z["pregame_report_status"].str.contains("QUESTION", na=False).astype(float)
    z["pregame_doubtful"] = z["pregame_report_status"].str.contains("DOUBT", na=False).astype(float)
    z["pregame_out"] = z["pregame_report_status"].str.contains("OUT", na=False).astype(float)
    z["pregame_practice_dnp"] = z["pregame_practice_status"].str.contains("DID NOT|DNP", regex=True, na=False).astype(float)
    z["pregame_practice_limited"] = z["pregame_practice_status"].str.contains("LIMIT", na=False).astype(float)

    x = x.merge(z, on=["season", "week", "team", "__name_key"], how="left")
    for c in ["pregame_injury_listed", "pregame_questionable", "pregame_doubtful", "pregame_out",
              "pregame_practice_dnp", "pregame_practice_limited"]:
        if c in x:
            x[c] = pd.to_numeric(x[c], errors="coerce").fillna(0.0)
    for c in ["pregame_report_status", "pregame_practice_status"]:
        if c in x:
            x[c] = x[c].fillna("NONE").astype(str)
    return x.drop(columns=["__name_key"], errors="ignore")


def add_pregame_context(base: pd.DataFrame, schedules: pd.DataFrame | None,
                        injuries: pd.DataFrame | None) -> pd.DataFrame:
    x = add_schedule_pregame(base, schedules)
    x = add_player_injury_pregame(x, injuries)
    return x
