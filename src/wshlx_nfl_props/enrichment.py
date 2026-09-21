from __future__ import annotations

import re
import unicodedata
import numpy as np
import pandas as pd

TEAM_MAP = {
    "LA": "LAR", "JAC": "JAX", "WSH": "WAS", "OAK": "LV", "SD": "LAC", "STL": "LAR"
}

NAME_ALIASES = ["player_display_name", "pfr_player_name", "player", "player_name", "full_name", "name"]
TEAM_ALIASES = ["team", "team_abbr", "recent_team", "club"]

SOURCE_KEEP = {
    "snap": ["offense_snaps", "offense_pct", "defense_snaps", "defense_pct"],
    "ngs_pass": [
        "avg_time_to_throw", "avg_completed_air_yards", "avg_intended_air_yards",
        "avg_air_yards_differential", "aggressiveness", "avg_air_yards_to_sticks",
        "passer_rating", "completion_percentage", "expected_completion_percentage",
        "completion_percentage_above_expectation",
    ],
    "ngs_rush": [
        "efficiency", "percent_attempts_gte_eight_defenders", "avg_time_to_los",
        "expected_rush_yards", "rush_yards_over_expected",
        "rush_yards_over_expected_per_att", "rush_pct_over_expected",
    ],
    "ngs_rec": [
        "avg_air_distance", "avg_cushion", "avg_separation",
        "percent_share_of_intended_air_yards", "catch_percentage",
        "avg_yac", "avg_expected_yac", "avg_yac_above_expectation",
    ],
    # PFR schemas evolve; curated common fields are preferred, then numeric extras
    # are picked up automatically below and safely lagged before modeling.
    "pfr_pass": [
        "passing_drops", "passing_drop_pct", "receiving_drop", "receiving_drop_pct",
        "passing_bad_throws", "passing_bad_throw_pct", "times_sacked", "times_blitzed",
        "times_hurried", "times_hit", "times_pressured", "times_pressured_pct",
        "def_times_blitzed", "def_times_hurried", "def_times_hitqb",
    ],
    "pfr_rush": [],
    "pfr_rec": [],
    "pfr_def": [],
}


def _first(df: pd.DataFrame, choices: list[str]) -> str | None:
    for c in choices:
        if c in df.columns:
            return c
    return None


def _name_key(value) -> str:
    s = "" if pd.isna(value) else str(value)
    s = unicodedata.normalize("NFKD", s).encode("ascii", "ignore").decode().lower()
    s = re.sub(r"\b(jr|sr|ii|iii|iv|v)\b", "", s)
    return re.sub(r"[^a-z0-9]", "", s)


def _clean_team(value):
    if pd.isna(value):
        return value
    v = str(value).upper()
    return TEAM_MAP.get(v, v)


def _prepare_aux(df: pd.DataFrame, prefix: str) -> pd.DataFrame:
    if df is None or df.empty or not {"season", "week"}.issubset(df.columns):
        return pd.DataFrame()
    a = df.copy()
    if "season_type" in a.columns:
        a = a[a["season_type"].astype(str).str.upper().isin(["REG", "REGULAR"])]
    name_col = _first(a, NAME_ALIASES)
    team_col = _first(a, TEAM_ALIASES)
    if name_col is None or team_col is None:
        return pd.DataFrame()
    a["__name_key"] = a[name_col].map(_name_key)
    a["team"] = a[team_col].map(_clean_team)
    a["season"] = pd.to_numeric(a["season"], errors="coerce")
    a["week"] = pd.to_numeric(a["week"], errors="coerce")

    preferred = [c for c in SOURCE_KEEP.get(prefix, []) if c in a.columns]
    identity = {
        "season", "week", name_col, team_col, "team", "__name_key", "game_id", "pfr_game_id",
        "player_gsis_id", "pfr_player_id", "player_position", "position", "opponent",
    }
    numeric_extra = [
        c for c in a.columns
        if c not in identity and pd.api.types.is_numeric_dtype(a[c])
        and c not in {"season", "week", "player_jersey_number"}
    ]
    # Curated metrics first; keep a bounded number of source-specific extras.
    metrics = list(dict.fromkeys(preferred + numeric_extra))[:40]
    if not metrics:
        return pd.DataFrame()
    keep = ["season", "week", "team", "__name_key"] + metrics
    a = a[keep].copy()
    for c in metrics:
        a[c] = pd.to_numeric(a[c], errors="coerce")
    a = a.groupby(["season", "week", "team", "__name_key"], as_index=False)[metrics].mean()
    return a.rename(columns={c: f"{prefix}_{c}" for c in metrics})


def merge_auxiliary(base: pd.DataFrame, data: dict[str, pd.DataFrame] | None) -> pd.DataFrame:
    """Merge postgame player-week sources onto the historical frame.

    Values are deliberately merged in raw same-week form here. build_feature_frame()
    shifts/rolls every auxiliary metric before model selection, so the current game's
    advanced result can never be used to predict itself.
    """
    if not data:
        return base
    x = base.copy()
    x["__name_key"] = x["player_display_name"].map(_name_key)
    x["team"] = x["team"].map(_clean_team)

    sources = [
        ("snap_counts", "snap"),
        ("ngs_passing", "ngs_pass"),
        ("ngs_rushing", "ngs_rush"),
        ("ngs_receiving", "ngs_rec"),
        ("adv_pass", "pfr_pass"),
        ("adv_rush", "pfr_rush"),
        ("adv_rec", "pfr_rec"),
        ("adv_def", "pfr_def"),
    ]
    for key, prefix in sources:
        aux = _prepare_aux(data.get(key), prefix)
        if aux.empty:
            continue
        x = x.merge(aux, on=["season", "week", "team", "__name_key"], how="left")
    return x.drop(columns=["__name_key"], errors="ignore")
