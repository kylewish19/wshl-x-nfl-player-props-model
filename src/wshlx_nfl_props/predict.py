from __future__ import annotations

from pathlib import Path
import numpy as np
import pandas as pd
from .features import build_feature_frame, normalize_player_stats
from .models import load_model


def american_to_implied(odds):
    if pd.isna(odds):
        return np.nan
    odds = float(odds)
    return 100 / (odds + 100) if odds > 0 else (-odds) / ((-odds) + 100)


def fair_pair(odds_a, odds_b):
    a = american_to_implied(odds_a)
    b = american_to_implied(odds_b)
    if np.isnan(a) or np.isnan(b) or (a + b) <= 0:
        return np.nan, np.nan
    return a / (a + b), b / (a + b)


def _next_game(team: str, schedules: pd.DataFrame, season: int):
    if schedules is None or schedules.empty:
        return None
    s = schedules[schedules.season == season].copy()
    if "game_type" in s:
        s = s[s.game_type.astype(str).str.upper().isin(["REG", "REGULAR"])]
    if {"home_score", "away_score"}.issubset(s.columns):
        s = s[s.home_score.isna() & s.away_score.isna()]
    s = s[(s.home_team == team) | (s.away_team == team)].sort_values("week")
    if s.empty:
        return None
    r = s.iloc[0]
    opp = r.away_team if r.home_team == team else r.home_team
    return {"week": int(r.week), "opponent_team": opp, "home": int(r.home_team == team)}


def build_prop_rows(history: pd.DataFrame, lines: pd.DataFrame, schedules: pd.DataFrame,
                    pbp: pd.DataFrame | None = None, windows=(3,5,8),
                    auxiliary: dict[str, pd.DataFrame] | None = None) -> pd.DataFrame:
    hist = normalize_player_stats(history)
    max_season = int(hist.season.max())
    src = lines.reset_index(drop=True).copy()

    # Sportsbooks often list many alternate thresholds for the same player.
    # Those rows describe one pregame state and MUST share exactly one feature
    # vector. Creating a placeholder per threshold lets prediction placeholders
    # leak into later rolling features for the same player.
    key_cols = ["player_display_name"]
    for c in ["team", "opponent", "week"]:
        if c in src.columns:
            key_cols.append(c)
    keys = []
    for _, r in src.iterrows():
        keys.append(tuple("__NA__" if pd.isna(r.get(c, np.nan)) else str(r.get(c)) for c in key_cols))
    codes, _ = pd.factorize(pd.Series(keys, dtype="object"), sort=False)
    first_indices = [int(np.flatnonzero(codes == i)[0]) for i in range(int(codes.max()) + 1)]
    unique_lines = src.iloc[first_indices].reset_index(drop=True)

    placeholders = []
    for i, line in unique_lines.iterrows():
        name = str(line.player_display_name)
        matches = hist[hist.player_display_name.str.lower() == name.lower()].copy()
        supplied_team = line.get("team", None)
        if not pd.isna(supplied_team) and supplied_team:
            team_matches = matches[matches["team"].astype(str).str.upper() == str(supplied_team).upper()]
            if not team_matches.empty:
                matches = team_matches
        matches = matches.sort_values(["season", "week"])
        if matches.empty:
            raise KeyError(f"Player not found in history: {name}")
        last = matches.iloc[-1]
        team = supplied_team if (not pd.isna(supplied_team) and supplied_team) else last.team
        ng = _next_game(team, schedules, max_season)
        opp = line.get("opponent", None)
        if pd.isna(opp): opp = None
        week = line.get("week", None)
        if pd.isna(week): week = None
        if ng:
            opp = opp or ng["opponent_team"]
            week = int(week or ng["week"])
        if week is None:
            week = int(hist[hist.season == max_season].week.max()) + 1
        row = {c: np.nan for c in hist.columns}
        row.update({
            "season": max_season, "week": week, "player_id": last.player_id,
            "player_display_name": last.player_display_name, "position": last.position,
            "team": team, "opponent_team": opp or "UNK", "__prediction_row": i,
        })
        if "season_type" in hist.columns:
            row["season_type"] = "REG"
        placeholders.append(row)

    h = hist.copy(); h["__prediction_row"] = np.nan
    combined = pd.concat([h, pd.DataFrame(placeholders)], ignore_index=True, sort=False)
    feat = build_feature_frame(combined, schedules=schedules, pbp=pbp, windows=windows, auxiliary=auxiliary)
    unique_rows = feat[feat["__prediction_row"].notna()].sort_values("__prediction_row").reset_index(drop=True)
    return unique_rows.iloc[codes].reset_index(drop=True)


def predict_lines(lines: pd.DataFrame, player_stats: pd.DataFrame, schedules: pd.DataFrame,
                  pbp: pd.DataFrame | None = None, model_root="models",
                  probability_floor=0.57, min_prob_edge=0.04,
                  auxiliary: dict[str, pd.DataFrame] | None = None) -> pd.DataFrame:
    rows = build_prop_rows(player_stats, lines, schedules, pbp=pbp, auxiliary=auxiliary)
    out = lines.reset_index(drop=True).copy()

    out["season"] = rows["season"].astype(int).to_numpy()
    out["week"] = rows["week"].astype(int).to_numpy()
    out["team"] = rows["team"].astype(str).to_numpy()
    out["opponent"] = rows["opponent_team"].astype(str).to_numpy()

    points, probs, sides, edges, labels, algorithms = [], [], [], [], [], []
    for i, line in out.iterrows():
        model = load_model(str(line.prop), root=model_root)
        X = rows.iloc[[i]]
        p_over = float(model.probability_over(X, float(line.line))[0])
        point = float(model.predict_point(X)[0])
        if line.prop == "sack_yes":
            side = "YES"
            p_side = p_over
            offered = line.get("yes_odds", np.nan)
        else:
            side = "OVER" if p_over >= 0.5 else "UNDER"
            p_side = p_over if p_over >= 0.5 else 1 - p_over
            offered = line.get("over_odds" if side == "OVER" else "under_odds", np.nan)
        if line.prop == "sack_yes":
            implied = american_to_implied(offered)
        else:
            fair_over, fair_under = fair_pair(line.get("over_odds", np.nan), line.get("under_odds", np.nan))
            implied = fair_over if side == "OVER" else fair_under
            if np.isnan(implied):
                implied = american_to_implied(offered)
        edge = p_side - implied if not np.isnan(implied) else np.nan
        official = p_side >= probability_floor and (np.isnan(edge) or edge >= min_prob_edge)
        points.append(point); probs.append(p_side); sides.append(side); edges.append(edge); labels.append("BET" if official else "PASS"); algorithms.append(model.estimator_name)
    out["model_point"] = points
    out["pick"] = sides
    out["model_probability"] = probs
    out["probability_edge_vs_price"] = edges
    out["status"] = labels
    out["algorithm"] = algorithms
    out["feature_set"] = [load_model(str(p), root=model_root).feature_set for p in out["prop"]]
    return out
