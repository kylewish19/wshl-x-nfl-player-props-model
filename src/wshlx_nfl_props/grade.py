from __future__ import annotations

import pandas as pd
from .features import normalize_player_stats

TARGETS = {
    "qb_pass_yds": "passing_yards",
    "qb_pass_tds": "passing_tds",
    "qb_rush_yds": "rushing_yards",
    "rb_rush_yds": "rushing_yards",
    "rec_yds": "receiving_yards",
    "sack_yes": "def_sacks",
}


def grade_locked(picks: pd.DataFrame, player_stats: pd.DataFrame) -> tuple[pd.DataFrame, pd.DataFrame]:
    stats = normalize_player_stats(player_stats)
    rows = []
    for _, p in picks.iterrows():
        q = stats[(stats.season == p.season) & (stats.week == p.week) &
                  (stats.player_display_name.str.lower() == str(p.player_display_name).lower())]
        r = p.to_dict()
        if q.empty:
            r.update(actual=None, grade="UNRESOLVED")
        else:
            target = TARGETS[p.prop]
            actual = float(q.iloc[-1][target])
            if p.prop == "sack_yes":
                win = actual > 0
                grade = "WIN" if win else "LOSS"
            else:
                line = float(p.line)
                if actual == line:
                    grade = "PUSH"
                else:
                    win = (actual > line and p.pick == "OVER") or (actual < line and p.pick == "UNDER")
                    grade = "WIN" if win else "LOSS"
            r.update(actual=actual, grade=grade)
        rows.append(r)
    graded = pd.DataFrame(rows)
    decided = graded[graded.grade.isin(["WIN", "LOSS", "PUSH"])]
    summary = decided.groupby("prop").grade.value_counts().unstack(fill_value=0)
    for c in ["WIN", "LOSS", "PUSH"]:
        if c not in summary: summary[c] = 0
    summary["win_rate"] = summary.WIN / (summary.WIN + summary.LOSS).replace(0, pd.NA)
    summary = summary.reset_index()
    return graded, summary
