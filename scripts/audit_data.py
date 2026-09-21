from pathlib import Path
import json
import pandas as pd
from wshlx_nfl_props.data import load_raw


def _scalar(v):
    if pd.isna(v):
        return None
    if hasattr(v, "item"):
        try:
            return v.item()
        except Exception:
            pass
    return v


def summarize(df: pd.DataFrame) -> dict:
    out = {"rows": int(len(df)), "columns": int(len(df.columns))}
    if "season" in df.columns and len(df):
        out["min_season"] = _scalar(pd.to_numeric(df["season"], errors="coerce").min())
        out["max_season"] = _scalar(pd.to_numeric(df["season"], errors="coerce").max())
    if "week" in df.columns and len(df):
        out["max_week"] = _scalar(pd.to_numeric(df["week"], errors="coerce").max())
    return out


d = load_raw()
report = {name: summarize(df) for name, df in sorted(d.items())}
ps = d.get("player_stats")
if ps is not None and not ps.empty and {"season", "week"}.issubset(ps.columns):
    reg = ps.copy()
    if "season_type" in reg.columns:
        reg = reg[reg["season_type"].astype(str).str.upper().isin(["REG", "REGULAR"])]
    max_season = int(pd.to_numeric(reg["season"], errors="coerce").max())
    cur = reg[pd.to_numeric(reg["season"], errors="coerce") == max_season]
    report["current_player_stats"] = {
        "season": max_season,
        "latest_week": int(pd.to_numeric(cur["week"], errors="coerce").max()),
        "rows": int(len(cur)),
        "players": int(cur["player_id"].nunique()) if "player_id" in cur.columns else None,
    }
Path("reports").mkdir(exist_ok=True)
Path("reports/data_audit.json").write_text(json.dumps(report, indent=2, default=str))
print(json.dumps(report, indent=2, default=str))
