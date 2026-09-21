import argparse, json, sys
from pathlib import Path
import pandas as pd

from wshlx_nfl_props.data import load_raw
from wshlx_nfl_props.features import normalize_player_stats

SUPPORTED={"qb_pass_yds","qb_pass_tds","qb_rush_yds","rb_rush_yds","rec_yds","sack_yes"}
REQUIRED={"player_display_name","prop","line"}

p=argparse.ArgumentParser(description="Validate an NFL player-prop card before locking picks.")
p.add_argument("lines_csv")
p.add_argument("--official-root",default="models/official")
p.add_argument("--shadow-roots",nargs="*",default=["models/challenger_v04","models/challenger_v05"])
a=p.parse_args()

issues=[]; warnings=[]
lines=pd.read_csv(a.lines_csv)
missing=REQUIRED-set(lines.columns)
if missing:
    issues.append(f"Missing required columns: {sorted(missing)}")

bad_props=sorted(set(lines.get("prop",pd.Series(dtype=str)).dropna())-SUPPORTED)
if bad_props:
    issues.append(f"Unsupported prop names: {bad_props}")

d=load_raw()
if "player_stats" not in d or d["player_stats"].empty:
    issues.append("No player_stats data. Run scripts/update_data.py.")
else:
    stats=normalize_player_stats(d["player_stats"])
    max_season=int(stats.season.max())
    cur=stats[stats.season==max_season]
    max_week=int(cur.week.max())
    print(f"[preflight] player data through {max_season} Week {max_week}")
    names={str(x).lower() for x in stats.player_display_name.dropna().unique()}
    for name in lines.get("player_display_name",pd.Series(dtype=str)).dropna():
        if str(name).lower() not in names:
            issues.append(f"Player not found in historical/current data: {name}")

    if "schedules" in d and not d["schedules"].empty:
        s=d["schedules"].copy()
        s=s[s.season==max_season]
        if "game_type" in s:
            s=s[s.game_type.astype(str).str.upper().isin(["REG","REGULAR"])]
        if {"home_score","away_score"}.issubset(s.columns):
            completed=s[s.home_score.notna() & s.away_score.notna()]
            if not completed.empty:
                sched_week=int(pd.to_numeric(completed.week,errors="coerce").max())
                if max_week < sched_week:
                    issues.append(f"Player stats lag completed schedule: stats Week {max_week}, schedule completed through Week {sched_week}.")

for prop in sorted(set(lines.get("prop",pd.Series(dtype=str)).dropna())):
    if not (Path(a.official_root)/f"{prop}.joblib").exists():
        issues.append(f"Official model missing for {prop}: {a.official_root}/{prop}.joblib")

for root in a.shadow_roots:
    rp=Path(root)
    if not rp.exists():
        warnings.append(f"Shadow model root not present locally: {root}")
        continue
    for prop in sorted(set(lines.get("prop",pd.Series(dtype=str)).dropna())):
        if prop in {"qb_pass_yds","rb_rush_yds","rec_yds"} and "v05" in root and not (rp/f"{prop}.joblib").exists():
            warnings.append(f"v0.5 shadow unavailable for {prop}")
        if "v04" in root and not (rp/f"{prop}.joblib").exists():
            warnings.append(f"v0.4 shadow unavailable for {prop}")

for _,r in lines.iterrows():
    prop=r.get("prop")
    if prop=="sack_yes":
        if "yes_odds" not in lines.columns or pd.isna(r.get("yes_odds")):
            warnings.append(f"Missing YES odds for sack prop: {r.get('player_display_name')}")
    else:
        if not (("over_odds" in lines.columns and pd.notna(r.get("over_odds"))) and ("under_odds" in lines.columns and pd.notna(r.get("under_odds")))):
            warnings.append(f"Need both Over and Under prices for clean vig/edge check: {r.get('player_display_name')} {prop}")

report={"ok":not issues,"issues":issues,"warnings":sorted(set(warnings))}
print(json.dumps(report,indent=2))
sys.exit(0 if not issues else 2)
