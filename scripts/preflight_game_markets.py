import argparse, json, sys
from pathlib import Path
import pandas as pd
import yaml

from wshlx_nfl_props.data import load_raw

REQUIRED={"away_team","home_team","home_spread","home_spread_odds","away_spread_odds",
          "home_ml","away_ml","total_line","over_odds","under_odds"}

p=argparse.ArgumentParser(description="Validate NFL spread/ML/total inputs before locking.")
p.add_argument("lines_csv")
p.add_argument("--model-root",default="models/game_markets")
a=p.parse_args()

lines=pd.read_csv(a.lines_csv)
d=load_raw()
issues=[]; warnings=[]

missing=REQUIRED-set(lines.columns)
if missing:
    issues.append(f"Missing required columns: {sorted(missing)}")

if "schedules" not in d or d["schedules"].empty:
    issues.append("Schedule data missing. Refresh nflverse data first.")
else:
    s=d["schedules"].copy()
    if "game_type" in s.columns:
        s=s[s.game_type.astype(str).str.upper().isin(["REG","REGULAR"])]
    for _,r in lines.iterrows():
        q=s[(s.home_team.astype(str)==str(r.get("home_team")))&
            (s.away_team.astype(str)==str(r.get("away_team")))]
        if "season" in lines.columns and pd.notna(r.get("season")):
            q=q[q.season==int(r.season)]
        if "week" in lines.columns and pd.notna(r.get("week")):
            q=q[q.week==int(r.week)]
        if q.empty:
            issues.append(f"Schedule matchup not found: {r.get('away_team')} at {r.get('home_team')}")

    completed=s[s["home_score"].notna() & s["away_score"].notna()] if {"home_score","away_score"}.issubset(s.columns) else pd.DataFrame()
    if not completed.empty:
        latest_season=int(completed.season.max())
        latest_sched_week=int(completed[completed.season==latest_season].week.max())
        if "team_stats" in d and not d["team_stats"].empty:
            ts=d["team_stats"]
            cur=ts[ts.season==latest_season]
            if not cur.empty:
                latest_stats_week=int(cur.week.max())
                if latest_stats_week<latest_sched_week:
                    issues.append(
                        f"Team stats lag completed schedule: team stats Week {latest_stats_week}, "
                        f"schedule completed through Week {latest_sched_week}."
                    )

for m in ["home_points","away_points","margin","total","home_win"]:
    if not (Path(a.model_root)/f"{m}.joblib").exists():
        issues.append(f"Missing trained game model: {a.model_root}/{m}.joblib")

for _,r in lines.iterrows():
    pairs=[
        ("spread",r.get("home_spread_odds"),r.get("away_spread_odds")),
        ("moneyline",r.get("home_ml"),r.get("away_ml")),
        ("total",r.get("over_odds"),r.get("under_odds")),
    ]
    for market,x,y in pairs:
        if pd.isna(x) or pd.isna(y):
            warnings.append(
                f"{r.get('away_team')} at {r.get('home_team')}: both {market} prices are needed for de-vigged edge."
            )

cfg=yaml.safe_load(Path("config/game_model_config.yaml").read_text())
report={
    "ok":not issues,
    "game_model_version":cfg["version"],
    "market_informed_features":cfg["market_informed_features"],
    "issues":issues,
    "warnings":sorted(set(warnings)),
}
print(json.dumps(report,indent=2))
sys.exit(0 if not issues else 2)
