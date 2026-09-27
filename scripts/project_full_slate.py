import argparse
from pathlib import Path
import pandas as pd
import numpy as np

from wshlx_nfl_props.data import load_raw
from wshlx_nfl_props.features import normalize_player_stats
from wshlx_nfl_props.predict import predict_lines

p=argparse.ArgumentParser()
p.add_argument("--date",required=True)
p.add_argument("--mode",choices=["official","v05","addon"],default="official")
p.add_argument("--model-root",required=True)
p.add_argument("--out",required=True)
a=p.parse_args()

d=load_raw()
stats=normalize_player_stats(d["player_stats"])
sched=d["schedules"].copy()
if "game_type" in sched.columns:
    sched=sched[sched.game_type.astype(str).str.upper().isin(["REG","REGULAR"])]
date_col="gameday" if "gameday" in sched.columns else ("game_date" if "game_date" in sched.columns else None)
if date_col is None:
    raise RuntimeError("Schedule has no date column")
games=sched[sched[date_col].astype(str).str[:10]==a.date].copy()
if games.empty:
    raise RuntimeError(f"No games found for {a.date}")
teams=set(games.home_team.astype(str))|set(games.away_team.astype(str))

season=int(stats.season.max())
s=stats[(stats.season==season)&(stats.team.astype(str).isin(teams))].copy()
s=s.sort_values(["player_id","week"]).groupby("player_id",as_index=False).tail(1)
s["position"]=s.position.astype(str).str.upper()

rows=[]
def add_players(df,prop,line=0.0):
    for _,r in df.iterrows():
        rows.append({
            "player_display_name":r.player_display_name,
            "team":r.team,
            "prop":prop,
            "line":line,
            "over_odds":np.nan,
            "under_odds":np.nan,
            "yes_odds":np.nan,
        })

if a.mode=="official":
    qb=s[s.position.eq("QB")]
    rb=s[s.position.isin(["RB","FB"])]
    rec=s[s.position.isin(["WR","RB","FB"])]
    de=s[s.position.isin(["DE","DT","DL","OLB","LB","EDGE"])]
    add_players(qb,"qb_pass_yds")
    add_players(qb,"qb_pass_tds",0.5)
    add_players(qb,"qb_rush_yds")
    add_players(rb,"rb_rush_yds")
    add_players(rec,"rec_yds")
    add_players(de,"sack_yes",0.5)
elif a.mode=="v05":
    qb=s[s.position.eq("QB")]
    rb=s[s.position.isin(["RB","FB"])]
    rec=s[s.position.isin(["WR","RB","FB"])]
    add_players(qb,"qb_pass_yds")
    add_players(rb,"rb_rush_yds")
    add_players(rec,"rec_yds")
else:
    eligible=s[s.position.isin(["WR","RB","FB","TE"])]
    te=s[s.position.eq("TE")]
    add_players(eligible,"receptions")
    add_players(te,"te_rec_yds")

lines=pd.DataFrame(rows).drop_duplicates(["player_display_name","team","prop"])
out=predict_lines(
    lines,stats,d["schedules"],d.get("pbp"),
    model_root=a.model_root,probability_floor=0.0,min_prob_edge=-1.0,auxiliary=d
)
# Keep point projections; dummy-line side/probability are not betting recommendations.
keep=["player_display_name","team","opponent","prop","model_point","algorithm","feature_set"]
out=out[keep].sort_values(["team","prop","model_point"],ascending=[True,True,False])
Path(a.out).parent.mkdir(parents=True,exist_ok=True)
out.to_csv(a.out,index=False)
print(out.to_string(index=False))
