import argparse
from pathlib import Path
import pandas as pd
import numpy as np

from wshlx_nfl_props.data import load_raw
from wshlx_nfl_props.features import normalize_player_stats
from wshlx_nfl_props.predict import predict_lines

p=argparse.ArgumentParser()
p.add_argument("--date",required=True)
p.add_argument("--mode",choices=["official","v05"],default="official")
p.add_argument("--model-root",required=True)
p.add_argument("--out",required=True)
a=p.parse_args()

d=load_raw()
stats=normalize_player_stats(d["player_stats"])
sched=d["schedules"].copy()
if "game_type" in sched.columns:
    sched=sched[sched.game_type.astype(str).str.upper().isin(["REG","REGULAR"])]
date_col="gameday" if "gameday" in sched.columns else ("game_date" if "game_date" in sched.columns else None)
games=sched[sched[date_col].astype(str).str[:10]==a.date].copy()
teams=set(games.home_team.astype(str))|set(games.away_team.astype(str))

season=int(stats.season.max())
ss=stats[(stats.season==season)&(stats.team.astype(str).isin(teams))].copy()
usage_cols=[c for c in ["passing_attempts","rushing_attempts","targets","def_sacks","def_qb_hits"] if c in ss.columns]
for c in usage_cols:
    ss[c]=pd.to_numeric(ss[c],errors="coerce").fillna(0)
agg=ss.groupby(["player_id","player_display_name","team","position"],as_index=False)[usage_cols].sum()
agg["position"]=agg.position.astype(str).str.upper()

rows=[]
def add(df,prop,line=0.0):
    for _,r in df.iterrows():
        rows.append({"player_display_name":r.player_display_name,"team":r.team,"prop":prop,"line":line,
                     "over_odds":np.nan,"under_odds":np.nan,"yes_odds":np.nan})

qb=agg[agg.position.eq("QB")]
rb=agg[agg.position.isin(["RB","FB"])]
rec=agg[agg.position.isin(["WR","RB","FB"])]
de=agg[agg.position.isin(["DE","DT","DL","OLB","LB","EDGE"])]

if "rushing_attempts" in agg:
    rb=rb[(rb.rushing_attempts>0) | (rb.get("targets",0)>0)]
if "targets" in agg:
    rec=rec[rec.targets>0]
if "def_sacks" in agg and "def_qb_hits" in agg:
    de=de[(de.def_sacks>0)|(de.def_qb_hits>0)]

if a.mode=="official":
    add(qb,"qb_pass_yds"); add(qb,"qb_pass_tds",0.5); add(qb,"qb_rush_yds")
    add(rb,"rb_rush_yds"); add(rec,"rec_yds"); add(de,"sack_yes",0.5)
else:
    add(qb,"qb_pass_yds"); add(rb,"rb_rush_yds"); add(rec,"rec_yds")

lines=pd.DataFrame(rows).drop_duplicates(["player_display_name","team","prop"])
out=predict_lines(lines,stats,d["schedules"],d.get("pbp"),model_root=a.model_root,
                  probability_floor=0.0,min_prob_edge=-1.0,auxiliary=d)
out=out[["player_display_name","team","opponent","prop","model_point","algorithm","feature_set"]]
Path(a.out).parent.mkdir(parents=True,exist_ok=True)
out.to_csv(a.out,index=False)
print(out.to_string(index=False))
