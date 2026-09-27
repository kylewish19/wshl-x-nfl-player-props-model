import argparse
from pathlib import Path
import numpy as np
import pandas as pd

from wshlx_nfl_props.data import load_raw
from wshlx_nfl_props.features import normalize_player_stats
from wshlx_nfl_props.predict import build_prop_rows
from wshlx_nfl_props.models import load_model

p=argparse.ArgumentParser()
p.add_argument("--date",required=True)
p.add_argument("--mode",choices=["official","v05"],required=True)
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

ss=stats[(stats.season==season)&stats.team.astype(str).isin(teams)].copy()
for c in ["passing_attempts","rushing_attempts","targets","def_sacks","def_qb_hits"]:
    if c not in ss: ss[c]=0.0
    ss[c]=pd.to_numeric(ss[c],errors="coerce").fillna(0.0)
agg=ss.groupby(["player_id","player_display_name","team","position"],as_index=False)[
    ["passing_attempts","rushing_attempts","targets","def_sacks","def_qb_hits"]
].sum()
agg["position"]=agg.position.astype(str).str.upper()

eligible=agg[
    ((agg.position=="QB") & (agg.passing_attempts>0)) |
    ((agg.position.isin(["RB","FB"])) & ((agg.rushing_attempts>0)|(agg.targets>0))) |
    ((agg.position.isin(["WR"])) & (agg.targets>0)) |
    ((agg.position.isin(["DE","DT","DL","OLB","LB","EDGE"])) & ((agg.def_sacks>0)|(agg.def_qb_hits>0)))
].copy()

# One future row per player, not one per prop.
base=eligible[["player_display_name","team"]].drop_duplicates().copy()
rows=build_prop_rows(stats,base,d["schedules"],pbp=d.get("pbp"),auxiliary=d)

# Match positions/IDs from the generated feature rows.
outputs=[]
specs = {
    "qb_pass_yds":["QB"],
    "qb_pass_tds":["QB"],
    "qb_rush_yds":["QB"],
    "rb_rush_yds":["RB","FB"],
    "rec_yds":["WR","RB","FB"],
    "sack_yes":["DE","DT","DL","OLB","LB","EDGE"],
} if a.mode=="official" else {
    "qb_pass_yds":["QB"],
    "rb_rush_yds":["RB","FB"],
    "rec_yds":["WR","RB","FB"],
}

for prop,positions in specs.items():
    model=load_model(prop,root=a.model_root)
    mask=rows.position.astype(str).str.upper().isin(positions)
    X=rows.loc[mask].copy()
    if X.empty: continue
    for cat_col in getattr(model,"categorical_features",[]):
        if cat_col in X.columns:
            s=X[cat_col].astype(object)
            X[cat_col]=s.where(pd.notna(s),"__MISSING__").astype(str)
    pred=model.predict_point(X)
    for (_,r),point in zip(X.iterrows(),pred):
        outputs.append({
            "player_display_name":r.player_display_name,
            "team":r.team,
            "opponent":r.opponent_team,
            "position":r.position,
            "prop":prop,
            "model_point":float(point),
            "algorithm":model.estimator_name,
            "feature_set":model.feature_set,
        })

out=pd.DataFrame(outputs).sort_values(["team","prop","model_point"],ascending=[True,True,False])
Path(a.out).parent.mkdir(parents=True,exist_ok=True)
out.to_csv(a.out,index=False)
print(out.to_string(index=False))
