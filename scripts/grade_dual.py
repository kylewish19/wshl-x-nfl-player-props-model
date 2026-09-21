import argparse
from pathlib import Path
import pandas as pd
from wshlx_nfl_props.data import load_raw
from wshlx_nfl_props.grade import grade_locked

p=argparse.ArgumentParser()
p.add_argument("dual_locked_csv")
p.add_argument("--outdir",default="grading")
a=p.parse_args()
d=load_raw(); dual=pd.read_csv(a.dual_locked_csv)

base_cols=["player_display_name","prop","line","season","week","team","opponent"]
cards={}
for label in ["official","shadow"]:
    card=dual[base_cols].copy()
    card["pick"]=dual[f"{label}_pick"]
    card["status"]=dual[f"{label}_status"]
    card["model_probability"]=dual[f"{label}_model_probability"]
    card["model_point"]=dual[f"{label}_model_point"]
    graded,summary=grade_locked(card,d["player_stats"])
    graded["model"]=label
    summary["model"]=label
    cards[label]=(graded,summary)

graded=pd.concat([cards["official"][0],cards["shadow"][0]],ignore_index=True)
summary=pd.concat([cards["official"][1],cards["shadow"][1]],ignore_index=True)
bets=graded[graded["status"].eq("BET") & graded["grade"].isin(["WIN","LOSS","PUSH"])].copy()
bet_summary=bets.groupby(["model","prop","grade"]).size().unstack(fill_value=0).reset_index()
for c in ["WIN","LOSS","PUSH"]:
    if c not in bet_summary: bet_summary[c]=0
bet_summary["hit_rate"]=bet_summary["WIN"]/(bet_summary["WIN"]+bet_summary["LOSS"]).replace(0,pd.NA)

Path(a.outdir).mkdir(parents=True,exist_ok=True)
graded.to_csv(Path(a.outdir)/"dual_graded.csv",index=False)
summary.to_csv(Path(a.outdir)/"dual_all_props_summary.csv",index=False)
bet_summary.to_csv(Path(a.outdir)/"dual_bet_summary.csv",index=False)
print(bet_summary.to_string(index=False))
