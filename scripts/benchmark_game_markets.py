from pathlib import Path
import json
import numpy as np
import pandas as pd
from sklearn.metrics import mean_absolute_error, brier_score_loss
from wshlx_nfl_props.data import load_raw
from wshlx_nfl_props.game_predict import fair_pair

d=load_raw()
s=d["schedules"].copy()
if "game_type" in s.columns:
    s=s[s.game_type.astype(str).str.upper().isin(["REG","REGULAR"])]
s=s[s.season.isin([2023,2024,2025])].copy()
s["actual_margin"]=pd.to_numeric(s["home_score"],errors="coerce")-pd.to_numeric(s["away_score"],errors="coerce")
s["actual_total"]=pd.to_numeric(s["home_score"],errors="coerce")+pd.to_numeric(s["away_score"],errors="coerce")
report={"description":"Historical nflverse closing-market benchmark; not assumed to be FanDuel-specific.","years":{}}

for year in [2023,2024,2025]:
    y=s[s.season==year].copy()
    out={"games":int(y["actual_margin"].notna().sum())}
    if "spread_line" in y:
        z=y[y.actual_margin.notna() & y.spread_line.notna()]
        if len(z):
            # nflverse spread_line is the home-team closing margin convention: positive means home favored.
            out["closing_spread_margin_mae"]=float(mean_absolute_error(z.actual_margin,pd.to_numeric(z.spread_line)))
            out["closing_spread_rows"]=int(len(z))
    if "total_line" in y:
        z=y[y.actual_total.notna() & y.total_line.notna()]
        if len(z):
            out["closing_total_mae"]=float(mean_absolute_error(z.actual_total,pd.to_numeric(z.total_line)))
            out["closing_total_rows"]=int(len(z))
    if {"home_moneyline","away_moneyline"}.issubset(y.columns):
        z=y[y.actual_margin.notna() & y.home_moneyline.notna() & y.away_moneyline.notna() & (y.actual_margin!=0)].copy()
        if len(z):
            probs=[]
            for _,r in z.iterrows():
                ph,_=fair_pair(r.home_moneyline,r.away_moneyline)
                probs.append(ph)
            target=(z.actual_margin>0).astype(int).to_numpy()
            out["closing_moneyline_brier"]=float(brier_score_loss(target,np.asarray(probs)))
            out["closing_moneyline_rows"]=int(len(z))
    report["years"][str(year)]=out

Path("reports").mkdir(exist_ok=True)
Path("reports/game_market_closing_benchmarks.json").write_text(json.dumps(report,indent=2))
print(json.dumps(report,indent=2))
