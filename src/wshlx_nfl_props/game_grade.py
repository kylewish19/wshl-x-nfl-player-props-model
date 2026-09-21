from __future__ import annotations

import numpy as np
import pandas as pd


def _result(value: float) -> str:
    if value > 0:
        return "WIN"
    if value < 0:
        return "LOSS"
    return "PUSH"


def grade_game_markets(locked: pd.DataFrame, schedules: pd.DataFrame):
    s=schedules.copy()
    if "game_type" in s.columns:
        s=s[s.game_type.astype(str).str.upper().isin(["REG","REGULAR"])]
    actual=s[["season","week","home_team","away_team","home_score","away_score"]].copy()
    g=locked.merge(actual,on=["season","week","home_team","away_team"],how="left",validate="many_to_one")
    g["actual_margin"]=pd.to_numeric(g["home_score"],errors="coerce")-pd.to_numeric(g["away_score"],errors="coerce")
    g["actual_total"]=pd.to_numeric(g["home_score"],errors="coerce")+pd.to_numeric(g["away_score"],errors="coerce")

    spread_results=[]; ml_results=[]; total_results=[]
    for _,r in g.iterrows():
        if pd.isna(r.get("home_score")) or pd.isna(r.get("away_score")):
            spread_results.append("PENDING"); ml_results.append("PENDING"); total_results.append("PENDING"); continue

        if r.get("spread_status")!="BET" or r.get("spread_pick") in [None,"N/A"] or pd.isna(r.get("home_spread")):
            spread_results.append("PASS")
        else:
            hs=float(r.home_spread)
            if str(r.spread_pick).startswith(str(r.home_team)):
                spread_results.append(_result(float(r.actual_margin)+hs))
            else:
                spread_results.append(_result(-float(r.actual_margin)-hs))

        if r.get("ml_status")!="BET":
            ml_results.append("PASS")
        elif float(r.home_score)==float(r.away_score):
            # FanDuel football moneyline ties are void/refunded.
            ml_results.append("PUSH")
        elif str(r.ml_pick)==str(r.home_team):
            ml_results.append("WIN" if float(r.home_score)>float(r.away_score) else "LOSS")
        else:
            ml_results.append("WIN" if float(r.away_score)>float(r.home_score) else "LOSS")

        if r.get("total_status")!="BET" or pd.isna(r.get("total_line")):
            total_results.append("PASS")
        else:
            line=float(r.total_line)
            side=str(r.total_pick)
            delta=float(r.actual_total)-line
            total_results.append(_result(delta if side.startswith("OVER") else -delta))

    g["spread_result"]=spread_results
    g["ml_result"]=ml_results
    g["total_result"]=total_results

    rows=[]
    for market in ["spread","ml","total"]:
        col=f"{market}_result"
        z=g[g[col].isin(["WIN","LOSS","PUSH"])]
        wins=int((z[col]=="WIN").sum()); losses=int((z[col]=="LOSS").sum()); pushes=int((z[col]=="PUSH").sum())
        rows.append({
            "market":market,
            "bets":wins+losses+pushes,
            "wins":wins,"losses":losses,"pushes":pushes,
            "win_rate_ex_push": wins/(wins+losses) if (wins+losses) else np.nan,
        })
    return g,pd.DataFrame(rows)
