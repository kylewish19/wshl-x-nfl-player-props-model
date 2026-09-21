from __future__ import annotations

from pathlib import Path
import numpy as np
import pandas as pd

from .game_features import build_game_frame
from .game_models import load_game_model


def american_to_implied(odds):
    if pd.isna(odds):
        return np.nan
    odds=float(odds)
    return 100/(odds+100) if odds>0 else (-odds)/((-odds)+100)


def fair_pair(odds_a, odds_b):
    a=american_to_implied(odds_a)
    b=american_to_implied(odds_b)
    if np.isnan(a) or np.isnan(b) or (a+b)<=0:
        return np.nan,np.nan
    return a/(a+b), b/(a+b)


def _choose(prob_a, prob_b, fair_a, fair_b, label_a, label_b, probability_floor, min_edge):
    if prob_a>=prob_b:
        pick=label_a; prob=prob_a; fair=fair_a
    else:
        pick=label_b; prob=prob_b; fair=fair_b
    edge=prob-fair if not np.isnan(fair) else np.nan
    bet=prob>=probability_floor and (np.isnan(edge) or edge>=min_edge)
    return pick,float(prob),float(edge) if not np.isnan(edge) else np.nan,("BET" if bet else "PASS")


def _match_game_row(frame: pd.DataFrame, line: pd.Series) -> pd.DataFrame:
    season=line.get("season",np.nan)
    week=line.get("week",np.nan)
    home=str(line["home_team"])
    away=str(line["away_team"])
    q=frame[(frame.home_team==home)&(frame.away_team==away)].copy()
    if not pd.isna(season):
        q=q[q.season==int(season)]
    else:
        q=q[q.season==frame.season.max()]
    if not pd.isna(week):
        q=q[q.week==int(week)]
    if q.empty:
        raise KeyError(f"Game not found in schedule: {away} at {home}")
    # Prefer the next unplayed matching game when week was omitted.
    if "home_score" in q and "away_score" in q and pd.isna(week):
        future=q[q.home_score.isna() | q.away_score.isna()]
        if not future.empty:
            q=future
    return q.sort_values(["season","week"]).iloc[[-1 if not pd.isna(week) else 0]]


def predict_game_lines(lines: pd.DataFrame, schedules: pd.DataFrame,
                       team_stats: pd.DataFrame | None=None, pbp: pd.DataFrame | None=None,
                       model_root="models/game_markets", probability_floor=0.57,
                       min_prob_edge=0.04, windows=(3,5,8)) -> pd.DataFrame:
    frame=build_game_frame(schedules,team_stats,pbp,windows=windows)
    out=lines.reset_index(drop=True).copy()

    rows=[]
    for _,line in out.iterrows():
        rows.append(_match_game_row(frame,line).iloc[0])
    X=pd.DataFrame(rows).reset_index(drop=True)

    models={m:load_game_model(m,root=model_root) for m in ["home_points","away_points","margin","total","home_win"]}
    home_pts=models["home_points"].predict(X)
    away_pts=models["away_points"].predict(X)
    margin=models["margin"].predict(X)
    total=models["total"].predict(X)
    home_win=models["home_win"].predict(X)

    out["season"]=X["season"].astype(int).to_numpy()
    out["week"]=X["week"].astype(int).to_numpy()
    out["model_home_points"]=home_pts
    out["model_away_points"]=away_pts
    out["model_score_margin"]=home_pts-away_pts
    out["model_score_total"]=home_pts+away_pts
    out["model_margin"]=margin
    out["model_total"]=total
    out["model_home_win_probability"]=home_win

    spread_picks=[]; spread_probs=[]; spread_edges=[]; spread_status=[]
    ml_picks=[]; ml_probs=[]; ml_edges=[]; ml_status=[]
    total_picks=[]; total_probs=[]; total_edges=[]; total_status=[]

    for i,line in out.iterrows():
        Xi=X.iloc[[i]]

        # Spread: home spread -3.5 means home must win by >3.5, so threshold is +3.5 margin.
        hs=line.get("home_spread",np.nan)
        if pd.isna(hs):
            spread_picks.append("N/A"); spread_probs.append(np.nan); spread_edges.append(np.nan); spread_status.append("PASS")
        else:
            p_home=float(models["margin"].probability_over(Xi,-float(hs))[0])
            p_away=1-p_home
            fair_home,fair_away=fair_pair(line.get("home_spread_odds",np.nan),line.get("away_spread_odds",np.nan))
            a,b,c,d=_choose(p_home,p_away,fair_home,fair_away,
                            f'{line.home_team} {float(hs):+g}',f'{line.away_team} {-float(hs):+g}',
                            probability_floor,min_prob_edge)
            spread_picks.append(a); spread_probs.append(b); spread_edges.append(c); spread_status.append(d)

        # Moneyline
        p_home=float(home_win[i]); p_away=1-p_home
        fair_home,fair_away=fair_pair(line.get("home_ml",np.nan),line.get("away_ml",np.nan))
        a,b,c,d=_choose(p_home,p_away,fair_home,fair_away,
                        str(line.home_team),str(line.away_team),probability_floor,min_prob_edge)
        ml_picks.append(a); ml_probs.append(b); ml_edges.append(c); ml_status.append(d)

        # Total
        tl=line.get("total_line",np.nan)
        if pd.isna(tl):
            total_picks.append("N/A"); total_probs.append(np.nan); total_edges.append(np.nan); total_status.append("PASS")
        else:
            p_over=float(models["total"].probability_over(Xi,float(tl))[0])
            p_under=1-p_over
            fair_over,fair_under=fair_pair(line.get("over_odds",np.nan),line.get("under_odds",np.nan))
            a,b,c,d=_choose(p_over,p_under,fair_over,fair_under,
                            f'OVER {float(tl):g}',f'UNDER {float(tl):g}',
                            probability_floor,min_prob_edge)
            total_picks.append(a); total_probs.append(b); total_edges.append(c); total_status.append(d)

    out["spread_pick"]=spread_picks
    out["spread_probability"]=spread_probs
    out["spread_edge_vs_devig"]=spread_edges
    out["spread_status"]=spread_status
    out["ml_pick"]=ml_picks
    out["ml_probability"]=ml_probs
    out["ml_edge_vs_devig"]=ml_edges
    out["ml_status"]=ml_status
    out["total_pick"]=total_picks
    out["total_probability"]=total_probs
    out["total_edge_vs_devig"]=total_edges
    out["total_status"]=total_status
    return out
