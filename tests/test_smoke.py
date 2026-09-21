import numpy as np
import pandas as pd
from wshlx_nfl_props.features import build_feature_frame
from wshlx_nfl_props.models import train_prop_model


def synthetic():
    rng=np.random.default_rng(42); rows=[]
    players=[("QB A","QB","AAA"),("QB B","QB","BBB"),("RB A","RB","AAA"),("RB B","RB","BBB"),("WR A","WR","AAA"),("WR B","WR","BBB"),("EDGE A","EDGE","AAA"),("EDGE B","EDGE","BBB")]
    for season in range(2021,2026):
        for week in range(1,10):
            for j,(name,pos,team) in enumerate(players):
                opp="BBB" if team=="AAA" else "AAA"
                pa=30+rng.integers(-5,6) if pos=="QB" else 0
                car=(12+rng.integers(-4,5)) if pos=="RB" else (4+rng.integers(0,4) if pos=="QB" else 0)
                tgt=(8+rng.integers(-3,4)) if pos=="WR" else (4+rng.integers(-2,3) if pos=="RB" else 0)
                rows.append(dict(season=season,week=week,player_id=name,player_display_name=name,position=pos,team=team,opponent_team=opp,
                                 passing_attempts=max(pa,0),completions=max(int(pa*.65),0),passing_yards=max(pa*7+rng.normal(0,35),0) if pos=="QB" else 0,
                                 passing_tds=max(rng.poisson(1.7),0) if pos=="QB" else 0,rushing_attempts=max(car,0),rushing_yards=max(car*4.4+rng.normal(0,15),0),
                                 targets=max(tgt,0),receptions=max(int(tgt*.68),0),receiving_yards=max(tgt*8.5+rng.normal(0,20),0),
                                 def_sacks=int(rng.random()<0.25) if pos=="EDGE" else 0,def_qb_hits=int(rng.random()<0.45) if pos=="EDGE" else 0,def_tackles=rng.integers(1,7)))
    return pd.DataFrame(rows)


def test_train_qb_pass_yards():
    f=build_feature_frame(synthetic(),windows=(3,5))
    m=train_prop_model(f,"qb_pass_yds","passing_yards",["QB"],"regression",holdout_season=2025,min_history_games=3)
    assert m.estimator_name
    assert "selected" in m.validation_metrics
