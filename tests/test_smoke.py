import numpy as np
import pandas as pd

from wshlx_nfl_props.features import build_feature_frame, model_feature_columns
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
                rows.append(dict(
                    season=season,week=week,player_id=name,player_display_name=name,position=pos,team=team,opponent_team=opp,
                    passing_attempts=max(pa,0),completions=max(int(pa*.65),0),
                    passing_yards=max(pa*7+rng.normal(0,35),0) if pos=="QB" else 0,
                    passing_tds=max(rng.poisson(1.7),0) if pos=="QB" else 0,
                    rushing_attempts=max(car,0),rushing_yards=max(car*4.4+rng.normal(0,15),0),
                    targets=max(tgt,0),receptions=max(int(tgt*.68),0),
                    receiving_yards=max(tgt*8.5+rng.normal(0,20),0),
                    def_sacks=int(rng.random()<0.25) if pos=="EDGE" else 0,
                    def_qb_hits=int(rng.random()<0.45) if pos=="EDGE" else 0,
                    def_tackles=rng.integers(1,7)
                ))
    return pd.DataFrame(rows)


def test_feature_selector_blocks_same_game_usage():
    f=build_feature_frame(synthetic(),windows=(3,5))
    numeric,_=model_feature_columns(f,feature_set="baseline")
    forbidden={"pass_attempt_share","carry_share","target_share","tw_pass_att","tw_rush_att","tw_targets"}
    assert forbidden.isdisjoint(numeric)
    assert "target_share_r3" in numeric
    assert "rushing_attempts_lag1" in numeric


def test_current_game_usage_does_not_change_selected_features():
    raw=synthetic()
    f1=build_feature_frame(raw,windows=(3,5))
    mask=(raw.player_id=="WR A") & (raw.season==2025) & (raw.week==9)
    raw2=raw.copy()
    raw2.loc[mask,"targets"]=99
    raw2.loc[mask,"receiving_yards"]=999
    f2=build_feature_frame(raw2,windows=(3,5))
    cols,_=model_feature_columns(f1,feature_set="baseline")
    r1=f1[(f1.player_id=="WR A")&(f1.season==2025)&(f1.week==9)][cols].reset_index(drop=True)
    r2=f2[(f2.player_id=="WR A")&(f2.season==2025)&(f2.week==9)][cols].reset_index(drop=True)
    pd.testing.assert_frame_equal(r1,r2)


def test_train_qb_pass_yards_chronologically():
    f=build_feature_frame(synthetic(),windows=(3,5))
    m=train_prop_model(
        f,"qb_pass_yds","passing_yards",["QB"],"regression",
        selection_season=2024,test_season=2025,min_history_games=3,
        feature_sets=("baseline",)
    )
    assert m.estimator_name
    assert m.feature_set=="baseline"
    assert m.validation_metrics["test_season"]==2025
    assert "test_metrics" in m.validation_metrics
