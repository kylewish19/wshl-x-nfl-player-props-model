import numpy as np
import pandas as pd

from wshlx_nfl_props.game_features import build_game_frame
from wshlx_nfl_props.game_predict import fair_pair


def _schedule():
    return pd.DataFrame([
        {"season":2025,"week":1,"home_team":"A","away_team":"B","home_score":24,"away_score":17,"game_type":"REG"},
        {"season":2025,"week":2,"home_team":"B","away_team":"A","home_score":20,"away_score":21,"game_type":"REG"},
        {"season":2025,"week":3,"home_team":"A","away_team":"B","home_score":np.nan,"away_score":np.nan,"game_type":"REG"},
    ])


def test_game_features_do_not_use_same_week_team_result():
    stats=pd.DataFrame([
        {"season":2025,"week":1,"team":"A","passing_yards":200.0},
        {"season":2025,"week":1,"team":"B","passing_yards":180.0},
        {"season":2025,"week":2,"team":"A","passing_yards":250.0},
        {"season":2025,"week":2,"team":"B","passing_yards":220.0},
    ])
    a=build_game_frame(_schedule(),stats,None,windows=(3,))
    stats2=stats.copy()
    stats2.loc[(stats2.team=="A")&(stats2.week==2),"passing_yards"]=9999.0
    b=build_game_frame(_schedule(),stats2,None,windows=(3,))

    # Week 2 pregame feature can only see Week 1, so changing Week 2's result must not move it.
    ca=a[(a.week==2)&(a.home_team=="B")].iloc[0]
    cb=b[(b.week==2)&(b.home_team=="B")].iloc[0]
    assert ca["away_stat_passing_yards_r3"] == cb["away_stat_passing_yards_r3"]

    # Week 3 is allowed to use the completed Week 2 result, so it should move.
    ca3=a[(a.week==3)&(a.home_team=="A")].iloc[0]
    cb3=b[(b.week==3)&(b.home_team=="A")].iloc[0]
    assert ca3["home_stat_passing_yards_r3"] != cb3["home_stat_passing_yards_r3"]


def test_devig_pair_sums_to_one():
    a,b=fair_pair(-110,-110)
    assert abs((a+b)-1.0)<1e-12
    assert abs(a-0.5)<1e-12
