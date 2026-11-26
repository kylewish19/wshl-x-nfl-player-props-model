import pandas as pd
from wshlx_nfl_props.matchup import add_matchup_context

def test_future_and_bye_weeks_use_completed_games_only():
    p=pd.DataFrame([dict(season=2026,week=w,posteam=t,defteam=o,pass_attempt=n,rush_attempt=1,sack=0,qb_hit=0) for w,t,o,n in [(1,"AAA","BBB",10),(1,"BBB","AAA",20),(3,"AAA","BBB",30),(3,"BBB","AAA",40),(6,"AAA","BBB",90),(6,"BBB","AAA",100)]])
    base=pd.DataFrame(dict(season=[2026]*5,week=[3,4,5,6,4],team=["AAA"]*5,opponent_team=["BBB"]*5))
    out=add_matchup_context(base,p,windows=(3,))
    assert len(out)==len(base)
    assert out.teamctx_team_dropbacks_r3.tolist()==[10.,20.,20.,20.,20.]
    assert out.oppctx_team_dropbacks_r3.tolist()==[20.,30.,30.,30.,30.]
    pd.testing.assert_frame_equal(out[base.columns],base)

def test_cross_season_and_aliases():
    p=pd.DataFrame([dict(season=2025,week=18,posteam="LA",defteam="JAC",pass_attempt=35,rush_attempt=1,sack=2,qb_hit=4),dict(season=2025,week=18,posteam="JAC",defteam="LA",pass_attempt=25,rush_attempt=1,sack=1,qb_hit=2)])
    base=pd.DataFrame(dict(season=[2026,2025],week=[1,18],team=["LAR","LAR"],opponent_team=["JAX","JAX"]))
    out=add_matchup_context(base,p,windows=(3,))
    assert out.loc[0,"teamctx_team_dropbacks_r3"]==35
    assert out.loc[0,"oppctx_sacks_allowed_r3"]==1
    assert pd.isna(out.loc[1,"teamctx_team_dropbacks_r3"])
