import pandas as pd
import pytest
from wshlx_nfl_props.reporting import build_cards

def row(name, prop, p, odds, pick="OVER", **kwargs):
    out=dict(season=2026,week=12,team="DET",player_display_name=name,prop=prop,line=1.5,pick=pick,model_probability=p,status="BET",validity_ok=True,context_ok=True)
    out[{"OVER":"over_odds","UNDER":"under_odds","YES":"yes_odds"}[pick]]=odds
    out.update(kwargs)
    return out

def test_all_prop_families_ranked_without_price_or_context():
    x=pd.DataFrame([row("WR","rec_yds",.58,-110),row("QB","qb_pass_tds",.61,-210,context_ok=False),row("DL","sack_yes",.57,-205,pick="YES"),row("RB","rb_rush_yds",.95,-110,validity_ok=False)])
    cards=build_cards(x,top_n=2)
    assert cards["top10_accuracy"].player_display_name.tolist()==["QB","WR"]
    assert cards["top10_accuracy"]["rank"].tolist()==[1,2]
    assert len(cards["raw_bets"])==4
    assert cards["official_value"].player_display_name.tolist()==["WR"]

def test_missing_price_and_negative_ev_not_official():
    x=pd.DataFrame([row("A","rec_yds",.8,None),row("B","rec_yds",.6,-200),row("C","rec_yds",.6,110),row("D","rec_yds",.7,-110,pick="UNDER"),row("E","rec_yds",.8,0)])
    cards=build_cards(x)
    assert len(cards["raw_bets"])==5
    assert cards["official_value"].player_display_name.tolist()==["C","D"]
    assert cards["audit"].loc[1,"expected_units_per_unit_risked"]<0

def test_duplicate_or_implicit_context_rejected():
    x=pd.DataFrame([row("A","rec_yds",.6,-110)]*2)
    with pytest.raises(ValueError,match="duplicate"): build_cards(x)
    with pytest.raises(ValueError,match="Missing required"): build_cards(x.iloc[:1].drop(columns="context_ok"))


def test_shadow_disagreement_retained_in_raw_not_official():
    x=pd.DataFrame([row("A","rec_yds",.7,-110,v05_agreement=False),row("B","rb_rush_yds",.7,-110,v05_agreement=True)])
    cards=build_cards(x)
    assert len(cards["raw_bets"])==2
    assert cards["official_value"].player_display_name.tolist()==["B"]
