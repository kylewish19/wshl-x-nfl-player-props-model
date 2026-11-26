"""Week 12 pregame inference smoke test: synthetic thresholds only, no betting card."""
import json
from pathlib import Path
import numpy as np
import pandas as pd
import yaml
from wshlx_nfl_props.data import load_raw
from wshlx_nfl_props.features import normalize_player_stats
from wshlx_nfl_props.enrichment import _clean_team
from wshlx_nfl_props.predict import build_prop_rows, fair_pair, american_to_implied
from wshlx_nfl_props.models import load_model
from wshlx_nfl_props.reporting import build_cards

cfg=yaml.safe_load(Path("config/model_config.yaml").read_text())
d=load_raw()
s=d["schedules"]
reg=s[(s.season==2026)&(s.game_type=="REG")].copy()
reg["kickoff"]=pd.to_datetime(reg.gameday.astype(str)+" "+reg.gametime.astype(str),errors="raise").dt.tz_localize("America/New_York").dt.tz_convert("UTC")
cutoff=pd.Timestamp("2026-11-26T14:27:00Z")
assert len(reg[reg.kickoff<cutoff])==164
assert reg.loc[reg.kickoff<cutoff,["home_score","away_score"]].notna().all().all()
assert reg.loc[reg.kickoff>=cutoff,["home_score","away_score"]].isna().all().all(),"Data advanced past pinned cutoff"
games=reg[(reg.gameday.astype(str).str[:10]=="2026-11-26")&(reg.kickoff>=cutoff)]
assert len(games)>0 and (games.week==12).all()
teams={}
for r in games.itertuples():
    teams[_clean_team(r.home_team)]=_clean_team(r.away_team)
    teams[_clean_team(r.away_team)]=_clean_team(r.home_team)
h=normalize_player_stats(d["player_stats"])
h["team"]=h.team.map(_clean_team)
assert not ((h.season==2026)&(h.week>=12)&h.team.isin(teams)).any()
recent=h[(h.season==2026)&(h.week<12)].sort_values(["season","week"]).groupby("player_id").tail(1)
counts=h.groupby("player_id").size()
recent=recent[recent.player_id.map(counts)>=cfg["min_history_games"]]
volume={"qb_pass_yds":"passing_attempts","qb_pass_tds":"passing_attempts","qb_rush_yds":"rushing_attempts","rb_rush_yds":"rushing_attempts","rec_yds":"targets","sack_yes":"def_qb_hits"}
lines=[]
for team,opp in teams.items():
    for prop,spec in cfg["models"].items():
        q=recent[(recent.team==team)&recent.position.isin(spec["positions"])].sort_values(volume[prop],ascending=False)
        if q.empty: continue
        r=q.iloc[0]
        match=h[h.player_display_name.str.lower()==str(r.player_display_name).lower()]
        assert match.player_id.nunique()==1,"Ambiguous name"
        target=spec["target"]
        line=.5 if prop=="sack_yes" else 1.5 if prop=="qb_pass_tds" else max(.5,float(np.floor(r[target])+.5))
        lines.append({"player_display_name":r.player_display_name,"team":team,"opponent":opp,"week":12,"prop":prop,"line":line,"over_odds":np.nan,"under_odds":np.nan,"yes_odds":np.nan,"expected_id":r.player_id})
lines=pd.DataFrame(lines)
assert set(lines.prop)==set(cfg["models"])
# Same player with multiple thresholds must share the same pregame state.
i=lines.index[lines.prop=="qb_pass_yds"][0]
alt=lines.iloc[[i]].copy(); alt["line"]+=10
lines=pd.concat([lines,alt],ignore_index=True)
rows=build_prop_rows(h,lines,d["schedules"],d.get("pbp"),auxiliary=d)
assert (rows.season==2026).all() and (rows.week==12).all()
assert rows.team.tolist()==lines.team.tolist()
assert rows.opponent_team.tolist()==lines.opponent.tolist()
assert rows.player_id.tolist()==lines.expected_id.tolist()
assert rows.iloc[i].history_games==rows.iloc[-1].history_games
report={"status":"smoke_test_only","cutoff_utc":str(cutoff),"games":len(games),"rows":len(rows),"official_artifact_id":15129935005,"shadow_artifact_id":15129795747,"shadow_status":"Week 12 refitted shadow, verified against matched cutoff","limitations":["Synthetic thresholds, no sportsbook odds or official picks.","Does not verify current active rosters or real-world player availability."],"models":{}}
for label,root,props in [("official","models/official",list(cfg["models"])),("shadow","models/challenger_v05",["qb_pass_yds","rb_rush_yds","rec_yds"])]:
    out=lines.drop(columns="expected_id").copy()
    out["season"]=2026
    records=[]
    for prop in props:
        m=load_model(prop,root=root)
        assert m.validation_metrics["latest_season"]==2026 and m.validation_metrics["latest_week"]==12
        idx=out.index[out.prop==prop]
        X=rows.loc[idx].copy()
        for c in m.categorical_features:
            q=X[c].astype(object); X[c]=q.where(q.notna(),"__MISSING__").astype(str)
        cols=m.numeric_features+m.categorical_features
        assert set(cols).issubset(X.columns)
        points=np.asarray(m.predict_point(X),dtype=float)
        assert np.isfinite(points).all()
        ps=np.array([m.probability_over(X.loc[[k]],float(out.loc[k,"line"]))[0] for k in idx],dtype=float)
        assert np.isfinite(ps).all() and ((ps>=0)&(ps<=1)).all()
        if prop=="qb_pass_yds":
            assert np.isclose(points[0],points[-1]) and ps[0]>=ps[-1],"Alternate-line invariance failed"
        for k,point,p in zip(idx,points,ps):
            row=out.loc[k].to_dict()
            side="YES" if prop=="sack_yes" else "OVER" if p>=.5 else "UNDER"
            prob=p if side in ("OVER","YES") else 1-p
            row.update(model_point=float(point),pick=side,model_probability=float(prob),status="BET" if prob>=cfg["official_probability_floor"] else "PASS",validity_ok=True,context_ok=False)
            records.append(row)
        report["models"][label+"_"+prop]={"n":len(idx),"finite":True,"numeric_feature_mean_nonnull_pct":float(X[m.numeric_features].notna().mean().mean()*100)}
    cards=build_cards(pd.DataFrame(records))
    assert len(cards["official_value"])==0
    assert cards["top10_accuracy"].model_probability.is_monotonic_decreasing
    report[label+"_reporting"]={"accuracy_rows":len(cards["top10_accuracy"]),"raw_rows":len(cards["raw_bets"]),"official_rows":0}
ctx=[c for c in rows if c.startswith(("teamctx_","oppctx_"))]
report["future_matchup_coverage"]={c:float(rows[c].notna().mean()) for c in ctx}
assert len(ctx)==32 and all(report["future_matchup_coverage"][c]==1 for c in ctx), "Future matchup coverage incomplete"
report["pass"]=True
Path("reports/2026_w12_future_matchup_smoke.json").write_text(json.dumps(report,indent=2))
print(json.dumps(report,indent=2))
