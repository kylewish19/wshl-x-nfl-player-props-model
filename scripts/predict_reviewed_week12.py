"""Week 12 FanDuel-only draft inference with explicit cutoff and model provenance."""
import argparse
import json
from pathlib import Path
import numpy as np
import pandas as pd
import yaml
from wshlx_nfl_props.data import load_raw
from wshlx_nfl_props.features import normalize_player_stats
from wshlx_nfl_props.enrichment import _clean_team
from wshlx_nfl_props.predict import predict_lines
from wshlx_nfl_props.models import load_model
from wshlx_nfl_props.reporting import KEY

p=argparse.ArgumentParser()
p.add_argument("lines")
p.add_argument("--outdir",required=True)
a=p.parse_args()
manifest=json.loads(Path("reports/2026_w12_active_artifacts.json").read_text())
assert manifest["status"]=="ready_for_reviewed_drafts", "Manifest not ready"
cfg=yaml.safe_load(Path("config/model_config.yaml").read_text())
d=load_raw()
cutoff=pd.Timestamp(manifest["as_of"])
now=pd.Timestamp.now(tz="UTC")
s=d["schedules"].copy()
reg=s[(s.season==2026)&(s.game_type=="REG")].copy()
reg["kickoff"]=pd.to_datetime(reg.gameday.astype(str)+" "+reg.gametime.astype(str),errors="raise").dt.tz_localize("America/New_York").dt.tz_convert("UTC")
done=reg[reg.kickoff<cutoff]
assert len(done)==manifest["completed_games"] and done[["home_score","away_score"]].notna().all().all()
assert reg.loc[reg.kickoff>=cutoff,["home_score","away_score"]].isna().all().all(),"New results present: require a fresh verified cutoff"
x=pd.read_csv(a.lines)
required=set(KEY+["opponent","sportsbook","observed_at_utc","over_odds","under_odds","yes_odds"])
if not required.issubset(x.columns) or x.empty:
    raise ValueError("Incomplete FanDuel input schema")
if not x.sportsbook.astype(str).str.strip().str.lower().eq("fanduel").all():
    raise ValueError("Only FanDuel is accepted")
if x[KEY+["opponent","observed_at_utc"]].isna().any().any() or x.duplicated(KEY).any():
    raise ValueError("Missing or duplicate prop identity")
assert x.season.eq(2026).all() and x.week.eq(12).all()
assert x.prop.isin(cfg["models"]).all(),"Separate unsupported props before inference"
x["line"]=pd.to_numeric(x.line,errors="raise")
assert np.isfinite(x.line).all() and (x.line%1==.5).all(),"Only half-point lines supported (no push ambiguity)"
assert x.loc[x.prop.eq("sack_yes"),"line"].eq(.5).all()
observed=pd.to_datetime(x.observed_at_utc,utc=True,errors="raise")
assert ((observed>=cutoff)&(observed<=now)).all(),"Invalid observation timestamp"
x["team"]=x.team.map(_clean_team); x["opponent"]=x.opponent.map(_clean_team)
h=normalize_player_stats(d["player_stats"])
h["team"]=h.team.map(_clean_team)
for row,t in zip(x.itertuples(),observed):
    match=reg[(reg.week==row.week)&(((reg.home_team.map(_clean_team)==row.team)&(reg.away_team.map(_clean_team)==row.opponent))|((reg.away_team.map(_clean_team)==row.team)&(reg.home_team.map(_clean_team)==row.opponent)))]
    assert len(match)==1 and match.kickoff.iloc[0]>max(now,t),"Target game is not verified pregame"
    name=h[h.player_display_name.str.lower()==str(row.player_display_name).lower()]
    assert name.player_id.nunique()==1, f"Unresolved or ambiguous player: {row.player_display_name}"
    latest=name.sort_values(["season","week"]).iloc[-1]
    assert latest.team==row.team and latest.position in cfg["models"][row.prop]["positions"],f"Team/position requires review: {row.player_display_name}"
    assert len(name)>=cfg["min_history_games"],f"Insufficient history: {row.player_display_name}"
    assert not ((name.season==2026)&(name.week>=row.week)).any(),"Target or future game in player history"
for c in ["over_odds","under_odds","yes_odds"]:
    v=pd.to_numeric(x[c],errors="coerce")
    assert (v.isna() | (np.isfinite(v)&(v.abs()>=100))).all()
    x[c]=v
for prop in cfg["models"]:
    m=load_model(prop,root=manifest["official"]["model_root"])
    assert m.validation_metrics["latest_season"]==2026 and m.validation_metrics["latest_week"]==12
shared=["qb_pass_yds","rb_rush_yds","rec_yds"]
for prop in shared:
    m=load_model(prop,root=manifest["shadow"]["model_root"])
    assert m.validation_metrics["latest_season"]==2026 and m.validation_metrics["latest_week"]==12
def predict(q,root):
    return predict_lines(q,h,d["schedules"],d.get("pbp"),model_root=root,probability_floor=cfg["official_probability_floor"],min_prob_edge=cfg["minimum_probability_edge"],auxiliary=d)
official=predict(x,manifest["official"]["model_root"])
shadow_input=x[x.prop.isin(shared)]
if not shadow_input.empty:
    shadow=predict(shadow_input,manifest["shadow"]["model_root"])
    rename={"pick":"v05_pick","model_probability":"v05_probability","model_point":"v05_point"}
    official=official.merge(shadow[KEY+list(rename)].rename(columns=rename),on=KEY,how="left",validate="one_to_one")
else:
    for c in ["v05_pick","v05_probability","v05_point"]: official[c]=np.nan
official["v05_required"]=official.prop.isin(shared)
official["v05_agreement"]=~official.v05_required|official.pick.eq(official.v05_pick)
assert len(official)==len(x)
assert official.model_probability.between(0,1).all() and np.isfinite(official.model_point).all()
out=Path(a.outdir)
out.mkdir(parents=True,exist_ok=False)
official.to_csv(out/"predictions.csv",index=False)
# Never auto-approve validity or context; human/source review must fill these fields.
context=official[KEY].copy()
context["validity_ok"]=""; context["context_ok"]=""; context["context_reason"]=""
context.to_csv(out/"context_template.csv",index=False)
(out/"provenance.json").write_text(json.dumps({"status":"draft_only","created_at_utc":str(now),"training_cutoff":str(cutoff),"official_artifact_id":manifest["official"]["artifact_id"],"shadow_artifact_id":manifest["shadow"]["artifact_id"],"input_rows":len(x)},indent=2))
print(f"Drafted {len(official)} predictions; no official card locked.")
