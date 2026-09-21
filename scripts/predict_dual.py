import argparse
from pathlib import Path
import pandas as pd
import yaml

from wshlx_nfl_props.data import load_raw
from wshlx_nfl_props.predict import predict_lines

p=argparse.ArgumentParser()
p.add_argument("lines_csv")
p.add_argument("--official-root",default="models/official")
p.add_argument("--shadow-root",default="models/challenger_v04")
p.add_argument("--out",default="predictions/dual_locked.csv")
a=p.parse_args()

cfg=yaml.safe_load(Path("config/model_config.yaml").read_text())
d=load_raw(); lines=pd.read_csv(a.lines_csv)
kwargs=dict(
    player_stats=d["player_stats"],schedules=d["schedules"],pbp=d.get("pbp"),
    probability_floor=cfg["official_probability_floor"],
    min_prob_edge=cfg["minimum_probability_edge"],auxiliary=d
)
off=predict_lines(lines,model_root=a.official_root,**kwargs)
sh=predict_lines(lines,model_root=a.shadow_root,**kwargs)

key=["player_display_name","prop","line","season","week","team","opponent"]
keep=["model_point","pick","model_probability","probability_edge_vs_price","status","algorithm","feature_set"]
out=off[key+keep].copy()
out=out.rename(columns={c:f"official_{c}" for c in keep})
for c in keep:
    out[f"shadow_{c}"]=sh[c].to_numpy()
out["models_agree"]=out["official_pick"].eq(out["shadow_pick"])
out["probability_gap"]=out["shadow_model_probability"]-out["official_model_probability"]
Path(a.out).parent.mkdir(parents=True,exist_ok=True)
out.to_csv(a.out,index=False)
print(out.to_string(index=False))
