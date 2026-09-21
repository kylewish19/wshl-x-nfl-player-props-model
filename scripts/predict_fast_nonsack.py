import argparse,yaml
from pathlib import Path
import pandas as pd
from wshlx_nfl_props.data import load_raw
from wshlx_nfl_props.predict import predict_lines

p=argparse.ArgumentParser()
p.add_argument("lines_csv")
p.add_argument("--out",default="predictions/fast_nonsack.csv")
p.add_argument("--model-root",default="models/official")
a=p.parse_args()
cfg=yaml.safe_load(Path("config/model_config.yaml").read_text())
d=load_raw()
lines=pd.read_csv(a.lines_csv)
lines=lines[lines["prop"]!="sack_yes"].reset_index(drop=True)
aux={"injuries":d.get("injuries",pd.DataFrame())}
out=predict_lines(lines,d["player_stats"],d["schedules"],pbp=None,model_root=a.model_root,
    probability_floor=cfg["official_probability_floor"],min_prob_edge=cfg["minimum_probability_edge"],auxiliary=aux)
Path(a.out).parent.mkdir(parents=True,exist_ok=True)
out.to_csv(a.out,index=False)
print(out.to_string(index=False))
