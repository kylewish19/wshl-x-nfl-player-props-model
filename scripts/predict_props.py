import argparse, yaml
from pathlib import Path
import pandas as pd
from wshlx_nfl_props.data import load_raw
from wshlx_nfl_props.predict import predict_lines

p=argparse.ArgumentParser()
p.add_argument("lines_csv")
p.add_argument("--out", default="predictions/latest.csv")
a=p.parse_args()
cfg=yaml.safe_load(Path("config/model_config.yaml").read_text())
d=load_raw()
lines=pd.read_csv(a.lines_csv)
out=predict_lines(lines,d["player_stats"],d["schedules"],d.get("pbp"),
                  probability_floor=cfg["official_probability_floor"], min_prob_edge=cfg["minimum_probability_edge"])
Path(a.out).parent.mkdir(parents=True,exist_ok=True)
out.to_csv(a.out,index=False)
print(out.to_string(index=False))
