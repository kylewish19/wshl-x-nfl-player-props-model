import argparse, yaml
from pathlib import Path
import pandas as pd
from wshlx_nfl_props.data import load_raw
from wshlx_nfl_props.game_predict import predict_game_lines

p=argparse.ArgumentParser()
p.add_argument("lines_csv")
p.add_argument("--out",default="predictions/game_markets_latest.csv")
p.add_argument("--model-root",default="models/game_markets")
a=p.parse_args()

cfg=yaml.safe_load(Path("config/game_model_config.yaml").read_text())
d=load_raw()
lines=pd.read_csv(a.lines_csv)
out=predict_game_lines(
    lines,d["schedules"],d.get("team_stats"),d.get("pbp"),
    model_root=a.model_root,
    probability_floor=cfg["official_probability_floor"],
    min_prob_edge=cfg["minimum_probability_edge"],
    windows=tuple(cfg["rolling_windows"])
)
Path(a.out).parent.mkdir(parents=True,exist_ok=True)
out.to_csv(a.out,index=False)
print(out.to_string(index=False))
