import argparse
from pathlib import Path
import pandas as pd
from wshlx_nfl_props.data import load_raw
from wshlx_nfl_props.game_grade import grade_game_markets

p=argparse.ArgumentParser()
p.add_argument("locked_csv")
p.add_argument("--outdir",default="grading/game_markets")
a=p.parse_args()

d=load_raw()
locked=pd.read_csv(a.locked_csv)
graded,summary=grade_game_markets(locked,d["schedules"])
Path(a.outdir).mkdir(parents=True,exist_ok=True)
graded.to_csv(Path(a.outdir)/"graded_latest.csv",index=False)
summary.to_csv(Path(a.outdir)/"summary_latest.csv",index=False)
print(summary.to_string(index=False))
