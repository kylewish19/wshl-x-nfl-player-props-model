import argparse
from pathlib import Path
import pandas as pd
from wshlx_nfl_props.data import load_raw
from wshlx_nfl_props.features import normalize_player_stats

p=argparse.ArgumentParser()
p.add_argument("lines_csv")
p.add_argument("--out",required=True)
p.add_argument("--missing",required=True)
a=p.parse_args()

d=load_raw()
hist=normalize_player_stats(d["player_stats"])
known={str(x).lower() for x in hist["player_display_name"].dropna().unique()}
lines=pd.read_csv(a.lines_csv)
mask=lines["player_display_name"].astype(str).str.lower().isin(known)
Path(a.out).parent.mkdir(parents=True,exist_ok=True)
lines[mask].to_csv(a.out,index=False)
missing=lines[~mask][["player_display_name","prop"]].copy()
missing.to_csv(a.missing,index=False)
print("[known]",int(mask.sum()),"of",len(lines))
if not missing.empty:
    print("[missing]")
    print(missing.to_string(index=False))
