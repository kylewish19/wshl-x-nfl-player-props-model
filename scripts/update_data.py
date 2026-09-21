import argparse, yaml
from pathlib import Path
from wshlx_nfl_props.data import load_nflverse, save_raw

p = argparse.ArgumentParser()
p.add_argument("--config", default="config/model_config.yaml")
p.add_argument("--no-pbp", action="store_true")
a = p.parse_args()
cfg = yaml.safe_load(Path(a.config).read_text())
data = load_nflverse(cfg["seasons"], include_pbp=not a.no_pbp)
save_raw(data)
