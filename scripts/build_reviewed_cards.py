"""Generate three auditable draft cards from actual model output and reviewed context."""
import argparse
from pathlib import Path
import pandas as pd
from wshlx_nfl_props.reporting import KEY, build_cards

p=argparse.ArgumentParser()
p.add_argument("predictions",help="Complete v0.3 output across supported markets")
p.add_argument("context",help="One reviewed decision per exact prop key")
p.add_argument("--outdir",required=True,help="New directory; never overwrite a prior card")
a=p.parse_args()
pred=pd.read_csv(a.predictions)
ctx=pd.read_csv(a.context)
fields=KEY+["validity_ok","context_ok","context_reason"]
if not set(fields).issubset(ctx.columns):
    raise ValueError("Context file requires exact prop keys, both approval flags and context_reason")
if not set(KEY).issubset(pred.columns):
    raise ValueError("Prediction file missing exact prop keys")
if ctx[fields].isna().any().any() or (ctx.context_reason.astype(str).str.strip()=="").any():
    raise ValueError("Context review must be explicit and complete")
if pred[KEY].isna().any().any():
    raise ValueError("Prediction identity cannot be missing")
for c in ["validity_ok","context_ok"]:
    normalized=ctx[c].astype(str).str.lower().str.strip()
    if not normalized.isin(["true","false"]).all():
        raise ValueError(f"Invalid boolean {c}")
    ctx[c]=normalized.eq("true")
if set(fields[len(KEY):])&set(pred.columns):
    raise ValueError("Context must come from independent reviewed file, not prediction output")
joined=pred.merge(ctx[fields],on=KEY,how="outer",validate="one_to_one",indicator=True)
if not joined["_merge"].eq("both").all():
    raise ValueError("Context rows must exactly match prediction rows")
cards=build_cards(joined.drop(columns="_merge"))
out=Path(a.outdir)
out.mkdir(parents=True,exist_ok=False)
for name,table in cards.items():
    table.to_csv(out/f"{name}.csv",index=False)
    print(f"{name}: {len(table)} rows")
print("Draft cards only; verify pregame timestamps and source prices before locking.")
