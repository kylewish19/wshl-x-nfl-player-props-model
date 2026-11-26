"""Deterministic reporting for frozen prop predictions; does not fit or modify models."""
from __future__ import annotations
import numpy as np
import pandas as pd

KEY = ["season", "week", "team", "player_display_name", "prop", "line"]

def build_cards(predictions: pd.DataFrame, *, top_n: int = 10):
    """Require explicit pregame validity/context decisions on every row.

    validity_ok is the minimal accuracy-card gate; context_ok is the stricter
    official gate. Preserve original model BET labels independently of price
    and context corrections. Missing odds never qualify for the official card.
    """
    required = set(KEY + ["pick", "model_probability", "status", "validity_ok", "context_ok"])
    missing = required - set(predictions.columns)
    if missing:
        raise ValueError(f"Missing required reporting fields: {sorted(missing)}")
    if top_n < 1:
        raise ValueError("top_n must be positive")
    x = predictions.copy()
    if x[KEY].isna().any().any() or x.duplicated(KEY).any():
        raise ValueError("Missing or duplicate prop identity")
    for flag in ("validity_ok", "context_ok"):
        if not pd.api.types.is_bool_dtype(x[flag]) or x[flag].isna().any():
            raise ValueError(f"{flag} must contain explicit boolean decisions")
    if not x["pick"].isin(["OVER", "UNDER", "YES"]).all():
        raise ValueError("Unrecognized model side")
    p = pd.to_numeric(x["model_probability"], errors="coerce")
    if not (p.notna() & p.between(0, 1)).all():
        raise ValueError("Model probabilities must be finite and in [0,1]")
    x["model_probability"] = p
    odds = pd.Series(np.nan, index=x.index, dtype=float)
    for side, col in [("OVER", "over_odds"), ("UNDER", "under_odds"), ("YES", "yes_odds")]:
        if col in x:
            mask = x["pick"].eq(side)
            odds.loc[mask] = pd.to_numeric(x.loc[mask, col], errors="coerce")
    valid_price = np.isfinite(odds) & (odds.abs() >= 100)
    odds = odds.where(valid_price)
    x["selected_odds"] = odds
    decimal = pd.Series(np.nan, index=x.index, dtype=float)
    pos = valid_price & (odds > 0)
    neg = valid_price & (odds < 0)
    decimal.loc[pos] = 1 + odds.loc[pos] / 100
    decimal.loc[neg] = 1 + 100 / odds.loc[neg].abs()
    x["break_even_probability"] = 1 / decimal
    x["expected_units_per_unit_risked"] = p * decimal - 1
    x["price_ok"] = valid_price & (x["expected_units_per_unit_risked"] > 0)
    # Sort every supported and valid side together, including QB TDs and sacks.
    top = x.loc[x.validity_ok].sort_values(["model_probability", *KEY], ascending=[False]+[True]*len(KEY)).head(top_n).copy()
    top.insert(0, "rank", range(1, len(top)+1))
    raw = x.loc[x.status.eq("BET")].copy()
    # Preserve the established cross-model direction gate for shared markets.
    if "v05_agreement" in x.columns:
        if not pd.api.types.is_bool_dtype(x["v05_agreement"]) or x["v05_agreement"].isna().any():
            raise ValueError("v05_agreement must be explicit boolean")
        agreement = raw["v05_agreement"]
    else:
        agreement = pd.Series(True, index=raw.index)
    official = raw.loc[raw.validity_ok & raw.context_ok & raw.price_ok & agreement].copy()
    return {"top10_accuracy":top, "raw_bets":raw, "official_value":official, "audit":x}
