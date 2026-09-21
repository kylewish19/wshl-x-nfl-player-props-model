from __future__ import annotations

from pathlib import Path
import warnings
import pandas as pd


def _to_pd(obj):
    if hasattr(obj, "to_pandas"):
        return obj.to_pandas()
    return pd.DataFrame(obj)


def _safe(name, fn, *args, **kwargs):
    try:
        out = _to_pd(fn(*args, **kwargs))
        print(f"[data] {name}: {len(out):,} rows x {len(out.columns):,} cols")
        return out
    except Exception as exc:
        warnings.warn(f"Could not load {name}: {exc}")
        return pd.DataFrame()


def load_nflverse(seasons: list[int], include_pbp: bool = True) -> dict[str, pd.DataFrame]:
    """Load public NFL data through the maintained nflreadpy interface.

    The model is designed to remain usable if one optional upstream table is late.
    Core player stats and schedules are required; advanced tables enrich features.
    """
    import nflreadpy as nfl

    d: dict[str, pd.DataFrame] = {}
    d["player_stats"] = _safe("player_stats", nfl.load_player_stats, seasons)
    d["team_stats"] = _safe("team_stats", nfl.load_team_stats, seasons)
    d["schedules"] = _safe("schedules", nfl.load_schedules)
    d["weekly_rosters"] = _safe("weekly_rosters", nfl.load_rosters_weekly, seasons)
    d["snap_counts"] = _safe("snap_counts", nfl.load_snap_counts, seasons)
    d["injuries"] = _safe("injuries", nfl.load_injuries, seasons)

    for stat_type in ("passing", "rushing", "receiving"):
        d[f"ngs_{stat_type}"] = _safe(
            f"ngs_{stat_type}", nfl.load_nextgen_stats, seasons, stat_type=stat_type
        )
    for stat_type in ("pass", "rush", "rec", "def"):
        d[f"adv_{stat_type}"] = _safe(
            f"adv_{stat_type}", nfl.load_pfr_advstats, seasons, stat_type=stat_type, summary_level="week"
        )
    if include_pbp:
        d["pbp"] = _safe("pbp", nfl.load_pbp, seasons)
    return d


def save_raw(data: dict[str, pd.DataFrame], root: str | Path = "data/raw") -> None:
    root = Path(root)
    root.mkdir(parents=True, exist_ok=True)
    for name, df in data.items():
        if df.empty:
            continue
        path = root / f"{name}.parquet"
        df.to_parquet(path, index=False)
        print(f"[data] wrote {path}")


def load_raw(root: str | Path = "data/raw") -> dict[str, pd.DataFrame]:
    root = Path(root)
    out = {}
    for path in root.glob("*.parquet"):
        out[path.stem] = pd.read_parquet(path)
    return out
