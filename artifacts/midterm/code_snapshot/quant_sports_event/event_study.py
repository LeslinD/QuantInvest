from __future__ import annotations

from typing import Dict, Iterable, List, Tuple

import numpy as np
import pandas as pd


def _window(df: pd.DataFrame, event_date: pd.Timestamp, left: int, right: int) -> pd.DataFrame:
    return df[(df["date"] >= event_date + pd.Timedelta(days=left)) & (df["date"] <= event_date + pd.Timedelta(days=right))]


def _fit_market_model(stock: pd.DataFrame, benchmark: pd.DataFrame, event_date: pd.Timestamp) -> Tuple[float, float] | None:
    s = stock[["date", "close"]].copy()
    b = benchmark[["date", "close"]].copy()
    s["ret_i"] = s["close"].pct_change()
    b["ret_m"] = b["close"].pct_change()
    merged = s.merge(b[["date", "ret_m"]], on="date").dropna()
    est = _window(merged, event_date, -180, -60)
    if len(est) < 30:
        return None
    x = est["ret_m"].to_numpy()
    y = est["ret_i"].to_numpy()
    beta, alpha = np.polyfit(x, y, 1)
    return float(alpha), float(beta)


def event_study(
    frames: Dict[str, pd.DataFrame],
    benchmark: pd.DataFrame,
    universe: Iterable[Dict],
    events: Iterable[Dict],
    windows: List[Tuple[int, int]] | None = None,
) -> pd.DataFrame:
    windows = windows or [(-60, -10), (-30, -10), (-10, 10)]
    rows = []
    for event in events:
        event_date = pd.to_datetime(event["event_date"])
        for row in universe:
            symbol = row["sina_symbol"]
            if symbol not in frames:
                continue
            model = _fit_market_model(frames[symbol], benchmark, event_date)
            if model is None:
                continue
            alpha, beta = model
            s = frames[symbol][["date", "close"]].copy()
            b = benchmark[["date", "close"]].copy()
            s["ret_i"] = s["close"].pct_change()
            b["ret_m"] = b["close"].pct_change()
            merged = s.merge(b[["date", "ret_m"]], on="date").dropna()
            merged["ar"] = merged["ret_i"] - (alpha + beta * merged["ret_m"])
            for left, right in windows:
                w = _window(merged, event_date, left, right)
                if w.empty:
                    continue
                rows.append(
                    {
                        "event_id": event["event_id"],
                        "ticker": row["ticker"],
                        "company": row["name"],
                        "industry": row["industry"],
                        "window": f"T{left:+d}_T{right:+d}",
                        "n_days": len(w),
                        "alpha": alpha,
                        "beta": beta,
                        "car": float(w["ar"].sum()),
                        "mean_ar": float(w["ar"].mean()),
                    }
                )
    return pd.DataFrame(rows)


def summarize_event_study(car: pd.DataFrame) -> pd.DataFrame:
    if car.empty:
        return pd.DataFrame()
    return (
        car.groupby(["event_id", "window"])
        .agg(caar=("car", "mean"), positive_rate=("car", lambda s: float((s > 0).mean())), n=("car", "size"))
        .reset_index()
        .sort_values(["event_id", "window"])
    )
