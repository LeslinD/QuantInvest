from __future__ import annotations

from dataclasses import dataclass
from typing import Dict, Iterable, List

import numpy as np
import pandas as pd


FEATURES = ["exposure", "attention", "momentum", "liquidity", "low_volatility"]


def zscore(series: pd.Series) -> pd.Series:
    std = series.std(ddof=0)
    if not np.isfinite(std) or std == 0:
        return pd.Series(0.0, index=series.index)
    return (series - series.mean()) / std


def safe_pct_change(series: pd.Series, periods: int) -> pd.Series:
    return series.pct_change(periods=periods).replace([np.inf, -np.inf], np.nan)


def compute_symbol_features(
    df: pd.DataFrame,
    decision_date: str,
    exposure: float,
    attention_lookback: int,
    momentum_lookback: int,
) -> Dict[str, float]:
    hist = df[df["date"] <= pd.to_datetime(decision_date)].copy()
    if len(hist) < max(attention_lookback, momentum_lookback) + 2:
        return {k: np.nan for k in FEATURES}
    hist["ret"] = hist["close"].pct_change()
    amount = hist["amount"].replace(0, np.nan)
    attention = np.log1p(amount.iloc[-1]) - np.log1p(amount.rolling(attention_lookback).mean().iloc[-1])
    momentum = hist["close"].iloc[-1] / hist["close"].iloc[-1 - momentum_lookback] - 1
    liquidity = np.log1p(amount.rolling(20, min_periods=5).mean().iloc[-1])
    low_volatility = -hist["ret"].rolling(20, min_periods=5).std().iloc[-1]
    return {
        "exposure": float(exposure),
        "attention": float(attention),
        "momentum": float(momentum),
        "liquidity": float(liquidity),
        "low_volatility": float(low_volatility),
    }


def factor_panel_for_date(
    frames: Dict[str, pd.DataFrame],
    universe: Iterable[Dict[str, str]],
    exposure_by_ticker: Dict[str, float],
    decision_date: str,
    attention_lookback: int,
    momentum_lookback: int,
) -> pd.DataFrame:
    rows: List[Dict[str, float | str]] = []
    for row in universe:
        symbol = row["sina_symbol"]
        features = compute_symbol_features(
            frames[symbol],
            decision_date=decision_date,
            exposure=exposure_by_ticker.get(row["ticker"], 0.0),
            attention_lookback=attention_lookback,
            momentum_lookback=momentum_lookback,
        )
        out = {"ticker": row["ticker"], "sina_symbol": symbol, "name": row["name"], "industry": row["industry"], **features}
        rows.append(out)
    panel = pd.DataFrame(rows)
    for col in FEATURES:
        panel[f"z_{col}"] = zscore(panel[col].astype(float))
    return panel


def estimate_rankic_weights(samples: pd.DataFrame, features: List[str] | None = None) -> Dict[str, float]:
    features = features or FEATURES
    weights: Dict[str, float] = {}
    if samples.empty or "label" not in samples:
        return {f: 1.0 / len(features) for f in features}
    for f in features:
        col = f"z_{f}" if f"z_{f}" in samples.columns else f
        tmp = samples[[col, "label"]].dropna()
        if len(tmp) < 3 or tmp[col].nunique() < 2 or tmp["label"].nunique() < 2:
            weights[f] = 0.0
        else:
            weights[f] = float(tmp[col].rank().corr(tmp["label"].rank()))
            if not np.isfinite(weights[f]):
                weights[f] = 0.0
    # Preserve sign, but normalize absolute exposure. If all ICs are zero, use equal weights.
    denom = sum(abs(v) for v in weights.values())
    if denom == 0:
        return {f: 1.0 / len(features) for f in features}
    return {f: v / denom for f, v in weights.items()}


def _normalize_positive(values: Dict[str, float], features: List[str]) -> Dict[str, float]:
    cleaned = {f: max(0.0, float(values.get(f, 0.0))) for f in features}
    total = sum(cleaned.values())
    if total <= 0:
        return {f: 1.0 / len(features) for f in features}
    return {f: cleaned[f] / total for f in features}


def estimate_constrained_rankic_weights(
    samples: pd.DataFrame,
    features: List[str] | None = None,
    directions: Dict[str, int] | None = None,
    priors: Dict[str, float] | None = None,
    prior_weight: float = 0.55,
) -> Dict[str, float]:
    """Estimate RankIC weights with finance-theory sign constraints."""

    features = features or FEATURES
    directions = directions or {f: 1 for f in features}
    prior_weight = min(1.0, max(0.0, float(prior_weight)))
    prior_norm = _normalize_positive(priors or {}, features)
    if samples.empty or "label" not in samples:
        return prior_norm

    raw: Dict[str, float] = {}
    for f in features:
        col = f"z_{f}" if f"z_{f}" in samples.columns else f
        tmp = samples[[col, "label"]].dropna()
        if len(tmp) < 3 or tmp[col].nunique() < 2 or tmp["label"].nunique() < 2:
            raw[f] = 0.0
            continue
        ic = float(tmp[col].rank().corr(tmp["label"].rank()))
        if not np.isfinite(ic):
            ic = 0.0
        direction = 1 if int(directions.get(f, 1)) >= 0 else -1
        raw[f] = max(0.0, direction * ic)

    data_norm = _normalize_positive(raw, features)
    combined = {
        f: prior_weight * prior_norm[f] + (1 - prior_weight) * data_norm[f]
        for f in features
    }
    return _normalize_positive(combined, features)


def score_panel(panel: pd.DataFrame, weights: Dict[str, float], attention_z_cap: float | None = None) -> pd.DataFrame:
    scored = panel.copy()
    score = pd.Series(0.0, index=scored.index)
    for f, w in weights.items():
        col = f"z_{f}"
        if col in scored:
            values = scored[col].fillna(0.0)
            if f == "attention" and attention_z_cap is not None:
                values = values.clip(upper=float(attention_z_cap))
            scored[f"contrib_{f}"] = w * values
            score = score + scored[f"contrib_{f}"]
    scored["alpha_score"] = score
    return scored.sort_values("alpha_score", ascending=False).reset_index(drop=True)


def event_time_score(event_date: str, decision_date: str) -> float:
    days = (pd.to_datetime(event_date) - pd.to_datetime(decision_date)).days
    if days < 0:
        return 0.0
    if days <= 10:
        return 0.4
    if days <= 30:
        return 0.8
    if days <= 90:
        return 1.0
    return 0.5


@dataclass
class FeatureSample:
    event_id: str
    ticker: str
    label: float
    features: Dict[str, float]
