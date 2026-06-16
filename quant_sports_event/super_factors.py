from __future__ import annotations

from typing import Dict, Iterable, Tuple

import pandas as pd


SUPER_FACTOR_WEIGHTS: Dict[str, float] = {
    "event_opportunity": 0.25,
    "stock_linkage": 0.25,
    "attention_confirmation": 0.20,
    "market_timing": 0.15,
    "tradable_risk_quality": 0.15,
}

SUPER_FACTOR_BOUNDS: Dict[str, Tuple[float, float]] = {
    "event_opportunity": (0.20, 0.30),
    "stock_linkage": (0.20, 0.30),
    "attention_confirmation": (0.10, 0.25),
    "market_timing": (0.05, 0.20),
    "tradable_risk_quality": (0.10, 0.20),
}


def validate_super_factor_weights(weights: Dict[str, float] | None = None) -> None:
    weights = weights or SUPER_FACTOR_WEIGHTS
    missing = set(SUPER_FACTOR_WEIGHTS) - set(weights)
    if missing:
        raise ValueError(f"Missing integrated factor weights: {sorted(missing)}")
    total = sum(float(weights[key]) for key in SUPER_FACTOR_WEIGHTS)
    if abs(total - 1.0) > 1e-9:
        raise ValueError(f"Integrated factor weights must sum to 1.0, got {total}")
    for key, (lower, upper) in SUPER_FACTOR_BOUNDS.items():
        value = float(weights[key])
        if value < lower or value > upper:
            raise ValueError(f"Integrated factor `{key}` weight {value} is outside [{lower}, {upper}]")


def _series(panel: pd.DataFrame, column: str) -> pd.Series:
    if column not in panel:
        return pd.Series(0.0, index=panel.index)
    return panel[column].astype(float).fillna(0.0)


def build_super_factor_panel(
    scores: pd.DataFrame,
    *,
    weights: Dict[str, float] | None = None,
    eligible_tickers: Iterable[str] | None = None,
) -> pd.DataFrame:
    validate_super_factor_weights(weights)
    weights = weights or SUPER_FACTOR_WEIGHTS
    panel = scores.copy()
    if panel.empty:
        return panel

    z_exposure = _series(panel, "z_exposure")
    z_attention = _series(panel, "z_attention")
    z_momentum = _series(panel, "z_momentum")
    z_liquidity = _series(panel, "z_liquidity")
    z_low_volatility = _series(panel, "z_low_volatility")

    panel["event_opportunity"] = z_exposure
    panel["stock_linkage"] = z_exposure
    panel["attention_confirmation"] = z_attention
    panel["market_timing"] = z_momentum.clip(upper=1.5)
    panel["tradable_risk_quality"] = 0.5 * z_liquidity + 0.5 * z_low_volatility
    panel["risk_penalty"] = (
        z_attention.sub(1.0).clip(lower=0.0) * 0.10
        + z_momentum.sub(1.0).clip(lower=0.0) * 0.05
        + (-z_low_volatility).sub(1.0).clip(lower=0.0) * 0.05
    )

    score = pd.Series(0.0, index=panel.index)
    for factor, weight in weights.items():
        contribution_col = f"integrated_contrib_{factor}"
        panel[contribution_col] = panel[factor] * float(weight)
        score = score + panel[contribution_col]
    panel["integrated_score"] = score - panel["risk_penalty"]
    if eligible_tickers is not None:
        eligible = set(eligible_tickers)
        panel["eligible_for_loop_001"] = panel["ticker"].isin(eligible)
    else:
        panel["eligible_for_loop_001"] = True
    return panel.sort_values("integrated_score", ascending=False).reset_index(drop=True)
