from __future__ import annotations

from typing import Dict, Iterable, List

import pandas as pd

from .costs import CostModel, round_lot_shares


def build_target_weights(
    scored: pd.DataFrame,
    max_event_exposure: float,
    max_position_per_stock: float,
    max_industry_exposure: float,
    top_k: int,
) -> pd.DataFrame:
    if scored.empty:
        return scored.copy()
    candidates = scored[scored["alpha_score"] > scored["alpha_score"].median()].head(top_k).copy()
    if candidates.empty:
        candidates = scored.head(max(1, min(top_k, len(scored)))).copy()
    raw = candidates["alpha_score"].clip(lower=0)
    if raw.sum() <= 0:
        raw = pd.Series(1.0, index=candidates.index)
    candidates["target_weight"] = raw / raw.sum() * max_event_exposure
    candidates["target_weight"] = candidates["target_weight"].clip(upper=max_position_per_stock)
    # Industry cap with one pass; residual remains cash.
    capped_rows = []
    industry_used: Dict[str, float] = {}
    for _, row in candidates.sort_values("alpha_score", ascending=False).iterrows():
        industry = row["industry"]
        remaining = max(0.0, max_industry_exposure - industry_used.get(industry, 0.0))
        weight = min(float(row["target_weight"]), remaining)
        industry_used[industry] = industry_used.get(industry, 0.0) + weight
        out = row.copy()
        out["target_weight"] = weight
        capped_rows.append(out)
    out = pd.DataFrame(capped_rows)
    out = out[out["target_weight"] > 0].copy()
    if out["target_weight"].sum() > max_event_exposure:
        out["target_weight"] *= max_event_exposure / out["target_weight"].sum()
    return out.reset_index(drop=True)


def generate_orders(
    targets: pd.DataFrame,
    frames: Dict[str, pd.DataFrame],
    universe: Iterable[Dict],
    cash: float,
    decision_date: str,
    order_date: str,
    cost_model: CostModel,
    lot_size: int,
) -> pd.DataFrame:
    universe_by_symbol = {row["sina_symbol"]: row for row in universe}
    orders: List[Dict] = []
    for _, row in targets.iterrows():
        symbol = row["sina_symbol"]
        hist = frames[symbol][frames[symbol]["date"] <= pd.to_datetime(decision_date)].copy()
        if hist.empty:
            continue
        last = hist.iloc[-1]
        prev = hist.iloc[-2] if len(hist) >= 2 else last
        board = universe_by_symbol[symbol].get("board", "main")
        target_value = cash * float(row["target_weight"])
        shares = round_lot_shares(target_value, float(last["close"]), lot_size=lot_size)
        trade_value = shares * float(last["close"])
        avg_amount = float(hist["amount"].tail(20).mean())
        expected_cost = cost_model.estimate(trade_value, "buy", avg_amount=avg_amount)
        reject_reason = ""
        if shares <= 0:
            reject_reason = "target_value_below_lot_size"
        # We do not mark close-limit as impossible unless high == low == close;
        # otherwise a next-day limit order can still be attempted.
        orders.append(
            {
                "decision_date": decision_date,
                "order_date": order_date,
                "ticker": row["ticker"],
                "sina_symbol": symbol,
                "company": row["name"],
                "industry": row["industry"],
                "alpha_score": float(row["alpha_score"]),
                "target_weight": float(row["target_weight"]),
                "reference_close": float(last["close"]),
                "prev_close": float(prev["close"]),
                "target_shares": int(shares),
                "estimated_trade_value": float(trade_value),
                "expected_cost": float(expected_cost),
                "avg_amount_20d": avg_amount,
                "reject_reason": reject_reason,
                "board": board,
            }
        )
    return pd.DataFrame(orders)
