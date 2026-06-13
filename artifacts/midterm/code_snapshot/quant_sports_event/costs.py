from __future__ import annotations

from dataclasses import dataclass
from typing import Dict, List

import math


@dataclass(frozen=True)
class CostModel:
    commission_rate: float
    min_commission: float
    stamp_tax_sell_rate: float
    default_slippage_rate: float
    liquidity_slippage: List[Dict[str, float]]

    @classmethod
    def from_config(cls, config: Dict) -> "CostModel":
        return cls(
            commission_rate=float(config["commission_rate"]),
            min_commission=float(config["min_commission"]),
            stamp_tax_sell_rate=float(config["stamp_tax_sell_rate"]),
            default_slippage_rate=float(config["default_slippage_rate"]),
            liquidity_slippage=list(config.get("liquidity_slippage", [])),
        )

    def slippage_rate_for_amount(self, avg_amount: float) -> float:
        for row in self.liquidity_slippage:
            if avg_amount >= float(row["min_amount"]):
                return float(row["slippage"])
        return self.default_slippage_rate

    def estimate(self, trade_value: float, side: str, avg_amount: float | None = None) -> float:
        if trade_value <= 0:
            return 0.0
        commission = max(self.min_commission, trade_value * self.commission_rate)
        stamp = trade_value * self.stamp_tax_sell_rate if side.lower() == "sell" else 0.0
        slippage = trade_value * (self.slippage_rate_for_amount(avg_amount or 0.0))
        return commission + stamp + slippage


def round_lot_shares(target_value: float, price: float, lot_size: int = 100) -> int:
    if price <= 0 or target_value <= 0:
        return 0
    lots = math.floor(target_value / price / lot_size)
    return int(lots * lot_size)


def is_limit_up(prev_close: float, close: float, board: str = "main") -> bool:
    limit = 0.20 if board in {"chi_next", "star"} else 0.10
    return close >= prev_close * (1 + limit) * 0.999


def is_limit_down(prev_close: float, close: float, board: str = "main") -> bool:
    limit = 0.20 if board in {"chi_next", "star"} else 0.10
    return close <= prev_close * (1 - limit) * 1.001
