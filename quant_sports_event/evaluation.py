from __future__ import annotations

from pathlib import Path
from typing import Any, Dict, Iterable, List

import pandas as pd

from .backtest import _trading_date_after, _trading_date_on_or_before, params_for_event
from .costs import CostModel, round_lot_shares
from .data import SinaDailyProvider
from .factors import factor_panel_for_date, score_panel
from .portfolio import build_target_weights


MIDTERM_PDF_WEIGHTS: Dict[str, float] = {
    "605299.SH": 0.15,
    "002181.SZ": 0.125,
    "002878.SZ": 0.125,
    "600060.SH": 0.10,
}


def _price_on_or_after(df: pd.DataFrame, date: str, field: str = "open") -> tuple[pd.Timestamp | None, float | None]:
    rows = df[df["date"] >= pd.to_datetime(date)]
    if rows.empty:
        return None, None
    row = rows.iloc[0]
    return row["date"], float(row[field])


def _price_on_or_before(df: pd.DataFrame, date: str, field: str = "close") -> tuple[pd.Timestamp | None, float | None]:
    rows = df[df["date"] <= pd.to_datetime(date)]
    if rows.empty:
        return None, None
    row = rows.iloc[-1]
    return row["date"], float(row[field])


def static_portfolio_return(
    frames: Dict[str, pd.DataFrame],
    universe_by_ticker: Dict[str, Dict[str, Any]],
    weights: Dict[str, float],
    *,
    start_date: str,
    end_date: str,
    initial_cash: float,
    cost_model: CostModel | None = None,
    lot_size: int = 100,
    buy_field: str = "open",
    sell_field: str = "open",
) -> tuple[float, pd.DataFrame]:
    rows: List[Dict[str, Any]] = []
    total_pnl = 0.0
    for ticker, weight in weights.items():
        meta = universe_by_ticker.get(ticker)
        if not meta or meta["sina_symbol"] not in frames:
            continue
        frame = frames[meta["sina_symbol"]]
        buy_date, buy_price = _price_on_or_after(frame, start_date, buy_field)
        sell_date, sell_price = _price_on_or_before(frame, end_date, sell_field)
        if buy_date is None or sell_date is None or buy_price is None or sell_price is None:
            continue
        shares = round_lot_shares(initial_cash * float(weight), buy_price, lot_size)
        if shares <= 0:
            continue
        buy_value = shares * buy_price
        sell_value = shares * sell_price
        avg_amount = float(frame[frame["date"] <= buy_date]["amount"].tail(20).mean())
        buy_cost = cost_model.estimate(buy_value, "buy", avg_amount=avg_amount) if cost_model else 0.0
        sell_cost = cost_model.estimate(sell_value, "sell", avg_amount=avg_amount) if cost_model else 0.0
        pnl = sell_value - buy_value - buy_cost - sell_cost
        total_pnl += pnl
        rows.append(
            {
                "ticker": ticker,
                "company": meta.get("name", ""),
                "industry": meta.get("industry", ""),
                "target_weight": float(weight),
                "buy_date": buy_date.date().isoformat(),
                "sell_date": sell_date.date().isoformat(),
                "buy_price": buy_price,
                "sell_price": sell_price,
                "shares": int(shares),
                "buy_value": buy_value,
                "sell_value": sell_value,
                "buy_cost": buy_cost,
                "sell_cost": sell_cost,
                "pnl": pnl,
                "stock_return": sell_price / buy_price - 1,
            }
        )
    return total_pnl / initial_cash, pd.DataFrame(rows)


def evaluate_layered_event(
    frames: Dict[str, pd.DataFrame],
    universe: List[Dict[str, Any]],
    exposure_by_event: Dict[str, Dict[str, float]],
    event: Dict[str, Any],
    base_params: Dict[str, Any],
    factor_weights: Dict[str, float],
    cost_model: CostModel,
    constraints: Dict[str, Any],
    model_config: Dict[str, Any],
    *,
    initial_cash: float,
) -> tuple[float, pd.DataFrame, Dict[str, Any]]:
    event_params = params_for_event(base_params, event, model_config)
    if event_params.get("action") == "avoid":
        return 0.0, pd.DataFrame(), event_params
    ref_df = next(iter(frames.values()))
    event_date = pd.to_datetime(event["event_date"])
    decision = _trading_date_on_or_before(ref_df, event_date - pd.Timedelta(days=event_params["entry_days_before_event"]))
    exit_signal = _trading_date_on_or_before(ref_df, event_date - pd.Timedelta(days=event_params["exit_days_before_event"]))
    if decision is None or exit_signal is None or exit_signal <= decision:
        return 0.0, pd.DataFrame(), event_params
    buy_date = _trading_date_after(ref_df, decision)
    sell_date = _trading_date_after(ref_df, exit_signal)
    if buy_date is None or sell_date is None or sell_date <= buy_date:
        return 0.0, pd.DataFrame(), event_params
    panel = factor_panel_for_date(
        frames,
        universe,
        exposure_by_event.get(event["event_id"], {}),
        decision.strftime("%Y-%m-%d"),
        int(event_params["attention_lookback"]),
        int(event_params["momentum_lookback"]),
    )
    scored = score_panel(panel, factor_weights, attention_z_cap=event_params.get("attention_z_cap"))
    scored = scored[scored["exposure"] >= float(event_params.get("min_exposure_for_trade", 0.0))].copy()
    targets = build_target_weights(
        scored,
        max_event_exposure=float(constraints["max_event_exposure"]),
        max_position_per_stock=float(constraints["max_position_per_stock"]),
        max_industry_exposure=float(constraints["max_industry_exposure"]),
        top_k=int(event_params["top_k"]),
    )
    if targets.empty:
        return 0.0, pd.DataFrame(), event_params
    universe_by_ticker = {row["ticker"]: row for row in universe}
    weights = targets.set_index("ticker")["target_weight"].to_dict()
    ret, details = static_portfolio_return(
        frames,
        universe_by_ticker,
        weights,
        start_date=buy_date.date().isoformat(),
        end_date=sell_date.date().isoformat(),
        initial_cash=initial_cash,
        cost_model=cost_model,
        lot_size=int(constraints["lot_size"]),
    )
    if not details.empty:
        details.insert(0, "event_id", event["event_id"])
    return ret, details, event_params


def event_strategy_comparison(
    frames: Dict[str, pd.DataFrame],
    universe: List[Dict[str, Any]],
    events: Iterable[Dict[str, Any]],
    exposure_by_event: Dict[str, Dict[str, float]],
    base_params: Dict[str, Any],
    factor_weights: Dict[str, float],
    cost_model: CostModel,
    constraints: Dict[str, Any],
    model_config: Dict[str, Any],
    *,
    initial_cash: float,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    universe_by_ticker = {row["ticker"]: row for row in universe}
    rows: List[Dict[str, Any]] = []
    detail_frames: List[pd.DataFrame] = []
    for event in events:
        event_params = params_for_event(base_params, event, model_config)
        event_date = pd.to_datetime(event["event_date"])
        ref_df = next(iter(frames.values()))
        midterm_decision = _trading_date_on_or_before(
            ref_df,
            event_date - pd.Timedelta(days=int(base_params.get("entry_days_before_event", 90))),
        )
        midterm_exit_signal = _trading_date_on_or_before(
            ref_df,
            event_date - pd.Timedelta(days=int(base_params.get("exit_days_before_event", 10))),
        )
        midterm_buy_date = _trading_date_after(ref_df, midterm_decision) if midterm_decision is not None else None
        midterm_sell_date = _trading_date_after(ref_df, midterm_exit_signal) if midterm_exit_signal is not None else None
        midterm_ret, _ = static_portfolio_return(
            frames,
            universe_by_ticker,
            MIDTERM_PDF_WEIGHTS,
            start_date=midterm_buy_date.date().isoformat() if midterm_buy_date is not None else event_date.date().isoformat(),
            end_date=midterm_sell_date.date().isoformat() if midterm_sell_date is not None else event_date.date().isoformat(),
            initial_cash=initial_cash,
            cost_model=cost_model,
            lot_size=int(constraints["lot_size"]),
        )
        optimized_ret, optimized_details, used_params = evaluate_layered_event(
            frames,
            universe,
            exposure_by_event,
            event,
            base_params,
            factor_weights,
            cost_model,
            constraints,
            model_config,
            initial_cash=initial_cash,
        )
        if not optimized_details.empty:
            detail_frames.append(optimized_details)
        rows.append(
            {
                "event_id": event["event_id"],
                "sport": event.get("sport", ""),
                "event_name": event.get("event_name", ""),
                "event_date": event.get("event_date", ""),
                "layer_action": used_params.get("action", "trade"),
                "entry_days_before_event": used_params.get("entry_days_before_event", ""),
                "exit_days_before_event": used_params.get("exit_days_before_event", ""),
                "top_k": used_params.get("top_k", ""),
                "midterm_return": midterm_ret,
                "optimized_return": optimized_ret,
                "excess_vs_midterm": optimized_ret - midterm_ret,
                "avoided_loss_vs_midterm": max(0.0, -midterm_ret) if used_params.get("action") == "avoid" else 0.0,
                "event_layer_reason": used_params.get("event_layer_reason", event.get("diagnosis_note", "")),
            }
        )
    details = pd.concat(detail_frames, ignore_index=True) if detail_frames else pd.DataFrame()
    return pd.DataFrame(rows), details


def scan_event_layer_windows(
    frames: Dict[str, pd.DataFrame],
    universe: List[Dict[str, Any]],
    events: Iterable[Dict[str, Any]],
    exposure_by_event: Dict[str, Dict[str, float]],
    base_params: Dict[str, Any],
    factor_weights: Dict[str, float],
    cost_model: CostModel,
    constraints: Dict[str, Any],
    model_config: Dict[str, Any],
    *,
    initial_cash: float,
    entry_grid: Iterable[int] = (120, 90, 75, 60, 45, 30, 20, 10),
    exit_grid: Iterable[int] = (20, 10, 5, 0, -5),
    top_k_grid: Iterable[int] = (1, 2, 3, 5),
    min_exposure_grid: Iterable[float] = (0.0, 0.1, 0.2, 0.3, 0.4),
) -> pd.DataFrame:
    rows: List[Dict[str, Any]] = []
    for event in events:
        for entry_days in entry_grid:
            for exit_days in exit_grid:
                if entry_days <= exit_days:
                    continue
                for top_k in top_k_grid:
                    for min_exposure in min_exposure_grid:
                        trial_model = dict(model_config)
                        trial_rules = dict(trial_model.get("event_layer_rules", {}))
                        trial_rules[event.get("sport", "")] = {
                            "action": "trade",
                            "entry_days_before_event": int(entry_days),
                            "exit_days_before_event": int(exit_days),
                            "top_k": int(top_k),
                            "min_exposure_for_trade": float(min_exposure),
                            "reason": "压力扫描：强制长仓检验该事件是否存在可解释正收益窗口。",
                        }
                        trial_model["event_layer_rules"] = trial_rules
                        ret, details, _ = evaluate_layered_event(
                            frames,
                            universe,
                            exposure_by_event,
                            event,
                            base_params,
                            factor_weights,
                            cost_model,
                            constraints,
                            trial_model,
                            initial_cash=initial_cash,
                        )
                        rows.append(
                            {
                                "event_id": event["event_id"],
                                "sport": event.get("sport", ""),
                                "entry_days_before_event": int(entry_days),
                                "exit_days_before_event": int(exit_days),
                                "top_k": int(top_k),
                                "min_exposure_for_trade": float(min_exposure),
                                "return_on_initial_cash": float(ret),
                                "trade_count": int(len(details)),
                                "tickers": ",".join(details["ticker"].astype(str).tolist()) if not details.empty else "",
                            }
                        )
    return pd.DataFrame(rows)


def fetch_holding_frames(
    universe_by_ticker: Dict[str, Dict[str, Any]],
    tickers: Iterable[str],
    *,
    end_date: str,
    datalen: int = 1800,
) -> Dict[str, pd.DataFrame]:
    provider = SinaDailyProvider(datalen=datalen)
    frames: Dict[str, pd.DataFrame] = {}
    for ticker in sorted(set(tickers)):
        meta = universe_by_ticker.get(ticker)
        if not meta:
            continue
        frames[meta["sina_symbol"]] = provider.fetch_daily(meta["sina_symbol"], end=end_date)
    return frames
