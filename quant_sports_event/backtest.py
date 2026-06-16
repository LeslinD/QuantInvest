from __future__ import annotations

from dataclasses import dataclass
from itertools import product
from typing import Any, Dict, Iterable, List, Tuple

import numpy as np
import pandas as pd

from .costs import CostModel, round_lot_shares
from .factors import FEATURES, estimate_constrained_rankic_weights, factor_panel_for_date, score_panel
from .portfolio import build_target_weights


@dataclass(frozen=True)
class BacktestResult:
    params: Dict
    weights: Dict[str, float]
    trades: pd.DataFrame
    summary: Dict[str, float]
    samples: pd.DataFrame


def params_for_event(params: Dict, event: Dict, model_config: Dict | None = None) -> Dict:
    model_config = model_config or {}
    rules = model_config.get("event_layer_rules", {})
    rule = rules.get(event.get("event_id")) or rules.get(event.get("sport")) or {}
    event_params = dict(params)
    for key, value in rule.items():
        if key not in {"reason"}:
            event_params[key] = value
    event_params["event_layer"] = event.get("sport", "")
    event_params["event_layer_reason"] = rule.get("reason", event.get("diagnosis_note", ""))
    return event_params


def exposure_for_event(exposure_by_ticker: Dict[str, Any], event_id: str) -> Dict[str, float]:
    if not exposure_by_ticker:
        return {}
    first_value = next(iter(exposure_by_ticker.values()))
    if isinstance(first_value, dict):
        return {ticker: float(value) for ticker, value in exposure_by_ticker.get(event_id, {}).items()}
    return {ticker: float(value) for ticker, value in exposure_by_ticker.items()}


def _trading_date_on_or_before(df: pd.DataFrame, date: pd.Timestamp) -> pd.Timestamp | None:
    dates = df.loc[df["date"] <= date, "date"]
    return dates.max() if not dates.empty else None


def _trading_date_after(df: pd.DataFrame, date: pd.Timestamp) -> pd.Timestamp | None:
    dates = df.loc[df["date"] > date, "date"]
    return dates.min() if not dates.empty else None


def _price_on_or_after(df: pd.DataFrame, date: pd.Timestamp, field: str = "open") -> Tuple[pd.Timestamp, float] | Tuple[None, None]:
    future = df[df["date"] >= date]
    if future.empty:
        return None, None
    row = future.iloc[0]
    return row["date"], float(row[field])


def make_training_samples(
    frames: Dict[str, pd.DataFrame],
    universe: List[Dict],
    exposure_by_ticker: Dict[str, float],
    events: List[Dict],
    params: Dict,
    benchmark: pd.DataFrame | None = None,
    model_config: Dict | None = None,
) -> pd.DataFrame:
    rows = []
    ref_df = next(iter(frames.values()))
    for event in events:
        event_params = params_for_event(params, event, model_config)
        if event_params.get("action") == "avoid":
            continue
        if float(event.get("event_fit", 1.0)) < float(event_params.get("min_event_fit_for_trade", 0.0)):
            continue
        event_date = pd.to_datetime(event["event_date"])
        decision = _trading_date_on_or_before(ref_df, event_date - pd.Timedelta(days=event_params["entry_days_before_event"]))
        exit_signal = _trading_date_on_or_before(ref_df, event_date - pd.Timedelta(days=event_params["exit_days_before_event"]))
        if decision is None or exit_signal is None or exit_signal <= decision:
            continue
        panel = factor_panel_for_date(
            frames,
            universe,
            exposure_for_event(exposure_by_ticker, event["event_id"]),
            decision.strftime("%Y-%m-%d"),
            event_params["attention_lookback"],
            event_params["momentum_lookback"],
        )
        for _, p in panel.iterrows():
            df = frames[p["sina_symbol"]]
            buy_date = _trading_date_after(df, decision)
            sell_date = _trading_date_after(df, exit_signal)
            if buy_date is None or sell_date is None or sell_date <= buy_date:
                continue
            buy_price = float(df.loc[df["date"] == buy_date, "open"].iloc[0])
            sell_price = float(df.loc[df["date"] == sell_date, "open"].iloc[0])
            gross = sell_price / buy_price - 1
            bench_ret = 0.0
            if benchmark is not None and not benchmark.empty:
                b_buy = _trading_date_after(benchmark, decision)
                b_sell = _trading_date_after(benchmark, exit_signal)
                if b_buy is not None and b_sell is not None:
                    bp0 = float(benchmark.loc[benchmark["date"] == b_buy, "open"].iloc[0])
                    bp1 = float(benchmark.loc[benchmark["date"] == b_sell, "open"].iloc[0])
                    bench_ret = bp1 / bp0 - 1
            row = p.to_dict()
            row.update({"event_id": event["event_id"], "decision_date": decision, "label": gross - bench_ret})
            rows.append(row)
    return pd.DataFrame(rows)


def run_event_backtest(
    frames: Dict[str, pd.DataFrame],
    universe: List[Dict],
    exposure_by_ticker: Dict[str, float],
    events: List[Dict],
    params: Dict,
    cost_model: CostModel,
    constraints: Dict,
    initial_cash: float,
    benchmark: pd.DataFrame | None = None,
    train_events: List[Dict] | None = None,
    model_config: Dict | None = None,
) -> BacktestResult:
    train_events = train_events if train_events is not None else events
    model_config = model_config or {}
    samples = make_training_samples(frames, universe, exposure_by_ticker, train_events, params, benchmark=benchmark, model_config=model_config)
    factor_weights = estimate_constrained_rankic_weights(
        samples,
        FEATURES,
        directions=model_config.get("factor_directions"),
        priors=model_config.get("factor_priors"),
        bounds=model_config.get("factor_weight_bounds"),
        prior_weight=float(params.get("rankic_prior_weight", model_config.get("rankic_prior_weight", 0.55))),
    )
    trades = []
    ref_df = next(iter(frames.values()))
    for event in events:
        event_params = params_for_event(params, event, model_config)
        if event_params.get("action") == "avoid":
            continue
        if float(event.get("event_fit", 1.0)) < float(event_params.get("min_event_fit_for_trade", 0.0)):
            continue
        event_date = pd.to_datetime(event["event_date"])
        decision = _trading_date_on_or_before(ref_df, event_date - pd.Timedelta(days=event_params["entry_days_before_event"]))
        exit_signal = _trading_date_on_or_before(ref_df, event_date - pd.Timedelta(days=event_params["exit_days_before_event"]))
        if decision is None or exit_signal is None or exit_signal <= decision:
            continue
        panel = factor_panel_for_date(
            frames,
            universe,
            exposure_for_event(exposure_by_ticker, event["event_id"]),
            decision.strftime("%Y-%m-%d"),
            event_params["attention_lookback"],
            event_params["momentum_lookback"],
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
        for _, row in targets.iterrows():
            df = frames[row["sina_symbol"]]
            buy_date = _trading_date_after(df, decision)
            sell_date = _trading_date_after(df, exit_signal)
            if buy_date is None or sell_date is None or sell_date <= buy_date:
                continue
            buy_price = float(df.loc[df["date"] == buy_date, "open"].iloc[0])
            sell_price = float(df.loc[df["date"] == sell_date, "open"].iloc[0])
            stop_hit = False
            effective_stop = float(event_params.get("stop_loss", 1.0))
            if "stop_loss" in event_params:
                ret_hist = df[df["date"] <= decision]["close"].pct_change().tail(20)
                vol_stop = float(event_params.get("stop_vol_multiplier", 0.0)) * float(ret_hist.std())
                effective_stop = max(float(event_params["stop_loss"]), vol_stop)
                effective_stop = min(effective_stop, float(event_params.get("max_effective_stop_loss", effective_stop)))
                stop_price = buy_price * (1 - effective_stop)
                hold = df[(df["date"] > buy_date) & (df["date"] <= sell_date)]
                stop_rows = hold[hold["low"] <= stop_price]
                if not stop_rows.empty:
                    stop_hit = True
                    sell_date = stop_rows.iloc[0]["date"]
                    # Conservative fill at stop price; actual backtests can
                    # later replace this with next-bar open or intraday VWAP.
                    sell_price = stop_price
            target_value = initial_cash * float(row["target_weight"])
            shares = round_lot_shares(target_value, buy_price, int(constraints["lot_size"]))
            if shares <= 0:
                continue
            buy_value = shares * buy_price
            sell_value = shares * sell_price
            avg_amount = float(df[df["date"] <= decision]["amount"].tail(20).mean())
            buy_cost = cost_model.estimate(buy_value, "buy", avg_amount=avg_amount)
            sell_cost = cost_model.estimate(sell_value, "sell", avg_amount=avg_amount)
            pnl = sell_value - buy_value - buy_cost - sell_cost
            trades.append(
                {
                    "event_id": event["event_id"],
                    "ticker": row["ticker"],
                    "company": row["name"],
                    "industry": row["industry"],
                    "decision_date": decision.date().isoformat(),
                    "buy_date": buy_date.date().isoformat(),
                    "sell_date": sell_date.date().isoformat(),
                    "exit_reason": "stop_loss" if stop_hit else "event_exit",
                    "event_fit": float(event.get("event_fit", 1.0)),
                    "event_layer": event_params.get("event_layer", ""),
                    "event_layer_action": event_params.get("action", "trade"),
                    "event_layer_reason": event_params.get("event_layer_reason", ""),
                    "effective_stop_loss": effective_stop,
                    "target_weight": float(row["target_weight"]),
                    "shares": int(shares),
                    "buy_price": buy_price,
                    "sell_price": sell_price,
                    "gross_return": sell_price / buy_price - 1,
                    "net_pnl": pnl,
                    "net_return_on_initial_cash": pnl / initial_cash,
                    "buy_cost": buy_cost,
                    "sell_cost": sell_cost,
                    "alpha_score": float(row["alpha_score"]),
                }
            )
    trades_df = pd.DataFrame(trades)
    summary = summarize_trades(trades_df, initial_cash, benchmark=benchmark)
    return BacktestResult(params=params, weights=factor_weights, trades=trades_df, summary=summary, samples=samples)


def summarize_trades(trades: pd.DataFrame, initial_cash: float, benchmark: pd.DataFrame | None = None) -> Dict[str, float]:
    if trades.empty:
        return {
            "num_trades": 0,
            "net_return": 0.0,
            "gross_return_mean": 0.0,
            "win_rate": 0.0,
            "total_cost": 0.0,
        }
    total_pnl = float(trades["net_pnl"].sum())
    costs = float(trades["buy_cost"].sum() + trades["sell_cost"].sum())
    return {
        "num_trades": float(len(trades)),
        "net_return": total_pnl / initial_cash,
        "gross_return_mean": float(trades["gross_return"].mean()),
        "win_rate": float((trades["net_pnl"] > 0).mean()),
        "total_cost": costs,
        "cost_to_initial_cash": costs / initial_cash,
        "avg_target_weight": float(trades["target_weight"].mean()),
    }


def _benchmark_return(benchmark: pd.DataFrame, buy_date: str, sell_date: str) -> float:
    if benchmark is None or benchmark.empty:
        return 0.0
    buy = pd.to_datetime(buy_date)
    sell = pd.to_datetime(sell_date)
    buy_rows = benchmark[benchmark["date"] >= buy]
    sell_rows = benchmark[benchmark["date"] >= sell]
    if buy_rows.empty or sell_rows.empty:
        return 0.0
    buy_price = float(buy_rows.iloc[0]["open"])
    sell_price = float(sell_rows.iloc[0]["open"])
    if buy_price <= 0:
        return 0.0
    return sell_price / buy_price - 1


def benchmark_comparison(
    trades: pd.DataFrame,
    benchmarks: Dict[str, pd.DataFrame],
    *,
    benchmark_names: Dict[str, str] | None = None,
) -> pd.DataFrame:
    benchmark_names = benchmark_names or {}
    strategy_return = float(trades["net_return_on_initial_cash"].sum()) if not trades.empty else 0.0
    rows = []
    for symbol, frame in benchmarks.items():
        implied_return = 0.0
        weighted_gross = 0.0
        observations = 0
        if not trades.empty:
            for _, trade in trades.iterrows():
                bench_ret = _benchmark_return(frame, trade["buy_date"], trade["sell_date"])
                implied_return += float(trade["target_weight"]) * bench_ret
                weighted_gross += bench_ret
                observations += 1
        rows.append(
            {
                "benchmark_symbol": symbol,
                "benchmark_name": benchmark_names.get(symbol, symbol),
                "strategy_net_return_on_initial_cash": strategy_return,
                "benchmark_event_book_return": implied_return,
                "benchmark_mean_trade_window_return": weighted_gross / observations if observations else 0.0,
                "excess_vs_benchmark_event_book": strategy_return - implied_return,
                "observations": observations,
            }
        )
    return pd.DataFrame(rows).sort_values("excess_vs_benchmark_event_book", ascending=False).reset_index(drop=True)


def walk_forward_search(
    frames: Dict[str, pd.DataFrame],
    universe: List[Dict],
    exposure_by_ticker: Dict[str, float],
    events: List[Dict],
    grid: Dict[str, List],
    cost_model: CostModel,
    constraints: Dict,
    initial_cash: float,
    benchmark: pd.DataFrame | None = None,
    model_config: Dict | None = None,
) -> Tuple[Dict, pd.DataFrame]:
    keys = list(grid)
    rows = []
    best_params = None
    best_score = -1e9
    ordered_events = sorted(events, key=lambda e: e["event_date"])
    for values in product(*[grid[k] for k in keys]):
        params = dict(zip(keys, values))
        validation_trades = []
        for i in range(1, len(ordered_events)):
            result = run_event_backtest(
                frames,
                universe,
                exposure_by_ticker,
                [ordered_events[i]],
                params,
                cost_model,
                constraints,
                initial_cash,
                benchmark=benchmark,
                train_events=ordered_events[:i],
                model_config=model_config,
            )
            if not result.trades.empty:
                holdout = result.trades.copy()
                holdout["validation_train_events"] = i
                validation_trades.append(holdout)
        if validation_trades:
            trades = pd.concat(validation_trades, ignore_index=True)
            summary = summarize_trades(trades, initial_cash, benchmark=benchmark)
        else:
            result = run_event_backtest(
                frames,
                universe,
                exposure_by_ticker,
                events,
                params,
                cost_model,
                constraints,
                initial_cash,
                benchmark=benchmark,
                model_config=model_config,
            )
            summary = dict(result.summary)
        full_result = run_event_backtest(
            frames,
            universe,
            exposure_by_ticker,
            events,
            params,
            cost_model,
            constraints,
            initial_cash,
            benchmark=benchmark,
            model_config=model_config,
        )
        full_summary = dict(full_result.summary)
        stop_rate = 0.0
        if not full_result.trades.empty and "exit_reason" in full_result.trades:
            stop_rate = float((full_result.trades["exit_reason"] == "stop_loss").mean())

        row = {f"validation_{k}": v for k, v in summary.items()}
        row.update({f"full_{k}": v for k, v in full_summary.items()})
        row["full_stop_loss_rate"] = stop_rate
        row.update(params)
        rows.append(row)
        score = (
            0.65 * row.get("validation_net_return", 0.0)
            + 0.35 * row.get("full_net_return", 0.0)
            - 0.25 * row.get("validation_cost_to_initial_cash", 0.0)
            - 0.15 * row.get("full_cost_to_initial_cash", 0.0)
            - 0.02 * stop_rate
        )
        row["selection_score"] = score
        if score > best_score and row.get("validation_num_trades", 0) >= 1:
            best_score = score
            best_params = params
    results = pd.DataFrame(rows).sort_values(["selection_score", "validation_net_return"], ascending=False).reset_index(drop=True)
    if best_params is None:
        best_params = {k: grid[k][0] for k in keys}
    return best_params, results


def expanding_walk_forward_validate(
    frames: Dict[str, pd.DataFrame],
    universe: List[Dict],
    exposure_by_ticker: Dict[str, float],
    events: List[Dict],
    params: Dict,
    cost_model: CostModel,
    constraints: Dict,
    initial_cash: float,
    benchmark: pd.DataFrame | None = None,
    model_config: Dict | None = None,
) -> Tuple[pd.DataFrame, Dict[str, float]]:
    ordered_events = sorted(events, key=lambda e: e["event_date"])
    validation_trades = []
    for i in range(1, len(ordered_events)):
        result = run_event_backtest(
            frames,
            universe,
            exposure_by_ticker,
            [ordered_events[i]],
            params,
            cost_model,
            constraints,
            initial_cash,
            benchmark=benchmark,
            train_events=ordered_events[:i],
            model_config=model_config,
        )
        if not result.trades.empty:
            holdout = result.trades.copy()
            holdout["validation_train_events"] = i
            validation_trades.append(holdout)
    if not validation_trades:
        empty = pd.DataFrame()
        return empty, summarize_trades(empty, initial_cash, benchmark=benchmark)
    trades = pd.concat(validation_trades, ignore_index=True)
    return trades, summarize_trades(trades, initial_cash, benchmark=benchmark)
