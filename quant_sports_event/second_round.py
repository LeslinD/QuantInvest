from __future__ import annotations

import json
from pathlib import Path
from typing import Any, Dict, Iterable, List

import pandas as pd

from .config import ROOT, load_project_config, read_json, write_json
from .costs import CostModel
from .data import SinaDailyProvider
from .evaluation import MIDTERM_PDF_WEIGHTS, fetch_holding_frames, static_portfolio_return


DEFAULT_OVERLAY: Dict[str, Any] = {
    "event_id": "FIFA_WC_2026_OPEN",
    "long_horizon_days_threshold": 300,
    "direct_core_relation_types": ["official_fifa_sponsor"],
    "max_long_exposure_without_manual_heat": 0.10,
    "delay_non_core_when_manual_heat_pending": True,
    "market_hedge": {
        "enabled": True,
        "benchmark_symbol": "sh000905",
        "benchmark_name": "中证500",
        "min_days_to_event": 0,
        "trigger_ret20_min": 0.03,
        "trigger_ret60_min": 0.0,
        "notional_cap": 0.10,
        "cost_rate_per_side": 0.00005,
    },
}


def _deep_merge(base: Dict[str, Any], override: Dict[str, Any]) -> Dict[str, Any]:
    out = dict(base)
    for key, value in override.items():
        if isinstance(value, dict) and isinstance(out.get(key), dict):
            out[key] = _deep_merge(out[key], value)
        else:
            out[key] = value
    return out


def load_second_round_overlay(root: Path = ROOT) -> Dict[str, Any]:
    path = root / "configs" / "strategy_final.json"
    if not path.exists():
        return dict(DEFAULT_OVERLAY)
    loaded = read_json(path)
    return _deep_merge(DEFAULT_OVERLAY, loaded.get("second_round_overlay", {}))


def event_days_to_event(events: Iterable[Dict[str, Any]], event_id: str, decision_date: str) -> int:
    decision = pd.to_datetime(decision_date)
    for event in events:
        if event.get("event_id") == event_id:
            return int((pd.to_datetime(event["event_date"]) - decision).days)
    raise ValueError(f"Event {event_id} not found")


def build_second_round_long_orders(
    final_orders: pd.DataFrame,
    universe: Iterable[Dict[str, Any]],
    *,
    overlay: Dict[str, Any],
    days_to_event: int,
    manual_attention_pending_count: int,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    if final_orders.empty:
        return pd.DataFrame(), pd.DataFrame()
    universe_by_ticker = {row["ticker"]: row for row in universe}
    direct_relations = set(overlay.get("direct_core_relation_types", []))
    delay_non_core = bool(overlay.get("delay_non_core_when_manual_heat_pending", True))
    manual_pending = manual_attention_pending_count > 0
    max_exposure = float(overlay.get("max_long_exposure_without_manual_heat", 0.10))

    rows: List[Dict[str, Any]] = []
    decisions: List[Dict[str, Any]] = []
    used_exposure = 0.0
    for _, order in final_orders.sort_values("final_target_weight", ascending=False).iterrows():
        ticker = str(order["ticker"])
        meta = universe_by_ticker.get(ticker, {})
        relations = set(meta.get("relation_types", []))
        is_direct_core = bool(relations & direct_relations)
        old_weight = float(order["final_target_weight"])
        action = "keep"
        reason = "保留直接赛事链条。"
        new_weight = old_weight
        if delay_non_core and manual_pending and not is_direct_core:
            action = "delay_until_attention_confirmed"
            reason = "该股票缺少直接世界杯权益和手工热度确认。"
            new_weight = 0.0
        elif used_exposure + new_weight > max_exposure:
            allowed = max(0.0, max_exposure - used_exposure)
            action = "cap_by_long_horizon_exposure"
            reason = "手工热度未补齐前控制长仓总风险。"
            new_weight = allowed
        used_exposure += new_weight
        decisions.append(
            {
                "ticker": ticker,
                "company": order.get("company", meta.get("name", "")),
                "industry": order.get("industry", meta.get("industry", "")),
                "relation_types": ";".join(sorted(relations)),
                "first_round_weight": old_weight,
                "second_round_weight": new_weight,
                "decision": action,
                "reason": reason,
            }
        )
        if new_weight <= 0:
            continue
        row = order.copy()
        row["second_round_target_weight"] = new_weight
        row["second_round_decision"] = action
        row["second_round_reason"] = reason
        rows.append(row.to_dict())
    return pd.DataFrame(rows), pd.DataFrame(decisions)


def index_regime(frame: pd.DataFrame, decision_date: str) -> Dict[str, float]:
    hist = frame[frame["date"] <= pd.to_datetime(decision_date)].copy()
    if hist.empty:
        return {"ret5": 0.0, "ret20": 0.0, "ret60": 0.0}
    last = hist.iloc[-1]

    def ret(days: int) -> float:
        sample = hist.tail(days + 1)
        if len(sample) <= 1:
            return 0.0
        return float(last["close"]) / float(sample.iloc[0]["close"]) - 1.0

    return {"ret5": ret(5), "ret20": ret(20), "ret60": ret(60)}


def hedge_decision(
    *,
    overlay: Dict[str, Any],
    regime: Dict[str, float],
    long_exposure: float,
    days_to_event: int,
) -> Dict[str, Any]:
    hedge = overlay.get("market_hedge", {})
    enabled = bool(hedge.get("enabled", True))
    min_days_to_event = int(hedge.get("min_days_to_event", 0))
    triggered = (
        enabled
        and long_exposure > 0
        and days_to_event >= min_days_to_event
        and float(regime.get("ret20", 0.0)) >= float(hedge.get("trigger_ret20_min", 0.03))
        and float(regime.get("ret60", 0.0)) >= float(hedge.get("trigger_ret60_min", 0.0))
    )
    notional = min(float(long_exposure), float(hedge.get("notional_cap", long_exposure))) if triggered else 0.0
    reason = (
        "宽基指数短期涨幅偏高，使用中证500股指期货空头代理降低市场回撤风险。"
        if triggered
        else "市场对冲条件未触发。"
    )
    return {
        "enabled": enabled,
        "triggered": triggered,
        "benchmark_symbol": hedge.get("benchmark_symbol", "sh000905"),
        "benchmark_name": hedge.get("benchmark_name", "中证500"),
        "hedge_notional_weight": notional,
        "cost_rate_per_side": float(hedge.get("cost_rate_per_side", 0.00005)),
        "reason": reason,
        **regime,
    }


def hedge_return(
    frame: pd.DataFrame,
    decision: Dict[str, Any],
    *,
    start_date: str,
    end_date: str,
    initial_cash: float,
) -> tuple[float, pd.DataFrame]:
    weight = float(decision.get("hedge_notional_weight", 0.0))
    rows = frame[frame["date"] >= pd.to_datetime(start_date)]
    end_rows = frame[frame["date"] <= pd.to_datetime(end_date)]
    if weight <= 0 or rows.empty or end_rows.empty:
        return 0.0, pd.DataFrame()
    start = rows.iloc[0]
    end = end_rows.iloc[-1]
    index_return = float(end["close"]) / float(start["open"]) - 1.0
    notional = initial_cash * weight
    cost = notional * float(decision.get("cost_rate_per_side", 0.0)) * 2.0
    pnl = -notional * index_return - cost
    detail = pd.DataFrame(
        [
            {
                "strategy": "second_round_hedge",
                "ticker": str(decision.get("benchmark_symbol", "")),
                "company": str(decision.get("benchmark_name", "")),
                "industry": "index_futures_proxy",
                "target_weight": -weight,
                "buy_date": pd.to_datetime(start["date"]).date().isoformat(),
                "sell_date": pd.to_datetime(end["date"]).date().isoformat(),
                "buy_price": float(start["open"]),
                "sell_price": float(end["close"]),
                "shares": 0,
                "buy_value": notional,
                "sell_value": notional * (1.0 - index_return),
                "buy_cost": cost / 2.0,
                "sell_cost": cost / 2.0,
                "pnl": pnl,
                "stock_return": -index_return,
            }
        ]
    )
    return pnl / initial_cash, detail


def run_second_round_evaluation(root: Path = ROOT, *, hold_end_date: str = "2026-06-16") -> Dict[str, Any]:
    root = Path(root)
    cfg = load_project_config(root)
    strategy = cfg.raw
    final_dir = root / "outputs" / "final"
    final_dir.mkdir(parents=True, exist_ok=True)
    overlay = load_second_round_overlay(root)
    first_round_summary_path = final_dir / "first_round_evaluation_summary.json"
    first_round_summary = read_json(first_round_summary_path) if first_round_summary_path.exists() else {}
    final_orders_path = final_dir / "paper_orders_final_2026-05-27.csv"
    final_orders = pd.read_csv(final_orders_path) if final_orders_path.exists() else pd.DataFrame()
    decision_date = strategy["paper_trading"]["decision_date"]
    order_date = strategy["paper_trading"]["order_date"]
    initial_cash = float(strategy["initial_cash"])
    cost_model = CostModel.from_config(strategy["costs"])
    manual_pending = int(first_round_summary.get("manual_attention_pending_count", 0))
    days_to_core_event = event_days_to_event(cfg.events, str(overlay["event_id"]), decision_date)
    second_orders, decision_audit = build_second_round_long_orders(
        final_orders,
        cfg.universe,
        overlay=overlay,
        days_to_event=days_to_core_event,
        manual_attention_pending_count=manual_pending,
    )
    second_orders_path = final_dir / "paper_orders_second_round_2026-05-27.csv"
    decision_audit_path = final_dir / "second_round_decision_audit.csv"
    second_orders.to_csv(second_orders_path, index=False)
    decision_audit.to_csv(decision_audit_path, index=False)

    universe_by_ticker = {row["ticker"]: row for row in cfg.universe}
    legacy_path = root / "configs" / "universe.json"
    if legacy_path.exists():
        for row in read_json(legacy_path):
            universe_by_ticker.setdefault(row["ticker"], row)
    first_weights = final_orders.set_index("ticker")["final_target_weight"].to_dict() if not final_orders.empty else {}
    second_weights = second_orders.set_index("ticker")["second_round_target_weight"].to_dict() if not second_orders.empty else {}
    tickers = set(MIDTERM_PDF_WEIGHTS) | set(first_weights) | set(second_weights)
    holding_frames = fetch_holding_frames(universe_by_ticker, tickers, end_date=hold_end_date)
    midterm_ret, midterm_details = static_portfolio_return(
        holding_frames,
        universe_by_ticker,
        MIDTERM_PDF_WEIGHTS,
        start_date=order_date,
        end_date=hold_end_date,
        initial_cash=initial_cash,
        cost_model=cost_model,
        lot_size=int(strategy["constraints"]["lot_size"]),
        buy_field="open",
        sell_field="close",
    )
    first_ret, first_details = static_portfolio_return(
        holding_frames,
        universe_by_ticker,
        first_weights,
        start_date=order_date,
        end_date=hold_end_date,
        initial_cash=initial_cash,
        cost_model=cost_model,
        lot_size=int(strategy["constraints"]["lot_size"]),
        buy_field="open",
        sell_field="close",
    )
    second_long_ret, second_long_details = static_portfolio_return(
        holding_frames,
        universe_by_ticker,
        second_weights,
        start_date=order_date,
        end_date=hold_end_date,
        initial_cash=initial_cash,
        cost_model=cost_model,
        lot_size=int(strategy["constraints"]["lot_size"]),
        buy_field="open",
        sell_field="close",
    )

    hedge_cfg = overlay.get("market_hedge", {})
    provider = SinaDailyProvider(datalen=1800)
    hedge_symbol = str(hedge_cfg.get("benchmark_symbol", "sh000905"))
    hedge_frame = provider.fetch_daily(hedge_symbol, end=hold_end_date)
    regime = index_regime(hedge_frame, decision_date)
    hedge_dec = hedge_decision(
        overlay=overlay,
        regime=regime,
        long_exposure=sum(float(v) for v in second_weights.values()),
        days_to_event=days_to_core_event,
    )
    hedge_ret, hedge_details = hedge_return(
        hedge_frame,
        hedge_dec,
        start_date=order_date,
        end_date=hold_end_date,
        initial_cash=initial_cash,
    )
    second_total_ret = second_long_ret + hedge_ret

    for label, details in [
        ("midterm", midterm_details),
        ("first_round", first_details),
        ("second_round_long", second_long_details),
    ]:
        if not details.empty:
            details.insert(0, "strategy", label)
    detail_frames = [frame for frame in [midterm_details, first_details, second_long_details, hedge_details] if not frame.empty]
    holding_comparison = pd.concat(detail_frames, ignore_index=True) if detail_frames else pd.DataFrame()
    holding_comparison_path = final_dir / f"second_round_holding_comparison_2026-05-27_to_{hold_end_date}.csv"
    holding_comparison.to_csv(holding_comparison_path, index=False)

    summary_rows = pd.DataFrame(
        [
            {"strategy": "midterm", "return": midterm_ret, "excess_vs_midterm": 0.0},
            {"strategy": "first_round", "return": first_ret, "excess_vs_midterm": first_ret - midterm_ret},
            {"strategy": "second_round_long", "return": second_long_ret, "excess_vs_midterm": second_long_ret - midterm_ret},
            {"strategy": "second_round_hedged", "return": second_total_ret, "excess_vs_midterm": second_total_ret - midterm_ret},
        ]
    )
    summary_table_path = final_dir / "second_round_holding_summary.csv"
    summary_rows.to_csv(summary_table_path, index=False)
    chart_dir = final_dir / "chart_data"
    chart_dir.mkdir(parents=True, exist_ok=True)
    summary_rows.to_csv(chart_dir / "second_round_holding_summary_chart.csv", index=False)
    if not decision_audit.empty:
        decision_audit[["ticker", "company", "first_round_weight", "second_round_weight", "decision"]].to_csv(
            chart_dir / "second_round_decision_chart.csv",
            index=False,
        )

    result = {
        "hold_end_date": hold_end_date,
        "event_id": overlay["event_id"],
        "days_to_event": days_to_core_event,
        "manual_attention_pending_count": manual_pending,
        "midterm_holding_return": midterm_ret,
        "first_round_holding_return": first_ret,
        "second_round_long_return": second_long_ret,
        "second_round_hedge_return": hedge_ret,
        "second_round_hedged_return": second_total_ret,
        "second_round_excess_vs_first_round": second_total_ret - first_ret,
        "second_round_excess_vs_midterm": second_total_ret - midterm_ret,
        "second_round_positive": second_total_ret > 0,
        "long_exposure": sum(float(v) for v in second_weights.values()),
        "hedge": hedge_dec,
        "removed_tickers": decision_audit.loc[
            decision_audit["second_round_weight"] <= 0, "ticker"
        ].astype(str).tolist()
        if not decision_audit.empty
        else [],
        "kept_tickers": list(second_weights),
        "output_files": {
            "second_round_orders": str(second_orders_path),
            "decision_audit": str(decision_audit_path),
            "holding_comparison": str(holding_comparison_path),
            "holding_summary": str(summary_table_path),
            "chart_data_dir": str(chart_dir),
        },
    }
    summary_path = final_dir / "second_round_evaluation_summary.json"
    result["output_files"]["summary"] = str(summary_path)
    write_json(summary_path, result)
    return result


def dumps_summary(summary: Dict[str, Any]) -> str:
    return json.dumps(summary, ensure_ascii=False, indent=2, sort_keys=True)
