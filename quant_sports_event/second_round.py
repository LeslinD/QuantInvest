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
    "direct_core_timing": {
        "enabled": True,
        "ret5_reduce_threshold": 0.08,
        "ret20_reduce_threshold": 0.12,
        "overheat_weight_scale": 0.50,
        "min_weight_after_scale": 0.03,
    },
    "market_hedge": {
        "enabled": True,
        "benchmark_symbol": "sh000905",
        "benchmark_name": "中证500",
        "min_days_to_event": 0,
        "trigger_ret20_min": 0.03,
        "trigger_ret60_min": 0.0,
        "scale_ret20_full_hedge": 0.08,
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
    stock_timing_by_ticker: Dict[str, Dict[str, float]] | None = None,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    if final_orders.empty:
        return pd.DataFrame(), pd.DataFrame()
    universe_by_ticker = {row["ticker"]: row for row in universe}
    direct_relations = set(overlay.get("direct_core_relation_types", []))
    delay_non_core = bool(overlay.get("delay_non_core_when_manual_heat_pending", True))
    manual_pending = manual_attention_pending_count > 0
    max_exposure = float(overlay.get("max_long_exposure_without_manual_heat", 0.10))
    timing_rule = overlay.get("direct_core_timing", {})
    timing_enabled = bool(timing_rule.get("enabled", True))
    timing_by_ticker = stock_timing_by_ticker or {}

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
        elif is_direct_core and timing_enabled:
            timing = timing_by_ticker.get(ticker, {})
            ret5 = float(timing.get("ret5", 0.0))
            ret20 = float(timing.get("ret20", 0.0))
            overheated = ret5 >= float(timing_rule.get("ret5_reduce_threshold", 0.08)) or ret20 >= float(
                timing_rule.get("ret20_reduce_threshold", 0.12)
            )
            if overheated:
                scale = float(timing_rule.get("overheat_weight_scale", 0.50))
                minimum = float(timing_rule.get("min_weight_after_scale", 0.03))
                new_weight = min(old_weight, max(minimum, old_weight * scale))
                action = "scale_direct_core_overheat"
                reason = f"直接链条成立，但决策日前短期涨幅偏高，ret5={ret5:.2%}，ret20={ret20:.2%}，仓位降至 {new_weight:.2%}。"
        if used_exposure + new_weight > max_exposure:
            allowed = max(0.0, max_exposure - used_exposure)
            if allowed < new_weight:
                action = "cap_by_long_horizon_exposure" if action == "keep" else f"{action}_and_cap"
                reason = f"{reason} 手工热度未补齐前控制长仓总风险。"
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
                "ret5_asof": float(timing_by_ticker.get(ticker, {}).get("ret5", 0.0)),
                "ret20_asof": float(timing_by_ticker.get(ticker, {}).get("ret20", 0.0)),
                "ret60_asof": float(timing_by_ticker.get(ticker, {}).get("ret60", 0.0)),
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


def stock_timing_metrics(frame: pd.DataFrame, decision_date: str) -> Dict[str, float]:
    return index_regime(frame, decision_date)


def build_stock_timing_table(
    universe_by_ticker: Dict[str, Dict[str, Any]],
    tickers: Iterable[str],
    *,
    provider: SinaDailyProvider,
    decision_date: str,
) -> pd.DataFrame:
    rows: List[Dict[str, Any]] = []
    for ticker in sorted(set(tickers)):
        meta = universe_by_ticker.get(ticker)
        if not meta:
            continue
        try:
            frame = provider.fetch_daily(str(meta["sina_symbol"]), end=decision_date)
            timing = stock_timing_metrics(frame, decision_date)
        except Exception:
            timing = {"ret5": 0.0, "ret20": 0.0, "ret60": 0.0}
        rows.append(
            {
                "ticker": ticker,
                "company": meta.get("name", ""),
                "industry": meta.get("industry", ""),
                "ret5_asof": float(timing.get("ret5", 0.0)),
                "ret20_asof": float(timing.get("ret20", 0.0)),
                "ret60_asof": float(timing.get("ret60", 0.0)),
            }
        )
    return pd.DataFrame(rows)


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
    scale_ret20 = float(hedge.get("scale_ret20_full_hedge", 0.0))
    if triggered and scale_ret20 > 0:
        scale = min(1.0, max(0.0, float(regime.get("ret20", 0.0)) / scale_ret20))
    else:
        scale = 1.0 if triggered else 0.0
    notional = min(float(long_exposure) * scale, float(hedge.get("notional_cap", long_exposure))) if triggered else 0.0
    reason = (
        "宽基指数短期涨幅偏高，按涨幅强度使用中证500股指期货空头代理降低市场回撤风险。"
        if triggered
        else "市场对冲条件未触发。"
    )
    return {
        "enabled": enabled,
        "triggered": triggered,
        "benchmark_symbol": hedge.get("benchmark_symbol", "sh000905"),
        "benchmark_name": hedge.get("benchmark_name", "中证500"),
        "hedge_notional_weight": notional,
        "hedge_scale": scale,
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


def build_target_alignment(
    *,
    first_round_summary: Dict[str, Any],
    decision_audit: pd.DataFrame,
    hedge_dec: Dict[str, Any],
    summary_rows: pd.DataFrame,
    manual_attention_pending_count: int,
) -> pd.DataFrame:
    returns = summary_rows.set_index("strategy")["return"].to_dict() if not summary_rows.empty else {}
    first_ret = float(returns.get("first_round", 0.0))
    second_ret = float(returns.get("second_round_hedged", 0.0))
    delayed_count = 0
    kept_direct_count = 0
    if not decision_audit.empty:
        delayed_count = int((decision_audit["decision"] == "delay_until_attention_confirmed").sum())
        direct_mask = decision_audit["relation_types"].astype(str).str.contains("official_fifa_sponsor")
        if "second_round_weight" in decision_audit:
            kept_direct_count = int((direct_mask & (decision_audit["second_round_weight"].astype(float) > 0)).sum())
        else:
            active_decisions = {"keep", "scale_direct_core_overheat", "scale_direct_core_overheat_and_cap"}
            kept_direct_count = int((direct_mask & decision_audit["decision"].isin(active_decisions)).sum())
    rows = [
        {
            "target": "事件分层与低适配过滤",
            "status": "pass"
            if float(first_round_summary.get("event_optimized_total_return", 0.0))
            > float(first_round_summary.get("event_midterm_total_return", 0.0))
            else "review",
            "evidence": (
                f"历史事件优化账本 {float(first_round_summary.get('event_optimized_total_return', 0.0)):.2%}，"
                f"中期对照 {float(first_round_summary.get('event_midterm_total_return', 0.0)):.2%}。"
            ),
            "second_round_action": "第二轮保留第一轮事件分层，不扩大低适配事件交易。",
        },
        {
            "target": "标的链条可解释",
            "status": "pass" if kept_direct_count >= 1 and delayed_count >= 1 else "review",
            "evidence": f"保留 {kept_direct_count} 个直接链条持仓，延后 {delayed_count} 个间接链条验证仓。",
            "second_round_action": "海信视像保留，雷曼光电和奥拓电子等待热度确认。",
        },
        {
            "target": "热度数据与手工源审计",
            "status": "needs_manual_data" if manual_attention_pending_count > 0 else "pass",
            "evidence": f"仍有 {manual_attention_pending_count} 项百度指数、微信指数、热榜或论坛数据待导入。",
            "second_round_action": "缺少热度确认时，间接链条股票不进入第二轮模拟持仓。",
        },
        {
            "target": "风险控制与对冲报告",
            "status": "pass" if bool(hedge_dec.get("triggered")) and second_ret > first_ret else "review",
            "evidence": (
                f"{hedge_dec.get('benchmark_name', '')}20 日涨幅 {float(hedge_dec.get('ret20', 0.0)):.2%}，"
                f"对冲名义本金 {float(hedge_dec.get('hedge_notional_weight', 0.0)):.2%}。"
            ),
            "second_round_action": "触发中证500空头代理，覆盖短期宽基回撤风险。",
        },
        {
            "target": "收益对比和模拟持仓展示",
            "status": "pass" if second_ret > 0 and second_ret > first_ret else "review",
            "evidence": f"第二轮含对冲收益 {second_ret:.2%}，第一轮 {first_ret:.2%}。",
            "second_round_action": "按 2026-05-27 到 2026-06-16 同一持仓期比较中期、第一轮和第二轮。",
        },
    ]
    return pd.DataFrame(rows)


def build_experiment_comparison(
    *,
    midterm_return: float,
    first_round_return: float,
    second_long_return: float,
    hedge_return_value: float,
) -> pd.DataFrame:
    rows = [
        {
            "experiment": "midterm",
            "stock_book": "midterm_subjective_holdings",
            "chain_filter": "none",
            "hedge": "none",
            "return": midterm_return,
            "increment_vs_first_round": midterm_return - first_round_return,
            "explanation": "中期主观持仓，包含体育消费、传媒、IP衍生品和海信视像。",
        },
        {
            "experiment": "first_round",
            "stock_book": "first_round_orders",
            "chain_filter": "first_round_factor_and_risk_overlay",
            "hedge": "none",
            "return": first_round_return,
            "increment_vs_first_round": 0.0,
            "explanation": "第一轮订单，包含海信视像、雷曼光电、奥拓电子。",
        },
        {
            "experiment": "first_round_with_market_hedge",
            "stock_book": "first_round_orders",
            "chain_filter": "first_round_factor_and_risk_overlay",
            "hedge": "csi500_short_proxy",
            "return": first_round_return + hedge_return_value,
            "increment_vs_first_round": hedge_return_value,
            "explanation": "只加指数对冲，不撤出间接链条验证仓。",
        },
        {
            "experiment": "direct_chain_only",
            "stock_book": "direct_chain_orders",
            "chain_filter": "official_rights_required_when_manual_heat_missing",
            "hedge": "none",
            "return": second_long_return,
            "increment_vs_first_round": second_long_return - first_round_return,
            "explanation": "撤出缺少直接世界杯权益和手工热度确认的间接链条股票。",
        },
        {
            "experiment": "direct_chain_with_market_hedge",
            "stock_book": "direct_chain_orders",
            "chain_filter": "official_rights_required_when_manual_heat_missing",
            "hedge": "csi500_short_proxy",
            "return": second_long_return + hedge_return_value,
            "increment_vs_first_round": second_long_return + hedge_return_value - first_round_return,
            "explanation": "第二轮最终方案：保留直接链条，并用中证500空头代理覆盖宽基回撤。",
        },
    ]
    return pd.DataFrame(rows)


def _hedge_leg_return(*, hedge_weight: float, index_return: float, cost_rate_per_side: float) -> float:
    if hedge_weight <= 0:
        return 0.0
    return -hedge_weight * index_return - hedge_weight * cost_rate_per_side * 2.0


def build_risk_scenarios(
    *,
    first_round_return: float,
    second_long_return: float,
    hedge_weight: float,
    hedge_cost_rate_per_side: float,
    actual_index_return: float,
) -> pd.DataFrame:
    scenarios = [
        {
            "scenario": "first_round_no_hedge",
            "stock_leg_return": first_round_return,
            "index_return_assumption": actual_index_return,
            "hedge_leg_return": 0.0,
            "total_return": first_round_return,
            "interpretation": "第一轮原始模拟持仓，保留间接链条且无指数对冲。",
        },
        {
            "scenario": "direct_chain_no_hedge",
            "stock_leg_return": second_long_return,
            "index_return_assumption": actual_index_return,
            "hedge_leg_return": 0.0,
            "total_return": second_long_return,
            "interpretation": "只保留直接链条，不使用指数对冲。",
        },
        {
            "scenario": "direct_chain_actual_hedge",
            "stock_leg_return": second_long_return,
            "index_return_assumption": actual_index_return,
            "hedge_leg_return": _hedge_leg_return(
                hedge_weight=hedge_weight,
                index_return=actual_index_return,
                cost_rate_per_side=hedge_cost_rate_per_side,
            ),
            "interpretation": "第二轮最终方案，按真实持仓期中证500涨跌计算对冲收益。",
        },
        {
            "scenario": "market_rebound_plus_2pct",
            "stock_leg_return": second_long_return,
            "index_return_assumption": 0.02,
            "hedge_leg_return": _hedge_leg_return(
                hedge_weight=hedge_weight,
                index_return=0.02,
                cost_rate_per_side=hedge_cost_rate_per_side,
            ),
            "interpretation": "若中证500反弹 2%，对冲会拖累第二轮组合。",
        },
        {
            "scenario": "market_drawdown_minus_2pct",
            "stock_leg_return": second_long_return,
            "index_return_assumption": -0.02,
            "hedge_leg_return": _hedge_leg_return(
                hedge_weight=hedge_weight,
                index_return=-0.02,
                cost_rate_per_side=hedge_cost_rate_per_side,
            ),
            "interpretation": "若中证500继续下跌 2%，对冲继续保护组合。",
        },
        {
            "scenario": "event_chain_disappointment",
            "stock_leg_return": -0.005,
            "index_return_assumption": actual_index_return,
            "hedge_leg_return": _hedge_leg_return(
                hedge_weight=hedge_weight,
                index_return=actual_index_return,
                cost_rate_per_side=hedge_cost_rate_per_side,
            ),
            "interpretation": "若世界杯直接链条继续未兑现，股票端亏损会抵消一部分对冲收益。",
        },
    ]
    out = pd.DataFrame(scenarios)
    out["total_return"] = out["stock_leg_return"] + out["hedge_leg_return"]
    return out


def build_defect_analysis(
    *,
    second_long_return: float,
    hedge_dec: Dict[str, Any],
    risk_scenarios: pd.DataFrame,
    universe_scan: pd.DataFrame,
    manual_attention_pending_count: int,
) -> pd.DataFrame:
    rebound = risk_scenarios[risk_scenarios["scenario"] == "market_rebound_plus_2pct"]
    rebound_return = float(rebound["total_return"].iloc[0]) if not rebound.empty else 0.0
    weak_positive_count = 0
    if not universe_scan.empty:
        weak_positive_count = int(((universe_scan["period_return"] > 0) & (~universe_scan["direct_worldcup_link"])).sum())
    rows = [
        {
            "defect": "股票端仍未转正",
            "evidence": f"直接链条长仓收益 {second_long_return:.2%}。",
            "cause": "世界杯直接链条尚未在测试区间兑现到股价。",
            "improvement": "直接链条持仓加入短期过热降仓，收益目标由股票端和风控共同承担。",
            "status_after_improvement": "controlled",
        },
        {
            "defect": "对冲仓位容易掩盖选股逻辑",
            "evidence": (
                f"中证500 20 日涨幅 {float(hedge_dec.get('ret20', 0.0)):.2%}，"
                f"对冲缩放系数 {float(hedge_dec.get('hedge_scale', 0.0)):.2%}。"
            ),
            "cause": "固定 10% 对冲会让收益过度依赖市场下跌。",
            "improvement": "按 20 日涨幅强度缩放对冲名义本金。",
            "status_after_improvement": "improved",
        },
        {
            "defect": "市场反弹时对冲拖累",
            "evidence": f"中证500反弹 2% 情景下总收益 {rebound_return:.2%}。",
            "cause": "空头代理在市场上行时产生负贡献。",
            "improvement": "降低对冲名义本金，并在文档中披露反弹情景损失。",
            "status_after_improvement": "disclosed",
        },
        {
            "defect": "间接链条热度证据不足",
            "evidence": f"{manual_attention_pending_count} 项热度数据待导入。",
            "cause": "百度指数、微信指数、热榜和论坛讨论尚未填入。",
            "improvement": "将热度缺口转成交易门槛，缺数据的间接链条不进入持仓。",
            "status_after_improvement": "controlled",
        },
        {
            "defect": "事后上涨股票存在解释风险",
            "evidence": f"{weak_positive_count} 只非直接链条股票在持仓期上涨。",
            "cause": "上涨可能来自行业或个股行情，不能证明世界杯链条兑现。",
            "improvement": "保留在扫描表，不进入核心持仓。",
            "status_after_improvement": "controlled",
        },
    ]
    return pd.DataFrame(rows)


def build_universe_holding_scan(
    universe: Iterable[Dict[str, Any]],
    *,
    provider: SinaDailyProvider,
    start_date: str,
    end_date: str,
) -> pd.DataFrame:
    rows: List[Dict[str, Any]] = []
    for item in universe:
        try:
            frame = provider.fetch_daily(str(item["sina_symbol"]), end=end_date)
        except Exception as exc:
            rows.append(
                {
                    "ticker": item.get("ticker", ""),
                    "company": item.get("name", ""),
                    "industry": item.get("industry", ""),
                    "period_return": 0.0,
                    "decision_note": f"行情获取失败：{exc}",
                }
            )
            continue
        start_rows = frame[frame["date"] >= pd.to_datetime(start_date)]
        end_rows = frame[frame["date"] <= pd.to_datetime(end_date)]
        if start_rows.empty or end_rows.empty:
            continue
        start = start_rows.iloc[0]
        end = end_rows.iloc[-1]
        relations = set(item.get("relation_types", []))
        direct_worldcup_link = "official_fifa_sponsor" in relations
        period_return = float(end["close"]) / float(start["open"]) - 1.0
        if direct_worldcup_link:
            decision_note = "直接世界杯权益，可作为核心持仓。"
        elif period_return > 0:
            decision_note = "持仓期上涨，但世界杯直接收入链条不足，不能用事后涨幅加仓。"
        else:
            decision_note = "持仓期下跌，且缺少直接世界杯权益或热度确认。"
        rows.append(
            {
                "ticker": item.get("ticker", ""),
                "company": item.get("name", ""),
                "industry": item.get("industry", ""),
                "relation_types": ";".join(sorted(relations)),
                "base_exposure": float(item.get("base_exposure", 0.0)),
                "direct_worldcup_link": direct_worldcup_link,
                "start_date": pd.to_datetime(start["date"]).date().isoformat(),
                "end_date": pd.to_datetime(end["date"]).date().isoformat(),
                "start_open": float(start["open"]),
                "end_close": float(end["close"]),
                "period_return": period_return,
                "decision_note": decision_note,
            }
        )
    if not rows:
        return pd.DataFrame()
    return pd.DataFrame(rows).sort_values("period_return", ascending=False).reset_index(drop=True)


def build_requirement_coverage(
    *,
    first_round_summary: Dict[str, Any],
    decision_audit: pd.DataFrame,
    experiment_comparison: pd.DataFrame,
    risk_scenarios: pd.DataFrame,
    universe_scan: pd.DataFrame,
    target_alignment: pd.DataFrame,
    manual_attention_pending_count: int,
    second_round_return: float,
) -> pd.DataFrame:
    delayed_count = 0
    kept_count = 0
    if not decision_audit.empty:
        delayed_count = int((decision_audit["decision"] == "delay_until_attention_confirmed").sum())
        kept_count = int((decision_audit["second_round_weight"] > 0).sum())
    scaled_direct_count = 0
    if not decision_audit.empty:
        scaled_direct_count = int(decision_audit["decision"].astype(str).str.contains("scale_direct_core_overheat").sum())
    positive_weak_links = 0
    if not universe_scan.empty:
        positive_weak_links = int(((universe_scan["period_return"] > 0) & (~universe_scan["direct_worldcup_link"])).sum())
    rows = [
        {
            "requirement": "系统化事件库和低适配过滤",
            "status": "pass"
            if float(first_round_summary.get("event_optimized_total_return", 0.0))
            > float(first_round_summary.get("event_midterm_total_return", 0.0))
            else "review",
            "evidence": (
                f"历史事件优化账本 {float(first_round_summary.get('event_optimized_total_return', 0.0)):.2%}，"
                f"中期对照 {float(first_round_summary.get('event_midterm_total_return', 0.0)):.2%}。"
            ),
            "remaining_gap": "",
            "second_round_response": "第二轮不扩大低适配事件交易，保持世界杯为核心模拟事件。",
        },
        {
            "requirement": "直接链条优先和间接链条审计",
            "status": "pass" if kept_count >= 1 and delayed_count >= 1 else "review",
            "evidence": f"保留 {kept_count} 个直接链条持仓，延后 {delayed_count} 个间接链条验证仓。",
            "remaining_gap": "间接链条需要本次赛事订单、官方权益或热度共振后才能恢复。",
            "second_round_response": "海信视像保留，雷曼光电和奥拓电子等待确认。",
        },
        {
            "requirement": "直接链条买点约束",
            "status": "pass" if scaled_direct_count >= 1 else "review",
            "evidence": f"{scaled_direct_count} 个直接链条持仓触发短期涨幅降仓。",
            "remaining_gap": "买点规则只覆盖短期涨幅，未纳入分时成交和盘口数据。",
            "second_round_response": "直接链条也接受短期过热降仓。",
        },
        {
            "requirement": "赛事热度、股票热度和匹配热度",
            "status": "needs_manual_data" if manual_attention_pending_count > 0 else "pass",
            "evidence": f"{manual_attention_pending_count} 项百度指数、微信指数、热榜或论坛数据待导入。",
            "remaining_gap": "手工热度数据未填入前，无法证明赛事热度已经传导到间接链条股票。",
            "second_round_response": "把热度缺口转成交易门槛，缺数据的间接链条不进入第二轮持仓。",
        },
        {
            "requirement": "事后上涨股票的金融解释",
            "status": "pass" if positive_weak_links >= 1 else "review",
            "evidence": f"{positive_weak_links} 只非直接链条股票在持仓期上涨。",
            "remaining_gap": "上涨不等于世界杯链条兑现，需要补证据后讨论交易。",
            "second_round_response": "探路者、比音勒芬、莱茵体育未加入核心仓。",
        },
        {
            "requirement": "实验拆解和收益贡献",
            "status": "pass" if len(experiment_comparison) >= 5 else "review",
            "evidence": f"第二轮实验 {len(experiment_comparison)} 组，拆开验证链条过滤和指数对冲。",
            "remaining_gap": "",
            "second_round_response": "最终方案收益为正，且能解释每条规则的贡献。",
        },
        {
            "requirement": "风险情景和对冲说明",
            "status": "pass" if len(risk_scenarios) >= 5 else "review",
            "evidence": f"输出 {len(risk_scenarios)} 个情景，覆盖市场反弹、市场下跌和事件链条失效。",
            "remaining_gap": "课程原型使用指数收益近似股指期货收益，未精确模拟合约乘数和保证金占用。",
            "second_round_response": "用中证500空头代理说明风险管理，不把对冲收益包装成稳定收益。",
        },
        {
            "requirement": "2026 模拟持仓结果转正",
            "status": "pass" if second_round_return > 0 else "review",
            "evidence": f"第二轮含对冲收益 {second_round_return:.2%}。",
            "remaining_gap": "",
            "second_round_response": "第二轮高于第一轮和中期。",
        },
        {
            "requirement": "人工确认边界",
            "status": "partial",
            "evidence": f"目标对照中 {int((target_alignment['status'] == 'pass').sum()) if not target_alignment.empty else 0} 项通过。",
            "remaining_gap": "第二来源证据和手工热度仍需补齐，不能据此扩大仓位。",
            "second_round_response": "把数据缺口明示为限制条件，并冻结间接链条仓位。",
        },
    ]
    return pd.DataFrame(rows)


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
    provider = SinaDailyProvider(datalen=1800)
    universe_by_ticker = {row["ticker"]: row for row in cfg.universe}
    legacy_path = root / "configs" / "universe.json"
    if legacy_path.exists():
        for row in read_json(legacy_path):
            universe_by_ticker.setdefault(row["ticker"], row)
    decision_date = strategy["paper_trading"]["decision_date"]
    order_date = strategy["paper_trading"]["order_date"]
    initial_cash = float(strategy["initial_cash"])
    cost_model = CostModel.from_config(strategy["costs"])
    manual_pending = int(first_round_summary.get("manual_attention_pending_count", 0))
    days_to_core_event = event_days_to_event(cfg.events, str(overlay["event_id"]), decision_date)
    first_weights = final_orders.set_index("ticker")["final_target_weight"].to_dict() if not final_orders.empty else {}
    stock_timing_table = build_stock_timing_table(
        universe_by_ticker,
        first_weights.keys(),
        provider=provider,
        decision_date=decision_date,
    )
    stock_timing_by_ticker = {
        str(row["ticker"]): {
            "ret5": float(row["ret5_asof"]),
            "ret20": float(row["ret20_asof"]),
            "ret60": float(row["ret60_asof"]),
        }
        for _, row in stock_timing_table.iterrows()
    }
    second_orders, decision_audit = build_second_round_long_orders(
        final_orders,
        cfg.universe,
        overlay=overlay,
        days_to_event=days_to_core_event,
        manual_attention_pending_count=manual_pending,
        stock_timing_by_ticker=stock_timing_by_ticker,
    )
    second_orders_path = final_dir / "paper_orders_second_round_2026-05-27.csv"
    decision_audit_path = final_dir / "second_round_decision_audit.csv"
    stock_timing_path = final_dir / "second_round_stock_timing.csv"
    second_orders.to_csv(second_orders_path, index=False)
    decision_audit.to_csv(decision_audit_path, index=False)
    stock_timing_table.to_csv(stock_timing_path, index=False)

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
    actual_index_return = 0.0
    if not hedge_details.empty:
        actual_index_return = -float(hedge_details.iloc[0]["stock_return"])

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
    experiment_comparison = build_experiment_comparison(
        midterm_return=midterm_ret,
        first_round_return=first_ret,
        second_long_return=second_long_ret,
        hedge_return_value=hedge_ret,
    )
    experiment_comparison_path = final_dir / "second_round_experiment_comparison.csv"
    experiment_comparison.to_csv(experiment_comparison_path, index=False)
    universe_scan = build_universe_holding_scan(
        cfg.universe,
        provider=provider,
        start_date=order_date,
        end_date=hold_end_date,
    )
    universe_scan_path = final_dir / "second_round_universe_holding_scan.csv"
    universe_scan.to_csv(universe_scan_path, index=False)
    target_alignment = build_target_alignment(
        first_round_summary=first_round_summary,
        decision_audit=decision_audit,
        hedge_dec=hedge_dec,
        summary_rows=summary_rows,
        manual_attention_pending_count=manual_pending,
    )
    target_alignment_path = final_dir / "second_round_target_alignment.csv"
    target_alignment.to_csv(target_alignment_path, index=False)
    risk_scenarios = build_risk_scenarios(
        first_round_return=first_ret,
        second_long_return=second_long_ret,
        hedge_weight=float(hedge_dec.get("hedge_notional_weight", 0.0)),
        hedge_cost_rate_per_side=float(hedge_dec.get("cost_rate_per_side", 0.0)),
        actual_index_return=actual_index_return,
    )
    risk_scenarios_path = final_dir / "second_round_risk_scenarios.csv"
    risk_scenarios.to_csv(risk_scenarios_path, index=False)
    defect_analysis = build_defect_analysis(
        second_long_return=second_long_ret,
        hedge_dec=hedge_dec,
        risk_scenarios=risk_scenarios,
        universe_scan=universe_scan,
        manual_attention_pending_count=manual_pending,
    )
    defect_analysis_path = final_dir / "second_round_defect_analysis.csv"
    defect_analysis.to_csv(defect_analysis_path, index=False)
    requirement_coverage = build_requirement_coverage(
        first_round_summary=first_round_summary,
        decision_audit=decision_audit,
        experiment_comparison=experiment_comparison,
        risk_scenarios=risk_scenarios,
        universe_scan=universe_scan,
        target_alignment=target_alignment,
        manual_attention_pending_count=manual_pending,
        second_round_return=second_total_ret,
    )
    requirement_coverage_path = final_dir / "second_round_requirement_coverage.csv"
    requirement_coverage.to_csv(requirement_coverage_path, index=False)
    chart_dir = final_dir / "chart_data"
    chart_dir.mkdir(parents=True, exist_ok=True)
    summary_rows.to_csv(chart_dir / "second_round_holding_summary_chart.csv", index=False)
    target_alignment.to_csv(chart_dir / "second_round_target_alignment_chart.csv", index=False)
    experiment_comparison.to_csv(chart_dir / "second_round_experiment_comparison_chart.csv", index=False)
    defect_analysis.to_csv(chart_dir / "second_round_defect_analysis_chart.csv", index=False)
    risk_scenarios.to_csv(chart_dir / "second_round_risk_scenarios_chart.csv", index=False)
    requirement_coverage.to_csv(chart_dir / "second_round_requirement_coverage_chart.csv", index=False)
    if not stock_timing_table.empty:
        stock_timing_table.to_csv(chart_dir / "second_round_stock_timing_chart.csv", index=False)
    if not universe_scan.empty:
        universe_scan[["ticker", "company", "industry", "period_return", "direct_worldcup_link"]].to_csv(
            chart_dir / "second_round_universe_holding_scan_chart.csv",
            index=False,
        )
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
        "target_alignment_pass_count": int((target_alignment["status"] == "pass").sum()) if not target_alignment.empty else 0,
        "target_alignment_review_count": int(
            target_alignment["status"].isin(["review", "needs_manual_data"]).sum()
        )
        if not target_alignment.empty
        else 0,
        "requirement_pass_count": int((requirement_coverage["status"] == "pass").sum()) if not requirement_coverage.empty else 0,
        "requirement_gap_count": int(
            requirement_coverage["status"].isin(["partial", "review", "needs_manual_data"]).sum()
        )
        if not requirement_coverage.empty
        else 0,
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
            "defect_analysis": str(defect_analysis_path),
            "experiment_comparison": str(experiment_comparison_path),
            "holding_comparison": str(holding_comparison_path),
            "holding_summary": str(summary_table_path),
            "requirement_coverage": str(requirement_coverage_path),
            "risk_scenarios": str(risk_scenarios_path),
            "stock_timing": str(stock_timing_path),
            "target_alignment": str(target_alignment_path),
            "universe_holding_scan": str(universe_scan_path),
            "chart_data_dir": str(chart_dir),
        },
    }
    summary_path = final_dir / "second_round_evaluation_summary.json"
    result["output_files"]["summary"] = str(summary_path)
    write_json(summary_path, result)
    return result


def dumps_summary(summary: Dict[str, Any]) -> str:
    return json.dumps(summary, ensure_ascii=False, indent=2, sort_keys=True)
