from __future__ import annotations

from copy import deepcopy
from typing import Any, Dict, Iterable, List

import pandas as pd

from .costs import CostModel
from .evaluation import event_strategy_comparison


def _as_float(value: Any, default: float = 0.0) -> float:
    try:
        return float(value)
    except (TypeError, ValueError):
        return default


def event_score(event: Dict[str, Any]) -> float:
    lifecycle = 1.0 if event.get("known_at") and event.get("event_date") else 0.5
    observability = 1.0 if event.get("source_url") and event.get("chain_weights") else 0.5
    return (
        0.30 * _as_float(event.get("certainty"), 0.0)
        + 0.25 * _as_float(event.get("domestic_attention_score"), 0.0)
        + 0.25 * _as_float(event.get("stock_market_fit"), 0.0)
        + 0.10 * lifecycle
        + 0.10 * observability
    )


def event_library_audit(events: Iterable[Dict[str, Any]], model_config: Dict[str, Any]) -> pd.DataFrame:
    rules = model_config.get("event_layer_rules", {})
    rows: List[Dict[str, Any]] = []
    for event in events:
        chain_count = len(event.get("chain_weights", {}))
        score = event_score(event)
        rule = rules.get(event.get("event_id")) or rules.get(event.get("sport")) or {}
        action = rule.get("action", "trade")
        missing = []
        if not event.get("source_url"):
            missing.append("source_url")
        if chain_count == 0:
            missing.append("chain_weights")
        if event.get("domestic_attention_score") is None:
            missing.append("domestic_attention_score")
        if event.get("stock_market_fit") is None:
            missing.append("stock_market_fit")
        if missing:
            status = "fail"
        elif action == "avoid" or score < 0.70 or _as_float(event.get("stock_market_fit")) < 0.45:
            status = "observe_or_filter"
        else:
            status = "pass"
        rows.append(
            {
                "event_id": event.get("event_id", ""),
                "sport": event.get("sport", ""),
                "event_date": event.get("event_date", ""),
                "status": event.get("status", ""),
                "certainty": _as_float(event.get("certainty")),
                "domestic_attention_score": _as_float(event.get("domestic_attention_score")),
                "stock_market_fit": _as_float(event.get("stock_market_fit")),
                "event_score": score,
                "chain_count": chain_count,
                "layer_action": action,
                "audit_status": status,
                "missing_items": ";".join(missing),
                "decision_reason": rule.get("reason", event.get("diagnosis_note", "")),
                "source_url": event.get("source_url", ""),
            }
        )
    return pd.DataFrame(rows)


def _chain_level(row: Dict[str, Any]) -> str:
    relation_types = set(row.get("relation_types", []))
    if "official_fifa_sponsor" in relation_types:
        return "L1"
    if relation_types & {"sports_facility", "sports_events", "sports_operation", "led_display", "venue_display", "artificial_turf"}:
        return "L2"
    if relation_types & {"media", "lottery_attention", "licensed_merchandise", "ip_products", "sports_marketing"}:
        return "L3"
    if relation_types & {"sports_consumption", "fitness_equipment", "sports_equipment", "outdoor", "winter_sports"}:
        return "L4"
    return "L5"


def universe_evidence_audit(universe: Iterable[Dict[str, Any]]) -> pd.DataFrame:
    rows: List[Dict[str, Any]] = []
    for stock in universe:
        evidence = stock.get("evidence", [])
        relation_types = stock.get("relation_types", [])
        level = _chain_level(stock)
        evidence_count = len(evidence)
        if evidence_count >= 2 and level in {"L1", "L2"}:
            status = "core_ready"
        elif evidence_count >= 1 and level in {"L1", "L2", "L3"}:
            status = "usable_first_round"
        elif evidence_count >= 1:
            status = "observe_only"
        else:
            status = "fail"
        rows.append(
            {
                "ticker": stock.get("ticker", ""),
                "name": stock.get("name", ""),
                "industry": stock.get("industry", ""),
                "chain_level": level,
                "base_exposure": _as_float(stock.get("base_exposure")),
                "relation_types": ",".join(relation_types),
                "evidence_count": evidence_count,
                "audit_status": status,
                "missing_items": "" if evidence_count >= 2 else "second_independent_source",
                "primary_source": evidence[0].get("source_url", "") if evidence else "",
            }
        )
    return pd.DataFrame(rows)


def attention_data_audit(events: Iterable[Dict[str, Any]], universe: Iterable[Dict[str, Any]]) -> pd.DataFrame:
    rows: List[Dict[str, Any]] = []
    for event in events:
        rows.append(
            {
                "object_type": "event",
                "object_id": event.get("event_id", ""),
                "source_name": "config_domestic_attention_score",
                "automation_level": "automatic_from_config",
                "status": "available" if event.get("domestic_attention_score") is not None else "missing",
                "current_proxy": event.get("domestic_attention_score", ""),
                "required_next": "baidu_index;wechat_index;news_count",
                "missing_reason": "" if event.get("domestic_attention_score") is not None else "missing_config_score",
            }
        )
        for source in ("baidu_index", "wechat_index", "news_count"):
            rows.append(
                {
                    "object_type": "event",
                    "object_id": event.get("event_id", ""),
                    "source_name": source,
                    "automation_level": "manual_or_semi_automatic",
                    "status": "pending_manual_import",
                    "current_proxy": "",
                    "required_next": "fill data/manual/event_heat_import_template.csv",
                    "missing_reason": "manual_source_not_imported_in_first_round",
                }
            )
    for stock in universe:
        rows.append(
            {
                "object_type": "stock",
                "object_id": stock.get("ticker", ""),
                "source_name": "amount_shock",
                "automation_level": "automatic_from_market_data",
                "status": "available",
                "current_proxy": "z_attention",
                "required_next": "",
                "missing_reason": "",
            }
        )
        for source in ("hot_rank", "forum_count"):
            rows.append(
                {
                    "object_type": "stock",
                    "object_id": stock.get("ticker", ""),
                    "source_name": source,
                    "automation_level": "manual_import",
                    "status": "pending_manual_import",
                    "current_proxy": "",
                    "required_next": "fill data/manual/stock_hot_rank_import_template.csv",
                    "missing_reason": "manual_source_not_imported_in_first_round",
                }
            )
    return pd.DataFrame(rows)


def ablation_comparison(
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
) -> pd.DataFrame:
    variants: List[tuple[str, str, Dict[str, Any], Dict[str, Any]]] = []
    variants.append(("final_full_model", "事件分层、低适配剔除和当前约束全部启用", deepcopy(base_params), deepcopy(model_config)))
    no_rules = deepcopy(model_config)
    no_rules["event_layer_rules"] = {}
    variants.append(("no_event_layer_rules", "取消事件分层窗口和剔除层，所有事件使用基础参数", deepcopy(base_params), no_rules))
    no_avoid = deepcopy(model_config)
    no_avoid.setdefault("event_layer_rules", {})
    no_avoid["event_layer_rules"] = deepcopy(no_avoid["event_layer_rules"])
    no_avoid["event_layer_rules"]["multi_sport_global"] = {
        "action": "trade",
        "entry_days_before_event": int(base_params.get("entry_days_before_event", 90)),
        "exit_days_before_event": int(base_params.get("exit_days_before_event", 10)),
        "top_k": int(base_params.get("top_k", 5)),
        "min_exposure_for_trade": 0.0,
        "reason": "去除低适配剔除层，强制检验境外综合赛事长仓。",
    }
    variants.append(("no_low_fit_filter", "保留其他分层规则，但强制交易境外综合赛事", deepcopy(base_params), no_avoid))
    no_fit_gate_params = deepcopy(base_params)
    no_fit_gate_params["min_event_fit_for_trade"] = 0.0
    no_fit_gate_params["min_exposure_for_trade"] = 0.0
    no_fit = deepcopy(model_config)
    no_fit["event_layer_rules"] = {}
    variants.append(("no_fit_gate", "取消事件适配和标的暴露门槛", no_fit_gate_params, no_fit))

    rows: List[Dict[str, Any]] = []
    full_total = None
    for experiment, description, params, model in variants:
        comparison, details = event_strategy_comparison(
            frames,
            universe,
            events,
            exposure_by_event,
            params,
            factor_weights,
            cost_model,
            constraints,
            model,
            initial_cash=initial_cash,
        )
        total_return = float(comparison["optimized_return"].sum()) if not comparison.empty else 0.0
        if full_total is None:
            full_total = total_return
        rows.append(
            {
                "experiment": experiment,
                "description": description,
                "total_return": total_return,
                "excess_vs_full_model": total_return - float(full_total),
                "positive_event_count": int((comparison["optimized_return"] > 0).sum()) if not comparison.empty else 0,
                "negative_event_count": int((comparison["optimized_return"] < 0).sum()) if not comparison.empty else 0,
                "trade_count": int(len(details)),
            }
        )
    return pd.DataFrame(rows)


EVENT_ANALYSIS_NOTES: Dict[str, Dict[str, str]] = {
    "FIFA_WC_2022_OPEN": {
        "root_cause": "足球链条有效，但 T-90 至 T-10 的保守退出会错过开幕前后后段行情；海信视像在主回测中触发止损，拖累单事件收益。",
        "chain_view": "显示设备、体育运营和体育营销链条有效，中体产业和莱茵体育补偿了海信视像的早期回撤。",
        "next_action": "第二轮只在样本外差距受控时测试 T-60/T-45 至开幕日窗口。",
    },
    "CHENGDU_FISU_2023_OPEN": {
        "root_cause": "综合赛事热度主要流向城市消费、旅游和线下服务，宽泛体育股票池没有同步兑现。",
        "chain_view": "收窄到体育运营和场馆设施后，中体产业和共创草坪能转正。",
        "next_action": "保留国内综合赛事层，继续要求场馆、运营或设施链条证据。",
    },
    "HANGZHOU_ASIAD_2023_OPEN": {
        "root_cause": "亚运热度高，但本地服务、平台消费和品牌广告收益分散，A股体育候选池同步性不足。",
        "chain_view": "共创草坪和雷曼光电贡献正收益，莱茵体育拖累，说明综合赛事需要分散子链条。",
        "next_action": "加入本地消费和官方供应商证据后再扩大仓位。",
    },
    "AFC_ASIAN_CUP_2024_OPEN": {
        "root_cause": "亚洲杯为海外举办赛事，国内关注度和商业转化弱于世界杯，A股收入链条不足。",
        "chain_view": "系统没有生成有效持仓，避免把低适配足球赛事等同于世界杯。",
        "next_action": "只有中国队关注度、官方合作或博彩传媒热度同步增强时才交易。",
    },
    "UEFA_EURO_2024_OPEN": {
        "root_cause": "欧洲杯有观赛热度，但海外商业权益和夜间观赛场景难直接传导到当前A股候选池。",
        "chain_view": "中期组合亏损，第一轮空仓后保住收益。",
        "next_action": "作为海外足球观察样本，不进入核心仓位。",
    },
    "PARIS_OLYMPICS_2024_OPEN": {
        "root_cause": "奥运社会热度高，赞助、转播和消费收益分散到海外主体、平台和非上市链条，当前A股候选池缺少直接现金流连接。",
        "chain_view": "740组强制长仓扫描没有正收益，剔除层比交易更有效。",
        "next_action": "继续作为高热低适配过滤测试，不为热度单独加仓。",
    },
    "HARBIN_ASIAN_WINTER_2025_OPEN": {
        "root_cause": "冰雪热度更多兑现到旅游、酒店、交通和地方消费，A股体育股票响应窗口短。",
        "chain_view": "短窗口单标的交易转正，但收益低于中期静态组合，说明冰雪链条还需要更细的标的池。",
        "next_action": "补充冰雪旅游、装备和本地消费证据，仓位保持小。",
    },
    "FIFA_CLUB_WORLD_CUP_2025_OPEN": {
        "root_cause": "海信官方合作明确，但世俱杯国内关注度低于世界杯，注意力扩散有限。",
        "chain_view": "海信视像贡献小幅正收益，单链条可交易但不应放大仓位。",
        "next_action": "保留显示设备核心链条，增加国内搜索热度确认。",
    },
    "NATIONAL_GAMES_2025_OPEN": {
        "root_cause": "全运会是国内综合赛事，但预热周期长，主题资金容易提前兑现；户外链条波动大。",
        "chain_view": "中体产业和海信视像赚钱，探路者触发止损，说明消费/户外链条需要更严格的风险过滤。",
        "next_action": "保留运营和显示链条，降低户外用品权重。",
    },
    "MILANO_CORTINA_2026_OPEN": {
        "root_cause": "境外冬奥对A股冰雪链条的传导弱，受益更偏海外旅游、装备品牌和本地运营主体。",
        "chain_view": "系统未生成正收益交易，观察层处理更合适。",
        "next_action": "只在国内搜索热度和股票热度同步时做小仓验证。",
    },
}


def event_failure_analysis(
    comparison: pd.DataFrame,
    event_details: pd.DataFrame,
    event_summary: pd.DataFrame,
) -> pd.DataFrame:
    rows: List[Dict[str, Any]] = []
    caar_map: Dict[str, float] = {}
    positive_map: Dict[str, float] = {}
    if not event_summary.empty:
        focus = event_summary[event_summary["window"] == "T-60_T-10"]
        caar_map = focus.set_index("event_id")["caar"].astype(float).to_dict()
        positive_map = focus.set_index("event_id")["positive_rate"].astype(float).to_dict()
    for _, row in comparison.iterrows():
        event_id = row["event_id"]
        details = event_details[event_details["event_id"] == event_id] if not event_details.empty else pd.DataFrame()
        losing = []
        winning = []
        if not details.empty:
            losing = details.loc[details["pnl"] < 0, "ticker"].astype(str).tolist()
            winning = details.loc[details["pnl"] > 0, "ticker"].astype(str).tolist()
        optimized_return = _as_float(row.get("optimized_return"))
        midterm_return = _as_float(row.get("midterm_return"))
        if optimized_return < 0:
            issue = "negative_return"
        elif optimized_return == 0:
            issue = "filtered_or_no_trade"
        elif optimized_return < midterm_return:
            issue = "positive_but_weak"
        else:
            issue = "improved"
        notes = EVENT_ANALYSIS_NOTES.get(event_id, {})
        rows.append(
            {
                "event_id": event_id,
                "event_name": row.get("event_name", ""),
                "issue": issue,
                "midterm_return": midterm_return,
                "optimized_return": optimized_return,
                "excess_vs_midterm": _as_float(row.get("excess_vs_midterm")),
                "event_caar_t60_t10": caar_map.get(event_id, 0.0),
                "positive_rate_t60_t10": positive_map.get(event_id, 0.0),
                "winning_tickers": ",".join(winning),
                "losing_tickers": ",".join(losing),
                "root_cause": notes.get("root_cause", ""),
                "chain_view": notes.get("chain_view", ""),
                "next_action": notes.get("next_action", ""),
            }
        )
    return pd.DataFrame(rows)


def holding_failure_analysis(holding_details: pd.DataFrame) -> pd.DataFrame:
    rows: List[Dict[str, Any]] = []
    if holding_details.empty:
        return pd.DataFrame()
    for _, row in holding_details.iterrows():
        strategy = row.get("strategy", "")
        ticker = row.get("ticker", "")
        stock_return = _as_float(row.get("stock_return"))
        pnl = _as_float(row.get("pnl"))
        if strategy == "midterm" and ticker in {"605299.SH", "002181.SZ", "002878.SZ"}:
            cause = "世界杯直接链条弱，主题热度没有传导到该股，持仓过重放大亏损。"
            action = "第一轮剔除该标的，改用显示设备和LED链条。"
        elif ticker == "600060.SH":
            cause = "FIFA合作链条最清晰，但2026-05-27至2026-06-16处于赛前一年，事件尚未进入强兑现期。"
            action = "保留核心仓位，上限维持10%。"
        elif ticker in {"300162.SZ", "002587.SZ"}:
            cause = "LED显示链条成立，但短期市场和小盘成长风格承压，事件热度还未覆盖价格波动。"
            action = "保留小仓，等待搜索热度和成交热度确认。"
        else:
            cause = "需要结合热度和业务证据复核。"
            action = "保留观察。"
        rows.append(
            {
                "strategy": strategy,
                "ticker": ticker,
                "company": row.get("company", ""),
                "industry": row.get("industry", ""),
                "target_weight": _as_float(row.get("target_weight")),
                "stock_return": stock_return,
                "pnl": pnl,
                "analysis": cause,
                "system_action": action,
            }
        )
    return pd.DataFrame(rows)
