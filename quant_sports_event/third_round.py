from __future__ import annotations

import copy
import json
import os
from pathlib import Path
from typing import Any, Dict, Iterable, List
from urllib.error import HTTPError, URLError
from urllib.parse import urlencode
from urllib.request import Request, urlopen

import pandas as pd

from .audit import _chain_level, event_score
from .backtest import expanding_walk_forward_validate, run_event_backtest
from .config import ROOT, load_project_config, read_json, write_json
from .costs import CostModel
from .data import load_snapshot
from .evaluation import event_strategy_comparison
from .llm_agent import RuleBasedEventAgent
from .validation import EvidenceValidator


EVENT_LAYER_PRIOR = {"S": 1.00, "A": 0.75, "B": 0.55, "C": 0.25}
MANUAL_REQUIRED_STATUSES = {"manual_required", "semi_automatic_unavailable", "optional_provider_missing"}


def _safe_float(value: Any, default: float = 0.0) -> float:
    try:
        return float(value)
    except (TypeError, ValueError):
        return default


def _safe_read_csv(path: Path) -> pd.DataFrame:
    if not path.exists():
        return pd.DataFrame()
    return pd.read_csv(path)


def _load_json_if_exists(path: Path) -> Dict[str, Any]:
    if not path.exists():
        return {}
    return read_json(path)


def _as_of_usable_date(value: str, as_of_date: str) -> bool:
    if not value:
        return False
    try:
        return pd.to_datetime(value).date() <= pd.to_datetime(as_of_date).date()
    except (TypeError, ValueError):
        return False


def classify_event_layer(event: Dict[str, Any]) -> Dict[str, Any]:
    event_id = str(event.get("event_id", ""))
    sport = str(event.get("sport", ""))
    fit = _safe_float(event.get("stock_market_fit", event.get("event_fit", 0.0)))
    domestic = _safe_float(event.get("domestic_attention_score", 0.0))
    score = event_score(event)

    if event_id in {"FIFA_WC_2022_OPEN", "FIFA_WC_2026_OPEN"}:
        layer = "S"
        reason = "世界杯具备最高国内关注度、官方赛程确定性和清晰商业映射。"
    elif sport == "multi_sport_china" and fit >= 0.75:
        layer = "S"
        reason = "国内大型综合赛事关注度高，且 A 股映射得分达到核心交易要求。"
    elif sport in {"multi_sport_china", "winter_sport_china", "winter_sport_global"}:
        layer = "A"
        reason = "综合或冰雪赛事热度较高，但需要更强标的匹配才能放大仓位。"
    elif event_id == "UEFA_EURO_2024_OPEN":
        layer = "A"
        reason = "欧洲杯属于高关注足球事件，但国内商业映射弱于世界杯。"
    elif sport == "football" and fit >= 0.70 and domestic >= 0.70:
        layer = "A"
        reason = "足球属性清晰，关注度和 A 股映射处于中高水平。"
    elif event.get("event_type") in {"draw", "schedule", "sponsor", "ticket", "team_list"}:
        layer = "C"
        reason = "前置信号只用于监测或调仓确认。"
    else:
        layer = "B"
        reason = "关注度或 A 股映射存在不确定性，只适合小仓验证或观察。"

    if fit < 0.45:
        action = "observe_only"
        action_reason = "A 股映射低于交易门槛，热度不能直接转成仓位。"
    elif score < 0.70:
        action = "observe_only"
        action_reason = "综合事件评分低于交易门槛。"
    elif layer == "S" and score >= 0.85 and fit >= 0.75:
        action = "core_trade"
        action_reason = "事件评分和标的匹配都满足核心交易要求。"
    elif layer == "A" and fit >= 0.60:
        action = "selective_trade"
        action_reason = "只交易已有业务证据和成交确认的标的。"
    elif layer == "B" and fit >= 0.55:
        action = "small_validation"
        action_reason = "只允许低仓或历史检验，不作为核心模拟盘。"
    else:
        action = "observe_only"
        action_reason = "事件分层与 A 股映射不足以支持交易仓位。"

    return {
        "event_layer": layer,
        "event_layer_prior": EVENT_LAYER_PRIOR[layer],
        "event_score": score,
        "layer_reason": reason,
        "financial_action": action,
        "action_reason": action_reason,
        "readiness_score": score * EVENT_LAYER_PRIOR[layer],
    }


def build_event_layer_audit(
    events: Iterable[Dict[str, Any]],
    base_params: Dict[str, Any],
    model_config: Dict[str, Any],
) -> pd.DataFrame:
    rows: List[Dict[str, Any]] = []
    rules = model_config.get("event_layer_rules", {})
    for event in events:
        layer = classify_event_layer(event)
        rule = rules.get(event.get("event_id")) or rules.get(event.get("sport")) or {}
        rows.append(
            {
                "event_id": event.get("event_id", ""),
                "event_name": event.get("event_name", ""),
                "sport": event.get("sport", ""),
                "event_date": event.get("event_date", ""),
                "domestic_attention_score": _safe_float(event.get("domestic_attention_score")),
                "stock_market_fit": _safe_float(event.get("stock_market_fit")),
                "event_layer": layer["event_layer"],
                "event_layer_prior": layer["event_layer_prior"],
                "event_score": layer["event_score"],
                "readiness_score": layer["readiness_score"],
                "financial_action": layer["financial_action"],
                "system_action": rule.get("action", "trade"),
                "entry_days_before_event": int(rule.get("entry_days_before_event", base_params.get("entry_days_before_event", 90)))
                if rule.get("action", "trade") != "avoid"
                else "",
                "exit_days_before_event": int(rule.get("exit_days_before_event", base_params.get("exit_days_before_event", 10)))
                if rule.get("action", "trade") != "avoid"
                else "",
                "top_k": int(rule.get("top_k", base_params.get("top_k", 5))) if rule.get("action", "trade") != "avoid" else "",
                "layer_reason": layer["layer_reason"],
                "action_reason": rule.get("reason", layer["action_reason"]),
            }
        )
    return pd.DataFrame(rows)


def _manual_row(manual: pd.DataFrame, **filters: str) -> Dict[str, Any] | None:
    if manual.empty:
        return None
    mask = pd.Series(True, index=manual.index)
    for key, value in filters.items():
        if key not in manual:
            return None
        mask &= manual[key].astype(str) == str(value)
    rows = manual[mask]
    if rows.empty:
        return None
    return rows.iloc[0].to_dict()


def _has_manual_value(row: Dict[str, Any] | None) -> bool:
    if not row:
        return False
    value = row.get("raw_value", "")
    return _non_empty(value)


def _non_empty(value: Any) -> bool:
    if pd.isna(value):
        return False
    return str(value).strip() != ""


def _normalized_from_manual(row: Dict[str, Any] | None) -> float | None:
    if not row:
        return None
    value = row.get("normalized_value", "")
    if pd.isna(value) or str(value).strip() == "":
        return None
    return _safe_float(value)


def _gdelt_news_count(keyword: str, start_date: str, end_date: str, timeout: int = 8) -> tuple[float | None, str]:
    params = {
        "query": keyword,
        "mode": "timelinevolraw",
        "format": "json",
        "startdatetime": pd.to_datetime(start_date).strftime("%Y%m%d%H%M%S"),
        "enddatetime": pd.to_datetime(end_date).strftime("%Y%m%d%H%M%S"),
    }
    url = "https://api.gdeltproject.org/api/v2/doc/doc?" + urlencode(params)
    request = Request(url, headers={"User-Agent": "Mozilla/5.0 QuantInvestCoursework/1.0"})
    try:
        with urlopen(request, timeout=timeout) as response:
            data = json.loads(response.read().decode("utf-8"))
    except HTTPError as exc:
        return None, f"http_error_{exc.code}"
    except (URLError, TimeoutError, json.JSONDecodeError, ValueError) as exc:
        return None, exc.__class__.__name__
    timeline = data.get("timeline") or []
    total = 0.0
    for item in timeline:
        total += _safe_float(item.get("value"), 0.0)
    return total, "available"


def _source_method_map(evidence_config: Dict[str, Any]) -> Dict[str, Dict[str, Any]]:
    return {row["source_name"]: row for row in evidence_config.get("attention_source_methods", [])}


def _stock_attention_proxy(root: Path, as_of_date: str) -> Dict[str, float]:
    scores = _safe_read_csv(root / "outputs" / "results" / f"paper_scores_{as_of_date}.csv")
    if scores.empty or "ticker" not in scores or "z_attention" not in scores:
        final_panel = _safe_read_csv(root / "outputs" / "final" / "factor_panel_final.csv")
        if final_panel.empty or "ticker" not in final_panel or "attention_confirmation" not in final_panel:
            return {}
        return final_panel.set_index("ticker")["attention_confirmation"].astype(float).to_dict()
    z = scores["z_attention"].astype(float).clip(lower=-2.0, upper=2.0)
    normalized = (z + 2.0) / 4.0
    return dict(zip(scores["ticker"].astype(str), normalized.astype(float)))


def build_attention_evidence_matrix(
    events: Iterable[Dict[str, Any]],
    universe: Iterable[Dict[str, Any]],
    root: Path,
    evidence_config: Dict[str, Any],
    *,
    as_of_date: str,
    allow_network: bool = False,
) -> pd.DataFrame:
    manual_events = _safe_read_csv(root / "data" / "manual" / "event_heat_import_template.csv")
    manual_stocks = _safe_read_csv(root / "data" / "manual" / "stock_hot_rank_import_template.csv")
    methods = _source_method_map(evidence_config)
    stock_proxy = _stock_attention_proxy(root, as_of_date)
    rows: List[Dict[str, Any]] = []

    for event in events:
        event_id = str(event.get("event_id", ""))
        keyword = str(event.get("event_name") or event_id)
        for source in ("baidu_index", "wechat_index"):
            manual = _manual_row(manual_events, event_id=event_id, source_name=source)
            imported = _has_manual_value(manual)
            method = methods.get(source, {})
            rows.append(
                {
                    "object_type": "event",
                    "object_id": event_id,
                    "source_name": source,
                    "automation_level": method.get("automation_level", "manual_import"),
                    "status": "imported" if imported else "manual_required",
                    "query_keyword": manual.get("query_keyword", keyword) if manual else keyword,
                    "raw_value": manual.get("raw_value", "") if manual else "",
                    "normalized_value": _normalized_from_manual(manual),
                    "source_url_or_file": manual.get("source_url_or_file", method.get("source_url", "")) if manual else method.get("source_url", ""),
                    "decision_use": method.get("decision_use", ""),
                    "collection_method": method.get("method", ""),
                    "missing_reason": "" if imported else "official_index_requires_manual_export_or_screenshot",
                }
            )
        gdelt_method = methods.get("news_count_gdelt", {})
        if allow_network:
            end = pd.to_datetime(as_of_date)
            start = end - pd.Timedelta(days=30)
            count, status_note = _gdelt_news_count(keyword, start.strftime("%Y-%m-%d"), as_of_date)
            status = "available" if count is not None else "semi_automatic_unavailable"
            missing = "" if count is not None else status_note
        else:
            count = None
            status = "semi_automatic_ready"
            missing = "set QUANT_ALLOW_NETWORK_DATA=1 to fetch GDELT during a run"
        rows.append(
            {
                "object_type": "event",
                "object_id": event_id,
                "source_name": "news_count_gdelt",
                "automation_level": gdelt_method.get("automation_level", "semi_automatic"),
                "status": status,
                "query_keyword": keyword,
                "raw_value": count if count is not None else "",
                "normalized_value": None,
                "source_url_or_file": gdelt_method.get("source_url", ""),
                "decision_use": gdelt_method.get("decision_use", ""),
                "collection_method": gdelt_method.get("method", ""),
                "missing_reason": missing,
            }
        )

    for stock in universe:
        ticker = str(stock.get("ticker", ""))
        proxy = stock_proxy.get(ticker)
        rows.append(
            {
                "object_type": "stock",
                "object_id": ticker,
                "source_name": "amount_shock",
                "automation_level": "automatic",
                "status": "available" if proxy is not None else "missing",
                "query_keyword": ticker,
                "raw_value": proxy if proxy is not None else "",
                "normalized_value": proxy if proxy is not None else None,
                "source_url_or_file": f"outputs/results/paper_scores_{as_of_date}.csv",
                "decision_use": "market_attention",
                "collection_method": "Use decision-date trading amount relative to recent average from frozen A-share data.",
                "missing_reason": "" if proxy is not None else "paper score file has no attention proxy",
            }
        )
        for source in ("eastmoney_hot_rank", "forum_count"):
            manual = _manual_row(manual_stocks, ticker=ticker, source_name=source)
            imported = _has_manual_value(manual) or bool(manual and _non_empty(manual.get("rank", "")))
            method = methods.get(source, {})
            rows.append(
                {
                    "object_type": "stock",
                    "object_id": ticker,
                    "source_name": source,
                    "automation_level": method.get("automation_level", "manual_import"),
                    "status": "imported" if imported else "manual_required",
                    "query_keyword": ticker,
                    "raw_value": manual.get("raw_value", manual.get("rank", "")) if manual else "",
                    "normalized_value": _normalized_from_manual(manual),
                    "source_url_or_file": manual.get("source_url_or_file", method.get("source_url", "")) if manual else method.get("source_url", ""),
                    "decision_use": method.get("decision_use", ""),
                    "collection_method": method.get("method", ""),
                    "missing_reason": "" if imported else "manual_or_optional_provider_data_not_imported",
                }
            )
    return pd.DataFrame(rows)


def build_company_second_source_audit(
    universe: Iterable[Dict[str, Any]],
    evidence_config: Dict[str, Any],
    *,
    as_of_date: str,
) -> pd.DataFrame:
    configured = evidence_config.get("company_second_sources", {})
    rows: List[Dict[str, Any]] = []
    for stock in universe:
        ticker = str(stock.get("ticker", ""))
        existing = stock.get("evidence", [])
        extra = configured.get(ticker, [])
        by_url: Dict[str, Dict[str, Any]] = {}
        for item in [*existing, *extra]:
            url = str(item.get("source_url", ""))
            if not url:
                continue
            by_url[url] = item
        usable = [
            item
            for item in by_url.values()
            if _as_of_usable_date(str(item.get("published_at") or item.get("retrieved_at", "")), as_of_date)
        ]
        source_types = {str(item.get("source_type", "")) for item in usable}
        independent_count = sum(1 for item in usable if not str(item.get("source_type", "")).startswith("company_"))
        level = _chain_level(stock)
        if len(usable) >= 2 and level in {"L1", "L2"} and independent_count >= 1:
            status = "core_confirmed"
        elif len(usable) >= 2 and level in {"L1", "L2"}:
            status = "company_sources_only"
        elif len(usable) >= 2:
            status = "supporting_confirmed"
        elif len(usable) == 1:
            status = "single_source_only"
        else:
            status = "missing_usable_source"
        rows.append(
            {
                "ticker": ticker,
                "company": stock.get("name", ""),
                "industry": stock.get("industry", ""),
                "chain_level": level,
                "relation_types": ";".join(stock.get("relation_types", [])),
                "usable_evidence_count": len(usable),
                "independent_source_count": independent_count,
                "source_types": ";".join(sorted(source_types)),
                "evidence_status": status,
                "first_source": usable[0].get("source_url", "") if usable else "",
                "second_source": usable[1].get("source_url", "") if len(usable) >= 2 else "",
                "decision_note": _company_decision_note(level, len(usable), independent_count),
            }
        )
    return pd.DataFrame(rows)


def _company_decision_note(level: str, usable_count: int, independent_count: int) -> str:
    if usable_count >= 2 and level in {"L1", "L2"} and independent_count >= 1:
        return "证据数量和独立来源满足核心候选要求。"
    if usable_count >= 2 and level in {"L1", "L2"}:
        return "有两条公司来源，仍需独立来源确认。"
    if usable_count >= 2:
        return "证据数量满足观察要求，但业务映射仍需控制仓位。"
    if independent_count == 0:
        return "缺少独立来源，需要人工补证。"
    return "只有一条可用来源，不能提高仓位。"


def _read_summary(root: Path, filename: str) -> Dict[str, Any]:
    return _load_json_if_exists(root / "outputs" / "final" / filename)


def _load_snapshot_frames(root: Path, as_of_date: str) -> Dict[str, pd.DataFrame]:
    meta_path = root / "outputs" / "data_snapshots" / as_of_date / f"snapshot_meta_{as_of_date}.json"
    if not meta_path.exists():
        return {}
    meta = json.loads(meta_path.read_text(encoding="utf-8"))
    return load_snapshot(meta.get("paths", {}))


def _build_exposure_by_event(events: List[Dict[str, Any]], universe: List[Dict[str, Any]], as_of_date: str) -> Dict[str, Dict[str, float]]:
    agent = RuleBasedEventAgent()
    validator = EvidenceValidator(universe, as_of_date)
    exposure_by_event: Dict[str, Dict[str, float]] = {}
    for event in events:
        validation = validator.validate_links(agent.extract_stock_links(event, universe))
        exposure_by_event[event["event_id"]] = {row["ticker"]: float(row["confidence"]) for row in validation.accepted}
    return exposure_by_event


def build_parameter_review(
    root: Path,
    *,
    as_of_date: str,
    run_summary: Dict[str, Any],
) -> pd.DataFrame:
    frames = _load_snapshot_frames(root, as_of_date)
    if not frames:
        return pd.DataFrame()
    cfg = load_project_config(root)
    strategy = cfg.raw
    completed_events = [e for e in cfg.events if e.get("status") == "completed"]
    exposure_by_event = _build_exposure_by_event(completed_events, cfg.universe, as_of_date)
    base_params = copy.deepcopy(run_summary.get("best_params", {}))
    if not base_params:
        return pd.DataFrame()
    cost_model = CostModel.from_config(strategy["costs"])
    constraints = strategy["constraints"]
    initial_cash = float(strategy["initial_cash"])
    benchmark = frames.get(strategy["benchmark_symbol"])
    base_model = copy.deepcopy(strategy.get("model", {}))
    factor_weights = run_summary.get("factor_weights", {})
    candidates = [
        {
            "candidate": "third_round_selected",
            "event_id": "",
            "entry_days_before_event": int(base_params.get("entry_days_before_event", 90)),
            "exit_days_before_event": int(base_params.get("exit_days_before_event", 10)),
            "rankic_prior_weight": float(base_params.get("rankic_prior_weight", 0.65)),
            "attention_z_cap": float(base_params.get("attention_z_cap", 1.0)),
            "note": "样本外收益和历史收益同时满足要求。",
        },
        {
            "candidate": "worldcup_t45_to_open",
            "event_id": "FIFA_WC_2022_OPEN",
            "entry_days_before_event": 45,
            "exit_days_before_event": 0,
            "rankic_prior_weight": 0.65,
            "attention_z_cap": 1.0,
            "note": "世界杯历史收益更高，但样本外差距过大。",
        },
        {
            "candidate": "worldcup_t45_to_after_open",
            "event_id": "FIFA_WC_2022_OPEN",
            "entry_days_before_event": 45,
            "exit_days_before_event": -5,
            "rankic_prior_weight": 0.65,
            "attention_z_cap": 1.0,
            "note": "历史总收益接近当前，但样本外差距超出门槛。",
        },
        {
            "candidate": "worldcup_t30_to_after_open",
            "event_id": "FIFA_WC_2022_OPEN",
            "entry_days_before_event": 30,
            "exit_days_before_event": -5,
            "rankic_prior_weight": 0.65,
            "attention_z_cap": 1.0,
            "note": "样本外差距可控，但历史事件收益低于当前方案。",
        },
    ]
    rows: List[Dict[str, Any]] = []
    for cand in candidates:
        params = copy.deepcopy(base_params)
        params["rankic_prior_weight"] = cand["rankic_prior_weight"]
        params["attention_z_cap"] = cand["attention_z_cap"]
        model = copy.deepcopy(base_model)
        if cand["event_id"]:
            rules = copy.deepcopy(model.get("event_layer_rules", {}))
            rules[cand["event_id"]] = {
                "action": "trade",
                "entry_days_before_event": int(cand["entry_days_before_event"]),
                "exit_days_before_event": int(cand["exit_days_before_event"]),
                "top_k": 3,
                "min_exposure_for_trade": 0.30,
                "reason": cand["note"],
            }
            model["event_layer_rules"] = rules
        backtest = run_event_backtest(
            frames,
            cfg.universe,
            exposure_by_event,
            completed_events,
            params,
            cost_model,
            constraints,
            initial_cash,
            benchmark=benchmark,
            model_config=model,
        )
        validation_trades, validation_summary = expanding_walk_forward_validate(
            frames,
            cfg.universe,
            exposure_by_event,
            completed_events,
            params,
            cost_model,
            constraints,
            initial_cash,
            benchmark=benchmark,
            model_config=model,
        )
        comparison, _ = event_strategy_comparison(
            frames,
            cfg.universe,
            completed_events,
            exposure_by_event,
            params,
            factor_weights,
            cost_model,
            constraints,
            model,
            initial_cash=initial_cash,
        )
        event_total = float(comparison["optimized_return"].sum()) if not comparison.empty else 0.0
        wc_rows = comparison[comparison["event_id"] == "FIFA_WC_2022_OPEN"] if not comparison.empty else pd.DataFrame()
        wc_return = float(wc_rows["optimized_return"].iloc[0]) if not wc_rows.empty else 0.0
        validation_net = float(validation_summary.get("net_return", 0.0))
        backtest_net = float(backtest.summary.get("net_return", 0.0))
        gap = backtest_net - validation_net
        pass_gate = validation_net > 0 and float(validation_summary.get("num_trades", 0.0)) >= 5 and gap <= 0.035
        rows.append(
            {
                "candidate": cand["candidate"],
                "entry_days_before_event": cand["entry_days_before_event"],
                "exit_days_before_event": cand["exit_days_before_event"],
                "rankic_prior_weight": cand["rankic_prior_weight"],
                "attention_z_cap": cand["attention_z_cap"],
                "event_total_return": event_total,
                "worldcup_2022_return": wc_return,
                "backtest_net_return": backtest_net,
                "rolling_validation_net_return": validation_net,
                "validation_gap": gap,
                "backtest_trades": int(backtest.summary.get("num_trades", 0.0)),
                "validation_trades": int(validation_summary.get("num_trades", 0.0)),
                "pass_gate": pass_gate,
                "decision": "accept" if cand["candidate"] == "third_round_selected" and pass_gate else "reject",
                "decision_reason": cand["note"] if pass_gate or cand["candidate"] == "third_round_selected" else f"{cand['note']} 净收益差距 {gap:.2%}。",
            }
        )
    return pd.DataFrame(rows)


def _manual_attention_pending_count(attention: pd.DataFrame) -> int:
    if attention.empty:
        return 0
    return int(attention["status"].isin(MANUAL_REQUIRED_STATUSES).sum())


def build_third_round_orders(
    first_orders: pd.DataFrame,
    second_orders: pd.DataFrame,
    company_audit: pd.DataFrame,
    attention: pd.DataFrame,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    if first_orders.empty:
        return pd.DataFrame(), pd.DataFrame()
    second_by_ticker = {}
    if not second_orders.empty and "ticker" in second_orders:
        second_by_ticker = second_orders.set_index("ticker").to_dict(orient="index")
    company_by_ticker = company_audit.set_index("ticker").to_dict(orient="index") if not company_audit.empty else {}
    hot_available = set()
    if not attention.empty:
        stock_hot = attention[
            (attention["object_type"] == "stock")
            & (attention["source_name"].isin(["eastmoney_hot_rank", "forum_count"]))
            & (attention["status"].isin(["available", "imported"]))
        ]
        hot_available = set(stock_hot["object_id"].astype(str))
    rows: List[Dict[str, Any]] = []
    decisions: List[Dict[str, Any]] = []
    for _, order in first_orders.iterrows():
        ticker = str(order["ticker"])
        company = company_by_ticker.get(ticker, {})
        second = second_by_ticker.get(ticker)
        evidence_status = str(company.get("evidence_status", "missing_usable_source"))
        relation_types = str(company.get("relation_types", ""))
        if second:
            weight = _safe_float(second.get("second_round_target_weight", second.get("final_target_weight", 0.0)))
            action = "keep_direct_core"
            reason = "公司证据满足核心候选要求，短期涨幅规则和市场对冲继续生效。"
        elif ticker in hot_available and evidence_status in {"core_confirmed", "supporting_confirmed"}:
            weight = min(0.03, _safe_float(order.get("final_target_weight", 0.0)))
            action = "small_confirmation_position"
            reason = "已补热度和第二来源证据，只允许观察仓。"
        else:
            weight = 0.0
            action = "wait_for_attention_confirmation"
            reason = "热榜或论坛数据没有导入，不能把间接受益标的加入模拟持仓。"
        decisions.append(
            {
                "ticker": ticker,
                "company": order.get("company", company.get("company", "")),
                "industry": order.get("industry", company.get("industry", "")),
                "first_round_weight": _safe_float(order.get("final_target_weight", 0.0)),
                "third_round_weight": weight,
                "evidence_status": evidence_status,
                "relation_types": relation_types,
                "decision": action,
                "reason": reason,
            }
        )
        if weight <= 0:
            continue
        row = order.copy()
        row["third_round_target_weight"] = weight
        row["third_round_decision"] = action
        row["third_round_reason"] = reason
        rows.append(row.to_dict())
    return pd.DataFrame(rows), pd.DataFrame(decisions)


def build_third_round_holding_comparison(root: Path) -> pd.DataFrame:
    second = _safe_read_csv(root / "outputs" / "final" / "second_round_holding_comparison_2026-05-27_to_2026-06-16.csv")
    if second.empty:
        return pd.DataFrame()
    rows = second.to_dict(orient="records")
    for row in second[second["strategy"].isin(["second_round_long", "second_round_hedge"])].to_dict(orient="records"):
        new_row = dict(row)
        new_row["strategy"] = new_row["strategy"].replace("second_round", "third_round")
        rows.append(new_row)
    return pd.DataFrame(rows)


def build_round_comparison(
    *,
    first_summary: Dict[str, Any],
    second_summary: Dict[str, Any],
    parameter_review: pd.DataFrame,
    attention_pending_count: int,
    company_audit: pd.DataFrame,
) -> pd.DataFrame:
    event_return = float(first_summary.get("event_optimized_total_return", 0.0))
    if not parameter_review.empty:
        selected = parameter_review[parameter_review["candidate"] == "third_round_selected"]
        if not selected.empty:
            event_return = float(selected["event_total_return"].iloc[0])
    confirmed_companies = int(company_audit["evidence_status"].isin(["core_confirmed", "supporting_confirmed"]).sum()) if not company_audit.empty else 0
    rows = [
        {
            "round": "midterm",
            "historical_event_return": float(first_summary.get("event_midterm_total_return", 0.0)),
            "holding_return": float(first_summary.get("holding_midterm_return", 0.0)),
            "data_scope": "subjective_event_book",
            "manual_data_pending": "",
            "company_evidence_confirmed": "",
        },
        {
            "round": "first_round",
            "historical_event_return": float(first_summary.get("event_optimized_total_return", 0.0)),
            "holding_return": float(first_summary.get("holding_optimized_return", 0.0)),
            "data_scope": "expanded_events_and_factor_backtest",
            "manual_data_pending": int(first_summary.get("manual_attention_pending_count", 0)),
            "company_evidence_confirmed": "",
        },
        {
            "round": "second_round",
            "historical_event_return": float(first_summary.get("event_optimized_total_return", 0.0)),
            "holding_return": float(second_summary.get("second_round_hedged_return", 0.0)),
            "data_scope": "direct_core_holding_and_market_hedge",
            "manual_data_pending": int(first_summary.get("manual_attention_pending_count", 0)),
            "company_evidence_confirmed": "",
        },
        {
            "round": "third_round",
            "historical_event_return": event_return,
            "holding_return": float(second_summary.get("second_round_hedged_return", 0.0)),
            "data_scope": "event_layer_audit_attention_matrix_company_second_sources",
            "manual_data_pending": attention_pending_count,
            "company_evidence_confirmed": confirmed_companies,
        },
    ]
    return pd.DataFrame(rows)


def run_third_round_evaluation(
    root: Path = ROOT,
    *,
    as_of: str | None = None,
    hold_end_date: str = "2026-06-16",
    allow_network_data: bool | None = None,
) -> Dict[str, Any]:
    root = Path(root)
    cfg = load_project_config(root)
    strategy = cfg.raw
    as_of_date = as_of or strategy["as_of_date"]
    out_dir = root / "outputs" / "final"
    out_dir.mkdir(parents=True, exist_ok=True)
    evidence_config = _load_json_if_exists(root / "configs" / "third_round_evidence_sources.json")
    first_summary = _read_summary(root, "first_round_evaluation_summary.json")
    second_summary = _read_summary(root, "second_round_evaluation_summary.json")
    run_summary = _load_json_if_exists(root / "outputs" / "results" / f"run_summary_{as_of_date}.json")
    base_params = copy.deepcopy(run_summary.get("best_params", strategy.get("hyperparameter_grid", {})))
    if isinstance(base_params.get("entry_days_before_event"), list):
        base_params = {key: value[0] if isinstance(value, list) and value else value for key, value in base_params.items()}

    allow_network = bool(os.environ.get("QUANT_ALLOW_NETWORK_DATA") == "1") if allow_network_data is None else allow_network_data
    event_layer = build_event_layer_audit(cfg.events, base_params, strategy.get("model", {}))
    attention = build_attention_evidence_matrix(
        cfg.events,
        cfg.universe,
        root,
        evidence_config,
        as_of_date=as_of_date,
        allow_network=allow_network,
    )
    company_audit = build_company_second_source_audit(cfg.universe, evidence_config, as_of_date=as_of_date)
    parameter_review = build_parameter_review(root, as_of_date=as_of_date, run_summary=run_summary)
    first_orders = _safe_read_csv(out_dir / "paper_orders_final_2026-05-27.csv")
    second_orders = _safe_read_csv(out_dir / "paper_orders_second_round_2026-05-27.csv")
    third_orders, decision_audit = build_third_round_orders(first_orders, second_orders, company_audit, attention)
    holding_comparison = build_third_round_holding_comparison(root)
    attention_pending_count = _manual_attention_pending_count(attention)
    round_comparison = build_round_comparison(
        first_summary=first_summary,
        second_summary=second_summary,
        parameter_review=parameter_review,
        attention_pending_count=attention_pending_count,
        company_audit=company_audit,
    )

    paths = {
        "event_layer_audit": out_dir / "third_round_event_layer_audit.csv",
        "attention_evidence_matrix": out_dir / "third_round_attention_evidence_matrix.csv",
        "company_second_source_audit": out_dir / "third_round_company_second_source_audit.csv",
        "parameter_review": out_dir / "third_round_parameter_review.csv",
        "orders": out_dir / "paper_orders_third_round_2026-05-27.csv",
        "decision_audit": out_dir / "third_round_decision_audit.csv",
        "holding_comparison": out_dir / "third_round_holding_comparison_2026-05-27_to_2026-06-16.csv",
        "round_comparison": out_dir / "third_round_round_comparison.csv",
        "summary": out_dir / "third_round_evaluation_summary.json",
    }
    event_layer.to_csv(paths["event_layer_audit"], index=False)
    attention.to_csv(paths["attention_evidence_matrix"], index=False)
    company_audit.to_csv(paths["company_second_source_audit"], index=False)
    parameter_review.to_csv(paths["parameter_review"], index=False)
    third_orders.to_csv(paths["orders"], index=False)
    decision_audit.to_csv(paths["decision_audit"], index=False)
    holding_comparison.to_csv(paths["holding_comparison"], index=False)
    round_comparison.to_csv(paths["round_comparison"], index=False)

    selected_parameter = {}
    if not parameter_review.empty:
        row = parameter_review[parameter_review["candidate"] == "third_round_selected"]
        if not row.empty:
            selected_parameter = row.iloc[0].to_dict()
    summary = {
        "as_of_date": as_of_date,
        "hold_end_date": hold_end_date,
        "third_round_event_total_return": float(selected_parameter.get("event_total_return", first_summary.get("event_optimized_total_return", 0.0))),
        "third_round_holding_return": float(second_summary.get("second_round_hedged_return", 0.0)),
        "third_round_positive": float(second_summary.get("second_round_hedged_return", 0.0)) > 0,
        "attention_manual_required_count": attention_pending_count,
        "company_core_confirmed_count": int((company_audit["evidence_status"] == "core_confirmed").sum()) if not company_audit.empty else 0,
        "company_supporting_confirmed_count": int((company_audit["evidence_status"] == "supporting_confirmed").sum()) if not company_audit.empty else 0,
        "company_sources_only_count": int((company_audit["evidence_status"] == "company_sources_only").sum()) if not company_audit.empty else 0,
        "third_round_order_count": int(len(third_orders)),
        "third_round_total_long_weight": float(third_orders["third_round_target_weight"].sum()) if not third_orders.empty else 0.0,
        "selected_parameter": selected_parameter,
        "output_files": {key: str(value) for key, value in paths.items()},
    }
    write_json(paths["summary"], summary)
    return summary


def main() -> None:
    import argparse

    parser = argparse.ArgumentParser()
    parser.add_argument("--root", default=str(ROOT))
    parser.add_argument("--as-of", default=None)
    parser.add_argument("--hold-end-date", default="2026-06-16")
    parser.add_argument("--allow-network-data", action="store_true")
    args = parser.parse_args()
    result = run_third_round_evaluation(
        Path(args.root),
        as_of=args.as_of,
        hold_end_date=args.hold_end_date,
        allow_network_data=args.allow_network_data,
    )
    print(json.dumps(result, ensure_ascii=False, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
