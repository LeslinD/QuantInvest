from __future__ import annotations

import argparse
import json
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict, Iterable, List

import pandas as pd

from .config import ROOT, stable_hash, write_json
from .costs import round_lot_shares
from .super_factors import SUPER_FACTOR_WEIGHTS, build_super_factor_panel, validate_super_factor_weights


DEFAULT_ITERATION_ID = "final_loop_001"
DEFAULT_HYPOTHESIS = "在现有模拟订单股票池内加入综合因子边界和风险约束，生成第一轮期末优化订单。"
DEFAULT_CHANGED_MODULE = "M4_integrated_factors;M5_portfolio_risk;M7_diagnostics;M8_iteration_decision"
DEFAULT_BASELINE_VERSION = "midterm_code_baseline"
ROLLING_VALIDATION_LABEL = "滚动样本外验证"
OPTIMIZED_MAX_EVENT_EXPOSURE = 0.25
OPTIMIZED_MAX_POSITION = 0.10
OPTIMIZED_MAX_INDUSTRY = 0.20
DEFAULT_LOT_SIZE = 100

MIDTERM_PDF_WEIGHTS = {
    "605299.SH": 0.15,   # 舒华体育: 30% of a 50% total event book
    "002181.SZ": 0.125,  # 粤传媒: 25% of a 50% total event book
    "002878.SZ": 0.125,  # 元隆雅图: 25% of a 50% total event book
    "600060.SH": 0.10,   # 海信视像: 20% of a 50% total event book
}


@dataclass(frozen=True)
class LoopPaths:
    experiment_registry: Path
    iteration_log: Path
    diagnostic_report: Path
    next_actions: Path
    holding_attribution: Path
    position_compare: Path
    event_diagnostics: Path
    final_factor_panel: Path
    final_orders: Path
    risk_report: Path
    hedge_report: Path
    summary: Path


def _read_json(path: Path) -> Dict[str, Any]:
    return json.loads(path.read_text(encoding="utf-8"))


def _safe_read_csv(path: Path) -> pd.DataFrame:
    if not path.exists():
        return pd.DataFrame()
    return pd.read_csv(path)


def _resolve_output_path(root: Path, recorded: str | None, fallback_name: str) -> Path:
    if recorded:
        path = Path(recorded)
        if path.exists():
            return path
        local = root / "outputs" / "results" / path.name
        if local.exists():
            return local
    return root / "outputs" / "results" / fallback_name


def _summary_path(root: Path, as_of: str | None) -> Path:
    results = root / "outputs" / "results"
    if as_of:
        return results / f"run_summary_{as_of}.json"
    matches = sorted(results.glob("run_summary_*.json"))
    if not matches:
        raise FileNotFoundError(f"No run_summary_*.json found under {results}")
    return matches[-1]


def _metric(summary: Dict[str, Any], section: str, name: str, default: float = 0.0) -> float:
    value = summary.get(section, {}).get(name, default)
    try:
        return float(value)
    except (TypeError, ValueError):
        return default


def _format_pct(value: float) -> str:
    return f"{value:.2%}"


def _load_final_strategy(root: Path) -> Dict[str, Any]:
    strategy = {
        "integrated_factor_weights": dict(SUPER_FACTOR_WEIGHTS),
        "constraints": {
            "max_event_exposure": OPTIMIZED_MAX_EVENT_EXPOSURE,
            "max_position_per_stock": OPTIMIZED_MAX_POSITION,
            "max_industry_exposure": OPTIMIZED_MAX_INDUSTRY,
            "lot_size": DEFAULT_LOT_SIZE,
        },
        "source_path": "",
    }
    path = root / "configs" / "strategy_final.json"
    if not path.exists():
        return strategy
    loaded = _read_json(path)
    if "integrated_factor_weights" in loaded:
        strategy["integrated_factor_weights"].update(
            {key: float(value) for key, value in loaded["integrated_factor_weights"].items()}
        )
    if "constraints" in loaded:
        strategy["constraints"].update({key: loaded["constraints"][key] for key in loaded["constraints"]})
    strategy["source_path"] = str(path)
    return strategy


def _diagnose(
    summary: Dict[str, Any],
    orders: pd.DataFrame,
    event_summary: pd.DataFrame,
) -> List[Dict[str, str]]:
    findings: List[Dict[str, str]] = []
    validation_trades = _metric(summary, "walk_forward_summary", "num_trades")
    backtest_return = _metric(summary, "backtest_summary", "net_return")
    validation_return = _metric(summary, "walk_forward_summary", "net_return")
    validation_cost = _metric(summary, "walk_forward_summary", "cost_to_initial_cash")
    full_cost = _metric(summary, "backtest_summary", "cost_to_initial_cash")

    if validation_trades < 5:
        findings.append(
            {
                "tag": "small_sample_warning",
                "severity": "high",
                "evidence": f"{ROLLING_VALIDATION_LABEL}只包含 {validation_trades:.0f} 笔交易。",
                "action": "将当前收益结论标记为探索性结论，下一轮优先扩展事件库。",
                "module": "M1_event_library",
            }
        )

    if backtest_return - validation_return > 0.05:
        findings.append(
            {
                "tag": "overfit_gap",
                "severity": "high",
                "evidence": f"完整回测净收益为 {_format_pct(backtest_return)}，{ROLLING_VALIDATION_LABEL}净收益为 {_format_pct(validation_return)}。",
                "action": f"冻结当前参数，缩小参数网格自由度，下一轮要求{ROLLING_VALIDATION_LABEL}改善。",
                "module": "M6_backtest",
            }
        )

    weights = {k: float(v) for k, v in summary.get("factor_weights", {}).items()}
    if weights:
        top_factor, top_weight = max(weights.items(), key=lambda item: item[1])
        if top_weight >= 0.40:
            findings.append(
                {
                    "tag": "factor_balance_warning",
                    "severity": "medium",
                    "evidence": f"最高权重因子 `{top_factor}` 权重为 {_format_pct(top_weight)}。",
                    "action": "引入综合因子权重边界，保证事件机会和标的映射主导信号。",
                    "module": "M4_integrated_factors",
                }
            )

    if not orders.empty and "target_weight" in orders:
        total_weight = float(orders["target_weight"].sum())
        max_weight = float(orders["target_weight"].max())
        if total_weight > 0 and max_weight / total_weight > 0.60:
            top = orders.sort_values("target_weight", ascending=False).iloc[0]
            findings.append(
                {
                    "tag": "risk_concentration",
                    "severity": "medium",
                    "evidence": f"{top['company']} 占入选仓位的 {_format_pct(max_weight / total_weight)}。",
                    "action": "单独报告集中度，下一轮测试更严格的单票上限或链条加权上限。",
                    "module": "M5_portfolio_risk",
                }
            )

    if full_cost > 0.003 or validation_cost > 0.003:
        findings.append(
            {
                "tag": "cost_drag",
                "severity": "medium",
                "evidence": f"交易成本占初始资金：完整回测 {_format_pct(full_cost)}，{ROLLING_VALIDATION_LABEL} {_format_pct(validation_cost)}。",
                "action": "将换手和交易成本拖累加入迭代闭环的通过门槛。",
                "module": "M5_portfolio_risk",
            }
        )

    if not event_summary.empty:
        focus = event_summary[event_summary["window"] == "T-60_T-10"].copy()
        for _, row in focus.iterrows():
            caar = float(row.get("caar", 0.0))
            positive_rate = float(row.get("positive_rate", 0.0))
            if caar < 0 and positive_rate <= 0.5:
                findings.append(
                    {
                        "tag": "event_misfit",
                        "severity": "medium",
                        "evidence": f"{row['event_id']} T-60_T-10 CAAR 为 {_format_pct(caar)}，正 CAR 比例为 {positive_rate:.0%}。",
                        "action": "将该事件保留为高热低适配压力测试；除非标的映射证据增强，否则不提高仓位。",
                        "module": "M1_event_library",
                    }
                )

    if not findings:
        findings.append(
            {
                "tag": "baseline_ready",
                "severity": "info",
                "evidence": "当前输出未触发自动阻断问题。",
                "action": "将当前运行作为下一轮基线。",
                "module": "M8_iteration_decision",
            }
        )
    return findings


def _primary_contributors(row: pd.Series) -> tuple[str, str]:
    contribs = {col.removeprefix("contrib_"): float(row[col]) for col in row.index if col.startswith("contrib_")}
    if not contribs:
        return "", ""
    positive = max(contribs.items(), key=lambda item: item[1])[0]
    negative = min(contribs.items(), key=lambda item: item[1])[0]
    return positive, negative


def build_holding_attribution(scores: pd.DataFrame, orders: pd.DataFrame) -> pd.DataFrame:
    if scores.empty:
        return pd.DataFrame()
    selected = {}
    if not orders.empty:
        selected = orders.set_index("ticker")["target_weight"].to_dict()
    rows: List[Dict[str, Any]] = []
    for _, row in scores.iterrows():
        ticker = row["ticker"]
        primary_positive, primary_negative = _primary_contributors(row)
        target_weight = float(selected.get(ticker, 0.0))
        status = "selected" if target_weight > 0 else "not_selected"
        status_text = "入选" if target_weight > 0 else "未入选"
        explanation = (
            f"{row['name']} {status_text}；最大正贡献为 {primary_positive or 'n/a'}，"
            f"最大拖累为 {primary_negative or 'n/a'}。"
        )
        out = {
            "ticker": ticker,
            "name": row["name"],
            "industry": row["industry"],
            "selected": status == "selected",
            "target_weight": target_weight,
            "alpha_score": float(row.get("alpha_score", 0.0)),
            "primary_positive_contributor": primary_positive,
            "primary_negative_contributor": primary_negative,
            "explanation": explanation,
        }
        for col in scores.columns:
            if col.startswith("contrib_"):
                out[col] = float(row[col])
        rows.append(out)
    return pd.DataFrame(rows).sort_values(["selected", "alpha_score"], ascending=[False, False]).reset_index(drop=True)


def build_position_compare(orders: pd.DataFrame, scores: pd.DataFrame) -> pd.DataFrame:
    names = {}
    industries = {}
    if not scores.empty:
        names.update(scores.set_index("ticker")["name"].to_dict())
        industries.update(scores.set_index("ticker")["industry"].to_dict())
    current = {}
    if not orders.empty:
        current = orders.set_index("ticker")["target_weight"].to_dict()
        names.update(orders.set_index("ticker")["company"].to_dict())
        industries.update(orders.set_index("ticker")["industry"].to_dict())
    tickers = sorted(set(MIDTERM_PDF_WEIGHTS) | set(current))
    rows = []
    for ticker in tickers:
        old = float(MIDTERM_PDF_WEIGHTS.get(ticker, 0.0))
        new = float(current.get(ticker, 0.0))
        if new > old:
            reason = "increased_by_current_alpha_and_constraints"
        elif new < old and new > 0:
            reason = "reduced_by_current_alpha_or_risk_constraints"
        elif old > 0 and new == 0:
            reason = "removed_by_current_selection_gate"
        else:
            reason = "new_current_selection"
        rows.append(
            {
                "ticker": ticker,
                "name": names.get(ticker, ""),
                "industry": industries.get(ticker, ""),
                "midterm_pdf_weight": old,
                "current_loop_weight": new,
                "weight_change_vs_pdf": new - old,
                "change_reason": reason,
            }
        )
    return pd.DataFrame(rows)


def build_event_diagnostics(event_summary: pd.DataFrame) -> pd.DataFrame:
    if event_summary.empty:
        return pd.DataFrame()
    rows = []
    focus = event_summary[event_summary["window"] == "T-60_T-10"].copy()
    for _, row in focus.iterrows():
        caar = float(row.get("caar", 0.0))
        positive_rate = float(row.get("positive_rate", 0.0))
        if caar >= 0.05 and positive_rate >= 0.6:
            status = "tradable_evidence"
            action = "keep_as_core_or_supporting_event"
        elif caar < 0 and positive_rate <= 0.5:
            status = "misfit_pressure_test"
            action = "lower_event_fit_or_require_stronger_stock_linkage"
        else:
            status = "watch"
            action = "keep_for_monitoring_and_more_samples"
        rows.append(
            {
                "event_id": row["event_id"],
                "window": row["window"],
                "caar": caar,
                "positive_rate": positive_rate,
                "n": int(row.get("n", 0)),
                "diagnostic_status": status,
                "next_action": action,
            }
        )
    return pd.DataFrame(rows)


def build_final_targets(
    factor_panel: pd.DataFrame,
    *,
    max_event_exposure: float = OPTIMIZED_MAX_EVENT_EXPOSURE,
    max_position: float = OPTIMIZED_MAX_POSITION,
    max_industry: float = OPTIMIZED_MAX_INDUSTRY,
    top_k: int = 3,
) -> pd.DataFrame:
    if factor_panel.empty:
        return pd.DataFrame()
    candidates = factor_panel[factor_panel["eligible_for_loop_001"]].copy()
    if candidates.empty:
        return pd.DataFrame()
    candidates = candidates.sort_values("integrated_score", ascending=False).head(top_k)
    raw = candidates["integrated_score"].clip(lower=0.0)
    if raw.sum() <= 0:
        raw = pd.Series(1.0, index=candidates.index)
    candidates["final_target_weight"] = raw / raw.sum() * max_event_exposure
    candidates["final_target_weight"] = candidates["final_target_weight"].clip(upper=max_position)

    rows = []
    industry_used: Dict[str, float] = {}
    for _, row in candidates.sort_values("integrated_score", ascending=False).iterrows():
        industry = row["industry"]
        remaining = max(0.0, max_industry - industry_used.get(industry, 0.0))
        weight = min(float(row["final_target_weight"]), remaining)
        industry_used[industry] = industry_used.get(industry, 0.0) + weight
        out = row.copy()
        out["final_target_weight"] = weight
        rows.append(out)
    out = pd.DataFrame(rows)
    if out.empty:
        return out
    out = out[out["final_target_weight"] > 0].copy()
    if out["final_target_weight"].sum() > max_event_exposure:
        out["final_target_weight"] *= max_event_exposure / out["final_target_weight"].sum()
    return out.reset_index(drop=True)


def build_final_orders(
    final_targets: pd.DataFrame,
    current_orders: pd.DataFrame,
    *,
    initial_cash: float,
    lot_size: int = 100,
) -> pd.DataFrame:
    if final_targets.empty or current_orders.empty:
        return pd.DataFrame()
    current = current_orders.set_index("ticker")
    rows = []
    for _, row in final_targets.iterrows():
        ticker = row["ticker"]
        if ticker not in current.index:
            continue
        old = current.loc[ticker]
        reference_close = float(old["reference_close"])
        target_value = initial_cash * float(row["final_target_weight"])
        shares = round_lot_shares(target_value, reference_close, lot_size=lot_size)
        trade_value = shares * reference_close
        old_value = float(old.get("estimated_trade_value", 0.0))
        old_cost = float(old.get("expected_cost", 0.0))
        cost_rate = old_cost / old_value if old_value > 0 else 0.0015
        rows.append(
            {
                "decision_date": old.get("decision_date", ""),
                "order_date": old.get("order_date", ""),
                "ticker": ticker,
                "sina_symbol": old.get("sina_symbol", ""),
                "company": old.get("company", row.get("name", "")),
                "industry": row["industry"],
                "integrated_score": float(row["integrated_score"]),
                "current_target_weight": float(old.get("target_weight", 0.0)),
                "final_target_weight": float(row["final_target_weight"]),
                "reference_close": reference_close,
                "target_shares": int(shares),
                "estimated_trade_value": float(trade_value),
                "expected_cost": float(trade_value * cost_rate),
                "board": old.get("board", "main"),
            }
        )
    return pd.DataFrame(rows)


def build_risk_report(
    current_orders: pd.DataFrame,
    final_orders: pd.DataFrame,
    *,
    max_event_exposure: float = OPTIMIZED_MAX_EVENT_EXPOSURE,
    max_position: float = OPTIMIZED_MAX_POSITION,
    max_industry: float = OPTIMIZED_MAX_INDUSTRY,
    min_cash_weight: float = 0.60,
) -> pd.DataFrame:
    rows = []
    for label, frame, weight_col in [
        ("current", current_orders, "target_weight"),
        ("final_loop_001", final_orders, "final_target_weight"),
    ]:
        if frame.empty or weight_col not in frame:
            rows.append({"portfolio": label, "metric": "has_orders", "value": 0.0, "status": "fail", "note": "no orders"})
            continue
        total_weight = float(frame[weight_col].sum())
        largest_position = float(frame[weight_col].max())
        max_share = largest_position / total_weight if total_weight > 0 else 0.0
        cash_weight = 1.0 - total_weight
        rows.extend(
            [
                {"portfolio": label, "metric": "total_weight", "value": total_weight, "status": "pass" if total_weight <= max_event_exposure or label == "current" else "warn", "note": ""},
                {"portfolio": label, "metric": "cash_weight", "value": cash_weight, "status": "pass" if cash_weight >= min_cash_weight else "warn", "note": ""},
                {"portfolio": label, "metric": "max_position", "value": largest_position, "status": "pass" if largest_position <= max_position or label == "current" else "warn", "note": ""},
                {"portfolio": label, "metric": "max_selected_exposure_share", "value": max_share, "status": "pass" if max_share <= 0.60 else "warn", "note": "single-name share within selected exposure"},
            ]
        )
        industry = frame.groupby("industry")[weight_col].sum()
        for industry_name, exposure in industry.items():
            rows.append(
                {
                    "portfolio": label,
                    "metric": f"industry_{industry_name}",
                    "value": float(exposure),
                    "status": "pass" if float(exposure) <= max_industry or label == "current" else "warn",
                    "note": "industry exposure",
                }
            )
    return pd.DataFrame(rows)


def build_hedge_report(final_orders: pd.DataFrame) -> pd.DataFrame:
    total_weight = float(final_orders["final_target_weight"].sum()) if not final_orders.empty else 0.0
    cash_weight = 1.0 - total_weight
    return pd.DataFrame(
        [
            {
                "scenario": "market_drawdown",
                "portfolio_exposure": total_weight,
                "cash_weight": cash_weight,
                "hedge_tool": "cash_buffer;沪深300ETF或中证500ETF情景分析",
                "action": "本轮不生成对冲订单，使用现金和指数情景报告说明风险。",
            },
            {
                "scenario": "theme_cooling",
                "portfolio_exposure": total_weight,
                "cash_weight": cash_weight,
                "hedge_tool": "position_cap",
                "action": "通过单票10%上限和高热低适配事件降权控制主题退潮风险。",
            },
        ]
    )


def _json_for_csv(data: Dict[str, Any]) -> str:
    return json.dumps(data, ensure_ascii=False, sort_keys=True)


def _write_markdown_report(
    path: Path,
    *,
    iteration_id: str,
    summary: Dict[str, Any],
    findings: List[Dict[str, str]],
    orders: pd.DataFrame,
    final_orders: pd.DataFrame,
    risk_report: pd.DataFrame,
    event_diagnostics: pd.DataFrame,
) -> None:
    backtest_return = _metric(summary, "backtest_summary", "net_return")
    validation_return = _metric(summary, "walk_forward_summary", "net_return")
    total_weight = float(orders["target_weight"].sum()) if not orders.empty and "target_weight" in orders else 0.0
    final_total_weight = float(final_orders["final_target_weight"].sum()) if not final_orders.empty and "final_target_weight" in final_orders else 0.0
    order_count = int(len(orders))
    lines = [
        f"# {iteration_id} 诊断报告",
        "",
        "## 核心指标",
        "",
        f"- 决策日期：`{summary.get('as_of_date', '')}`",
        f"- 完整回测净收益：{_format_pct(backtest_return)}",
        f"- {ROLLING_VALIDATION_LABEL}净收益：{_format_pct(validation_return)}",
        f"- 当前模拟持仓数量：{order_count}",
        f"- 当前模拟目标仓位：{_format_pct(total_weight)}",
        f"- 第一轮优化目标仓位：{_format_pct(final_total_weight)}",
        "",
        "## 诊断结论",
        "",
    ]
    for finding in findings:
        lines.append(f"- `{finding['tag']}` [{finding['severity']}]: {finding['evidence']} 动作：{finding['action']}")
    lines.extend(["", "## 当前模拟持仓", ""])
    if orders.empty:
        lines.append("- 无模拟订单。")
    else:
        for _, row in orders.sort_values("target_weight", ascending=False).iterrows():
            lines.append(f"- {row['company']} `{row['ticker']}`：目标仓位 {_format_pct(float(row['target_weight']))}，行业 {row['industry']}。")
    lines.extend(["", "## 第一轮优化后持仓", ""])
    if final_orders.empty:
        lines.append("- 无优化后订单。")
    else:
        for _, row in final_orders.sort_values("final_target_weight", ascending=False).iterrows():
            lines.append(
                f"- {row['company']} `{row['ticker']}`：目标仓位 {_format_pct(float(row['final_target_weight']))}，"
                f"综合评分 {float(row['integrated_score']):.4f}。"
            )
    lines.extend(["", "## 风险检查", ""])
    if risk_report.empty:
        lines.append("- 无风险检查数据。")
    else:
        focus = risk_report[risk_report["portfolio"].isin(["current", "final_loop_001"])]
        for _, row in focus.iterrows():
            if row["metric"] in {"total_weight", "cash_weight", "max_position", "max_selected_exposure_share"}:
                lines.append(f"- {row['portfolio']} `{row['metric']}`：{_format_pct(float(row['value']))}，状态 {row['status']}。")
    lines.extend(["", "## 事件诊断", ""])
    if event_diagnostics.empty:
        lines.append("- 无事件诊断数据。")
    else:
        for _, row in event_diagnostics.iterrows():
            lines.append(
                f"- `{row['event_id']}`：{row['diagnostic_status']}，T-60_T-10 CAAR {_format_pct(float(row['caar']))}，"
                f"正 CAR 比例 {float(row['positive_rate']):.0%}。"
            )
    lines.extend(["", "## 第一轮结论", ""])
    lines.append(f"本轮建立诊断闭环，并生成综合因子与风险覆盖后的期末优化订单。下一轮优先处理事件样本少、完整回测与{ROLLING_VALIDATION_LABEL}差距、综合因子历史回测和热度数据接入。")
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")


def _write_next_actions(path: Path, iteration_id: str, findings: Iterable[Dict[str, str]]) -> None:
    lines = [f"iteration_id: {iteration_id}", "actions:"]
    for finding in findings:
        lines.extend(
            [
                f"  - tag: {finding['tag']}",
                f"    severity: {finding['severity']}",
                f"    module: {finding['module']}",
                f"    action: \"{finding['action']}\"",
            ]
        )
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")


def run_research_loop(
    root: Path = ROOT,
    *,
    as_of: str | None = None,
    iteration_id: str = DEFAULT_ITERATION_ID,
    hypothesis: str = DEFAULT_HYPOTHESIS,
    changed_module: str = DEFAULT_CHANGED_MODULE,
    baseline_version: str = DEFAULT_BASELINE_VERSION,
) -> Dict[str, Any]:
    root = Path(root)
    summary_path = _summary_path(root, as_of)
    summary = _read_json(summary_path)
    as_of = str(summary.get("as_of_date", as_of or ""))
    output_files = summary.get("output_files", {})

    orders = _safe_read_csv(_resolve_output_path(root, output_files.get("paper_orders"), f"paper_orders_{summary.get('best_params', {}).get('order_date', '2026-05-27')}.csv"))
    if orders.empty:
        orders = _safe_read_csv(root / "outputs" / "results" / "paper_orders_2026-05-27.csv")
    scores = _safe_read_csv(_resolve_output_path(root, output_files.get("paper_scores"), f"paper_scores_{as_of}.csv"))
    event_summary = _safe_read_csv(_resolve_output_path(root, output_files.get("event_study_summary"), f"event_study_summary_{as_of}.csv"))

    final_dir = root / "outputs" / "final"
    final_dir.mkdir(parents=True, exist_ok=True)
    paths = LoopPaths(
        experiment_registry=final_dir / "experiment_registry.csv",
        iteration_log=final_dir / "iteration_log.csv",
        diagnostic_report=final_dir / "diagnostic_report.md",
        next_actions=final_dir / "next_actions.yaml",
        holding_attribution=final_dir / "holding_attribution.csv",
        position_compare=final_dir / "position_compare_midterm_vs_current.csv",
        event_diagnostics=final_dir / "event_diagnostics.csv",
        final_factor_panel=final_dir / "factor_panel_final.csv",
        final_orders=final_dir / "paper_orders_final_2026-05-27.csv",
        risk_report=final_dir / "risk_report.csv",
        hedge_report=final_dir / "hedge_report.csv",
        summary=final_dir / "research_loop_summary.json",
    )

    findings = _diagnose(summary, orders, event_summary)
    eligible_tickers = orders["ticker"].tolist() if not orders.empty and "ticker" in orders else []
    final_strategy = _load_final_strategy(root)
    factor_weights = final_strategy["integrated_factor_weights"]
    constraints = final_strategy["constraints"]
    validate_super_factor_weights(factor_weights)
    final_factor_panel = build_super_factor_panel(scores, weights=factor_weights, eligible_tickers=eligible_tickers)
    final_targets = build_final_targets(
        final_factor_panel,
        max_event_exposure=float(constraints["max_event_exposure"]),
        max_position=float(constraints["max_position_per_stock"]),
        max_industry=float(constraints["max_industry_exposure"]),
    )
    final_orders = build_final_orders(
        final_targets,
        orders,
        initial_cash=float(summary.get("initial_cash", 1000000.0)),
        lot_size=int(constraints["lot_size"]),
    )
    risk_report = build_risk_report(
        orders,
        final_orders,
        max_event_exposure=float(constraints["max_event_exposure"]),
        max_position=float(constraints["max_position_per_stock"]),
        max_industry=float(constraints["max_industry_exposure"]),
        min_cash_weight=float(constraints.get("min_cash_weight", 0.60)),
    )
    hedge_report = build_hedge_report(final_orders)
    holding_attribution = build_holding_attribution(scores, orders)
    position_compare = build_position_compare(orders, scores)
    event_diagnostics = build_event_diagnostics(event_summary)

    holding_attribution.to_csv(paths.holding_attribution, index=False)
    position_compare.to_csv(paths.position_compare, index=False)
    event_diagnostics.to_csv(paths.event_diagnostics, index=False)
    final_factor_panel.to_csv(paths.final_factor_panel, index=False)
    final_orders.to_csv(paths.final_orders, index=False)
    risk_report.to_csv(paths.risk_report, index=False)
    hedge_report.to_csv(paths.hedge_report, index=False)
    _write_next_actions(paths.next_actions, iteration_id, findings)
    _write_markdown_report(
        paths.diagnostic_report,
        iteration_id=iteration_id,
        summary=summary,
        findings=findings,
        orders=orders,
        final_orders=final_orders,
        risk_report=risk_report,
        event_diagnostics=event_diagnostics,
    )

    metrics = {
        "backtest_net_return": _metric(summary, "backtest_summary", "net_return"),
        "rolling_oos_net_return": _metric(summary, "walk_forward_summary", "net_return"),
        "paper_total_target_weight": float(orders["target_weight"].sum()) if not orders.empty and "target_weight" in orders else 0.0,
        "paper_order_count": int(len(orders)),
        "final_total_target_weight": float(final_orders["final_target_weight"].sum()) if not final_orders.empty else 0.0,
        "final_order_count": int(len(final_orders)),
        "diagnostic_count": int(len(findings)),
    }
    decision = "accept_first_round_risk_overlay"
    review_note = "已建立基线诊断，并生成第一轮综合因子与风险覆盖后的期末优化订单。"
    changed_files = [
        "quant_sports_event/research_loop.py",
        "quant_sports_event/run_research_loop.py",
        "quant_sports_event/super_factors.py",
        "configs/strategy_final.json",
    ]
    registry = pd.DataFrame(
        [
            {
                "iteration_id": iteration_id,
                "hypothesis": hypothesis,
                "changed_module": changed_module,
                "changed_files": ";".join(changed_files),
                "baseline_version": baseline_version,
                "as_of_date": as_of,
                "run_hash": summary.get("run_hash", ""),
                **metrics,
                "decision": decision,
                "review_note": review_note,
            }
        ]
    )
    registry.to_csv(paths.experiment_registry, index=False)

    iteration_log = pd.DataFrame(
        [
            {
                "iteration_id": iteration_id,
                "hypothesis": hypothesis,
                "changed_module": changed_module,
                "changed_files": ";".join(changed_files),
                "baseline_version": baseline_version,
                "metrics_before": _json_for_csv({"baseline": baseline_version}),
                "metrics_after": _json_for_csv(metrics),
                "ablation_result": "去除对照实验在下一轮扩展；本轮完成综合因子边界和风险覆盖。",
                "decision": decision,
                "review_note": review_note,
                "next_action_count": len(findings),
            }
        ]
    )
    iteration_log.to_csv(paths.iteration_log, index=False)

    loop_summary = {
        "iteration_id": iteration_id,
        "as_of_date": as_of,
        "baseline_version": baseline_version,
        "decision": decision,
        "metrics": metrics,
        "findings": findings,
        "final_strategy": final_strategy,
        "output_files": {field: str(getattr(paths, field)) for field in paths.__dataclass_fields__},
    }
    loop_summary["loop_hash"] = stable_hash(loop_summary)
    write_json(paths.summary, loop_summary)
    return loop_summary


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--root", default=str(ROOT))
    parser.add_argument("--as-of", default=None)
    parser.add_argument("--iteration-id", default=DEFAULT_ITERATION_ID)
    args = parser.parse_args()
    result = run_research_loop(Path(args.root), as_of=args.as_of, iteration_id=args.iteration_id)
    print(json.dumps(result, ensure_ascii=False, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
