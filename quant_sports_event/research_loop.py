from __future__ import annotations

import argparse
import json
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict, Iterable, List

import pandas as pd

from .config import ROOT, stable_hash, write_json


DEFAULT_ITERATION_ID = "final_loop_001"
DEFAULT_HYPOTHESIS = "建立现有事件驱动流水线的基线诊断，暂不修改交易逻辑。"
DEFAULT_CHANGED_MODULE = "M7_diagnostics;M8_iteration_decision"
DEFAULT_BASELINE_VERSION = "midterm_code_baseline"

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
                "evidence": f"走步验证只包含 {validation_trades:.0f} 笔交易。",
                "action": "将当前收益结论标记为探索性结论，下一轮优先扩展事件库。",
                "module": "M1_event_library",
            }
        )

    if backtest_return - validation_return > 0.05:
        findings.append(
            {
                "tag": "overfit_gap",
                "severity": "high",
                "evidence": f"完整回测净收益为 {_format_pct(backtest_return)}，走步验证净收益为 {_format_pct(validation_return)}。",
                "action": "冻结当前参数，缩小参数网格自由度，下一轮要求走步验证改善。",
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
                    "action": "引入超因子权重边界，保证事件机会和标的映射主导信号。",
                    "module": "M4_super_factors",
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
                "evidence": f"Cost to cash: full {_format_pct(full_cost)}, validation {_format_pct(validation_cost)}.",
                "action": "Add turnover and cost drag to loop acceptance gates.",
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


def _json_for_csv(data: Dict[str, Any]) -> str:
    return json.dumps(data, ensure_ascii=False, sort_keys=True)


def _write_markdown_report(
    path: Path,
    *,
    iteration_id: str,
    summary: Dict[str, Any],
    findings: List[Dict[str, str]],
    orders: pd.DataFrame,
    event_diagnostics: pd.DataFrame,
) -> None:
    backtest_return = _metric(summary, "backtest_summary", "net_return")
    validation_return = _metric(summary, "walk_forward_summary", "net_return")
    total_weight = float(orders["target_weight"].sum()) if not orders.empty and "target_weight" in orders else 0.0
    order_count = int(len(orders))
    lines = [
        f"# {iteration_id} 诊断报告",
        "",
        "## 核心指标",
        "",
        f"- 决策日期：`{summary.get('as_of_date', '')}`",
        f"- 完整回测净收益：{_format_pct(backtest_return)}",
        f"- 走步验证净收益：{_format_pct(validation_return)}",
        f"- 模拟持仓数量：{order_count}",
        f"- 模拟目标仓位：{_format_pct(total_weight)}",
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
    lines.append("本轮只建立诊断闭环，不改变交易逻辑。下一轮优先处理事件样本少、完整回测与走步验证差距、低波动因子权重偏高和持仓集中问题。")
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
        summary=final_dir / "research_loop_summary.json",
    )

    findings = _diagnose(summary, orders, event_summary)
    holding_attribution = build_holding_attribution(scores, orders)
    position_compare = build_position_compare(orders, scores)
    event_diagnostics = build_event_diagnostics(event_summary)

    holding_attribution.to_csv(paths.holding_attribution, index=False)
    position_compare.to_csv(paths.position_compare, index=False)
    event_diagnostics.to_csv(paths.event_diagnostics, index=False)
    _write_next_actions(paths.next_actions, iteration_id, findings)
    _write_markdown_report(
        paths.diagnostic_report,
        iteration_id=iteration_id,
        summary=summary,
        findings=findings,
        orders=orders,
        event_diagnostics=event_diagnostics,
    )

    metrics = {
        "backtest_net_return": _metric(summary, "backtest_summary", "net_return"),
        "walk_forward_net_return": _metric(summary, "walk_forward_summary", "net_return"),
        "paper_total_target_weight": float(orders["target_weight"].sum()) if not orders.empty and "target_weight" in orders else 0.0,
        "paper_order_count": int(len(orders)),
        "diagnostic_count": int(len(findings)),
    }
    decision = "observe"
    review_note = "已建立基线诊断；loop 001 不修改交易逻辑。"
    changed_files = [
        "quant_sports_event/research_loop.py",
        "quant_sports_event/run_research_loop.py",
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
                "ablation_result": "not_run_in_loop_001",
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
