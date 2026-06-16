from __future__ import annotations

import argparse
import json
from pathlib import Path

import pandas as pd

from .audit import (
    ablation_comparison,
    attention_data_audit,
    event_failure_analysis,
    event_library_audit,
    holding_failure_analysis,
    universe_evidence_audit,
)
from .config import ROOT, load_project_config, read_json, write_json
from .costs import CostModel
from .data import SinaDailyProvider, load_snapshot
from .evaluation import (
    MIDTERM_PDF_WEIGHTS,
    event_strategy_comparison,
    fetch_holding_frames,
    scan_event_layer_windows,
    static_portfolio_return,
)


def run_evaluation(root: Path = ROOT, *, hold_end_date: str = "2026-06-16") -> dict:
    cfg = load_project_config(root)
    strategy = cfg.raw
    out_dir = root / "outputs" / "final"
    out_dir.mkdir(parents=True, exist_ok=True)

    summary = read_json(root / "outputs" / "results" / f"run_summary_{strategy['as_of_date']}.json")
    snapshot_meta = read_json(summary["data_snapshot_meta"])
    frames = load_snapshot(snapshot_meta["paths"])
    raw_links = read_json(root / "outputs" / "results" / f"llm_links_{strategy['as_of_date']}.json")
    exposure_by_event = {}
    for row in raw_links:
        exposure_by_event.setdefault(row["event_id"], {})[row["ticker"]] = float(row["confidence"])

    cost_model = CostModel.from_config(strategy["costs"])
    comparison, event_details = event_strategy_comparison(
        frames,
        cfg.universe,
        [event for event in cfg.events if event.get("status") == "completed"],
        exposure_by_event,
        summary["best_params"],
        summary["factor_weights"],
        cost_model,
        strategy["constraints"],
        strategy.get("model", {}),
        initial_cash=float(strategy["initial_cash"]),
    )
    comparison_path = out_dir / "event_strategy_comparison.csv"
    event_details_path = out_dir / "event_strategy_trade_details.csv"
    comparison.to_csv(comparison_path, index=False)
    event_details.to_csv(event_details_path, index=False)
    event_audit = event_library_audit(cfg.events, strategy.get("model", {}))
    universe_audit = universe_evidence_audit(cfg.universe)
    attention_audit = attention_data_audit(cfg.events, cfg.universe)
    event_audit_path = out_dir / "event_library_audit.csv"
    universe_audit_path = out_dir / "universe_evidence_audit.csv"
    attention_audit_path = out_dir / "attention_data_audit.csv"
    event_audit.to_csv(event_audit_path, index=False)
    universe_audit.to_csv(universe_audit_path, index=False)
    attention_audit.to_csv(attention_audit_path, index=False)
    completed_events = [event for event in cfg.events if event.get("status") == "completed"]
    ablation = ablation_comparison(
        frames,
        cfg.universe,
        completed_events,
        exposure_by_event,
        summary["best_params"],
        summary["factor_weights"],
        cost_model,
        strategy["constraints"],
        strategy.get("model", {}),
        initial_cash=float(strategy["initial_cash"]),
    )
    ablation_path = out_dir / "ablation_comparison.csv"
    ablation.to_csv(ablation_path, index=False)
    avoid_ids = set(comparison.loc[comparison["layer_action"] == "avoid", "event_id"]) if not comparison.empty else set()
    stress_scan = scan_event_layer_windows(
        frames,
        cfg.universe,
        [event for event in cfg.events if event.get("event_id") in avoid_ids],
        exposure_by_event,
        summary["best_params"],
        summary["factor_weights"],
        cost_model,
        strategy["constraints"],
        strategy.get("model", {}),
        initial_cash=float(strategy["initial_cash"]),
    )
    stress_scan_path = out_dir / "event_layer_stress_scan.csv"
    stress_scan.to_csv(stress_scan_path, index=False)

    final_orders = pd.read_csv(out_dir / "paper_orders_final_2026-05-27.csv")
    final_weights = final_orders.set_index("ticker")["final_target_weight"].to_dict()
    universe_by_ticker = {row["ticker"]: row for row in cfg.universe}
    # Include midterm-only tickers if they are present in the legacy universe file.
    legacy_universe_path = root / "configs" / "universe.json"
    if legacy_universe_path.exists():
        for row in read_json(legacy_universe_path):
            universe_by_ticker.setdefault(row["ticker"], row)
    holding_tickers = set(MIDTERM_PDF_WEIGHTS) | set(final_weights)
    holding_frames = fetch_holding_frames(universe_by_ticker, holding_tickers, end_date=hold_end_date)
    midterm_holding_return, midterm_holding_details = static_portfolio_return(
        holding_frames,
        universe_by_ticker,
        MIDTERM_PDF_WEIGHTS,
        start_date="2026-05-27",
        end_date=hold_end_date,
        initial_cash=float(strategy["initial_cash"]),
        cost_model=cost_model,
        lot_size=int(strategy["constraints"]["lot_size"]),
        buy_field="open",
        sell_field="close",
    )
    optimized_holding_return, optimized_holding_details = static_portfolio_return(
        holding_frames,
        universe_by_ticker,
        final_weights,
        start_date="2026-05-27",
        end_date=hold_end_date,
        initial_cash=float(strategy["initial_cash"]),
        cost_model=cost_model,
        lot_size=int(strategy["constraints"]["lot_size"]),
        buy_field="open",
        sell_field="close",
    )
    midterm_holding_details.insert(0, "strategy", "midterm")
    optimized_holding_details.insert(0, "strategy", "first_round")
    holding_details = pd.concat([midterm_holding_details, optimized_holding_details], ignore_index=True)
    holding_path = out_dir / f"holding_comparison_2026-05-27_to_{hold_end_date}.csv"
    holding_details.to_csv(holding_path, index=False)
    holding_analysis = holding_failure_analysis(holding_details)
    holding_analysis_path = out_dir / "holding_failure_analysis.csv"
    holding_analysis.to_csv(holding_analysis_path, index=False)
    event_summary_path = root / "outputs" / "results" / f"event_study_summary_{strategy['as_of_date']}.csv"
    event_summary = pd.read_csv(event_summary_path) if event_summary_path.exists() else pd.DataFrame()
    event_analysis = event_failure_analysis(comparison, event_details, event_summary)
    event_analysis_path = out_dir / "event_failure_analysis.csv"
    event_analysis.to_csv(event_analysis_path, index=False)

    provider = SinaDailyProvider(datalen=1800)
    market_rows = []
    for symbol in strategy.get("benchmark_symbols", []):
        frame = provider.fetch_daily(symbol, end=hold_end_date)
        start_rows = frame[frame["date"] >= pd.to_datetime("2026-05-27")]
        end_rows = frame[frame["date"] <= pd.to_datetime(hold_end_date)]
        if start_rows.empty or end_rows.empty:
            continue
        start = start_rows.iloc[0]
        end = end_rows.iloc[-1]
        market_rows.append(
            {
                "symbol": symbol,
                "name": strategy.get("benchmark_names", {}).get(symbol, symbol),
                "start_date": start["date"].date().isoformat(),
                "end_date": end["date"].date().isoformat(),
                "start_open": float(start["open"]),
                "end_close": float(end["close"]),
                "period_return": float(end["close"]) / float(start["open"]) - 1,
            }
        )
    market_context = pd.DataFrame(market_rows)
    market_context_path = out_dir / f"holding_market_context_2026-05-27_to_{hold_end_date}.csv"
    market_context.to_csv(market_context_path, index=False)
    chart_dir = out_dir / "chart_data"
    chart_dir.mkdir(parents=True, exist_ok=True)
    comparison[["event_id", "midterm_return", "optimized_return", "excess_vs_midterm"]].to_csv(
        chart_dir / "event_strategy_comparison_chart.csv",
        index=False,
    )
    holding_details[["strategy", "ticker", "company", "target_weight", "stock_return", "pnl"]].to_csv(
        chart_dir / "holding_comparison_chart.csv",
        index=False,
    )
    ablation[["experiment", "total_return", "excess_vs_full_model", "trade_count"]].to_csv(
        chart_dir / "ablation_comparison_chart.csv",
        index=False,
    )
    market_context[["symbol", "name", "period_return"]].to_csv(
        chart_dir / "holding_market_context_chart.csv",
        index=False,
    )

    total_midterm = float(comparison["midterm_return"].sum()) if not comparison.empty else 0.0
    total_optimized = float(comparison["optimized_return"].sum()) if not comparison.empty else 0.0
    avoid_rows = comparison[comparison["layer_action"] == "avoid"]
    full_ablation = ablation[ablation["experiment"] == "final_full_model"]
    no_fit_ablation = ablation[ablation["experiment"] == "no_fit_gate"]
    result = {
        "event_comparison_path": str(comparison_path),
        "event_trade_details_path": str(event_details_path),
        "event_layer_stress_scan_path": str(stress_scan_path),
        "event_library_audit_path": str(event_audit_path),
        "universe_evidence_audit_path": str(universe_audit_path),
        "attention_data_audit_path": str(attention_audit_path),
        "ablation_comparison_path": str(ablation_path),
        "event_failure_analysis_path": str(event_analysis_path),
        "holding_failure_analysis_path": str(holding_analysis_path),
        "holding_market_context_path": str(market_context_path),
        "chart_data_dir": str(chart_dir),
        "holding_comparison_path": str(holding_path),
        "event_midterm_total_return": total_midterm,
        "event_optimized_total_return": total_optimized,
        "event_excess_vs_midterm": total_optimized - total_midterm,
        "event_win_count_vs_midterm": int((comparison["excess_vs_midterm"] > 0).sum()) if not comparison.empty else 0,
        "event_count": int(len(comparison)),
        "avoided_loss_vs_midterm": float(avoid_rows["avoided_loss_vs_midterm"].sum()) if not avoid_rows.empty else 0.0,
        "stress_scan_positive_count": int((stress_scan["return_on_initial_cash"] > 0).sum()) if not stress_scan.empty else 0,
        "stress_scan_best_nonzero_return": float(
            stress_scan.loc[stress_scan["trade_count"] > 0, "return_on_initial_cash"].max()
        )
        if not stress_scan.empty and not stress_scan.loc[stress_scan["trade_count"] > 0].empty
        else 0.0,
        "event_audit_pass_count": int((event_audit["audit_status"] == "pass").sum()) if not event_audit.empty else 0,
        "event_audit_count": int(len(event_audit)),
        "universe_core_ready_count": int((universe_audit["audit_status"] == "core_ready").sum()) if not universe_audit.empty else 0,
        "universe_usable_first_round_count": int((universe_audit["audit_status"] == "usable_first_round").sum()) if not universe_audit.empty else 0,
        "manual_attention_pending_count": int((attention_audit["status"] == "pending_manual_import").sum()) if not attention_audit.empty else 0,
        "no_fit_gate_return": float(no_fit_ablation["total_return"].iloc[0]) if not no_fit_ablation.empty else 0.0,
        "full_model_ablation_return": float(full_ablation["total_return"].iloc[0]) if not full_ablation.empty else 0.0,
        "holding_midterm_return": midterm_holding_return,
        "holding_optimized_return": optimized_holding_return,
        "holding_excess_vs_midterm": optimized_holding_return - midterm_holding_return,
        "holding_market_min_return": float(market_context["period_return"].min()) if not market_context.empty else 0.0,
        "holding_market_max_return": float(market_context["period_return"].max()) if not market_context.empty else 0.0,
        "hold_end_date": hold_end_date,
    }
    write_json(out_dir / "first_round_evaluation_summary.json", result)
    return result


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--root", default=str(ROOT))
    parser.add_argument("--hold-end-date", default="2026-06-16")
    args = parser.parse_args()
    result = run_evaluation(Path(args.root), hold_end_date=args.hold_end_date)
    print(json.dumps(result, ensure_ascii=False, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
