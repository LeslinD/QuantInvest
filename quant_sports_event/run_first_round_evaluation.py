from __future__ import annotations

import argparse
import json
from pathlib import Path

import pandas as pd

from .config import ROOT, load_project_config, read_json, write_json
from .costs import CostModel
from .data import load_snapshot
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
    midterm_holding_details.insert(0, "strategy", "midterm_pdf")
    optimized_holding_details.insert(0, "strategy", "first_round")
    holding_details = pd.concat([midterm_holding_details, optimized_holding_details], ignore_index=True)
    holding_path = out_dir / f"holding_comparison_2026-05-27_to_{hold_end_date}.csv"
    holding_details.to_csv(holding_path, index=False)

    total_midterm = float(comparison["midterm_return"].sum()) if not comparison.empty else 0.0
    total_optimized = float(comparison["optimized_return"].sum()) if not comparison.empty else 0.0
    avoid_rows = comparison[comparison["layer_action"] == "avoid"]
    result = {
        "event_comparison_path": str(comparison_path),
        "event_trade_details_path": str(event_details_path),
        "event_layer_stress_scan_path": str(stress_scan_path),
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
        "holding_midterm_return": midterm_holding_return,
        "holding_optimized_return": optimized_holding_return,
        "holding_excess_vs_midterm": optimized_holding_return - midterm_holding_return,
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
