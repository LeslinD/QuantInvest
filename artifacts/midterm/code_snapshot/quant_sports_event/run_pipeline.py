from __future__ import annotations

import argparse
import json
from pathlib import Path

import pandas as pd

from .backtest import expanding_walk_forward_validate, run_event_backtest, walk_forward_search
from .config import ROOT, load_project_config, stable_hash, write_json
from .costs import CostModel
from .data import SinaDailyProvider, freeze_market_snapshot, load_snapshot
from .event_study import event_study, summarize_event_study
from .factors import factor_panel_for_date, score_panel
from .llm_agent import OpenAICompatibleAgent, RuleBasedEventAgent
from .portfolio import build_target_weights, generate_orders
from .validation import EvidenceValidator


def run_all(root: Path = ROOT) -> dict:
    cfg = load_project_config(root)
    strategy = cfg.raw
    as_of = strategy["as_of_date"]
    out_dir = root / "outputs" / "results"
    out_dir.mkdir(parents=True, exist_ok=True)
    snapshot_dir = root / strategy["data"]["snapshot_dir"] / as_of

    symbols = [row["sina_symbol"] for row in cfg.universe]
    benchmarks = [strategy["benchmark_symbol"], strategy["secondary_benchmark_symbol"]]
    provider = SinaDailyProvider(datalen=int(strategy["data"]["datalen"]))
    paths = freeze_market_snapshot(provider, symbols, as_of, snapshot_dir, benchmark_symbols=benchmarks)
    frames = load_snapshot(paths)

    event = next(e for e in cfg.events if e["event_id"] == strategy["paper_trading"]["event_id"])
    agent = OpenAICompatibleAgent() if OpenAICompatibleAgent().is_enabled() else RuleBasedEventAgent()
    raw_links = agent.extract_stock_links(event, cfg.universe)
    validator = EvidenceValidator(cfg.universe, as_of)
    validation = validator.validate_links(raw_links)
    exposure_by_ticker = {row["ticker"]: float(row["confidence"]) for row in validation.accepted}

    completed_events = [e for e in cfg.events if e.get("status") == "completed"]
    cost_model = CostModel.from_config(strategy["costs"])
    benchmark = frames.get(strategy["benchmark_symbol"])
    best_params, search_results = walk_forward_search(
        frames,
        cfg.universe,
        exposure_by_ticker,
        completed_events,
        strategy["hyperparameter_grid"],
        cost_model,
        strategy["constraints"],
        strategy["initial_cash"],
        benchmark=benchmark,
        model_config=strategy.get("model", {}),
    )
    validation_trades, validation_summary = expanding_walk_forward_validate(
        frames,
        cfg.universe,
        exposure_by_ticker,
        completed_events,
        best_params,
        cost_model,
        strategy["constraints"],
        strategy["initial_cash"],
        benchmark=benchmark,
        model_config=strategy.get("model", {}),
    )
    backtest = run_event_backtest(
        frames,
        cfg.universe,
        exposure_by_ticker,
        completed_events,
        best_params,
        cost_model,
        strategy["constraints"],
        strategy["initial_cash"],
        benchmark=benchmark,
        model_config=strategy.get("model", {}),
    )
    car = event_study(frames, benchmark, cfg.universe, completed_events)
    car_summary = summarize_event_study(car)

    decision_date = strategy["paper_trading"]["decision_date"]
    panel = factor_panel_for_date(
        frames,
        cfg.universe,
        exposure_by_ticker,
        decision_date,
        best_params["attention_lookback"],
        best_params["momentum_lookback"],
    )
    scored = score_panel(panel, backtest.weights, attention_z_cap=best_params.get("attention_z_cap"))
    scored = scored[scored["exposure"] >= float(best_params.get("min_exposure_for_trade", 0.0))].copy()
    max_exposure = min(float(strategy["constraints"]["max_event_exposure"]), float(strategy["paper_trading"]["fallback_max_event_exposure"]))
    targets = build_target_weights(
        scored,
        max_event_exposure=max_exposure,
        max_position_per_stock=float(strategy["constraints"]["max_position_per_stock"]),
        max_industry_exposure=float(strategy["constraints"]["max_industry_exposure"]),
        top_k=int(best_params["top_k"]),
    )
    orders = generate_orders(
        targets,
        frames,
        cfg.universe,
        cash=float(strategy["initial_cash"]),
        decision_date=decision_date,
        order_date=strategy["paper_trading"]["order_date"],
        cost_model=cost_model,
        lot_size=int(strategy["constraints"]["lot_size"]),
    )

    raw_links_path = out_dir / f"llm_links_{as_of}.json"
    validation_path = out_dir / f"validation_metrics_{as_of}.json"
    search_path = out_dir / f"hyperparam_search_{as_of}.csv"
    trades_path = out_dir / f"backtest_trades_{as_of}.csv"
    validation_trades_path = out_dir / f"walk_forward_trades_{as_of}.csv"
    samples_path = out_dir / f"training_samples_{as_of}.csv"
    scores_path = out_dir / f"paper_scores_{as_of}.csv"
    orders_path = out_dir / f"paper_orders_{strategy['paper_trading']['order_date']}.csv"
    car_path = out_dir / f"event_study_car_{as_of}.csv"
    car_summary_path = out_dir / f"event_study_summary_{as_of}.csv"
    summary_path = out_dir / f"run_summary_{as_of}.json"

    raw_links_path.write_text(json.dumps(raw_links, ensure_ascii=False, indent=2, sort_keys=True), encoding="utf-8")
    write_json(validation_path, validation.metrics)
    search_results.to_csv(search_path, index=False)
    backtest.trades.to_csv(trades_path, index=False)
    validation_trades.to_csv(validation_trades_path, index=False)
    backtest.samples.to_csv(samples_path, index=False)
    scored.to_csv(scores_path, index=False)
    orders.to_csv(orders_path, index=False)
    car.to_csv(car_path, index=False)
    car_summary.to_csv(car_summary_path, index=False)

    summary = {
        "as_of_date": as_of,
        "config_hash": cfg.config_hash,
        "code_version": "quant_sports_event-0.1.0",
        "data_snapshot_meta": paths["_meta"],
        "validation": validation.metrics,
        "best_params": best_params,
        "factor_weights": backtest.weights,
        "walk_forward_summary": validation_summary,
        "backtest_summary": backtest.summary,
        "paper_total_target_weight": float(orders["target_weight"].sum()) if not orders.empty else 0.0,
        "paper_total_estimated_trade_value": float(orders["estimated_trade_value"].sum()) if not orders.empty else 0.0,
        "paper_total_expected_cost": float(orders["expected_cost"].sum()) if not orders.empty else 0.0,
        "output_files": {
            "raw_links": str(raw_links_path),
            "validation": str(validation_path),
            "hyperparam_search": str(search_path),
            "backtest_trades": str(trades_path),
            "walk_forward_trades": str(validation_trades_path),
            "training_samples": str(samples_path),
            "paper_scores": str(scores_path),
            "paper_orders": str(orders_path),
            "event_study_car": str(car_path),
            "event_study_summary": str(car_summary_path),
        },
        "run_hash": stable_hash(
            {
                "config": cfg.config_hash,
                "best_params": best_params,
                "factor_weights": backtest.weights,
                "orders": orders.to_dict(orient="records"),
            }
        ),
    }
    summary["output_files"]["summary"] = str(summary_path)
    write_json(summary_path, summary)
    return summary


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--root", default=str(ROOT))
    args = parser.parse_args()
    summary = run_all(Path(args.root))
    print(json.dumps(summary, ensure_ascii=False, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
