from __future__ import annotations

import argparse
import json
from pathlib import Path

from .config import ROOT
from .run_first_round_evaluation import run_evaluation
from .run_pipeline import run_all
from .research_loop import run_research_loop


def run_final_pipeline(root: Path = ROOT, *, as_of: str | None = None, hold_end_date: str = "2026-06-16") -> dict:
    root = Path(root)
    pipeline_summary = run_all(root)
    resolved_as_of = as_of or str(pipeline_summary.get("as_of_date", ""))
    research_summary = run_research_loop(root, as_of=resolved_as_of)
    evaluation_summary = run_evaluation(root, hold_end_date=hold_end_date)
    return {
        "as_of_date": resolved_as_of,
        "hold_end_date": hold_end_date,
        "pipeline_summary_path": pipeline_summary.get("output_files", {}).get("summary", ""),
        "research_loop_summary_path": research_summary.get("output_files", {}).get("summary", ""),
        "first_round_evaluation_summary_path": str(root / "outputs" / "final" / "first_round_evaluation_summary.json"),
        "pipeline": {
            "backtest_net_return": pipeline_summary.get("backtest_summary", {}).get("net_return", 0.0),
            "rolling_oos_net_return": pipeline_summary.get("walk_forward_summary", {}).get("net_return", 0.0),
            "paper_total_target_weight": pipeline_summary.get("paper_total_target_weight", 0.0),
        },
        "research_loop": research_summary.get("metrics", {}),
        "first_round_evaluation": {
            "event_midterm_total_return": evaluation_summary.get("event_midterm_total_return", 0.0),
            "event_optimized_total_return": evaluation_summary.get("event_optimized_total_return", 0.0),
            "holding_midterm_return": evaluation_summary.get("holding_midterm_return", 0.0),
            "holding_optimized_return": evaluation_summary.get("holding_optimized_return", 0.0),
            "manual_attention_pending_count": evaluation_summary.get("manual_attention_pending_count", 0),
        },
    }


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--root", default=str(ROOT))
    parser.add_argument("--as-of", default=None)
    parser.add_argument("--hold-end-date", default="2026-06-16")
    args = parser.parse_args()
    result = run_final_pipeline(Path(args.root), as_of=args.as_of, hold_end_date=args.hold_end_date)
    print(json.dumps(result, ensure_ascii=False, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
