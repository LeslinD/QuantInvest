from __future__ import annotations

import argparse
from pathlib import Path

from .config import ROOT
from .second_round import dumps_summary, run_second_round_evaluation


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--root", default=str(ROOT))
    parser.add_argument("--hold-end-date", default="2026-06-16")
    args = parser.parse_args()
    result = run_second_round_evaluation(Path(args.root), hold_end_date=args.hold_end_date)
    print(dumps_summary(result))


if __name__ == "__main__":
    main()
