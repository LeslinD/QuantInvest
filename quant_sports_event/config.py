from __future__ import annotations

import hashlib
import json
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict, List


ROOT = Path(__file__).resolve().parents[1]


def read_json(path: str | Path) -> Any:
    with Path(path).open("r", encoding="utf-8") as f:
        return json.load(f)


def write_json(path: str | Path, data: Any) -> None:
    p = Path(path)
    p.parent.mkdir(parents=True, exist_ok=True)
    with p.open("w", encoding="utf-8") as f:
        json.dump(data, f, ensure_ascii=False, indent=2, sort_keys=True)


def stable_hash(data: Any) -> str:
    payload = json.dumps(data, ensure_ascii=False, sort_keys=True, default=str)
    return hashlib.sha256(payload.encode("utf-8")).hexdigest()


@dataclass(frozen=True)
class ProjectConfig:
    raw: Dict[str, Any]
    events: List[Dict[str, Any]]
    universe: List[Dict[str, Any]]
    config_hash: str


def load_project_config(root: str | Path = ROOT) -> ProjectConfig:
    root = Path(root)
    raw = read_json(root / "configs" / "strategy.json")
    events = read_json(root / "configs" / "events.json")
    universe = read_json(root / "configs" / "universe.json")
    config_hash = stable_hash({"strategy": raw, "events": events, "universe": universe})
    return ProjectConfig(raw=raw, events=events, universe=universe, config_hash=config_hash)
