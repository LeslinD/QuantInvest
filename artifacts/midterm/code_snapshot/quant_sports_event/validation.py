from __future__ import annotations

from dataclasses import dataclass
from datetime import datetime
from typing import Any, Dict, Iterable, List, Tuple


REQUIRED_STOCK_LINK_FIELDS = {"ticker", "company", "evidence", "confidence"}


@dataclass(frozen=True)
class ValidationResult:
    passed: bool
    accepted: List[Dict[str, Any]]
    rejected: List[Dict[str, Any]]
    metrics: Dict[str, float]


def _parse_datetime(value: str) -> datetime:
    value = value.replace("Z", "+00:00")
    try:
        return datetime.fromisoformat(value)
    except ValueError:
        return datetime.fromisoformat(value + "T00:00:00")


class EvidenceValidator:
    """Five-gate LLM output validator: schema, source, time, entity, quant."""

    def __init__(self, universe: Iterable[Dict[str, Any]], as_of_date: str):
        self.universe = {row["ticker"]: row for row in universe}
        self.as_of = _parse_datetime(as_of_date)

    def validate_links(self, links: Iterable[Dict[str, Any]]) -> ValidationResult:
        accepted: List[Dict[str, Any]] = []
        rejected: List[Dict[str, Any]] = []
        total = 0
        for link in links:
            total += 1
            ok, reason = self._validate_one(link)
            row = dict(link)
            if ok:
                accepted.append(row)
            else:
                row["reject_reason"] = reason
                rejected.append(row)
        metrics = {
            "total": float(total),
            "accepted": float(len(accepted)),
            "rejected": float(len(rejected)),
            "accept_rate": len(accepted) / total if total else 0.0,
            "hallucination_proxy_rate": len(rejected) / total if total else 0.0,
        }
        return ValidationResult(passed=len(rejected) == 0, accepted=accepted, rejected=rejected, metrics=metrics)

    def _validate_one(self, link: Dict[str, Any]) -> Tuple[bool, str]:
        missing = REQUIRED_STOCK_LINK_FIELDS - set(link)
        if missing:
            return False, f"schema_missing:{','.join(sorted(missing))}"
        ticker = link["ticker"]
        if ticker not in self.universe:
            return False, "entity_unknown_ticker"
        if link.get("company") != self.universe[ticker].get("name"):
            return False, "entity_company_mismatch"
        evidence = link.get("evidence") or {}
        if not evidence.get("source_url") or not evidence.get("claim"):
            return False, "source_missing"
        published_at = evidence.get("published_at") or evidence.get("retrieved_at")
        if not published_at:
            return False, "time_missing"
        try:
            if _parse_datetime(published_at).replace(tzinfo=None) > self.as_of.replace(tzinfo=None):
                return False, "time_after_decision_date"
        except ValueError:
            return False, "time_invalid"
        confidence = float(link.get("confidence", 0.0))
        if confidence <= 0:
            return False, "quant_non_positive_confidence"
        return True, ""
