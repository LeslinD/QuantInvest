from __future__ import annotations

import json
import os
from typing import Any, Dict, Iterable, List
from urllib.request import Request, urlopen


class RuleBasedEventAgent:
    """Deterministic fallback for LLM extraction.

    It mirrors the required LLM JSON contract using curated evidence. The real
    OpenAI-backed agent can be enabled later, but tests and backtests should not
    depend on stochastic remote calls.
    """

    def extract_stock_links(self, event: Dict[str, Any], universe: Iterable[Dict[str, Any]]) -> List[Dict[str, Any]]:
        links = []
        for row in universe:
            evidence = (row.get("evidence") or [{}])[0]
            confidence = self._confidence(row, event)
            links.append(
                {
                    "event_id": event["event_id"],
                    "ticker": row["ticker"],
                    "company": row["name"],
                    "relation_types": row.get("relation_types", []),
                    "evidence": evidence,
                    "confidence": confidence,
                    "risk_flags": self._risk_flags(row, event),
                }
            )
        return links

    @staticmethod
    def _confidence(row: Dict[str, Any], event: Dict[str, Any]) -> float:
        relation_types = row.get("relation_types", [])
        chain_weights = event.get("chain_weights", {})
        default_chain = float(event.get("default_chain_weight", chain_weights.get("default", 0.35)))
        chain_score = max([float(chain_weights.get(item, default_chain)) for item in relation_types] or [default_chain])
        base = float(row.get("base_exposure", 0.0))
        impact = float(event.get("impact_proxy", 1.0))
        domestic_attention = float(event.get("domestic_attention_score", 0.7))
        stock_market_fit = float(event.get("stock_market_fit", event.get("event_fit", 1.0)))
        event_multiplier = (0.55 + 0.45 * chain_score) * (0.65 + 0.35 * domestic_attention) * stock_market_fit
        return min(0.98, max(0.01, base * impact * event_multiplier))

    @staticmethod
    def _risk_flags(row: Dict[str, Any], event: Dict[str, Any]) -> List[str]:
        flags = []
        if row.get("base_exposure", 0) < 0.55:
            flags.append("事件暴露较弱，需限制仓位")
        if float(event.get("stock_market_fit", event.get("event_fit", 1.0))) < 0.55:
            flags.append("赛事关注度和A股标的链条匹配较弱")
        if event["event_id"].endswith("2026_OPEN"):
            flags.append("临近开幕，可能已被部分定价")
        return flags


class OpenAICompatibleAgent:
    """Optional OpenAI-compatible extractor using only the Python standard library.

    The implementation is intentionally isolated. Production code should pass
    its output through EvidenceValidator before any factor or trading module can
    consume it.
    """

    def __init__(self, model: str = "gpt-4.1-mini", base_url: str = "https://api.openai.com/v1/chat/completions"):
        self.model = model
        self.base_url = base_url
        self.api_key = os.getenv("OPENAI_API_KEY")

    def is_enabled(self) -> bool:
        return bool(self.api_key)

    def extract_stock_links(self, event: Dict[str, Any], universe: Iterable[Dict[str, Any]]) -> List[Dict[str, Any]]:
        if not self.api_key:
            raise RuntimeError("OPENAI_API_KEY is not set")
        prompt = (
            "You are a financial information extraction agent. Return strict JSON only. "
            "Use the provided universe and evidence; do not invent sources. "
            f"Event: {json.dumps(event, ensure_ascii=False)}\n"
            f"Universe: {json.dumps(list(universe), ensure_ascii=False)}\n"
            "Return {\"stock_links\":[...]}, each with event_id,ticker,company,relation_types,evidence,confidence,risk_flags."
        )
        body = {
            "model": self.model,
            "messages": [{"role": "user", "content": prompt}],
            "temperature": 0,
            "response_format": {"type": "json_object"},
        }
        req = Request(
            self.base_url,
            data=json.dumps(body).encode("utf-8"),
            headers={"Authorization": f"Bearer {self.api_key}", "Content-Type": "application/json"},
        )
        data = json.loads(urlopen(req, timeout=60).read().decode("utf-8"))
        content = data["choices"][0]["message"]["content"]
        return json.loads(content)["stock_links"]
