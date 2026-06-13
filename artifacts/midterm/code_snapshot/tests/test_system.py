import unittest
from pathlib import Path

import pandas as pd

from quant_sports_event.backtest import run_event_backtest, walk_forward_search
from quant_sports_event.costs import CostModel, is_limit_down, is_limit_up, round_lot_shares
from quant_sports_event.factors import compute_symbol_features, estimate_constrained_rankic_weights, estimate_rankic_weights
from quant_sports_event.event_study import event_study, summarize_event_study
from quant_sports_event.portfolio import build_target_weights
from quant_sports_event.validation import EvidenceValidator


def synthetic_frame(symbol="sh600000", start="2024-01-01", periods=160):
    dates = pd.bdate_range(start, periods=periods)
    base = pd.Series(range(periods), dtype=float)
    close = 10 + base * 0.03
    df = pd.DataFrame(
        {
            "date": dates,
            "open": close * 0.998,
            "high": close * 1.01,
            "low": close * 0.99,
            "close": close,
            "volume": 1000000 + base * 1000,
        }
    )
    df["amount"] = df["close"] * df["volume"]
    df["sina_symbol"] = symbol
    return df


class ValidationTests(unittest.TestCase):
    def setUp(self):
        self.universe = [
            {
                "ticker": "600060.SH",
                "sina_symbol": "sh600060",
                "name": "海信视像",
                "industry": "显示设备",
                "board": "main",
            }
        ]

    def test_rejects_future_evidence(self):
        validator = EvidenceValidator(self.universe, "2026-05-26")
        result = validator.validate_links(
            [
                {
                    "ticker": "600060.SH",
                    "company": "海信视像",
                    "confidence": 0.8,
                    "evidence": {
                        "source_url": "https://example.com",
                        "published_at": "2026-05-27",
                        "claim": "future claim",
                    },
                }
            ]
        )
        self.assertEqual(len(result.accepted), 0)
        self.assertIn("time_after_decision_date", result.rejected[0]["reject_reason"])

    def test_accepts_supported_link(self):
        validator = EvidenceValidator(self.universe, "2026-05-26")
        result = validator.validate_links(
            [
                {
                    "ticker": "600060.SH",
                    "company": "海信视像",
                    "confidence": 0.8,
                    "evidence": {
                        "source_url": "https://example.com",
                        "published_at": "2026-05-26",
                        "claim": "supported",
                    },
                }
            ]
        )
        self.assertEqual(len(result.accepted), 1)


class CostTests(unittest.TestCase):
    def test_round_lot(self):
        self.assertEqual(round_lot_shares(10500, 10, 100), 1000)
        self.assertEqual(round_lot_shares(999, 10, 100), 0)

    def test_cost_model(self):
        model = CostModel(
            commission_rate=0.0003,
            min_commission=5,
            stamp_tax_sell_rate=0.0005,
            default_slippage_rate=0.001,
            liquidity_slippage=[{"min_amount": 0, "slippage": 0.001}],
        )
        self.assertAlmostEqual(model.estimate(10000, "buy", 1000000), 15.0)
        self.assertAlmostEqual(model.estimate(10000, "sell", 1000000), 20.0)

    def test_limit_flags(self):
        self.assertTrue(is_limit_up(10, 11, "main"))
        self.assertTrue(is_limit_down(10, 9, "main"))


class FactorAndBacktestTests(unittest.TestCase):
    def test_features_are_finite(self):
        df = synthetic_frame()
        feats = compute_symbol_features(df, "2024-04-30", 0.8, 10, 20)
        self.assertTrue(all(pd.notna(v) for v in feats.values()))

    def test_rankic_weights_normalized(self):
        samples = pd.DataFrame(
            {
                "z_exposure": [1, 2, 3],
                "z_attention": [3, 2, 1],
                "z_momentum": [1, 1, 2],
                "z_liquidity": [2, 2, 2],
                "z_low_volatility": [1, 2, 3],
                "label": [0.1, 0.2, 0.3],
            }
        )
        w = estimate_rankic_weights(samples)
        self.assertAlmostEqual(sum(abs(v) for v in w.values()), 1.0)

    def test_constrained_rankic_weights_are_non_negative(self):
        samples = pd.DataFrame(
            {
                "z_exposure": [3, 2, 1],
                "z_attention": [1, 2, 3],
                "z_momentum": [1, 1, 2],
                "z_liquidity": [2, 3, 4],
                "z_low_volatility": [1, 2, 3],
                "label": [0.1, 0.2, 0.3],
            }
        )
        w = estimate_constrained_rankic_weights(
            samples,
            priors={"exposure": 0.3, "attention": 0.3, "momentum": 0.2, "liquidity": 0.1, "low_volatility": 0.1},
            directions={"exposure": 1, "attention": 1, "momentum": 1, "liquidity": 1, "low_volatility": 1},
            prior_weight=0.5,
        )
        self.assertTrue(all(v >= 0 for v in w.values()))
        self.assertAlmostEqual(sum(w.values()), 1.0)

    def test_portfolio_caps(self):
        scored = pd.DataFrame(
            {
                "ticker": ["a", "b", "c"],
                "sina_symbol": ["a", "b", "c"],
                "name": ["a", "b", "c"],
                "industry": ["x", "x", "y"],
                "alpha_score": [3.0, 2.0, 1.0],
            }
        )
        targets = build_target_weights(scored, 0.4, 0.2, 0.25, 3)
        self.assertLessEqual(targets["target_weight"].sum(), 0.4 + 1e-9)
        self.assertLessEqual(targets.groupby("industry")["target_weight"].sum().max(), 0.25 + 1e-9)

    def test_backtest_runs(self):
        universe = [
            {"ticker": "A.SH", "sina_symbol": "sha", "name": "A", "industry": "x", "board": "main"},
            {"ticker": "B.SH", "sina_symbol": "shb", "name": "B", "industry": "y", "board": "main"},
        ]
        frames = {"sha": synthetic_frame("sha"), "shb": synthetic_frame("shb")}
        events = [
            {
                "event_id": "E1",
                "event_date": "2024-06-10",
                "status": "completed",
            }
        ]
        params = {
            "attention_lookback": 5,
            "momentum_lookback": 10,
            "entry_days_before_event": 30,
            "exit_days_before_event": 5,
            "top_k": 2,
            "stop_loss": 0.08,
        }
        cost = CostModel(0.0003, 5, 0.0005, 0.001, [{"min_amount": 0, "slippage": 0.001}])
        constraints = {
            "max_event_exposure": 0.4,
            "max_position_per_stock": 0.2,
            "max_industry_exposure": 0.25,
            "lot_size": 100,
        }
        result = run_event_backtest(frames, universe, {"A.SH": 0.8, "B.SH": 0.5}, events, params, cost, constraints, 100000)
        self.assertGreaterEqual(result.summary["num_trades"], 1)

    def test_event_study_runs(self):
        universe = [{"ticker": "A.SH", "sina_symbol": "sha", "name": "A", "industry": "x", "board": "main"}]
        stock = synthetic_frame("sha", periods=220)
        bench = synthetic_frame("bench", periods=220)
        car = event_study(
            {"sha": stock},
            bench,
            universe,
            [{"event_id": "E1", "event_date": "2024-08-01"}],
            windows=[(-30, -10)],
        )
        self.assertFalse(car.empty)
        self.assertFalse(summarize_event_study(car).empty)


if __name__ == "__main__":
    unittest.main()
