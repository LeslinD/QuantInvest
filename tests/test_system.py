import unittest
import json
import tempfile
from pathlib import Path

import pandas as pd

from quant_sports_event.audit import (
    attention_data_audit,
    event_library_audit,
    universe_evidence_audit,
)
from quant_sports_event.backtest import benchmark_comparison, exposure_for_event, params_for_event, run_event_backtest, walk_forward_search
from quant_sports_event.costs import CostModel, is_limit_down, is_limit_up, round_lot_shares
from quant_sports_event.factors import compute_symbol_features, estimate_constrained_rankic_weights, estimate_rankic_weights
from quant_sports_event.event_study import event_study, summarize_event_study
from quant_sports_event.evaluation import static_portfolio_return
from quant_sports_event.llm_agent import RuleBasedEventAgent
from quant_sports_event.portfolio import build_target_weights
from quant_sports_event.research_loop import run_research_loop
from quant_sports_event.super_factors import build_super_factor_panel, validate_super_factor_weights
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

    def test_rule_based_agent_uses_event_specific_chain_weights(self):
        universe = [
            {
                "ticker": "600060.SH",
                "sina_symbol": "sh600060",
                "name": "海信视像",
                "industry": "显示设备",
                "board": "main",
                "base_exposure": 0.86,
                "relation_types": ["official_fifa_sponsor", "display_device"],
                "evidence": [{"source_url": "https://example.com", "published_at": "2026-05-26", "claim": "supported"}],
            }
        ]
        agent = RuleBasedEventAgent()
        fifa = {
            "event_id": "FIFA_WC_2026_OPEN",
            "impact_proxy": 1.0,
            "domestic_attention_score": 0.95,
            "stock_market_fit": 0.95,
            "chain_weights": {"official_fifa_sponsor": 1.0, "display_device": 0.95},
        }
        olympics = {
            "event_id": "PARIS_OLYMPICS_2024_OPEN",
            "impact_proxy": 0.9,
            "domestic_attention_score": 0.95,
            "stock_market_fit": 0.4,
            "chain_weights": {"official_fifa_sponsor": 0.1, "display_device": 0.35},
        }
        fifa_confidence = agent.extract_stock_links(fifa, universe)[0]["confidence"]
        olympics_confidence = agent.extract_stock_links(olympics, universe)[0]["confidence"]
        self.assertGreater(fifa_confidence, olympics_confidence)
        self.assertLess(olympics_confidence, 0.35)


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

    def test_event_specific_exposure_lookup(self):
        exposures = {"E1": {"A.SH": 0.8}, "E2": {"A.SH": 0.2}}
        self.assertEqual(exposure_for_event(exposures, "E1")["A.SH"], 0.8)
        self.assertEqual(exposure_for_event({"A.SH": 0.5}, "E1")["A.SH"], 0.5)

    def test_event_layer_params_override_base_params(self):
        params = {"entry_days_before_event": 90, "exit_days_before_event": 10, "top_k": 5}
        event = {"event_id": "E1", "sport": "multi_sport_global"}
        model = {"event_layer_rules": {"multi_sport_global": {"action": "avoid", "reason": "low fit"}}}
        out = params_for_event(params, event, model)
        self.assertEqual(out["action"], "avoid")
        self.assertEqual(out["event_layer_reason"], "low fit")

    def test_benchmark_comparison_runs(self):
        trades = pd.DataFrame(
            [
                {
                    "buy_date": "2024-01-02",
                    "sell_date": "2024-01-05",
                    "target_weight": 0.1,
                    "net_return_on_initial_cash": 0.02,
                }
            ]
        )
        bench = synthetic_frame("bench", start="2024-01-01", periods=10)
        report = benchmark_comparison(trades, {"bench": bench}, benchmark_names={"bench": "测试基准"})
        self.assertEqual(report.loc[0, "benchmark_name"], "测试基准")
        self.assertIn("excess_vs_benchmark_event_book", report)

    def test_static_portfolio_return_runs(self):
        frame = synthetic_frame("sha", start="2024-01-01", periods=20)
        universe = {"A.SH": {"ticker": "A.SH", "sina_symbol": "sha", "name": "A", "industry": "x"}}
        ret, details = static_portfolio_return(
            {"sha": frame},
            universe,
            {"A.SH": 0.1},
            start_date="2024-01-02",
            end_date="2024-01-20",
            initial_cash=100000,
            lot_size=100,
        )
        self.assertFalse(details.empty)
        self.assertIn("pnl", details)
        self.assertTrue(pd.notna(ret))

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


class ResearchLoopTests(unittest.TestCase):
    def test_integrated_factor_weight_bounds(self):
        validate_super_factor_weights()
        with self.assertRaises(ValueError):
            validate_super_factor_weights(
                {
                    "event_opportunity": 0.10,
                    "stock_linkage": 0.40,
                    "attention_confirmation": 0.20,
                    "market_timing": 0.15,
                    "tradable_risk_quality": 0.15,
                }
            )

    def test_integrated_factor_panel_scores(self):
        scores = pd.DataFrame(
            [
                {
                    "ticker": "A.SH",
                    "name": "A",
                    "industry": "x",
                    "z_exposure": 1.0,
                    "z_attention": 1.0,
                    "z_momentum": 0.5,
                    "z_liquidity": 0.2,
                    "z_low_volatility": 0.4,
                },
                {
                    "ticker": "B.SH",
                    "name": "B",
                    "industry": "y",
                    "z_exposure": -1.0,
                    "z_attention": -1.0,
                    "z_momentum": -0.5,
                    "z_liquidity": -0.2,
                    "z_low_volatility": -0.4,
                },
            ]
        )
        panel = build_super_factor_panel(scores, eligible_tickers=["A.SH"])
        self.assertIn("integrated_score", panel)
        self.assertTrue(panel.loc[panel["ticker"] == "A.SH", "eligible_for_loop_001"].iloc[0])
        self.assertFalse(panel.loc[panel["ticker"] == "B.SH", "eligible_for_loop_001"].iloc[0])

    def test_research_loop_writes_diagnostics(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            results = root / "outputs" / "results"
            results.mkdir(parents=True)
            summary = {
                "as_of_date": "2026-05-26",
                "run_hash": "abc",
                "factor_weights": {
                    "exposure": 0.16,
                    "attention": 0.25,
                    "momentum": 0.10,
                    "liquidity": 0.05,
                    "low_volatility": 0.44,
                },
                "backtest_summary": {
                    "net_return": 0.11,
                    "cost_to_initial_cash": 0.0025,
                    "num_trades": 6,
                },
                "walk_forward_summary": {
                    "net_return": 0.02,
                    "cost_to_initial_cash": 0.001,
                    "num_trades": 3,
                },
                "output_files": {
                    "paper_orders": "/stale/path/paper_orders_2026-05-27.csv",
                    "paper_scores": "/stale/path/paper_scores_2026-05-26.csv",
                    "event_study_summary": "/stale/path/event_study_summary_2026-05-26.csv",
                },
            }
            (results / "run_summary_2026-05-26.json").write_text(json.dumps(summary), encoding="utf-8")
            pd.DataFrame(
                [
                    {
                        "decision_date": "2026-05-26",
                        "order_date": "2026-05-27",
                        "ticker": "600060.SH",
                        "sina_symbol": "sh600060",
                        "company": "海信视像",
                        "industry": "显示设备",
                        "target_weight": 0.12,
                        "reference_close": 25.0,
                        "estimated_trade_value": 120000.0,
                        "expected_cost": 150.0,
                        "board": "main",
                    },
                    {
                        "decision_date": "2026-05-26",
                        "order_date": "2026-05-27",
                        "ticker": "605099.SH",
                        "sina_symbol": "sh605099",
                        "company": "共创草坪",
                        "industry": "体育设施",
                        "target_weight": 0.03,
                        "reference_close": 40.0,
                        "estimated_trade_value": 30000.0,
                        "expected_cost": 45.0,
                        "board": "main",
                    },
                ]
            ).to_csv(results / "paper_orders_2026-05-27.csv", index=False)
            pd.DataFrame(
                [
                    {
                        "ticker": "600060.SH",
                        "name": "海信视像",
                        "industry": "显示设备",
                        "alpha_score": 1.2,
                        "contrib_exposure": 0.2,
                        "contrib_attention": 0.3,
                    },
                    {
                        "ticker": "605099.SH",
                        "name": "共创草坪",
                        "industry": "体育设施",
                        "alpha_score": 0.2,
                        "contrib_exposure": 0.1,
                        "contrib_attention": -0.1,
                    },
                ]
            ).to_csv(results / "paper_scores_2026-05-26.csv", index=False)
            pd.DataFrame(
                [
                    {"event_id": "PARIS_OLYMPICS_2024_OPEN", "window": "T-60_T-10", "caar": -0.05, "positive_rate": 0.3, "n": 10},
                    {"event_id": "FIFA_WC_2022_OPEN", "window": "T-60_T-10", "caar": 0.20, "positive_rate": 0.9, "n": 10},
                ]
            ).to_csv(results / "event_study_summary_2026-05-26.csv", index=False)

            loop = run_research_loop(root, as_of="2026-05-26")
            final = root / "outputs" / "final"
            self.assertTrue((final / "experiment_registry.csv").exists())
            self.assertTrue((final / "iteration_log.csv").exists())
            self.assertTrue((final / "diagnostic_report.md").exists())
            self.assertTrue((final / "holding_attribution.csv").exists())
            self.assertTrue((final / "event_diagnostics.csv").exists())
            self.assertTrue((final / "factor_panel_final.csv").exists())
            self.assertTrue((final / "paper_orders_final_2026-05-27.csv").exists())
            self.assertTrue((final / "risk_report.csv").exists())
            final_orders = pd.read_csv(final / "paper_orders_final_2026-05-27.csv")
            self.assertLessEqual(final_orders["final_target_weight"].max(), 0.10 + 1e-9)
            tags = {finding["tag"] for finding in loop["findings"]}
            self.assertIn("small_sample_warning", tags)
            self.assertIn("overfit_gap", tags)
            self.assertIn("event_misfit", tags)
            self.assertEqual(loop["decision"], "accept_first_round_risk_overlay")


class AuditTests(unittest.TestCase):
    def test_event_and_universe_audits_run(self):
        events = [
            {
                "event_id": "E1",
                "sport": "football",
                "event_date": "2024-01-01",
                "status": "completed",
                "certainty": 1,
                "domestic_attention_score": 0.9,
                "stock_market_fit": 0.8,
                "known_at": "2023-01-01",
                "source_url": "https://example.com",
                "chain_weights": {"display_device": 1.0},
            },
            {
                "event_id": "E2",
                "sport": "multi_sport_global",
                "event_date": "2024-02-01",
                "status": "completed",
                "certainty": 1,
                "domestic_attention_score": 0.9,
                "stock_market_fit": 0.4,
                "known_at": "2023-01-01",
                "source_url": "https://example.com",
                "chain_weights": {"media": 0.4},
            },
        ]
        model = {"event_layer_rules": {"multi_sport_global": {"action": "avoid", "reason": "low fit"}}}
        event_audit = event_library_audit(events, model)
        self.assertEqual(event_audit.loc[event_audit["event_id"] == "E1", "audit_status"].iloc[0], "pass")
        self.assertEqual(event_audit.loc[event_audit["event_id"] == "E2", "layer_action"].iloc[0], "avoid")

        universe = [
            {
                "ticker": "600060.SH",
                "name": "海信视像",
                "industry": "显示设备",
                "base_exposure": 0.8,
                "relation_types": ["official_fifa_sponsor", "display_device"],
                "evidence": [{"source_url": "https://example.com"}],
            }
        ]
        universe_audit = universe_evidence_audit(universe)
        attention_audit = attention_data_audit(events, universe)
        self.assertIn("chain_level", universe_audit)
        self.assertIn("pending_manual_import", set(attention_audit["status"]))


if __name__ == "__main__":
    unittest.main()
