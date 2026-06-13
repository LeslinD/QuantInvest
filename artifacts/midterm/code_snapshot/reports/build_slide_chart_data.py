from __future__ import annotations

import csv
import json
from collections import Counter, defaultdict
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
RESULTS = ROOT / "outputs" / "results"
OUT = ROOT / "outputs" / "ppt" / "chart_data"

EVENT_LABELS = {
    "FIFA_WC_2022_OPEN": "2022世界杯",
    "UEFA_EURO_2024_OPEN": "2024欧洲杯",
    "PARIS_OLYMPICS_2024_OPEN": "2024巴黎奥运会",
}

WINDOW_LABELS = {
    "T-60_T-10": "赛前60至10日",
    "T-30_T-10": "赛前30至10日",
    "T-10_T+10": "开幕前后10日",
}


def read_csv(name: str) -> list[dict[str, str]]:
    with (RESULTS / name).open(newline="", encoding="utf-8") as f:
        return list(csv.DictReader(f))


def write_csv(name: str, rows: list[dict[str, object]], fieldnames: list[str] | None = None) -> None:
    if not fieldnames and rows:
        fieldnames = list(rows[0].keys())
    elif not fieldnames:
        fieldnames = ["empty"]
    with (OUT / name).open("w", newline="", encoding="utf-8-sig") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)


def pct(x: float) -> float:
    return round(x * 100, 4)


def money(x: float) -> float:
    return round(x, 2)


def main() -> None:
    OUT.mkdir(parents=True, exist_ok=True)

    summary = json.loads((RESULTS / "run_summary_2026-05-26.json").read_text(encoding="utf-8"))
    backtest = read_csv("backtest_trades_2026-05-26.csv")
    walk = read_csv("walk_forward_trades_2026-05-26.csv")
    event_summary = read_csv("event_study_summary_2026-05-26.csv")
    paper_orders = read_csv("paper_orders_2026-05-27.csv")
    paper_scores = read_csv("paper_scores_2026-05-26.csv")
    stop_diag = read_csv("stop_loss_diagnostic_fixed_calendar.csv")
    hyper = read_csv("hyperparam_search_2026-05-26.csv")

    write_csv(
        "chart_01_midterm_requirement_checklist.csv",
        [
            {"中期要求": "至少一轮回测分析", "完成状态": "已完成", "本项目对应内容": "2022世界杯、2024欧洲杯已进入当前交易回测；巴黎奥运会用于事件适配压力测试"},
            {"中期要求": "开始模拟交易测试", "完成状态": "已完成", "本项目对应内容": "使用2026-05-26收盘数据生成2026-05-27模拟盘订单"},
            {"中期要求": "系统或策略基本思想", "完成状态": "已完成", "本项目对应内容": "体育大事件临近时的注意力扩散、产业链暴露和预期兑现"},
            {"中期要求": "系统或策略核心流程", "完成状态": "已完成", "本项目对应内容": "事件库、证据闭环、因子评分、组合约束、回测、模拟盘"},
            {"中期要求": "回测及结果分析", "完成状态": "已完成", "本项目对应内容": "净收益、交易成本、事件研究CAAR、止损压力测试、走步验证"},
            {"中期要求": "模拟交易配置和理由", "完成状态": "已完成", "本项目对应内容": "给出持仓权重、股数、成本、约束和配置原因"},
        ],
    )

    event_rows = []
    for r in event_summary:
        event_rows.append(
            {
                "事件": EVENT_LABELS.get(r["event_id"], r["event_id"]),
                "窗口": WINDOW_LABELS.get(r["window"], r["window"]),
                "CAAR(%)": pct(float(r["caar"])),
                "正收益股票占比(%)": pct(float(r["positive_rate"])),
                "样本数": int(float(r["n"])),
            }
        )
    order = {"赛前60至10日": 0, "赛前30至10日": 1, "开幕前后10日": 2}
    event_rows.sort(key=lambda x: (x["事件"], order.get(str(x["窗口"]), 99)))
    write_csv("chart_02_event_study_caar.csv", event_rows)

    bt = summary["backtest_summary"]
    wf = summary["walk_forward_summary"]
    write_csv(
        "chart_03_backtest_kpi.csv",
        [
            {"指标": "回测净收益率", "数值": pct(bt["net_return"]), "单位": "%", "解读": "扣除佣金、印花税、滑点后的策略收益"},
            {"指标": "交易笔数", "数值": int(bt["num_trades"]), "单位": "笔", "解读": "进入真实交易规则后的成交记录"},
            {"指标": "单笔平均毛收益率", "数值": pct(bt["gross_return_mean"]), "单位": "%", "解读": "未扣成本的平均价格涨幅"},
            {"指标": "胜率", "数值": pct(bt["win_rate"]), "单位": "%", "解读": "当前回测样本较少，不单独作为最终有效性证据"},
            {"指标": "总交易成本", "数值": money(bt["total_cost"]), "单位": "元", "解读": "佣金、印花税和滑点估计"},
            {"指标": "成本/初始资金", "数值": pct(bt["cost_to_initial_cash"]), "单位": "%", "解读": "策略频率低，成本侵蚀可控"},
        ],
    )

    write_csv(
        "chart_04_backtest_trade_detail.csv",
        [
            {
                "事件": EVENT_LABELS.get(r["event_id"], r["event_id"]),
                "股票": r["company"],
                "代码": r["ticker"],
                "买入日": r["buy_date"],
                "卖出日": r["sell_date"],
                "退出原因": "事件窗口退出" if r["exit_reason"] == "event_exit" else r["exit_reason"],
                "目标权重(%)": pct(float(r["target_weight"])),
                "毛收益率(%)": pct(float(r["gross_return"])),
                "净收益贡献(元)": money(float(r["net_pnl"])),
                "净收益贡献(%)": pct(float(r["net_return_on_initial_cash"])),
            }
            for r in backtest
        ],
    )

    by_event: dict[str, list[dict[str, str]]] = defaultdict(list)
    for r in backtest:
        by_event[r["event_id"]].append(r)
    event_contrib = []
    for event_id, rows in by_event.items():
        event_contrib.append(
            {
                "事件": EVENT_LABELS.get(event_id, event_id),
                "交易笔数": len(rows),
                "净收益贡献(元)": money(sum(float(r["net_pnl"]) for r in rows)),
                "净收益贡献(%)": pct(sum(float(r["net_return_on_initial_cash"]) for r in rows)),
                "平均毛收益率(%)": pct(sum(float(r["gross_return"]) for r in rows) / len(rows)),
                "交易成本(元)": money(sum(float(r["buy_cost"]) + float(r["sell_cost"]) for r in rows)),
                "止损笔数": sum(1 for r in rows if r["exit_reason"] == "stop_loss"),
            }
        )
    write_csv("chart_05_backtest_event_contribution.csv", event_contrib)

    write_csv(
        "chart_06_walk_forward_kpi.csv",
        [
            {"指标": "走步验证净收益率", "数值": pct(wf["net_return"]), "单位": "%", "解读": "用前一事件训练权重，后一事件验证"},
            {"指标": "交易笔数", "数值": int(wf["num_trades"]), "单位": "笔", "解读": "2024欧洲杯三个入选标的"},
            {"指标": "单笔平均毛收益率", "数值": pct(wf["gross_return_mean"]), "单位": "%", "解读": "验证样本仍为正，但样本数有限"},
            {"指标": "胜率", "数值": pct(wf["win_rate"]), "单位": "%", "解读": "不能过度外推，结项需继续积累模拟盘"},
            {"指标": "交易成本", "数值": money(wf["total_cost"]), "单位": "元", "解读": "扣除交易成本后仍为正收益"},
        ],
    )

    write_csv(
        "chart_07_walk_forward_trade_detail.csv",
        [
            {
                "事件": EVENT_LABELS.get(r["event_id"], r["event_id"]),
                "股票": r["company"],
                "代码": r["ticker"],
                "目标权重(%)": pct(float(r["target_weight"])),
                "毛收益率(%)": pct(float(r["gross_return"])),
                "净收益贡献(元)": money(float(r["net_pnl"])),
                "训练事件数": int(float(r["validation_train_events"])),
            }
            for r in walk
        ],
    )

    diag_by_event: dict[str, list[dict[str, str]]] = defaultdict(list)
    for r in stop_diag:
        diag_by_event[r["event_id"]].append(r)
    diag_rows = []
    for event_id, rows in diag_by_event.items():
        c = Counter(r["exit_reason"] for r in rows)
        diag_rows.append(
            {
                "压力测试事件": EVENT_LABELS.get(event_id, event_id),
                "交易笔数": len(rows),
                "止损笔数": c["stop_loss"],
                "事件退出笔数": c["event_exit"],
                "净收益贡献(%)": pct(sum(float(r["net_return_on_initial_cash"]) for r in rows)),
                "净收益贡献(元)": money(sum(float(r["net_pnl"]) for r in rows)),
                "主要解释": "入场过早" if event_id == "FIFA_WC_2022_OPEN" else ("事件适配度不足" if event_id == "PARIS_OLYMPICS_2024_OPEN" else "赛前窗口有效"),
            }
        )
    write_csv("chart_08_stop_loss_diagnostic.csv", diag_rows)

    write_csv(
        "chart_09_strategy_risk_comparison.csv",
        [
            {
                "框架": "固定日历压力测试",
                "交易笔数": len(stop_diag),
                "净收益率(%)": pct(sum(float(r["net_return_on_initial_cash"]) for r in stop_diag)),
                "止损笔数": sum(1 for r in stop_diag if r["exit_reason"] == "stop_loss"),
                "止损率(%)": pct(sum(1 for r in stop_diag if r["exit_reason"] == "stop_loss") / len(stop_diag)),
                "交易成本(元)": money(sum(float(r["buy_cost"]) + float(r["sell_cost"]) for r in stop_diag)),
            },
            {
                "框架": "当前中期系统",
                "交易笔数": len(backtest),
                "净收益率(%)": pct(bt["net_return"]),
                "止损笔数": sum(1 for r in backtest if r["exit_reason"] == "stop_loss"),
                "止损率(%)": pct(sum(1 for r in backtest if r["exit_reason"] == "stop_loss") / len(backtest)),
                "交易成本(元)": money(bt["total_cost"]),
            },
        ],
    )

    top_hyper = sorted(hyper, key=lambda r: float(r["selection_score"]), reverse=True)[:10]
    write_csv(
        "chart_10_hyperparameter_top10.csv",
        [
            {
                "排名": i + 1,
                "综合评分": round(float(r["selection_score"]), 6),
                "验证净收益率(%)": pct(float(r["validation_net_return"])),
                "全样本净收益率(%)": pct(float(r["full_net_return"])),
                "全样本止损率(%)": pct(float(r["full_stop_loss_rate"])),
                "入场日": f"T-{r['entry_days_before_event']}",
                "退出日": f"T-{r['exit_days_before_event']}",
                "TopK": r["top_k"],
                "止损线": pct(float(r["stop_loss"])),
                "波动止损倍数": r["stop_vol_multiplier"],
                "暴露门槛": r["min_exposure_for_trade"],
                "注意力截尾": r["attention_z_cap"],
            }
            for i, r in enumerate(top_hyper)
        ],
    )

    sens_rows = []
    for param in ["entry_days_before_event", "top_k", "stop_loss", "stop_vol_multiplier", "min_exposure_for_trade", "attention_z_cap"]:
        groups: dict[str, list[dict[str, str]]] = defaultdict(list)
        for r in hyper:
            groups[r[param]].append(r)
        for value, rows in sorted(groups.items(), key=lambda kv: kv[0]):
            sens_rows.append(
                {
                    "参数": param,
                    "取值": value,
                    "组合数量": len(rows),
                    "平均验证净收益率(%)": pct(sum(float(r["validation_net_return"]) for r in rows) / len(rows)),
                    "平均全样本净收益率(%)": pct(sum(float(r["full_net_return"]) for r in rows) / len(rows)),
                    "平均止损率(%)": pct(sum(float(r["full_stop_loss_rate"]) for r in rows) / len(rows)),
                }
            )
    write_csv("chart_11_hyperparameter_sensitivity.csv", sens_rows)

    factor_weights = summary["factor_weights"]
    write_csv(
        "chart_12_factor_weights.csv",
        [
            {"因子": "低波动", "系统字段": "low_volatility", "权重(%)": pct(factor_weights["low_volatility"]), "金融含义": "控制事件交易中不必要的下行波动"},
            {"因子": "注意力", "系统字段": "attention", "权重(%)": pct(factor_weights["attention"]), "金融含义": "成交额冲击代表关注度升温"},
            {"因子": "事件暴露", "系统字段": "exposure", "权重(%)": pct(factor_weights["exposure"]), "金融含义": "公司业务与体育事件产业链的相关性"},
            {"因子": "动量", "系统字段": "momentum", "权重(%)": pct(factor_weights["momentum"]), "金融含义": "捕捉反应不足后的趋势延续"},
            {"因子": "流动性", "系统字段": "liquidity", "权重(%)": pct(factor_weights["liquidity"]), "金融含义": "保证组合能在真实市场成交"},
        ],
    )

    write_csv(
        "chart_13_paper_orders.csv",
        [
            {
                "决策日": r["decision_date"],
                "模拟订单日": r["order_date"],
                "股票": r["company"],
                "代码": r["ticker"],
                "行业": r["industry"],
                "Alpha评分": round(float(r["alpha_score"]), 4),
                "目标权重(%)": pct(float(r["target_weight"])),
                "参考收盘价": float(r["reference_close"]),
                "目标股数": int(float(r["target_shares"])),
                "预估成交额(元)": money(float(r["estimated_trade_value"])),
                "预估成本(元)": money(float(r["expected_cost"])),
                "20日均成交额(元)": money(float(r["avg_amount_20d"])),
            }
            for r in paper_orders
        ],
    )

    score_rows = []
    for r in sorted(paper_scores, key=lambda x: float(x["alpha_score"]), reverse=True):
        score_rows.append(
            {
                "股票": r["name"],
                "代码": r["ticker"],
                "行业": r["industry"],
                "Alpha评分": round(float(r["alpha_score"]), 4),
                "事件暴露贡献": round(float(r["contrib_exposure"]), 4),
                "注意力贡献": round(float(r["contrib_attention"]), 4),
                "动量贡献": round(float(r["contrib_momentum"]), 4),
                "流动性贡献": round(float(r["contrib_liquidity"]), 4),
                "低波动贡献": round(float(r["contrib_low_volatility"]), 4),
                "是否进入模拟盘": "是" if any(o["ticker"] == r["ticker"] for o in paper_orders) else "否",
            }
        )
    write_csv("chart_14_paper_scores_factor_contribution.csv", score_rows)

    write_csv(
        "chart_15_paper_order_summary.csv",
        [
            {"指标": "模拟盘决策日", "数值": summary["as_of_date"], "单位": "", "说明": "只使用当日收盘前可获得的数据"},
            {"指标": "模拟盘订单日", "数值": "2026-05-27", "单位": "", "说明": "用于开盘后模拟跟踪，不纳入回测调参"},
            {"指标": "策略目标仓位", "数值": pct(summary["paper_total_target_weight"]), "单位": "%", "说明": "当前系统输出的真实配置比例"},
            {"指标": "预估成交额", "数值": money(summary["paper_total_estimated_trade_value"]), "单位": "元", "说明": "按100股整数手后的订单金额"},
            {"指标": "预估交易成本", "数值": money(summary["paper_total_expected_cost"]), "单位": "元", "说明": "含佣金、印花税、滑点估计"},
        ],
    )

    write_csv(
        "chart_16_validation_metrics.csv",
        [
            {"指标": "候选关系数", "数值": int(summary["validation"]["total"]), "单位": "条", "说明": "事件-股票关系候选"},
            {"指标": "通过数", "数值": int(summary["validation"]["accepted"]), "单位": "条", "说明": "通过Schema、来源、时间、实体校验"},
            {"指标": "拒绝数", "数值": int(summary["validation"]["rejected"]), "单位": "条", "说明": "本次规则Agent未触发拒绝"},
            {"指标": "通过率", "数值": pct(summary["validation"]["accept_rate"]), "单位": "%", "说明": "中期样本偏小，结项需扩充证据集"},
            {"指标": "幻觉代理率", "数值": pct(summary["validation"]["hallucination_proxy_rate"]), "单位": "%", "说明": "无来源或实体不匹配视为幻觉代理"},
        ],
    )

    write_csv(
        "chart_17_market_constraints.csv",
        [
            {"约束": "交易方向", "设置": "仅做多", "理由": "A股普通股票模拟盘不使用融券做空"},
            {"约束": "交易单位", "设置": "100股整数手", "理由": "订单需符合A股最小交易单位"},
            {"约束": "单股上限", "设置": "12%", "理由": "降低单一概念股回撤风险"},
            {"约束": "行业上限", "设置": "25%", "理由": "防止组合集中在同一体育产业链环节"},
            {"约束": "交易成本", "设置": "佣金0.03%、最低5元；卖出印花税0.05%；滑点0.05%-0.20%", "理由": "用真实交易摩擦修正收益"},
            {"约束": "数据时点", "设置": "2026-05-26收盘快照", "理由": "模拟盘配置不使用2026-05-27之后信息"},
        ],
    )

    index_rows = [
        {"文件": p.name, "用途": purpose}
        for p, purpose in [
            (OUT / "chart_01_midterm_requirement_checklist.csv", "中期要求对照表"),
            (OUT / "chart_02_event_study_caar.csv", "事件研究CAAR柱状图/热力表"),
            (OUT / "chart_03_backtest_kpi.csv", "回测绩效总览KPI"),
            (OUT / "chart_04_backtest_trade_detail.csv", "分标的交易明细表"),
            (OUT / "chart_05_backtest_event_contribution.csv", "分事件收益贡献图"),
            (OUT / "chart_06_walk_forward_kpi.csv", "走步验证KPI"),
            (OUT / "chart_07_walk_forward_trade_detail.csv", "走步验证交易表"),
            (OUT / "chart_08_stop_loss_diagnostic.csv", "止损触发原因诊断"),
            (OUT / "chart_09_strategy_risk_comparison.csv", "固定日历压力测试与当前系统对比"),
            (OUT / "chart_10_hyperparameter_top10.csv", "超参数Top10"),
            (OUT / "chart_11_hyperparameter_sensitivity.csv", "超参数敏感性汇总"),
            (OUT / "chart_12_factor_weights.csv", "因子权重图"),
            (OUT / "chart_13_paper_orders.csv", "模拟盘订单表"),
            (OUT / "chart_14_paper_scores_factor_contribution.csv", "模拟盘评分贡献拆解"),
            (OUT / "chart_15_paper_order_summary.csv", "模拟盘配置总览"),
            (OUT / "chart_16_validation_metrics.csv", "LLM/Agent证据闭环校验结果"),
            (OUT / "chart_17_market_constraints.csv", "市场约束与成本假设"),
        ]
    ]
    write_csv("chart_00_index.csv", index_rows)

    readme = OUT / "README.md"
    readme.write_text(
        "# 中期汇报图表数据\n\n"
        "这些 CSV 均由 `outputs/results/` 下 2026-05-26 的真实运行结果派生，"
        "可直接导入 PPT、Excel 或 Keynote 生成图表。百分比列已乘以 100，金额单位为人民币元。\n",
        encoding="utf-8",
    )


if __name__ == "__main__":
    main()
