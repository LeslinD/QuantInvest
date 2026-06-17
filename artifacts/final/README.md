# 期末提交归档

本目录保留期末提交需要审阅的报告和关键数据。完整运行时输出由程序生成，不放入提交归档。

## 报告

- `reports/06_第三轮优化方案与测试结果.md`：第三轮主要提升、结果和分析。
- `reports/07_期末最终报告.md`：完整期末报告。

## 关键数据

`key_data/` 保存报告使用的核心表格：

| 文件 | 用途 |
|---|---|
| `third_round_round_comparison.csv` | 中期、第一轮、第二轮、第三轮结果对比 |
| `event_strategy_comparison.csv` | 分事件的中期策略与最终策略回测对比 |
| `benchmark_comparison.csv` | 指数基准对比 |
| `ablation_comparison.csv` | 事件分层和适配规则的模块贡献对比 |
| `second_round_holding_comparison_2026-05-27_to_2026-06-16.csv` | 中期、第一轮、第二轮模拟持仓对比 |
| `third_round_holding_comparison_2026-05-27_to_2026-06-16.csv` | 第三轮模拟持仓对比 |
| `paper_orders_final_2026-05-27.csv` | 第一轮模拟订单 |
| `paper_orders_second_round_2026-05-27.csv` | 第二轮模拟订单 |
| `paper_orders_third_round_2026-05-27.csv` | 第三轮模拟订单 |
| `third_round_event_layer_audit.csv` | 事件分层和交易动作审计 |
| `third_round_company_second_source_audit.csv` | 公司第二来源证据审计 |
| `third_round_attention_evidence_matrix.csv` | 热度数据设计和可用性矩阵 |
| `third_round_decision_audit.csv` | 第三轮持仓决策审计 |
| `third_round_parameter_review.csv` | 事件窗口参数复核 |
| `first_round_evaluation_summary.json` | 第一轮核心指标 |
| `second_round_evaluation_summary.json` | 第二轮核心指标 |
| `third_round_evaluation_summary.json` | 第三轮核心指标 |

`outputs/` 是运行管线时生成的临时目录。需要复现实验时运行 README 中的命令即可生成。
