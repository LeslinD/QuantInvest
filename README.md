# QuantInvest

体育大事件事件驱动量化投资系统。项目围绕世界杯、亚运会、奥运会、全运会等体育事件，构建可解释的 A 股事件驱动策略，并输出回测、模拟持仓、风险和证据审计结果。

## 项目结构

| 路径 | 内容 |
|---|---|
| `quant_sports_event/` | 策略、回测、研究闭环和三轮评估代码 |
| `configs/` | 事件库、股票池、策略参数和证据来源配置 |
| `data/manual/` | 百度指数、微信指数、热榜、论坛等人工数据导入模板 |
| `docs/final/` | 期末研究文档和最终报告 |
| `artifacts/final/` | 可提交的最终报告和关键结果数据 |
| `artifacts/midterm/` | 中期冻结归档 |
| `tests/` | 单元测试 |
| `outputs/` | 运行时生成目录，默认不提交 |

## 环境要求

- Python 3.10+
- Linux/macOS shell
- 网络可用时，系统可以抓取新浪行情；无网络时，可使用已有归档数据阅读报告和关键结果。

本项目只依赖 Python 标准库和 `pandas`、`numpy`。如果本机没有依赖，先安装：

```bash
python3 -m pip install pandas numpy
```

## 快速验证

在项目根目录运行：

```bash
python3 -m unittest tests/test_system.py
```

当前测试覆盖成本、因子、组合约束、事件分层、公司证据、热度矩阵、二轮对冲和三轮持仓决策。

## 运行完整流程

完整流程会生成 `outputs/`，包括行情快照、回测结果、研究诊断、第一轮评估、第二轮评估和第三轮评估。

```bash
python3 -m quant_sports_event.run_final_pipeline \
  --root /workspace/coursework/QuantInvest \
  --hold-end-date 2026-06-16
```

常用单独运行命令：

```bash
python3 -m quant_sports_event.run_pipeline
python3 -m quant_sports_event.run_first_round_evaluation --root /workspace/coursework/QuantInvest
python3 -m quant_sports_event.run_second_round_evaluation --root /workspace/coursework/QuantInvest --hold-end-date 2026-06-16
python3 -m quant_sports_event.third_round --root /workspace/coursework/QuantInvest --hold-end-date 2026-06-16
```

`outputs/` 是生成目录，清理后可以通过上述命令恢复。

## 最终交付物

最终可提交材料位于 `artifacts/final/`：

| 路径 | 内容 |
|---|---|
| `artifacts/final/reports/06_第三轮优化方案与测试结果.md` | 第三轮相对第二轮的提升、结果和分析 |
| `artifacts/final/reports/07_期末最终报告.md` | 完整期末报告 |
| `artifacts/final/key_data/` | 最终报告使用的关键结果表和 summary |

`docs/final/` 保留研究过程文档，`artifacts/final/` 保留提交用的精简结果。

## 数据说明

系统使用三类数据：

| 数据 | 位置 | 说明 |
|---|---|---|
| 行情数据 | 运行时写入 `outputs/data_snapshots/` | 新浪日线，运行时冻结 |
| 人工热度模板 | `data/manual/` | 百度指数、微信指数、财经热榜、论坛讨论等导入模板 |
| 关键结果数据 | `artifacts/final/key_data/` | 支撑最终报告的汇总表、订单表、参数表和审计表 |

热度数据按设计拆为事件热度、股票热度和热度匹配。百度指数、微信指数、财经热榜、论坛讨论需要人工导入或外部数据源；系统不会把缺失热度填成估计值。

## 主要结果

| 指标 | 中期 | 最终系统 |
|---|---:|---:|
| 历史事件账本收益 | 1.30% | 8.50% |
| 完整回测净收益 | - | 6.38% |
| 样本外验证净收益 | - | 3.17% |
| 2026-05-27 至 2026-06-16 模拟持仓 | -5.25% | +0.03% |

最终模拟盘为海信视像 5% 多头和中证500空头代理 2.74%。完整解释见 `artifacts/final/reports/07_期末最终报告.md`。
