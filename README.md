# QuantInvest

体育大事件事件驱动量化投资系统。

## 目录说明

- `quant_sports_event/`：当前主代码，后续提升阶段优先在这里迭代。
- `configs/`：事件、股票池、策略参数等配置。
- `tests/`：系统测试。
- `reports/`：中期报告和 PPT 的生成脚本，不再直接存放中期成品文档。
- `outputs/`：当前运行输出、结果表、图表数据和预览文件。
- `docs/midterm/`：整理后的中期文档入口。
- `artifacts/midterm/`：中期冻结归档，包含文档交付物、结果数据和当前代码快照。
- `reference/`：参考模板和前期资料。

## 下一阶段开发约定

后续功能提升直接修改主工程目录，例如 `quant_sports_event/`、`configs/`、`tests/`、`reports/` 和 `outputs/`。

`artifacts/midterm/` 作为中期状态备份，默认只读使用；需要参考中期内容时从这里读取，不要在后续迭代中覆盖它。
