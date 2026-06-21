# 策略记忆与持续模拟优化

`simulation_memory` 是本地策略记忆层，用来把已有训练、回测、AI 优化、runner 实时模拟和 paper replay 结果沉淀成长期可分析的记录。

## 它会读取什么

- `reports/model_optimization_*.json`
- `reports/*_model_optimization.json`
- `reports/runner_live_*.json`
- `reports/*_runner_live_train_metrics.json`
- `reports/*_live_train_metrics.json`
- `reports/*_ai_optimization.json`
- `reports/*_backtest_summary.json`
- `reports/*_paper_summary.json`

## 它会输出什么

- `state/simulation_memory.json`：长期记忆状态。
- `reports/simulation_memory_latest.json`：最新策略记忆报告。
- `reports/simulation_memory_*.json`：带时间戳的历史记忆快照。

每个币种周期会记录：

- 历史 observation 数量
- 最新模型与最佳模型
- 平均收益、正收益比例、最大回撤、profit factor、交易次数
- 方向预测质量，包括 balanced accuracy、AUC 等
- 是否适合继续模拟盘观察
- 下一步建议，例如继续 paper simulation、提高阈值、降低风险、改特征/标签

## 手动更新

```powershell
.\.venv\Scripts\python.exe -m crypto_ai_trader.cli memory-update
```

桌面控制台也提供 `更新策略记忆` 按钮。

## 持续优化入口

`scheduled-optimize` 会在每轮优化前后自动刷新策略记忆，让新的训练、回测、实时闭合 K 线和模拟盘结果继续沉淀：

```powershell
.\.venv\Scripts\python.exe -m crypto_ai_trader.cli scheduled-optimize --symbols ETHUSDT BNBUSDT --intervals 5m 15m --include-realtime --complexity expanded --rolling-folds 2 --time-budget-minutes 15 --max-model-trials 12 --max-targets 3
```

桌面控制台的 `持续优化一次` 按钮使用同一条链路。

## 与智能体审查的关系

`autonomous-review` 每次运行时会自动刷新策略记忆，并把记忆摘要写入 `reports/autonomous_review_latest.json`。

这个功能只做模拟盘、测试网和优化建议，不启用实盘、不读取 API Key、不下单。
