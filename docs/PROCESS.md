# 执行流程

这份文档描述从历史训练到本地实时模拟的推荐流程。当前阶段只做研究、回测、离线模拟和本地报告，不启用实盘交易。

## 阶段 1：环境检查

```powershell
cd "<项目解压或克隆目录>"
.\.venv\Scripts\python.exe -m crypto_ai_trader.cli smoke
.\.venv\Scripts\python.exe -m crypto_ai_trader.cli doctor
```

目标：

- 确认本地 Python 环境可用。
- 确认训练、回测、报告生成链路可用。
- 确认 Binance 数据下载通路可用，必要时走 `http://127.0.0.1:7890`。

可选启动 Windows 桌面工具 app：

```powershell
powershell -ExecutionPolicy Bypass -File ".\scripts\start_desktop.ps1"
```

如果当前机器的 Node.js/npm 不可用，继续使用浏览器控制台：

```powershell
powershell -ExecutionPolicy Bypass -File ".\scripts\start_control_panel.ps1"
```

## 阶段 2：历史训练

运行完整周期：

```powershell
.\scripts\run_cycle.ps1
```

或手动运行下载、训练、回测、模拟盘命令。

重点检查：

```text
reports/*_train_metrics.json
reports/*_backtest_summary.json
reports/*_paper_summary.json
reports/cycle_*.json
reports/progress.json
```

回测和模拟盘报告至少检查：

- `total_return`、`annualized_return` 和 `duration_days`；
- `sharpe_like`、`sortino_ratio`、`calmar_ratio`；
- `max_drawdown`、`profit_factor`、`win_rate`；
- `fee_ratio`、`total_cost_drag`、`notional_turnover` 和暴露；
- `performance_by_year`、`performance_by_month`、`performance_by_symbol`。

年月绩效按北京时间聚合。短时间窗口的年化收益和风险调整比率可能被
放大，不能脱离持续时间、交易数、成本和回撤单独用于晋级。

如果大多数币种测试集仍为负收益，优先修改特征、标签、阈值和交易过滤，不要进入测试网。

## 阶段 3：AI 优化

运行 AI 优化：

```powershell
.\.venv\Scripts\python.exe -m crypto_ai_trader.cli ai-optimize --symbols BTCUSDT ETHUSDT SOLUSDT BNBUSDT --interval 1h --trials 80 --max-leverage 3 --min-trades 12 --max-drawdown-limit 0.35
```

模型池包括：

- Numpy Logistic / MLP
- sklearn 模型
- LightGBM
- XGBoost
- CatBoost
- `ensemble_probability_average`

优化逻辑：

- 使用验证集选择模型和参数。
- 使用测试集只评估一次。
- 收益优先，但过滤极端回撤、交易次数过少和 profit factor 过低的配置。

重点检查：

```text
reports/ai_optimization_*.json
reports/*_ai_optimization.json
models/*_ai.pkl
```

## 阶段 3.5：每小时准确率优先优化

当你在线时，可以运行轻量准确率优化，用来提高方向预测质量，而不是直接追求实盘收益：

```powershell
.\.venv\Scripts\python.exe -m crypto_ai_trader.cli model-optimize --symbols ETHUSDT BNBUSDT --intervals 5m 15m --time-budget-minutes 15 --max-model-trials 24 --include-realtime
```

默认候选：

- 币种：`ETHUSDT`、`BNBUSDT`
- 周期：`5m`、`15m`
- 特征：`feature_version = v2`
- 模型：Numpy Logistic / MLP、sklearn、LightGBM、XGBoost、CatBoost、概率平均集成

评分逻辑：

- 主评分：`balanced_accuracy + 0.10 * auc - 0.05 * log_loss`
- AUC 无法计算时按 `0.5` 处理。
- 交易指标只做过滤：交易次数过少、profit factor 过低、回撤过大时降级。
- 验证集用于选择模型，测试集只最终评估一次。

重点检查：

```text
reports/model_optimization_*.json
reports/*_model_optimization.json
models/*_accuracy_ai.pkl
```

当前最近报告的判断：

- `ETHUSDT 1h`：较适合继续模拟盘观察。
- `BNBUSDT 1h`：测试收益强，但验证收益偏弱，需要延长模拟观察。
- `BTCUSDT`：观察。
- `SOLUSDT`：暂不进入候选。

## 阶段 4：实时闭合 K 线训练

手动运行：

```powershell
.\.venv\Scripts\python.exe -m crypto_ai_trader.cli live-train --symbols ETHUSDT --interval 5m --limit 500 --iterations 1
```

推荐先用：

- `5m`
- `15m`
- 候选币种优先 `ETHUSDT`、`BNBUSDT`

不要过早依赖 `1m`，因为噪音和交易成本更敏感。

重点检查：

```text
reports/live_training_*.json
reports/*_live_train_metrics.json
```

## 阶段 5：本地 runner 持续模拟

启动图形控制台：

```powershell
powershell -ExecutionPolicy Bypass -File ".\scripts\start_control_panel.ps1"
```

打开：

```text
http://127.0.0.1:8765
```

桌面 app 和浏览器控制台使用同一套后端 API。桌面工具区可以启动准确率优化、实时训练、便携包导出，并显示最新优化摘要；实盘交易区域保持禁用。

点击 `开始` 后，runner 会持续：

1. 拉取最近已闭合 K 线。
2. 合并并去重本地实时数据。
3. 训练本地 runner 模型。
4. 写入报告。
5. 等待下一轮训练。

控制台里的实时决策图会读取最新 runner 回测明细，并只刷新图表区域：

- 价格线来自 `close`
- 上涨概率线来自 `prob_up`
- 买点/做多来自 `position` 从空仓或空头切换到多头
- 卖点/做空来自 `position` 从空仓或多头切换到空头
- 平仓点来自 `position` 回到 0

重点检查：

```text
reports/runner_live_latest.json
reports/runner_live_{INTERVAL}_*.json
reports/*_runner_live_train_metrics.json
reports/*_runner_live_backtest.csv
state/runner_state.json
```

进入继续模拟的最低标准：

- 连续多轮 runner 报告为正收益。
- 回撤可控。
- 交易次数足够。
- `profit_factor > 1`。
- 当前上涨概率和模型方向不要与历史 AI 优化结果明显冲突。

## 阶段 6：测试网模拟

只有当历史 AI 优化和本地 runner 都稳定后，才考虑 Binance Futures Testnet。

测试网前必须确认：

- 使用 testnet API。
- API Key 没有提现权限。
- 实盘开关仍然关闭。
- 先 dry-run 打印订单，不真实下单。
- 测试网连续稳定后再讨论小资金实盘。

## 暂不建议开启

- 自动实盘
- 自动全仓
- 高杠杆
- 马丁格尔加仓
- 无止损持仓
- 未经过测试网的真实订单

## 每次优化后的决策规则

- 若 4 个币种多数为负收益：改特征和标签。
- 若个别币种历史和实时都稳定：只保留这些币种进入模拟候选池。
- 若收益来自高频换仓但成本过高：提高阈值，加入最低置信度过滤。
- 若验证集差但测试集好：延长模拟观察，不直接升级。
- 若实时 runner 连续负收益：降低频率，优先回到 `5m` 或 `15m`。
