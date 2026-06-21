# Crypto AI Trader

当前架构、关键接口、训练/回测/模拟盘命令见
[`docs/SYSTEM_GUIDE.md`](docs/SYSTEM_GUIDE.md)；原始优化文档的逐条验收证据见
[`docs/REQUIREMENTS_TRACEABILITY.md`](docs/REQUIREMENTS_TRACEABILITY.md)。

本项目用于 Binance 加密货币历史训练、AI 模型优化、回测、离线模拟盘和本地实时闭合 K 线训练。

重要提醒：本项目不保证盈利。合约和杠杆风险很高，当前阶段默认只做历史训练、回测、模拟盘和本地报告。`live_trading_enabled` 必须保持 `false`，不接入实盘 API Key，不自动下实盘订单。

## 当前能力

- 下载 Binance USDT-M Futures 公开历史 K 线。
- 生成技术特征和涨跌标签。
- 训练模型池：
  - Numpy Logistic Regression
  - Numpy MLP
  - sklearn MLP / LogisticRegression / RandomForest / ExtraTrees / HistGradientBoosting
  - 可选 LightGBM / XGBoost / CatBoost
  - 概率平均集成模型 `ensemble_probability_average`
- 使用验证集选择模型、阈值、杠杆、止损、止盈和仓位比例。
- 使用测试集只做最终评估，减少测试集泄漏。
- 同步实时已闭合 K 线，继续本地训练 runner。
- 提供本地网页控制台，用按钮开始、暂停、恢复、停止 runner。
- 生成多币种组合规划快照，按波动率、置信度、流动性、风险额度和
  相关性簇约束目标多空权重；该快照只用于模拟研究，不下单。
- 正式配置使用闭合收益 EWMA 日化波动率缩放仓位，默认目标为 `3%`；
- 策略校准风险档案位于 `config/strategy_calibration_profiles.toml`，支持固定值、基础配置复制以及动态上下限，并在报告中记录 catalog 版本；
  回测报告记录周期、日化波动率、缩放比例和实际风险口径。
- 高正资金费否决拥挤多头，高负资金费否决拥挤空头；默认阈值为
  每 8 小时 `0.05%`，反方向仓位不受该门禁影响。
- 历史回测和单币种模拟盘共用统一绩效评估器，输出总收益、年化收益、
  Sharpe、Sortino、Calmar、最大回撤、费用率、换手率、暴露及按北京时间
  划分的年/月/币种绩效。短样本年化值必须结合 `duration_days` 判断。
- 模型优化报告分别输出 Alpha 预测质量排名和策略门槛排名；验证集按时间
  再拆分为校准段与门槛段，测试集只在模型和策略参数冻结后评估。

## 快速开始

进入项目目录：

```powershell
cd "<项目解压或克隆目录>"
```

运行本地检查：

```powershell
.\.venv\Scripts\python.exe -m crypto_ai_trader.cli smoke
.\.venv\Scripts\python.exe -m crypto_ai_trader.cli doctor
```

Refresh public Binance funding history and symbol order filters for offline
backtests:

```powershell
.\.venv\Scripts\python.exe -m crypto_ai_trader.cli sync-market-context --symbols BTCUSDT ETHUSDT SOLUSDT BNBUSDT --start 2024-01 --end 2025-12
```

This command only downloads public market data. It does not use API keys or
place orders. Cached files are stored under `data/market_context/`.
The cache includes funding settlements, minimum/maximum market quantity,
quantity step, minimum notional, and price tick. Missing K-line recovery bars
are blocked by the execution layer rather than converted into fake fills.

启动图形控制台：

```powershell
powershell -ExecutionPolicy Bypass -File ".\scripts\start_control_panel.ps1"
```

启动 Windows 桌面工具 app：

```powershell
powershell -ExecutionPolicy Bypass -File ".\scripts\start_desktop.ps1"
```

桌面 app 使用 Electron 外壳承载本地控制台。当前机器如果提示 Node.js/npm 不可用或 `node.exe` 被拒绝访问，需要先修复或重装 Node.js，也可以把 portable Node 解压到 `tools\node`；不影响 Python 训练链路继续通过浏览器控制台运行。

浏览器会自动打开控制台。默认优先使用：

```text
http://127.0.0.1:8765
```

如果 8765 已经被旧控制台占用，脚本会自动改用 8766 到 8775 之间的空闲端口，并在 PowerShell 里打印实际地址。

在页面里选择币种、周期、K 线数量、训练间隔，然后点击 `开始`。页面会显示 runner 状态、训练进度、候选模型、收益和回撤。
桌面工具区还提供一键准确率优化、一键实时训练、查看最新优化报告、导出便携包和打开报告目录。

控制台的“实时决策图”会每 5 秒单独刷新图表区域，不会刷新整个页面。图上会标注：

- 价格线
- 上涨概率线
- 买点/做多
- 卖点/做空
- 平仓点

这些信号来自本地模型和回测明细，只是模拟决策展示，不是实盘下单。

## 本地 runner

runner 用于在本地持续同步 Binance 已闭合 K 线，并继续训练 `_runner_live` 模型。

常用命令：

```powershell
powershell -ExecutionPolicy Bypass -File ".\scripts\status_runner.ps1"
powershell -ExecutionPolicy Bypass -File ".\scripts\pause_runner.ps1"
powershell -ExecutionPolicy Bypass -File ".\scripts\resume_runner.ps1"
powershell -ExecutionPolicy Bypass -File ".\scripts\stop_runner.ps1"
```

注意：图形控制台只是控制面板，不会自动替你点击开始。要让 runner 自动运行，需要点击页面的 `开始`，或安装 Windows 启动项。

安装 Windows 任务和登录启动快捷方式：

```powershell
powershell -ExecutionPolicy Bypass -File ".\scripts\install_runner_task.ps1"
powershell -ExecutionPolicy Bypass -File ".\scripts\start_runner_task.ps1"
```

如果提示脚本不存在，说明当前 PowerShell 不在项目目录。先运行：

```powershell
cd "<项目解压或克隆目录>"
```

## 历史训练和 AI 优化

下载 24 个月、4 个主流合约币种历史数据并跑完整训练周期：

```powershell
.\scripts\run_cycle.ps1
```

运行收益优先的 AI 优化：

```powershell
.\.venv\Scripts\python.exe -m crypto_ai_trader.cli ai-optimize --symbols BTCUSDT ETHUSDT SOLUSDT BNBUSDT --interval 1h --trials 80 --max-leverage 3 --min-trades 12 --max-drawdown-limit 0.35
```

报告会写入：

```text
reports/ai_optimization_*.json
reports/{SYMBOL}_{INTERVAL}_ai_optimization.json
models/{SYMBOL}_{INTERVAL}_ai.pkl
```

运行准确率优先的轻量模型优化，适合每小时在线迭代：

```powershell
.\.venv\Scripts\python.exe -m crypto_ai_trader.cli model-optimize --symbols ETHUSDT BNBUSDT --intervals 5m 15m --time-budget-minutes 15 --max-model-trials 12 --include-realtime --complexity expanded --rolling-folds 2
```

这条链路会使用增强特征 `v3_small_account`，按验证集的 balanced accuracy、AUC 和 log loss 选择模型，并用回测收益、回撤、交易次数和 profit factor 做过滤。默认定时优化使用 `standard`，避免每小时任务过重；`expanded`/`deep`/`blackbox` 只适合手动长跑。Transformer 仍属于神经网络，但只在 `blackbox` 档位作为更慢的候选。测试集只在最佳模型确定后评估一次，不参与模型或阈值选择，但发布正式候选仍必须通过测试集硬风控。

报告和模型会写入：

```text
reports/model_optimization_*.json
reports/{SYMBOL}_{INTERVAL}_model_optimization.json
reports/{SYMBOL}_{INTERVAL}_model_optimization_backtest.csv
models/{SYMBOL}_{INTERVAL}_accuracy_ai.pkl
```

每次 `live-train`/runner 训练还会生成：

```text
reports/{SYMBOL}_{INTERVAL}_monitoring.json
```

监控包含特征 PSI/KS、概率校准、置信度、滚动收益/回撤、近期回放与冻结测试偏差以及状态分布变化。触发条件只会让下一次受时间预算限制的 `scheduled-optimize` 优先处理该目标，不会自动发布模型或启用实盘。

运行 walk-forward 策略验证门禁，适合在更新本地策略前确认是否过拟合：

```powershell
.\.venv\Scripts\python.exe -m crypto_ai_trader.cli strategy-validate --symbols BTCUSDT ETHUSDT SOLUSDT BNBUSDT --intervals 1h 5m --include-realtime --folds 3 --time-budget-minutes 55 --max-model-trials 1 --min-profit-factor 1.0
```

验证报告会写入：

```text
reports/strategy_validation_latest.json
reports/{SYMBOL}_{INTERVAL}_strategy_validation.json
state/local_strategy_state.json
```

只有通过多窗口外推验证的目标才会进入继续模拟候选。

## 便携迁移

在控制台或桌面 app 点击 `导出便携包` 会在 `exports/` 生成一个 zip。默认包含代码、文档、脚本、Web 控制台、`config/` 策略档案、配置模板、`data/`、`models/`、`reports/` 和 `state/`，并排除 `.venv/`、`__pycache__/`、`.cache/`、`downloads/`、`logs/`、`node_modules/` 和旧计划任务文件。

在新机器或新目录解压后运行：

```powershell
powershell -ExecutionPolicy Bypass -File ".\scripts\restore_portable.ps1"
```

恢复脚本会重建 `.venv`、安装依赖并运行 `smoke` 和 `doctor`。计划任务和开机启动必须在新机器重新安装，不要复制旧机器的任务。

## 实时闭合 K 线训练

手动跑一次实时训练：

```powershell
.\.venv\Scripts\python.exe -m crypto_ai_trader.cli live-train --symbols ETHUSDT --interval 5m --limit 500 --iterations 1
```

runner 的持续训练报告：

```text
reports/runner_live_latest.json
reports/runner_live_{INTERVAL}_*.json
reports/*_runner_live_train_metrics.json
reports/*_runner_live_backtest.csv
reports/portfolio_snapshot_latest.json
reports/portfolio_paper_latest.json
reports/portfolio_paper_{INTERVAL}.jsonl
state/portfolio_paper_{INTERVAL}.json
reports/shadow_portfolio_snapshot_latest.json
reports/shadow_portfolio_paper_latest.json
reports/shadow_portfolio_paper_{INTERVAL}.jsonl
state/shadow_portfolio_paper_{INTERVAL}.json
```

`portfolio_snapshot_latest.json` 是组合构建层的只读规划结果。控制台每
10 秒只刷新组合区域，显示总敞口、净敞口、相关簇和逐币目标权重。
组合输出不会连接实盘执行。

当严格候选因多窗口验证不足而没有成交时，runner 还会维护一条完全
独立的 Shadow 学习链。它只允许验证集上具有正收益、正期望、足够
信号数和足够 profit factor 的单边模型进入低风险模拟，支持做多和
做空。默认杠杆为 `1x`、单币种最大仓位为 `5%`、组合总敞口不超过
`20%`。Shadow 结果不会反向放宽严格策略，也不会连接实盘执行。

组合 paper 账本只在多个币种最新闭合 K 线时间一致时更新，并先按上一轮
权重计算本根 K 线收益，再应用新目标权重。它会记录组合权益、峰值回撤、
北京时间自然日收益、换仓手续费和滑点；重复 K 线不会重复记账。

## 报告怎么看

重点看这些文件：

```text
reports/progress.json
reports/runner_live_latest.json
reports/ai_optimization_*.json
reports/*_ai_optimization.json
reports/model_optimization_*.json
reports/*_model_optimization.json
reports/*_backtest_summary.json
reports/*_optimized_thresholds.json
reports/*_live_train_metrics.json
```

候选币种进入继续模拟盘的基本标准：

- 验证集和测试集不能明显冲突。
- 测试集 `total_return` 为正。
- `max_drawdown` 可控。
- `profit_factor` 大于 1。
- 交易次数不能太少。
- 实时 runner 连续多轮表现稳定，而不是只靠一次结果。

当前最近一次报告结论：

- 暂无币种/周期满足继续模拟盘或测试网候选门槛，全部保持观察/研究模式。
- `BNBUSDT 1h` 最新 3 折 walk-forward 被降级为观察：收益中位数为正，但高滑点成本压力通过率不足。
- `BNBUSDT 15m` 只有 1 折轻微盈利，交易数太少且短周期波动分层门槛失败。
- `ETHUSDT 15m` 最新验证回撤偏大，稳定性不足。
- 最新 `1m`/`5m` runner 和短周期验证仍不能作为测试网或实盘依据。

## 目录说明

```text
crypto_ai_trader/      核心代码
data/                  历史和实时 K 线数据
models/                训练后的模型
reports/               训练、回测、优化、runner 报告
state/                 runner 控制和状态文件
logs/                  runner 和控制台日志
scripts/               常用启动脚本
web/                   本地网页控制台
docs/                  详细流程文档
```

绩效指标公式与报告字段见
[`docs/PERFORMANCE_EVALUATION.md`](docs/PERFORMANCE_EVALUATION.md)。
模型选择和策略校准边界见
[`docs/MODEL_SELECTION_CALIBRATION.md`](docs/MODEL_SELECTION_CALIBRATION.md)。
Alpha 分类、预期收益回归和跨币种 Ranker 接口见
[`docs/ALPHA_MODEL_LAYER.md`](docs/ALPHA_MODEL_LAYER.md)。
市场状态规则与走步式 KMeans 见
[`docs/REGIME_LAYER.md`](docs/REGIME_LAYER.md)。

## 安全规则

- 不启用实盘交易。
- 不写入或修改 API Key。
- 不授予提现权限。
- 不自动全仓、高杠杆、马丁格尔。
- 模拟盘连续稳定后，才考虑 Binance Futures Testnet。
