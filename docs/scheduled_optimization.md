# 定时模型优化与数据整理

本功能用于把“模型优化、策略记忆、实时闭合 K 线、数据整理”合成一条本地闭环。它只服务历史训练、回测、模拟盘和报告分析，不启用实盘，不读取或保存 API Key，也不下单。

## 核心命令

运行一次持续优化：

```powershell
.\.venv\Scripts\python.exe -m crypto_ai_trader.cli scheduled-optimize --symbols BNBUSDT ETHUSDT --intervals 15m 1h --include-realtime --complexity standard --rolling-folds 0 --time-budget-minutes 8 --max-model-trials 2 --max-training-rows 8000 --max-targets 1
```

只整理数据：

```powershell
.\.venv\Scripts\python.exe -m crypto_ai_trader.cli data-maintenance
```

预演整理，不移动文件：

```powershell
.\.venv\Scripts\python.exe -m crypto_ai_trader.cli data-maintenance --dry-run
```

## scheduled-optimize 做什么

1. 先刷新 `simulation_memory`，读取历史训练、回测、实时 runner、AI 优化和模拟盘结果。
2. 如果命令没有指定币种周期，则从策略记忆中优先选择弱项和高潜力目标。
3. 可选运行 `data-maintenance`，压缩实时闭合 K 线文件，清理旧临时文件。
4. 运行 `model-optimize` 的准确率优先训练链路，并启用更复杂模型池。
5. 训练结束后再次刷新策略记忆，让新结果进入长期积累。
6. 输出：
   - `reports/scheduled_optimization_latest.json`
   - `reports/scheduled_optimization_*.json`
   - `reports/model_optimization_*.json`
   - `reports/simulation_memory_latest.json`

## 复杂度参数

- `standard`：原有轻量模型池。
- `expanded`：增加更多 sklearn、LightGBM、XGBoost、CatBoost 参数组合，适合手动加深优化。
- `deep`：在 `expanded` 基础上优先尝试 MLP/表格神经网络，并启用滚动验证惩罚。
- `blackbox`：在 `deep` 基础上允许更黑箱的模型；只有这个档位会加入可选 PyTorch 序列 Transformer。

每次搜索只用验证集选模型和参数，测试集只做最终评估，避免反复拿测试集调参。

## 神经网络与防过拟合

Transformer 本身也是神经网络，而且属于深度神经网络。本项目当前默认把 `deep` 定义为“MLP/表格神经网络优先”，因为最近结果显示这类模型准确率更高；Transformer 保留在 `blackbox` 档位，作为更慢、更黑箱的候选。

当本机安装了 `torch` 且使用 `--complexity blackbox` 时，会加入小型 Transformer 序列模型：

- 使用最近 24 或 48 根 K 线的特征序列。
- 只用 CPU 单线程训练，避免拖垮本机。
- 使用 dropout、weight decay、早停、梯度裁剪和时间顺序验证集。
- 模型选择时加入训练/验证差距惩罚，训练集明显好于验证集会降分。
- `--rolling-folds` 会在训练段内部做滚动验证，滚动表现差的模型会被降分。

如果没有安装 `torch`，报告会记录 `missing_dependency: torch`，其他模型继续运行。使用 `deep` 时，报告会记录 Transformer 已保留给 `blackbox` 档位。

需要启用 Transformer 候选时，可以单独安装可选依赖：

```powershell
.\.venv\Scripts\python.exe -m pip install -r requirements-neural.txt
```

## 未来 5/10 分钟预测匹配

在 `5m` 周期上，训练会额外生成两个辅助方向模型：

- `next_5m`：预测下一根 5 分钟 K 线涨跌。
- `next_10m`：预测未来两根 5 分钟 K 线累计涨跌。

报告会把预测与实际涨跌对齐：

- `reports/{SYMBOL}_5m_model_optimization.json` 中的 `model_report.multi_horizon`
- `reports/{SYMBOL}_5m_*_train_metrics.json` 中的 `latest_horizon_probabilities`
- `reports/{SYMBOL}_5m_*_train_metrics.json` 中的 `recent_horizon_matches`

桌面控制台的实时决策区会显示未来 5 分钟、未来 10 分钟上涨概率，并同时展示最近已闭合样本的方向匹配率。

## 数据整理策略

`data-maintenance` 保守处理数据：

- 不删除历史月度 K 线。
- 不删除 `archive_cache`。
- 只合并 `data/realtime/{SYMBOL}/{interval}` 下的实时闭合 K 线文件。
- 默认只保留每个币种/周期最近 6000 条实时闭合 K 线，历史月度 K 线不删，用它们扩展训练样本。
- 合并后旧实时文件会移动到 `data/maintenance_archive/`，方便回溯。
- 只清理 `reports/`、`state/`、`logs/` 下过旧的 `*.tmp` 临时文件。

输出：

- `reports/data_maintenance_latest.json`
- `reports/data_maintenance_*.json`

## 桌面 App

桌面控制台新增：

- `持续优化一次`：调用 `scheduled-optimize`。
- `整理训练数据`：调用 `data-maintenance`。
- 后台任务列表会显示运行状态和日志。
- 状态卡会显示最新持续优化与数据整理摘要。

## Windows 每小时计划任务

安装本地每小时优化任务：

```powershell
powershell -ExecutionPolicy Bypass -File ".\scripts\install_scheduled_optimizer_task.ps1"
```

立即手动触发一次：

```powershell
powershell -ExecutionPolicy Bypass -File ".\scripts\start_scheduled_optimizer_task.ps1"
```

查询状态：

```powershell
powershell -ExecutionPolicy Bypass -File ".\scripts\query_scheduled_optimizer_task.ps1"
```

停止当前正在运行的优化：

```powershell
powershell -ExecutionPolicy Bypass -File ".\scripts\stop_scheduled_optimizer_task.ps1"
```

计划任务默认每小时运行一次 `scheduled-optimize`，目标优先为 `BNBUSDT`、`ETHUSDT` 的 `15m`、`1h`，复杂度为 `standard`，滚动验证折数为 0，时间预算为 8 分钟，最多 2 组模型参数、最近 8000 行和 1 个目标。新设备迁移后需要重新安装计划任务。

如果同一计划任务允许的币种存在有效、未过期的监控再训练触发器，触发器的实际周期会优先于命令行默认周期。例如命令默认 `1h`，`ETHUSDT 5m` 连续漂移触发后仍会先优化 `ETHUSDT 5m`。成功完成该目标后，触发器会写入确认时间和优化报告路径。

## 安全边界

所有入口都保持：

- `live_trading_enabled=false`
- 不接 API Key
- 不下单
- 不修改真实交易开关
