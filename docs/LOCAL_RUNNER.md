# 本地 Runner 和图形控制台

本地 runner 用于在 Codex 不在线时继续同步 Binance USDT-M Futures 已闭合 K 线，并训练本地 `_runner_live` 模型。它只写入数据、模型和报告，不会开启实盘交易。

## 推荐方式：图形控制台

先进入项目目录：

```powershell
cd "<项目解压或克隆目录>"
```

启动控制台：

```powershell
powershell -ExecutionPolicy Bypass -File ".\scripts\start_control_panel.ps1"
```

如果要使用 Windows 桌面工具 app，运行：

```powershell
powershell -ExecutionPolicy Bypass -File ".\scripts\start_desktop.ps1"
```

桌面 app 使用 Electron 外壳打开同一个本地控制台，并额外提供一键准确率优化、实时训练、便携包导出和最新优化摘要。当前机器如果 Node.js/npm 不可用，先使用浏览器控制台，不影响 runner。

浏览器访问：

```text
http://127.0.0.1:8765
```

如果 8765 已经被旧控制台占用，启动脚本会自动使用 8766 到 8775 之间的空闲端口，并打印实际地址。

控制台功能：

- 选择币种：`BTCUSDT`、`ETHUSDT`、`SOLUSDT`、`BNBUSDT`
- 选择周期：`1m`、`3m`、`5m`、`15m`、`30m`、`1h`
- 设置 K 线数量和训练间隔
- 点击 `开始`、`暂停`、`恢复`、`停止`
- 查看 runner 状态、模型排名、上涨概率、收益、回撤和最近事件
- 查看实时决策图，图表每 5 秒单独刷新，不会刷新整个页面
- 在图上查看买点/做多、卖点/做空、平仓点和上涨概率线

注意：控制台不会自动点击开始。打开页面后需要手动点击 `开始`，或者使用 Windows 启动项让 runner 登录后自动运行。

## 命令行控制

查看状态：

```powershell
powershell -ExecutionPolicy Bypass -File ".\scripts\status_runner.ps1"
```

暂停：

```powershell
powershell -ExecutionPolicy Bypass -File ".\scripts\pause_runner.ps1"
```

恢复：

```powershell
powershell -ExecutionPolicy Bypass -File ".\scripts\resume_runner.ps1"
```

停止：

```powershell
powershell -ExecutionPolicy Bypass -File ".\scripts\stop_runner.ps1"
```

手动启动后台 runner：

```powershell
powershell -ExecutionPolicy Bypass -File ".\scripts\start_runner.ps1"
```

## Windows 自动启动

安装计划任务和登录启动快捷方式：

```powershell
powershell -ExecutionPolicy Bypass -File ".\scripts\install_runner_task.ps1"
```

手动启动计划任务：

```powershell
powershell -ExecutionPolicy Bypass -File ".\scripts\start_runner_task.ps1"
```

查询计划任务：

```powershell
powershell -ExecutionPolicy Bypass -File ".\scripts\query_runner_task.ps1"
```

结束计划任务：

```powershell
powershell -ExecutionPolicy Bypass -File ".\scripts\stop_runner_task.ps1"
```

安装脚本会生成：

```text
%USERPROFILE%\crypto_ai_runner.ps1
%APPDATA%\Microsoft\Windows\Start Menu\Programs\Startup\CryptoAiRunner.lnk
```

这是为了避开 `cmd.exe` 对中文路径的编码问题。

## 默认参数

- 默认币种：`ETHUSDT`、`BNBUSDT`
- 默认周期：`5m`
- 默认 K 线数量：`800`
- 默认训练间隔：`900` 秒
- 每轮最多尝试模型数：`4`
- 单个训练目标时间预算：`6` 分钟
- 滚动验证折数：`1`
- 单次训练最多使用最近 `8000` 行；完整历史仍保留在本地数据目录
- 默认代理：自动检测，优先使用 `http://127.0.0.1:7890`
- 实盘交易：关闭

安装任务时可以覆盖参数：

```powershell
powershell -ExecutionPolicy Bypass -File ".\scripts\install_runner_task.ps1" -Symbols ETHUSDT,BNBUSDT -Interval 5m -Limit 800 -TrainEverySeconds 900
```

## 判断 runner 是否真的在运行

不要只看 `state/runner_state.json` 里的 `status: waiting`。如果进程已经退出，状态文件可能是旧的。

需要同时确认：

- `state/runner_state.json` 的 `updated_utc` 是否持续更新
- `reports/runner_live_latest.json` 的修改时间是否持续更新
- 图形控制台里 `进程` 是否显示运行中
- `logs/runner.gui.err.log` 或 `logs/runner.task.err.log` 是否有错误

如果报告只更新了一次，说明 runner 只执行了一轮，没有常驻运行。重新打开控制台并点击 `开始`。

## 输出文件

```text
state/runner_control.json
state/runner_state.json
data/realtime/{SYMBOL}/{INTERVAL}/
reports/runner_live_latest.json
reports/runner_live_{INTERVAL}_*.json
reports/*_runner_live_train_metrics.json
reports/*_runner_live_backtest.csv
models/{SYMBOL}_{INTERVAL}_runner_live.pkl
reports/portfolio_snapshot_latest.json
reports/portfolio_paper_latest.json
state/portfolio_paper_{INTERVAL}.json
reports/shadow_portfolio_snapshot_latest.json
reports/shadow_portfolio_paper_latest.json
state/shadow_portfolio_paper_{INTERVAL}.json
logs/runner.gui.out.log
logs/runner.gui.err.log
logs/runner.task.out.log
logs/runner.task.err.log
```

严格组合和 Shadow 学习组合完全分账：

- 严格组合继续使用原有多窗口验证、交易门控和风险约束，不会为了增加成交而降低门槛。
- Shadow 在 `0.54` 到 `0.70` 的低原始阈值上搜索趋势、区间、成交量、流动性和波动过滤策略。
- 策略必须在验证校准段和后续门控段为正，测试集只做最终发布否决；连续重叠信号按预测周期去重，触发后按该周期保持模拟方向。
- 风险关闭、交易所不可用、流动性不足和资金费率拥挤可以在持仓周期结束前强制退出。
- Shadow 支持做多和做空，默认 `10x` 逐仓；单币保证金最多 `2.5%`，名义敞口最多 `25%`，组合名义总敞口最多 `50%`。
- Shadow 只写模拟账本，不连接 API Key，也不会产生实盘订单。

## VS Code

打开项目：

```powershell
powershell -ExecutionPolicy Bypass -File ".\scripts\open_vscode.ps1"
```

VS Code 已配置使用项目内 `.venv`：

```text
.vscode/settings.json
.vscode/tasks.json
```

可以在 VS Code 任务里运行 `Control Panel`、`Runner Once`、`Runner Status`。
