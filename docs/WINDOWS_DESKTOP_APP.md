# Windows 桌面工具 App

本文记录 Windows 桌面工具 app、桌面任务 API 和便携恢复流程。当前版本采用 Electron 外壳承载现有 Python 本地服务，核心训练、回测和模拟盘逻辑仍在 `crypto_ai_trader` Python 包中。

## 当前目标

- 在 Windows 上提供一个可恢复、可重新安装依赖的本地运行环境。
- 继续使用本地控制台和 dashboard 作为主要界面。
- 桌面外壳只负责启动、显示和关闭本地服务，不改变交易安全边界。
- 便携包在新机器或新路径解压后，通过 `scripts/restore_portable.ps1` 重建 `.venv` 并执行健康检查。

## 便携恢复流程

在新机器或新目录解压便携包后：

```powershell
cd "C:\path\to\extracted\project"
powershell -ExecutionPolicy Bypass -File ".\scripts\restore_portable.ps1"
```

恢复脚本会：

- 从脚本目录或当前目录向上定位项目根目录。
- 解除项目内 `.ps1` 文件的 Windows 阻止标记。
- 优先使用 `py -3` 创建 `.venv`，找不到时使用 `python`。
- 安装 `requirements.txt`。
- 运行 `crypto_ai_trader.cli smoke` 和 `crypto_ai_trader.cli doctor`。
- 提醒计划任务必须在当前 Windows 用户下重新安装。

计划任务不要从旧机器复制。恢复完成后如需后台 runner，重新运行：

```powershell
powershell -ExecutionPolicy Bypass -File ".\scripts\install_runner_task.ps1"
```

如需本地每小时模型优化，重新运行：

```powershell
powershell -ExecutionPolicy Bypass -File ".\scripts\install_scheduled_optimizer_task.ps1"
```

## 桌面 app 入口

启动 Windows 桌面工具 app：

```powershell
powershell -ExecutionPolicy Bypass -File ".\scripts\start_desktop.ps1"
```

如果 Node.js/npm 不可用，先继续使用浏览器控制台，或安装 Node.js，也可以把 portable Node 解压到 `tools\node`：

```powershell
powershell -ExecutionPolicy Bypass -File ".\scripts\start_control_panel.ps1"
```

浏览器访问：

```text
http://127.0.0.1:8765
```

如果端口被占用，启动脚本会尝试 8766 到 8775 之间的空闲端口，并打印实际地址。

Electron 外壳会自动寻找本地空闲端口，启动：

```text
python -m crypto_ai_trader.dashboard_server --host 127.0.0.1 --port <port>
```

然后在桌面窗口中打开本地控制台。退出 Electron 时会关闭它启动的 dashboard server；runner 是否继续运行由用户在界面中明确控制。

## 桌面工具能力

控制台保留原有功能：

- runner 开始、暂停、恢复、停止
- 实时决策图
- 训练进度
- 回测和训练摘要

新增桌面工具区：

- 一键准确率优化：启动 `model-optimize`
- 持续优化一次：启动 `scheduled-optimize`，先刷新策略记忆、整理实时数据，再运行复杂模型搜索
- 一键实时训练：启动 `live-train`
- 整理训练数据：启动 `data-maintenance`，合并实时闭合 K 线并清理旧临时文件
- 查看最新优化：读取 `reports/model_optimization_*.json`
- 导出便携包：在 `exports/` 生成 zip
- 打开报告目录
- 实盘交易占位区，但按钮全部禁用

## 后端 API 边界

新增 API 只允许白名单任务，不接受任意 shell 命令：

```text
GET  /api/app/health
GET  /api/app/paths
POST /api/tasks/start
GET  /api/tasks
GET  /api/tasks/{task_id}
POST /api/tasks/{task_id}/cancel
GET  /api/tasks/{task_id}/logs
GET  /api/model-optimization/latest
GET  /api/scheduled-optimization/latest
GET  /api/data-maintenance/latest
GET  /api/model-optimization/reports
POST /api/export/portable
POST /api/system/open-path
```

Electron 启动时会生成本地 token，并通过 `X-Desktop-Token` 发送给写接口。普通浏览器启动控制台时不设置 token，仍保持原有本地控制体验。

## 安全边界

桌面外壳必须遵守：

- 不直接修改交易配置中的实盘开关。
- 不绕过 `crypto_ai_trader.cli`、runner 和 dashboard server 的既有入口。
- 不把旧计划任务、旧绝对路径或旧用户目录写入新安装。
- 不把 `.venv`、`logs`、`node_modules` 或构建产物视为便携包必需内容。
- 不把 `tools\node` 视为便携包必需内容；新机器可重新安装或重新放置 portable Node。
- 不在桌面层实现独立交易逻辑。
- 不录入、保存或导出 API Key。
- `live_trading_enabled` 必须保持 `false`。

## 打包与验收清单

桌面应用进入实现阶段前，至少确认：

- Electron 外壳能启动 dashboard server 并打开窗口。
- 写接口在 Electron 中携带本地 token。
- `scripts/restore_portable.ps1` 可以在无 `.venv` 的解压目录中完成恢复。
- `smoke` 和 `doctor` 通过。
- 控制台可以打开并显示 runner 状态。
- 计划任务通过 `scripts/install_runner_task.ps1` 在当前用户下重新安装。
- 每小时模型优化任务通过 `scripts/install_scheduled_optimizer_task.ps1` 在当前用户下重新安装。
- 便携包不包含本机敏感配置，例如 `config.json`、`.env`、`secrets.json`。
- 便携包包含版本化的
  `config/strategy_calibration_profiles.toml`，新设备会使用同一套风险档案。
- 实盘交易仍保持关闭，桌面层没有新增下单入口。

## 常见问题

如果 PowerShell 拒绝运行脚本，使用：

```powershell
powershell -ExecutionPolicy Bypass -File ".\scripts\restore_portable.ps1"
```

如果找不到 Python，安装 Python 3，并确保 `py -3` 或 `python` 在 PATH 中可用。

如果恢复后 runner 没有自动启动，重新安装并启动计划任务：

```powershell
powershell -ExecutionPolicy Bypass -File ".\scripts\install_runner_task.ps1"
powershell -ExecutionPolicy Bypass -File ".\scripts\start_runner_task.ps1"
```
