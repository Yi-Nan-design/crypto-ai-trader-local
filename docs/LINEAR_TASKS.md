# Linear 任务拆分

当前会话没有可用的 Linear 写入工具，所以先把任务拆成可复制到 Linear 的条目。

## A. Electron 桌面壳

状态：初稿完成

验收标准：

- `desktop/electron/main.js` 自动查找空闲端口。
- Electron 使用 `.venv\Scripts\python.exe` 启动 dashboard server。
- 使用 `spawn(..., { shell: false })`，不拼接 shell 字符串。
- Electron 请求本地 API 时自动附带 `X-Desktop-Token`。
- 关闭窗口时清理它启动的 dashboard server。

## B. Python 桌面任务 API

状态：初稿完成

验收标准：

- 新增 app health、paths、tasks、logs、model optimization、portable export、open path API。
- 任务类型只允许 runner、model-optimize、live-train、ai-optimize、cycle、doctor、export-portable。
- 每种任务同一时间最多运行一个。
- 日志只能从 `logs/desktop_tasks/` 读取。
- 写接口在 Electron 模式下要求本地 token。

## C. 便携导出与恢复

状态：初稿完成

验收标准：

- 导出 zip 写入 `exports/`。
- zip 包含源码、文档、脚本、Web 控制台、配置模板、`data/`、`models/`、`reports/`、`state/`。
- zip 排除 `.venv/`、`__pycache__/`、`.cache/`、`downloads/`、`logs/`、`node_modules/`、构建产物和敏感配置。
- `manifest.json` 记录文件数量、大小、关键 hash 和 `live_trading_enabled=false` 检查结果。
- `scripts/restore_portable.ps1` 在新机器重建 `.venv`、安装依赖、运行 `smoke` 和 `doctor`。

## D. 控制台 UI 与文档

状态：初稿完成

验收标准：

- 控制台新增桌面工具区。
- 一键准确率优化启动 `model-optimize`。
- 一键实时训练启动 `live-train`。
- 可以查看最新准确率优化摘要。
- 可以启动便携包导出。
- 实盘交易区域全部禁用并显示锁死说明。
- README 和 Windows 桌面文档包含启动、恢复、导出说明。

## E. QA

状态：待继续扩展

验收标准：

- `python -m compileall crypto_ai_trader` 通过。
- `python -m crypto_ai_trader.cli smoke` 通过。
- `python -m crypto_ai_trader.cli doctor` 可运行。
- dashboard server API 可本地访问。
- `POST /api/tasks/start` 可启动轻量 `doctor` 或 `model-optimize`。
- `POST /api/export/portable` 可生成 zip。
- Node/npm 修复后补测 Electron 启动和打包。

## 风险

- 当前机器的 `node.exe` 有拒绝访问问题，Electron 运行和打包需要先修复 Node/npm。
- 中文路径下不要使用 `.cmd` 或 shell 字符串拼接启动任务。
- 计划任务、开机启动和 `.venv` 都是机器绑定资源，迁移后必须重新生成。
- 实盘交易保持锁死，不创建 API Key 配置任务。
