# 无 VPN 数据与训练模式

目标：当当前网络无法直接访问 Binance 时，仍然可以用本地历史 K 线缓存继续训练、回测、实时模拟和模型优化。

## 推荐流程

1. 挂 VPN 时同步完整历史数据：

```powershell
.\.venv\Scripts\python.exe -m crypto_ai_trader.cli download --symbols BTCUSDT ETHUSDT SOLUSDT BNBUSDT --interval 5m --start 2024-01 --end 2025-12
```

程序会保存两类本地数据：

- `data/raw/.../*.csv`：训练流程直接读取的 K 线文件。
- `data/archive_cache/.../*.zip`：原始 Binance zip 缓存，方便以后重建 CSV。

2. 不挂 VPN 时只用本地缓存训练：

```powershell
.\.venv\Scripts\python.exe -m crypto_ai_trader.cli model-optimize --symbols ETHUSDT BNBUSDT --intervals 5m 15m --time-budget-minutes 15
```

3. 不挂 VPN 时检查缓存是否足够：

```powershell
.\.venv\Scripts\python.exe -m crypto_ai_trader.cli download --symbols ETHUSDT --interval 5m --start 2025-01 --end 2025-12 --cache-only --allow-partial-cache
```

如果输出 `skipped`，说明对应月份没有本地 CSV 或 zip 缓存，需要等下次挂 VPN 补齐，或配置你自己可访问的合规数据镜像。

## 桌面控制台按钮

- `验证币安下载`：直接测试 Binance 历史归档 K 线和实时 REST K 线是否可访问，并把结果写入 `reports/binance_download_check_latest.json`。
- `无 VPN 缓存检查`：不联网，只检查本地已有缓存是否足够被训练流程读取。

如果 `验证币安下载` 显示不可下载，说明当前网络、VPN、代理、防火墙或 `data_base_urls` 镜像源不可用；这时仍然可以使用本地缓存继续训练。

## 自定义可访问数据源

如果你有可访问的镜像源，可以在 `config.default.json` 里添加：

```json
{
  "data_base_urls": [
    "https://your-data-host/data/futures/um"
  ]
}
```

镜像目录结构需要与 Binance public data 一致，例如：

```text
monthly/klines/ETHUSDT/5m/ETHUSDT-5m-2025-01.zip
daily/klines/ETHUSDT/5m/ETHUSDT-5m-2025-01-01.zip
```

## Runner 行为

实时 Runner 连接 Binance 失败时，如果本地已有 `data/realtime/...` 数据，会继续使用缓存训练，并在报告 `sync[].skipped` 中记录原因。

实盘交易仍然关闭。这个模式只影响数据下载、训练、回测和模拟盘。
