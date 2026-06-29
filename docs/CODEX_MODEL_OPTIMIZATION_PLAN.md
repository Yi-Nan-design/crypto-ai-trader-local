# Codex 模型优化实施计划：crypto-ai-trader-local

> 目标：把当前模型从“能训练、能回测”升级为更稳健的本地研究系统。
>
> 本文件只面向离线研究、回测、模拟盘一致性评估和工程质量优化，不构成投资建议，也不承诺任何收益结果。

---

## 1. 总体判断

当前项目已经具备较完整的本地量化研究链路：历史 K 线、已闭合实时 K 线、因果特征、前视标签、时间顺序切分、purge、Alpha 模型训练、验证集排序、测试集评估、回测、paper simulation、monitoring，以及 PSI、KS、ECE、Brier、rolling Sharpe、drawdown、paper deviation 等监控逻辑。

下一步优先优化的不是直接堆更复杂的模型，而是把以下能力打通到训练、验证、选模和发布门控中：

1. 概率校准；
2. purge + embargo + 多路径 walk-forward；
3. risk-aware publish score；
4. 分类 + 回归 + ranker 的多任务 Alpha；
5. bounded HPO；
6. 样本权重与 regime 平衡；
7. champion-challenger 与回滚机制。

---

## 2. 优先级总表

| 优先级 | 模块 | 要解决的问题 | 预期价值 |
|---|---|---|---|
| P0 | 概率校准前移 | 概率不可靠、阈值漂移、信号触发不稳定 | 降低 ECE/Brier，提高阈值稳定性 |
| P0 | Purge + Embargo + 多路径验证 | 单路径 walk-forward 容易过拟合 | 降低回测幻觉，提高验证可信度 |
| P0 | Risk-aware publish score | 只看预测分数或单次回测结果会选错模型 | 将风险、校准、策略质量统一为选模标准 |
| P1 | 多任务 Alpha | 只做方向分类会浪费收益标签和跨品种排序信息 | 提升机会排序与信号质量 |
| P1 | Optuna 分层 HPO | 固定候选模型与阈值搜索效率低 | 更系统地搜索模型与参数 |
| P1 | 样本权重与 regime 平衡 | 稀有高价值行情学习不足 | 提升稀有样本识别能力 |
| P2 | Champion-Challenger + 回滚 | 新模型退化时缺少明确回退机制 | 提升系统安全性与可追踪性 |

---

## 3. P0-1：概率校准前移

### 问题

当前系统是阈值驱动型系统。模型输出概率是否可靠，比单纯分类准确率更重要。未校准概率会导致：

- 验证集阈值迁移到测试集或 paper simulation 后漂移；
- `0.62` 的预测概率不一定代表真实命中率接近 62%；
- 策略层误判 edge，导致信号质量不稳定；
- monitoring 虽然记录 ECE/Brier，但训练侧没有充分利用这些指标。

### Codex 任务

在模型训练与模型选择阶段加入校准路径：

1. 将数据划分扩展为：`train / valid_model / valid_calibration / test`。
2. 支持两种校准方法：`sigmoid` 和 `isotonic`。
3. 增加配置项：
   - `enable_calibration: bool`
   - `calibration_method: sigmoid | isotonic`
   - `calibration_min_samples`
   - `calibration_bins`
4. 增加指标：
   - `raw_log_loss`
   - `calibrated_log_loss`
   - `brier`
   - `ece`
   - `threshold_drift`
5. 将校准指标纳入模型选择。

建议 publish score 初版：

```python
publish_score = (
    0.15 * auc
    - 0.15 * calibrated_log_loss
    - 0.15 * brier
    - 0.15 * ece
    + 0.20 * signal_expectancy_after_cost
    + 0.10 * profit_factor
    - 0.10 * max_drawdown_penalty
)
```

### 推荐实现位置

- `crypto_ai_trader/models.py`
- `crypto_ai_trader/model_optimization.py`
- `crypto_ai_trader/model_selection.py`
- `tests/test_model_calibration.py`

### 验收标准

- 未开启校准时，旧训练流程保持兼容；
- 开启校准后，报告包含 raw 与 calibrated 两组概率指标；
- 校准集不能与测试集重叠；
- `ece`、`brier` 能稳定计算；
- 样本不足时自动降级或跳过校准，不崩溃；
- 校准后 ECE 或 Brier 至少一个不劣于未校准结果；若变差，报告中必须明确记录。

---

## 4. P0-2：Purge 升级为 Embargo + 多路径 Walk-forward

### 问题

项目已有 `purge_rows`，但金融时间序列存在标签重叠、行情滞后反应、特征窗口残留等问题。单一路径 walk-forward 容易选出“在某一条历史路径上最好”的模型，而不是稳健模型。

### Codex 任务

1. 在 walk-forward 构造函数中新增：
   - `embargo_rows`
   - `min_train_rows`
   - `min_valid_rows`
   - `min_test_rows`
2. 支持多路径验证摘要：
   - `median_profit_factor`
   - `p25_profit_factor`
   - `p10_profit_factor`
   - `worst_path_drawdown`
   - `selected_model_stability`
3. 模型晋级不能只看单次最优结果，必须同时满足：
   - median 表现达标；
   - p25 表现达标；
   - worst-path drawdown 未触发风险红线。
4. CLI 增加参数：

```bash
--embargo-rows 12
--wf-paths 5
--wf-score-quantile p25
```

### 推荐实现位置

- `crypto_ai_trader/strategy_validation.py`
- `crypto_ai_trader/cli.py`
- `tests/test_strategy_validation.py`

### 验收标准

- `purge_rows` 与 `embargo_rows` 不能产生训练/验证/测试交叉；
- 多路径验证报告输出分位数，不只输出均值；
- 任一 split 样本不足时应明确报错或跳过，不得静默产生错误结果；
- 默认参数保持当前行为兼容。

---

## 5. P0-3：Risk-aware Publish Score

### 问题

项目在 monitoring 层记录了 PSI、KS、calibration error、rolling Sharpe、drawdown、paper deviation 等指标，但训练选模阶段没有完全把这些风险语言前移。

### Codex 任务

新增统一的 `publish_score`，用于决定模型是否能成为 champion。建议由以下部分构成：

```text
predictive_quality:
  auc
  log_loss
  balanced_accuracy

calibration_quality:
  brier
  ece
  reliability_slope

strategy_quality:
  signal_expectancy_after_cost
  profit_factor
  sharpe
  turnover

risk_quality:
  max_drawdown
  drawdown_duration
  worst_path_drawdown
  exposure_concentration

stability_quality:
  psi
  ks
  threshold_drift
  paper_vs_backtest_deviation
```

建议新增函数：

```python
def compute_publish_score(metrics: dict, weights: PublishScoreWeights) -> PublishDecision:
    ...
```

返回：

```python
@dataclass
class PublishDecision:
    score: float
    passed: bool
    hard_fail_reasons: list[str]
    soft_warnings: list[str]
    metric_breakdown: dict[str, float]
```

### 硬性失败条件

以下情况直接禁止模型成为 champion：

- 测试集样本数不足；
- ECE 超过上限；
- max drawdown 超过上限；
- p25 profit factor 低于阈值；
- paper deviation 连续触发；
- 特征列版本与训练版本不一致；
- 测试集与校准集或训练集存在时间重叠。

### 验收标准

- 每次模型优化必须输出 publish decision；
- 如果模型预测分数高但风险门控失败，报告必须解释失败原因；
- publish score 必须可配置、可复现、可记录版本。

---

## 6. P1-1：分类 + 回归 + Ranker 多任务 Alpha

### 问题

`labeling.py` 已经生成大量有价值标签，例如：`future_return`、`future_realized_volatility`、`future_risk_adjusted_return`、`long_target`、`short_target`、`tradable_label`、`actionable_label`、`big_move_target`、`future_return_h*`、`future_return_net_edge_h*`。

但主训练路径仍偏方向分类。这样会浪费连续收益、净边际、多 horizon 与跨品种排序信息。

### Codex 任务

设计三头 Alpha 输出：

1. 方向分类头：`P(edge_trade_target = 1)`
2. 连续收益回归头：`E[future_return_net_edge]`
3. 跨品种排序头：`rank(symbols at same timestamp)`

最终策略信号：

```text
final_edge_score = calibrated_probability
                 * expected_return_after_cost
                 * rank_score
                 * risk_gate_multiplier
```

### 实施顺序

1. 先实现分类 + 回归双头；
2. 再做多品种时间对齐；
3. 最后启用 `LGBMRanker`。

### 推荐实现位置

- `crypto_ai_trader/alpha_models.py`
- `crypto_ai_trader/model_optimization.py`
- `crypto_ai_trader/labeling.py`
- `crypto_ai_trader/portfolio.py`
- `tests/test_alpha_multitask.py`

### 验收标准

- 单品种模式下，Ranker 自动禁用；
- 多品种未时间对齐时，Ranker 不允许训练；
- 回归头必须输出 IC / Spearman / MAE / Huber loss；
- 多任务融合后，至少报告分类、回归、策略三组指标；
- 不允许只因为回归指标变好就发布模型，必须通过策略与风险门控。

---

## 7. P1-2：Optuna 分层 HPO

### Codex 任务

实现两层 HPO。

第一层：模型参数搜索。

搜索内容：

- LightGBM / XGBoost / CatBoost 的核心参数；
- Logistic / MLP 的基础参数；
- sample weight 策略参数。

第一层目标函数：

```text
AUC + calibrated_log_loss + ECE + balanced_accuracy
```

第二层：策略阈值搜索。只对第一层 top 20% 候选搜索：

- entry threshold；
- exit threshold；
- min edge；
- position cap；
- turnover cap；
- risk gate 参数。

第二层目标函数：

```text
signal_expectancy_after_cost + profit_factor - drawdown_penalty - turnover_penalty
```

### 验收标准

- 支持 timeout；
- 支持 max trials；
- 支持 pruning；
- 每个 trial 记录参数、指标、失败原因；
- HPO 不能直接覆盖 champion，只能产生 challenger。

---

## 8. P1-3：样本权重、Regime 平衡与稀有机会学习

### Codex 任务

将当前 sample weight 扩展为多因子：

```python
sample_weight = (
    edge_strength_weight
    * regime_rarity_weight
    * recency_weight
    * symbol_balance_weight
    * actionable_weight
)
```

建议新增：

- `regime_rarity_weight`：稀有行情 regime 提权；
- `recency_weight`：近期样本适度提权；
- `symbol_balance_weight`：避免 BTC/ETH 过度主导；
- `actionable_weight`：可研究样本提权，但不能过度放大噪声。

### 验收标准

- 权重必须有上限，避免极端样本支配训练；
- 报告中输出权重分布：min / p25 / median / p75 / p95 / max；
- 权重变更必须可配置；
- 权重打开与关闭要能做 A/B 对比。

---

## 9. P2：Champion-Challenger、回滚与阈值版本化

### Codex 任务

建立模型注册表，保存：

```text
model_id
model_family
feature_version
label_version
threshold_version
calibration_version
training_data_range
validation_protocol
publish_score
publish_decision
artifact_paths
created_at
parent_champion_id
```

规则：

- 新模型只能先成为 challenger；
- challenger 通过离线测试和 paper 验证后才能成为 champion；
- 阈值必须单独版本化，不能隐含在模型文件中；
- monitoring 连续触发 breach 时，系统应回滚到上一版 champion。

---

## 10. Codex 可直接执行的总提示词

```text
你正在修改 GitHub 仓库 Yi-Nan-design/crypto-ai-trader-local。

目标：提升模型稳定性、泛化能力、风险调整表现和离线/模拟盘一致性。不要承诺任何收益，不要修改任何凭证或密钥，不要加入真实执行逻辑。本任务只面向研究、回测、模拟盘和工程验证。

请按以下优先级分阶段实现：

Phase 0A：概率校准前移
- 在模型训练和 model optimization 流程中加入 train / valid_model / valid_calibration / test 切分。
- 支持 sigmoid 和 isotonic calibration。
- 新增 raw_log_loss、calibrated_log_loss、brier、ece、threshold_drift。
- 将 calibration metrics 纳入 model selection / publish score。
- 添加测试，确保校准集与测试集不重叠，样本不足时安全降级。

Phase 0B：Purge + Embargo + 多路径验证
- 在 strategy_validation 中增加 embargo_rows。
- 输出 median / p25 / p10 / worst-path 指标。
- 模型晋级不能只看单路径最好结果。
- CLI 增加 --embargo-rows、--wf-paths、--wf-score-quantile。
- 添加测试，确保时间切分无泄漏。

Phase 0C：Risk-aware publish score
- 新增 compute_publish_score(metrics, weights)。
- 输出 PublishDecision：score、passed、hard_fail_reasons、soft_warnings、metric_breakdown。
- 把 AUC、log_loss、Brier、ECE、signal expectancy after cost、profit factor、max drawdown、PSI/KS、paper deviation 纳入发布门控。
- 模型预测质量高但风险失败时，报告必须解释失败原因。

Phase 1A：分类 + 回归 + Ranker 多任务 Alpha
- 在已有 labeling 输出基础上，增加 expected_return_after_cost 回归头。
- 多品种时间对齐后，再启用 LGBMRanker。
- 最终 final_edge_score = calibrated_probability * expected_return_after_cost * rank_score * risk_gate_multiplier。
- 单品种或未对齐数据时，Ranker 必须自动禁用。

Phase 1B：Optuna bounded HPO
- 使用 time budget 和 max trials。
- 第一层搜索模型参数，第二层只对 top candidates 搜索策略阈值。
- 支持 pruning、early stopping、trial 记录。
- HPO 只能产生 challenger，不得直接覆盖 champion。

Phase 1C：样本权重和 regime 平衡
- 将 sample_weight 扩展为 edge_strength_weight * regime_rarity_weight * recency_weight * symbol_balance_weight * actionable_weight。
- 输出权重分布。
- 权重必须有上限并支持配置关闭。

开发要求：
- 保持现有 CLI 默认行为兼容。
- 每个阶段必须增加单元测试。
- 所有新配置必须有默认值。
- 所有模型发布相关结果必须写入 report 或 artifact。
- 禁止使用未来数据，禁止 train/validation/test 时间重叠。
- 修改完成后运行现有测试，并新增必要测试。

优先完成 Phase 0A、0B、0C。Phase 1 可以在 Phase 0 测试通过后继续。
```

---

## 11. 推荐测试命令

```bash
python -m pytest tests/test_monitoring.py -q
python -m pytest tests/test_strategy_validation.py -q
python -m pytest tests/test_model_calibration.py -q
python -m pytest tests/test_model_selection.py -q
```

如果测试文件尚不存在，请创建对应测试。

端到端 dry-run 示例：

```bash
python -m crypto_ai_trader.cli model-optimize \
  --symbols BTCUSDT ETHUSDT SOLUSDT \
  --intervals 1h 15m \
  --include-realtime \
  --time-budget-minutes 30 \
  --max-model-trials 16
```

更严格验证示例：

```bash
python -m crypto_ai_trader.cli strategy-validate \
  --symbols BTCUSDT ETHUSDT SOLUSDT BNBUSDT \
  --intervals 1h 15m \
  --include-realtime \
  --folds 5 \
  --purge-rows 24 \
  --embargo-rows 12 \
  --holdout-fraction 0.15 \
  --wf-train-rows 4000 \
  --wf-valid-rows 1000 \
  --wf-test-rows 1000 \
  --max-threshold-evals 20
```

---

## 12. 判断优化是否有效

不要只看 accuracy。建议以如下顺序判断：

1. 无泄漏：时间切分、purge、embargo、校准集、测试集均无交叉；
2. 概率可靠：ECE、Brier、log loss 改善或至少不恶化；
3. 阈值稳定：验证集最佳阈值迁移到测试集后不过度漂移；
4. 策略指标有效：扣除成本后的 signal expectancy 为正；
5. 风险可控：max drawdown、p25 PF、worst-path drawdown 达标；
6. 模拟盘一致：paper 与 frozen test 偏差在可接受范围内；
7. 可回滚：新模型失败时能回到旧 champion。

真正值得保留的模型，不是历史回测结果最高的模型，而是在多路径、校准、成本、风险和模拟盘一致性下仍然稳定的模型。
