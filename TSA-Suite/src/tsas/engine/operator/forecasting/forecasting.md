# 工业时序预测算子库实现说明

## 1. 实现逻辑概述

参考 `src/tsas/engine/operator/detection` 的四层检测算子架构，本次为 TSA-Suite 新增了工业时序预测算子库。由于时序预测任务的输入输出维度与现有 `NumericOperator` 的 2-D 假设不同（预测需要 3-D 窗口张量：`(batch, seq_len, num_features) → (batch, pred_len, num_targets)`），因此新增了一个专门的 `BaseForecaster` 基类。

`BaseForecaster` 直接继承 `LearnableOperatorMixin` + `BaseOperator`，在公共边界上仍然接受 `pd.DataFrame` / `np.ndarray`，但内部提供 3-D 友好的 IO 辅助方法。具体架构如下：

```
BaseOperator
  └── BaseForecaster
        └── ITransformerForecaster
```

iTransformer 模型代码来自 `HBHD_predict_v1.5`，经过适配后封装为算子。模型采用以下工业时序预测最佳实践：

- **Dense NPU-compatible Transformer Encoder**（通过 `num_experts=1, top_k=1` 将 MoE 退化为标准 Dense FFN，降低部署复杂度）
- **KAN 预测头**：比传统 MLP 更适合捕捉非线性映射
- **Lag-Aware Refiner**：利用互相关先验增强目标变量的时序建模
- **目标历史平线掩码（Target Flatline Masking）**：破坏自回归捷径，迫使模型从其他特征学习因果关系
- **残差预测策略**：模型预测相对最后已知值的增量，提升对漂移和趋势的稳定性
- **训练损失**：加权 MSE + 一阶趋势损失，配合 `ReduceLROnPlateau` 与早停

评价指标方面，参考 `HBHD_predict_v1.5/predict_check.py` 中的 `calculate_metrics`，在 `src/tsas/engine/operator/evaluation` 下新增 `ForecastingMetrics` 算子，统一提供 MSE、RMSE、MAE、MAPE、SMAPE、MASE、DTW、R² 八项指标。

## 2. 新增文件清单

### 2.1 时序预测算子库 (`src/tsas/engine/operator/forecasting/`)

| 文件 | 说明 |
|------|------|
| `__init__.py` | 包导出：`BaseForecaster`、`ForecastExtraOutput`、`ITransformerForecaster`、`ITransformerForecasterConfig` |
| `base.py` | `BaseForecaster` 基类，定义 3-D 时序预测的输入输出约定与模板方法 |
| `itransformer.py` | `ITransformerForecaster` 算子：Config、训练、推理、save/load |
| `_models/npu_transformer.py` | NPU 兼容的 Transformer 底层组件（ attention、MoE、Encoder） |
| `_models/itransformer_kan_res.py` | iTransformer 回归模型（KAN 头 + 残差预测 + Lag-Aware Refiner） |
| `forecasting.md` | 本文档 |

### 2.2 评价指标算子库 (`src/tsas/engine/operator/evaluation/`)

| 文件 | 说明 |
|------|------|
| `forecasting_metrics.py` | `ForecastingMetrics` 算子及 `ForecastingMetricResult` / `ForecastingMetricConfig` |
| `__init__.py` | 导出 `ForecastingMetrics` 等 |

### 2.3 CLI 集成 (`src/tsas/engine/operator/cli/`)

| 文件 | 说明 |
|------|------|
| `forecasting.py` | 新增 `forecasting` 模块 CLI：`help`、`fit`、`run` |
| `__main__.py` | 在 `_MODULE_MAP` 中注册 `forecasting` 模块 |

## 3. 新增算子库

### 3.1 `BaseForecaster`

基类，定义时序预测算子的公共接口：

- `fit(x, y)`：
  - `x`: `(timesteps, num_features)`，DataFrame 或 ndarray
  - `y`: `(timesteps, num_targets)`，DataFrame 或 ndarray（当前 `ITransformerForecaster` 仅支持单目标）
- `run(x)`：
  - `x`: `(seq_len, num_features)` 或 `(batch, seq_len, num_features)`
  - 返回：`(pred_len, num_targets)` 或 `(batch, pred_len, num_targets)`，类型与输入保持一致

### 3.2 `ITransformerForecaster`

基于 iTransformer 的具体预测算子，名称 `itransformer_forecaster`。

**可配置参数（`ITransformerForecasterConfig`）**：

| 参数 | 默认值 | 说明 | HPO 可搜索 |
|------|--------|------|------------|
| `seq_len` | 100 | 输入历史窗口长度 | ✓ |
| `pred_len` | 20 | 预测未来步长 | ✓ |
| `d_model` | 128 | 模型嵌入维度 | ✓ |
| `nhead` | 4 | 注意力头数 | ✓ |
| `num_layers` | 2 | Encoder 层数 | ✓ |
| `dim_feedforward` | `None` | FFN 隐藏层维度，None 时取 `2*d_model` | ✓ |
| `dropout` | 0.2 | Dropout 比率 | ✓ |
| `step_cond_head` | `False` | 是否使用步长条件预测头 | - |
| `lag_aware` | `True` | 是否启用 Lag-Aware Refiner | - |
| `lag_max` | 16 | 最大滞后步数 | ✓ |
| `lag_bias_scale` | 2.0 | 互相关先验偏置缩放 | ✓ |
| `lag_dropout` | 0.1 | Lag Refiner Dropout | ✓ |
| `kan_grid_size` | 5 | KAN 网格大小 | ✓ |
| `target_idx` | -1 | 目标变量列索引，-1 表示最后一列 | - |
| `epochs` | 30 | 最大训练轮数 | ✓ |
| `batch_size` | 128 | 训练批次大小 | ✓ |
| `lr` | 0.001 | 学习率 | ✓ |
| `weight_decay` | 1e-5 | 权重衰减 | ✓ |
| `early_stop_patience` | 12 | 早停耐心轮数 | ✓ |
| `train_ratio` | 0.7 | 训练集占比 | ✓ |
| `val_ratio` | 0.15 | 验证集占剩余数据比例 | ✓ |
| `trend_weight` | 1.0 | 趋势损失权重 | ✓ |
| `time_weight_start` | 0.1 | 时间加权损失起始权重 | ✓ |
| `time_weight_end` | 1.0 | 时间加权损失结束权重 | ✓ |
| `max_grad_norm` | 1.0 | 梯度裁剪范数 | ✓ |
| `scheduler_factor` | 0.5 | 学习率衰减因子 | ✓ |
| `scheduler_patience` | 3 | 学习率衰减耐心轮数 | ✓ |
| `device` | `'auto'` | 计算设备：`auto` / `cpu` / `cuda` | - |

**训练流程**：

1. 使用 `StandardScaler` 对训练数据拟合标准化参数。
2. 通过滑动窗口构造样本：输入为历史窗口，目标为未来 `pred_len` 步的残差（相对窗口最后一时刻目标值的增量）。
3. 按时间顺序划分训练集 / 验证集。
4. 训练循环：加权 MSE（近期预测权重更高） + 一阶趋势损失；使用 `Adam` 优化器、`ReduceLROnPlateau` 调度器、梯度裁剪与早停。
5. 保存验证损失最低的模型权重。

**推理流程**：

1. 对输入窗口做标准化。
2. 模型输出归一化残差。
3. 反标准化得到物理量残差： `delta_physical = pred_res_norm * scale[target_idx]`。
4. 加上输入窗口最后一个时间步的目标物理值，得到未来预测值。

**持久化**：

- `_scaler.npz`：保存 `mean_`、`scale_`、`n_features_in_`。
- `_model_weights.pt`：保存 PyTorch 模型权重。
- `_forecaster_state.npz`：保存 `target_idx`、`num_features`、`num_targets`。
- `last_fit_params.json` / `config.json`：继承自 `BaseOperator` / `LearnableOperatorMixin`。

## 4. 新增评价指标算子库

### 4.1 `ForecastingMetrics`

名称 `forecasting_metrics`，输入为 `(y_true, y_pred)` 元组，返回结构化的 `ForecastingMetricResult`。

**指标定义**：

| 指标 | 公式 |
|------|------|
| MSE | `mean((y_pred - y_true)^2)` |
| RMSE | `sqrt(MSE)` |
| MAE | `mean(\|y_pred - y_true\|)` |
| MAPE | `mean(\|(y_true - y_pred) / (\|y_true\| + ε)\|) * 100` |
| SMAPE | `mean(2 * \|y_pred - y_true\| / (\|y_true\| + \|y_pred\| + ε)) * 100` |
| MASE | `MAE / naive_error`（`naive_error` 通过 Config 传入） |
| R² | `1 - SS_res / SS_tot` |
| DTW | 归一化 Dynamic Time Warping；大样本时均匀降采样到 `max_dtw_len` |

**配置项（`ForecastingMetricConfig`）**：

- `main_scores`：默认映射全部 8 项指标，可覆写以选择 HPO 优化目标。
- `epsilon`：零值保护常数，默认 `1e-8`。
- `max_dtw_len`：DTW 最大采样长度，默认 `2000`。
- `naive_error`：训练集随机游走基线误差，用于 MASE；未提供时返回 `nan`。

**DTW 依赖说明**：`fastdtw` 为可选依赖。未安装时自动回退为 MAE，并发出 `UserWarning`。

## 5. CLI 用法

### 5.1 列出预测算子

```bash
python -m tsas.engine.operator.cli forecasting help
```

### 5.2 查看单个算子帮助

```bash
python -m tsas.engine.operator.cli forecasting help itransformer_forecaster
```

### 5.3 训练模型

```bash
python -m tsas.engine.operator.cli forecasting fit \
  --input train.csv \
  --target target_col \
  --config forecaster.yaml \
  --save model_dir/
```

配置文件示例 `forecaster.yaml`：

```yaml
operator:
  name: "itransformer_forecaster"
  input_columns:
    - "feat_0"
    - "feat_1"
    - "feat_2"
  config:
    seq_len: 100
    pred_len: 20
    d_model: 128
    nhead: 4
    num_layers: 2
    epochs: 30
    batch_size: 128
```

### 5.4 执行预测

```bash
python -m tsas.engine.operator.cli forecasting run \
  --input test_window.csv \
  --config forecaster.yaml \
  --load model_dir/ \
  --output pred.csv
```

### 5.5 计算预测指标

```bash
python -m tsas.engine.operator.cli evaluation run \
  --input predictions.csv \
  --output metrics.json \
  --config eval.yaml
```

配置文件示例 `eval.yaml`：

```yaml
operators:
  - name: "forecasting_metrics"
    truth_columns: ["true"]
    predict_columns: ["pred"]
    config:
      naive_error: 1.0
```

## 6. HPO 集成

`ITransformerForecasterConfig` 中大量参数使用 `Field(ge=..., le=...)` 声明边界，因此可被 `tsas.engine.hpo.search_hint.extract_search_space` 自动识别为搜索空间。

示例：使用 `HPOTrainer` 搜索 `d_model`、`nhead`、`num_layers`。

```python
from tsas.engine.hpo.trainer import HPOTrainer
from tsas.engine.operator.forecasting import ITransformerForecaster
from tsas.engine.operator.evaluation import ForecastingMetrics, ForecastingMetricConfig

metric_cfg = ForecastingMetricConfig(
    main_scores={"rmse": "rmse"},
    naive_error=1.0,
)
metric_op = ForecastingMetrics(config=metric_cfg)

trainer = HPOTrainer(
    operator=ITransformerForecaster,
    metric_op=metric_op,
    directions=["minimize"],
)
# 在 (x_train, y_train, x_val, y_val) 上执行搜索
```

> 注：`HPOTrainer` 要求算子 `fit(x, y)` / `run(x)` 接口与本文约定一致；
> `ForecastingMetrics.scores()` 返回 `{"rmse": value}` 可直接作为优化目标。

## 7. 验证情况

由于当前运行环境为 Python 3.8，而项目 `requirements.txt` 指定 Python 3.11，以下验证项已完成：

1. **语法检查**：所有新增文件均通过 `python -m py_compile`。
2. **指标公式验证**：使用纯 NumPy 复现了 `ForecastingMetrics` 的计算逻辑，结果符合预期。
3. **模型结构**：iTransformer 模型代码直接改编自已运行通过的 `HBHD_predict_v1.5/models/npu_itransformer_kan_res.py`，结构保持一致。

在 Python 3.11 环境（安装 `requirements.txt`）下，建议执行以下完整验证：

```bash
# 列出算子
PYTHONPATH=src python -m tsas.engine.operator.cli forecasting help

# 训练与预测
PYTHONPATH=src python -m tsas.engine.operator.cli forecasting fit \
  --input data.csv --target target --config forecaster.yaml --save model_dir/
PYTHONPATH=src python -m tsas.engine.operator.cli forecasting run \
  --input window.csv --config forecaster.yaml --load model_dir/ --output pred.csv

# 计算指标
PYTHONPATH=src python -m tsas.engine.operator.cli evaluation help
PYTHONPATH=src python -m tsas.engine.operator.cli evaluation run \
  --input pred.csv --output metrics.json --config eval.yaml
```

## 8. 关键设计决策

| 决策 | 选择 | 原因 |
|------|------|------|
| 预测基类 | 自定义 `BaseForecaster`，不继承 `NumericOperator` | `NumericOperator` 的 2-D 同样本数假设不适用于时序窗口 |
| 模型代码复用 | 复制并适配到 `forecasting/_models/` | HBHD 目录名含空格且含项目特定硬编码，不适合直接 import |
| 预测策略 | 残差预测 + 目标平线掩码 | 抑制自回归复制，增强工业场景下的因果学习与漂移鲁棒性 |
| Scaler 持久化 | `StandardScaler` 参数保存为 `.npz` | 避免额外依赖，加载时重建 scaler |
| DTW | `fastdtw` 可选，缺失时回退 MAE | 降低算子依赖门槛 |
| MASE | 通过 Config 传入 `naive_error` | 指标算子无训练数据访问权，由训练方计算后传入 |
