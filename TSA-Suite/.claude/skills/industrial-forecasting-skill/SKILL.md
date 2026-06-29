---
name: industrial-forecasting-pipeline
description: 基于 TSA-Suite 的工业时序预测端到端流程（数据生成、训练、推理、评估、报告）
version: 1.0.0
---

# 工业时序预测端到端流程

本 skill 用于在 TSA-Suite 项目中执行工业时序预测的完整流程。

## 触发条件

当用户请求以下任务时触发：
- "运行工业时序预测端到端流程"
- "执行时序预测完整流程"
- "跑通 forecasting pipeline"
- "工业时序预测"
- "训练并评估 ITransformer 预测模型"

## 功能范围

1. **数据生成**：自动生成示例时序数据（如不存在）
2. **模型训练**：使用 `ITransformerForecaster` 训练预测模型
3. **模型推理**：在测试窗口上执行预测
4. **指标评估**：使用 `ForecastingMetrics` 计算 MSE/RMSE/MAE/MAPE/SMAPE/MASE/DTW/R²
5. **报告输出**：打印并保存结果报告

## 执行步骤

### 快速执行（推荐）

```bash
python .claude/skills/industrial-forecasting-skill/run_all_pipelines.py
```

该命令会自动：
1. 生成示例数据到 `data/synthetic/`
2. 训练 `ITransformerForecaster`
3. 执行预测并保存结果
4. 计算评估指标并输出报告

### 分步执行

```bash
# 1. 生成示例数据
python .claude/skills/industrial-forecasting-skill/generate_synthetic_data.py

# 2. 运行训练、推理、评估
python .claude/skills/industrial-forecasting-skill/run_forecast_pipeline.py
```

### 使用指定超参数训练（供 HPO / Orchestrator 调用）

本 skill 还提供参数化入口 `run_forecast_with_params.py`，接收 JSON 超参数，训练并返回 8 项指标：

```bash
python .claude/skills/industrial-forecasting-skill/run_forecast_with_params.py \
  --params '{"d_model": 64, "nhead": 2, "num_layers": 1, "dropout": 0.1, \
             "lag_max": 8, "kan_grid_size": 3, "epochs": 10, \
             "batch_size": 32, "lr": 0.001, "trend_weight": 1.0}' \
  --model_tag trial_001 \
  --output -
```

输出为 JSON：

```json
{
  "params": { ... },
  "metrics": {
    "mse": ..., "rmse": ..., "mae": ..., "mape": ...,
    "smape": ..., "mase": ..., "dtw": ..., "r2": ...
  },
  "model_dir": "..."
}
```

该脚本主要用于被 `hebo-forecasting-hpo` 等 orchestrator skill 在超参数寻优循环中调用。

---

### 使用 TSA-Suite CLI

```bash
# 生成数据
python .claude/skills/industrial-forecasting-skill/generate_synthetic_data.py

# 训练
python -m tsas.engine.operator.cli forecasting fit \
  --input data/synthetic/forecast_demo.csv \
  --target target_shuiwei_chazhi \
  --config .claude/skills/industrial-forecasting-skill/configs/forecast_config.yaml \
  --save results/models/forecast_demo

# 推理
python -m tsas.engine.operator.cli forecasting run \
  --input data/synthetic/forecast_demo.csv \
  --config .claude/skills/industrial-forecasting-skill/configs/forecast_config.yaml \
  --load results/models/forecast_demo \
  --output results/forecasts/pred.csv
```

## 输出物

| 路径 | 说明 |
|------|------|
| `data/synthetic/forecast_demo.csv` | 训练/推理/评估数据（单一文件，含 `split` 列与 30 个工业特征） |
| `results/models/forecast_demo/` | 保存的模型 |
| `results/forecasts/pred.csv` | 预测结果（含 y_true / y_pred） |
| `results/metrics/forecast_metrics.json` | 评估指标 JSON |

## 关键指标

- MSE、RMSE、MAE
- MAPE、SMAPE
- MASE（需训练集随机游走基线误差）
- DTW（动态时间规整，fastdtw 为可选依赖）
- R²

## 环境要求

- Python >= 3.11
- torch
- numpy、 pandas、 scikit-learn
- TSA-Suite 已安装或在 `PYTHONPATH` 中

## 注意事项

- 首次运行会自动生成示例数据。
- 默认使用 CPU，可在配置中改为 `cuda` 或 `npu`。
- 结果保存在项目根目录的 `results/` 下。
