---
name: hebo-forecasting-hpo
description: 基于 HEBO 的工业时序预测超参数多目标寻优 orchestrator。通过调用 /bo-for-experiment 和 /industrial-forecasting-skill 实现闭环超参数优化，自身不包含 HEBO 引擎或训练代码。
version: 2.0.0
---

# hebo-forecasting-hpo：工业时序预测超参数 HEBO 多目标寻优 Orchestrator

本 skill 是一个**编排器（orchestrator）**，不直接实现贝叶斯优化或模型训练逻辑。它通过 Claude Code 框架的 Skill 调用机制，协调以下两个子 skill 完成闭环寻优：

- `/bo-for-experiment` — 负责 HEBO 参数空间管理与下一组参数推荐
- `/industrial-forecasting-skill` — 负责工业时序预测模型训练与评估

---

## 触发条件

当用户请求以下任务时触发：
- "用 HEBO 优化工业时序预测超参数"
- "时序预测模型超参数自动寻优"
- "多目标优化预测模型参数"
- "基于 MSE/RMSE/MAE/R² 等指标的预测模型参数推荐"
- 例："帮我自动找一组 ITransformer 超参数，让 MSE 和 R² 都最好"

---

## 功能范围

1. **初始化 HPO 任务**：调用 `/bo-for-experiment` 创建任务，传入预定义的参数空间和优化目标
2. **HEBO 推荐超参数**：调用 `/bo-for-experiment` 的 iterate 模式获取候选参数
3. **训练与评估**：调用 `/industrial-forecasting-skill` 的 `run_forecast_with_params.py` 执行训练并获取 8 项指标
4. **指标反馈**：将指标转换为 HEBO 观测格式，调用 `/bo-for-experiment` 追加记录
5. **循环迭代**：重复步骤 2-4 直到达到最大 trial 数
6. **保存最优参数**：使用 rank-sum 方法综合评分，保存 `{Task_ID}_best_params.json`

---

## 优化目标

| 指标 | 方向 | 说明 |
|------|------|------|
| MSE  | min | 均方误差 |
| RMSE | min | 均方根误差 |
| MAE  | min | 平均绝对误差 |
| MAPE | min | 平均绝对百分比误差 |
| SMAPE| min | 对称平均绝对百分比误差 |
| MASE | min | 平均绝对尺度误差 |
| DTW  | min | 动态时间规整距离 |
| R²   | max | 决定系数（越大越好） |

---

## 可调超参数空间

| 参数名 | 类型 | 范围 | 说明 |
|--------|------|------|------|
| d_model | int | [16, 256] | Transformer 嵌入维度 |
| nhead | int | [1, 8] | 注意力头数 |
| num_layers | int | [1, 4] | Encoder 层数 |
| dropout | num | [0.0, 0.5] | Dropout 比率 |
| lag_max | int | [0, 32] | Lag-Aware 最大滞后步数 |
| kan_grid_size | int | [1, 10] | KAN 网格大小 |
| epochs | int | [5, 50] | 训练轮数 |
| batch_size | int | [16, 128] | 批次大小 |
| lr | pow | [1e-4, 1e-2] | 学习率（对数尺度） |
| trend_weight | num | [0.0, 5.0] | 趋势损失权重 |

---

## 工作流程（Conversation 级别）

当用户在 Claude Code 对话中触发本 skill 时，按以下步骤执行：

### Step 1: 初始化 HPO 任务

调用 `/bo-for-experiment` 的 `init` 模式（非交互）：

```bash
python .claude/skills/bo-for-experiment/main.py --mode init \
  --non_interactive \
  --params_config '[{"name":"d_model","type":"int","lb":16,"ub":256},{"name":"nhead","type":"int","lb":1,"ub":8},{"name":"num_layers","type":"int","lb":1,"ub":4},{"name":"dropout","type":"num","lb":0.0,"ub":0.5},{"name":"lag_max","type":"int","lb":0,"ub":32},{"name":"kan_grid_size","type":"int","lb":1,"ub":10},{"name":"epochs","type":"int","lb":5,"ub":50},{"name":"batch_size","type":"int","lb":16,"ub":128},{"name":"lr","type":"pow","lb":1e-4,"ub":1e-2,"base":10},{"name":"trend_weight","type":"num","lb":0.0,"ub":5.0}]' \
  --objectives '[{"name":"mse","direction":"min"},{"name":"rmse","direction":"min"},{"name":"mae","direction":"min"},{"name":"mape","direction":"min"},{"name":"smape","direction":"min"},{"name":"mase","direction":"min"},{"name":"dtw","direction":"min"},{"name":"r2","direction":"max"}]' \
  --task_id HPO_FORECAST_20260626_143022 \
  --data_dir ./hpo_results
```

### Step 2: 迭代循环

对于每一轮（直到 `max_trials`）：

**2a. 获取推荐参数**

调用 `/bo-for-experiment` 的 `iterate` 模式，使用 `--format json` 以便解析：

```bash
python .claude/skills/bo-for-experiment/main.py --mode iterate \
  --task_id HPO_FORECAST_20260626_143022 \
  --x_new '[...已评估参数...]' \
  --y_new '[...对应指标...]' \
  --n_suggest 2 \
  --data_dir ./hpo_results \
  --format json
```

- 首次迭代：若历史为空，可传入 `--x_new '[]' --y_new '[]'`，`/bo-for-experiment` 会返回随机采样参数。
- 后续迭代：传入所有已累积的 `(params, metrics)` 对。

解析 stdout 最后一行的 JSON，得到 `suggestions` 列表。

**2b. 训练与评估**

对每一组推荐参数，调用 `/industrial-forecasting-skill` 的参数化训练入口：

```bash
python .claude/skills/industrial-forecasting-skill/run_forecast_with_params.py \
  --params '{"d_model": 64, "nhead": 2, "num_layers": 1, ...}' \
  --model_tag trial_0001 \
  --output -
```

捕获 stdout 中的 metrics JSON。

**2c. 反馈**

将 `(params, metrics)` 追加到观测列表中，用于下一轮 iterate 调用。

### Step 3: 输出最优参数

循环结束后，读取 `{Task_ID}_history.json`，对每个指标独立排序（R² 取反使其与越小越好指标方向一致），计算 8 个排名之和，返回 rank-sum 最小的那组参数，并保存为 `{Task_ID}_best_params.json`。

---

## 子 Skill 调用详情

### /bo-for-experiment

| 模式 | 参数 | 说明 |
|------|------|------|
| init | `--mode init --non_interactive --params_config <json> --objectives <json> --task_id <id> --data_dir <dir>` | 创建 HPO 任务 |
| iterate | `--mode iterate --task_id <id> --x_new <json> --y_new <json> --n_suggest <n> --data_dir <dir> --format json` | 追加观测并获取推荐 |

### /industrial-forecasting-skill

| 脚本 | 参数 | 说明 |
|------|------|------|
| `run_forecast_with_params.py` | `--params <json> --model_tag <tag> --output -` | 训练并返回指标 |

---

## 输出文件

| 文件 | 说明 |
|------|------|
| `{Task_ID}_history.json` | 由 `/bo-for-experiment` 维护的完整 HPO 历史 |
| `{Task_ID}_best_params.json` | 由本 orchestrator 生成的综合最优参数（rank-sum 方法） |
| `results/models_hpo/trial_xxxx/` | 每次 trial 的模型（由 `/industrial-forecasting-skill` 生成） |

---

## 环境要求

- Python 3.10+
- `bo-for-experiment` 已配置（HEBO、OpenAI 环境变量等）
- `industrial-forecasting-skill` 已配置（torch、TSA-Suite 等）
- 首次运行前请确保已通过 `/industrial-forecasting-skill` 生成示例数据 `data/synthetic/forecast_demo.csv`

---

## 注意事项

- 本 skill 是**编排器**，自身不包含 HEBO 引擎和模型训练代码；实际工作由 `/bo-for-experiment` 和 `/industrial-forecasting-skill` 完成。
- 本 skill **不再支持** `python .claude/skills/hebo-forecasting-hpo/main.py --mode run` 离线全自助运行。如需脚本化执行，请直接使用 `bo-for-experiment` + `run_forecast_with_params.py`。
- 每次 trial 都会重新训练模型，运行时间与 `max_trials` 和 `n_suggest` 成正比。
- 若某组参数训练失败，该 trial 会被跳过，不影响后续推荐。
- `--max_trials` 建议从 10-20 开始，效果较好时可增加到 50+。
- 当前实现固定使用 CPU 训练，可在 `/industrial-forecasting-skill/run_forecast_with_params.py` 中修改 `DEVICE` 使用 cuda/npu。
