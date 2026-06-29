---
name: bo-for-experiment
description: 贝叶斯优化实验助手。适用于指导迭代寻优及输出下一步推荐参数场景。支持两个核心步骤——①自然语言定义实验参数空间与优化目标，生成《寻优任务建议书》；②接收新的实验观测数据，基于全量历史记录推荐下一组最优实验参数。
version: 1.0.0
---

# bo-for-experiment：贝叶斯优化闭环实验助手

通过 HEBO (Heteroscedastic Evolutionary Bayesian Optimization) 框架，为科研人员提供从"自然语言定义实验空间"到"基于历史数据持续推荐参数"的闭环实验寻优工具。

---

## 触发条件

当用户请求以下任务时触发：
- "实验参数优化"、"实验寻优"、"贝叶斯优化实验"
- "推荐下一组实验参数"、"根据历史实验数据推荐"
- "多目标实验优化"、"实验闭环优化"
- 例："根据之前 10 组实验数据，帮我推荐下一组催化剂合成参数"
- 例："我想用贝叶斯优化来指导我的实验，参数有温度、压力、时间"

---

## 功能范围

1. **任务初始化**：将自然语言描述解析为结构化的参数空间、优化目标和约束
2. **建议书输出**：生成《寻优任务建议书》并等待用户确认
3. **历史记录管理**：以 `{Task_ID}_history.json` 持久化实验记录
4. **参数推荐**：基于全量历史数据调用 HEBO 推荐下一组实验参数
5. **Pareto 简报**：多目标场景下输出 Pareto 前沿分析

---

## 工作流程

### Scenario 1：任务初始化（首次使用）

**触发时机**：用户首次描述实验寻优需求，尚未创建任务。

**步骤**：
1. **收集信息**：从用户描述中提取参数空间和优化目标
2. **解析空间**：调用 LLM 将自然语言转化为结构化的 HEBO 参数配置
3. **展示建议书**：输出《寻优任务建议书》，包含：
   - 参数表（名称、类型、范围/类别）
   - 目标表（目标名、优化方向 min/max）
   - 约束说明（如有）
4. **等待确认**：展示建议书后阻塞等待用户输入，**绝不提前写文件**：
   - 直接回车 / 输入"确认" → 创建任务
   - 输入修改意见（任意文字）→ 追加到描述后重新解析，再次展示，继续等待（支持多轮累积修改）
   - 输入"取消" → 退出，不创建任务
5. **持久化**：用户确认后创建唯一 `Task_ID`，初始化 `{Task_ID}_history.json`

**执行命令**：

```bash
python skills/bo-for-experiment/main.py --mode init \
  --description "催化剂合成：温度200-400°C，压力1-10bar，催化剂用量0.1-2g，目标最大化转化率"
```

**非交互模式（适用于 orchestrator 调用）**：

当由其他 skill（如 `hebo-forecasting-hpo`）编排调用时，可直接提供结构化的参数空间与优化目标，跳过 LLM 解析和确认循环：

```bash
python skills/bo-for-experiment/main.py --mode init \
  --non_interactive \
  --params_config '[{"name":"temperature","type":"num","lb":200,"ub":400},{"name":"pressure","type":"num","lb":1,"ub":10},{"name":"catalyst_amount","type":"num","lb":0.1,"ub":2.0}]' \
  --objectives '[{"name":"conversion_rate","direction":"max"}]' \
  --task_id BO20260626_143022 \
  --data_dir ./experiments
```

**输出示例**：

```
============================================================
  [寻优任务建议书]
============================================================

### 参数空间

| 参数名 | 类型 | 下界 | 上界 / 类别 | 备注 |
|--------|------|------|-------------|------|
| temperature | num | 200 | 400 | 连续数值 |
| pressure | num | 1 | 10 | 连续数值 |
| catalyst_amount | num | 0.1 | 2.0 | 连续数值 |

### 优化目标

| 目标名 | 优化方向 | 说明 |
|--------|---------|------|
| conversion_rate | 最大化 (max) | 反应转化率 |

### 约束条件

（无约束）

============================================================
  [提示] 请检查以上解析是否符合您的实验设计。
  输入 "确认" 创建任务，或告诉我需要修改的地方。
============================================================

  [确认] 直接按回车 或 输入 '确认' — 创建任务
  [修改] 输入修改意见（如'把温度上限改为500'）— 重新解析
  [退出] 输入 '取消' — 放弃，不创建任务

请输入您的选择:
```

---

### Scenario 2：数据迭代与参数推荐

**触发时机**：用户已有 `Task_ID`，并提供了新的实验观测结果；如果没有 `Task_ID`，则触发 Scenario 1，然后将用户输入再匹配到构建的任务中。

**步骤**：
1. **载入历史**：读取 `{Task_ID}_history.json`
2. **追加数据**：将新观测数据（参数值 + 目标值）写入历史记录
3. **样本检查**：若历史记录 < 5 条，输出模型不稳定警告
4. **拟合模型**：重新实例化 GeneralBO，注入所有历史数据
5. **生成推荐**：调用 `suggest()` 输出下一组实验参数
6. **Pareto 简报**：多目标时额外输出帕累托前沿分析

**执行命令**：

```bash
# 单条输入
python skills/bo-for-experiment/main.py --mode iterate \
  --task_id BO20260415_143022 \
  --x_new '{"temperature": 300, "pressure": 5, "catalyst_amount": 1.0}' \
  --y_new '{"conversion_rate": 0.85}' \
  --n_suggest 3

# 批量输入（一次提交多组实验结果）
python skills/bo-for-experiment/main.py --mode iterate \
  --task_id BO20260415_143022 \
  --x_new '[{"temperature": 300, "pressure": 5, "catalyst_amount": 1.0}, {"temperature": 350, "pressure": 7, "catalyst_amount": 1.5}]' \
  --y_new '[{"conversion_rate": 0.85}, {"conversion_rate": 0.71}]' \
  --n_suggest 3

# 指定历史文件目录（默认当前工作目录）
python skills/bo-for-experiment/main.py --mode iterate \
  --task_id BO20260415_143022 \
  --x_new '{"temperature": 300, "pressure": 5, "catalyst_amount": 1.0}' \
  --y_new '{"conversion_rate": 0.85}' \
  --data_dir ./experiments

# JSON 输出（便于 orchestrator 解析）
python skills/bo-for-experiment/main.py --mode iterate \
  --task_id BO20260415_143022 \
  --x_new '{"temperature": 300, "pressure": 5, "catalyst_amount": 1.0}' \
  --y_new '{"conversion_rate": 0.85}' \
  --n_suggest 3 \
  --format json

# 空历史时直接随机推荐（orchestrator 首次迭代）
python skills/bo-for-experiment/main.py --mode iterate \
  --task_id BO20260415_143022 \
  --x_new '[]' \
  --y_new '[]' \
  --n_suggest 3 \
  --format json
```

**输出示例**：

```
[数据更新] 已追加 1 条新记录（历史累计 4 条）(2026-04-15 14:35:10)
[警告] 当前历史数据仅 4 条 (<5)，代理模型可能不稳定，推荐结果仅供参考。

[推荐的下一组实验参数]:

| # | temperature | pressure | catalyst_amount |
|---|------------|---------|----------------|
| 1 | 352.3 | 7.8 | 1.45 |
| 2 | 280.1 | 9.2 | 0.85 |
| 3 | 375.0 | 6.5 | 1.80 |

[文件] 历史记录已更新：BO20260415_143022_history.json (4 条记录)
```

---

## 参数类型参考

| 类型 | 说明 | 示例 |
|------|------|------|
| `num` | 连续数值型 | 温度 200-400°C |
| `int` | 整数型 | 反应步骤数 1-5 |
| `pow` | 对数尺度（如学习率） | 浓度 1e-4 到 1e-1（base=10）|
| `cat` | 类别型 | 溶剂类型 ['乙醇', '丙酮', '甲醇'] |
| `bool` | 布尔型 | 是否加热搅拌 True/False |

---

## 输出文件说明

| 文件 | 说明 |
|------|------|
| `{Task_ID}_history.json` | 完整历史记录（参数空间定义 + 所有实验记录 + 时间戳） |

---

## 参考文档

- `skills/bo-for-experiment/references/hebo_api.md` — HEBO DesignSpace / GeneralBO 接口说明
- `skills/bo-for-experiment/references/experiment_guide.md` — 实验容错与健壮性指南
- `skills/bo-for-experiment/references/scenario_examples.md` — 完整使用示例（单目标 / 多目标）

---

## 环境要求

```bash
# Python 环境：建议使用 Python 3.10+
pip install openai pandas numpy

# HEBO 从源码安装：
pip install -e path/to/HEBO
```

环境变量（OpenAI 兼容协议）：

```bash
export OPENAI_API_KEY=your_key
export OPENAI_BASE_URL=https://...
export OPENAI_MODEL=your_model_name
```

---

## 注意事项

- **首次使用**必须先执行 Scenario 1 创建任务；之后每轮实验后执行 Scenario 2
- **目标方向**：HEBO 内部统一最小化；`max` 目标会自动取负值传入，展示时还原
- **样本数建议**：< 5 条时模型不稳定；5-10 条可用但不确定性大；10 条以上效果较好
- **`--task_id` 的历史文件**默认保存在执行命令的当前目录；可用 `--data_dir` 指定路径
- 多目标时，推荐结果会展示 Top-5 均衡解（rank-sum 方法）及各目标最优解
- **`--non_interactive` 模式**：适用于被其他 skill 编排调用，必须同时提供 `--params_config` 和 `--objectives`；不会调用 LLM，也不会阻塞等待用户输入
- **`--format json` 模式**：`iterate` 模式下 stdout 最后一行会输出 `{"suggestions": [...], "task_id": "...", "n_records": N}`，便于 orchestrator 解析；默认 `text` 模式保持原有 Markdown 表格输出
- **空历史随机推荐**：当历史记录为空时，`iterate` 可接收 `--x_new '[]' --y_new '[]'`，直接返回 `n_suggest` 组随机参数