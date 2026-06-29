# HPO（超参数优化）模块开发指南

本模块基于 Optuna 实现异常检测算子的自动化超参数搜索，支持 Detector/Scorer（含 Composite）的自动超参寻优。HPO 模块已从检测算子模块中独立，作为通用优化能力提供给上层使用。

本文档旨在指导开发者理解和使用 HPO 模块。

**目录结构**：

```
src/tsas/engine/hpo/
├── __init__.py          # 模块入口，导出公共 API
├── search_hint.py      # SearchHint 标记 + 搜索空间提取
├── result.py           # HPOResult + TrialInfo 结果容器
├── trainer.py          # HPOTrainer 编排器
└── hpo.md              # 本文档
```

---

## 1. 模块概述

HPO 模块提供三个核心组件：

| 组件                    | 文件               | 功能                                       |
|-----------------------|--------------------|------------------------------------------|
| `SearchHint` + `extract_search_space` | search_hint.py | 基于 Pydantic 原生约束的搜索空间声明与提取 |
| `HPOTrainer`          | trainer.py         | HPO 编排器，自动搜索最优超参数组合                   |
| `HPOResult` / `TrialInfo` | result.py         | 优化结果数据容器                                 |

---

## 2. 可搜索超参声明

### 2.1 核心设计：Pydantic 原生约束 + 零侵入

搜索空间声明基于 Pydantic 原生 `Field(ge/le/gt/lt)` 和 `Enum`/`Literal` 类型注解，零侵入。
绝大多数场景下无需额外标注——搜索范围直接由 Pydantic 原生约束表达。

仅在需要 log 尺度采样、非1步长等 Pydantic 原生无法表达的分布参数时，通过 `Annotated` 注入 `SearchHint` 标记。

### 2.2 基本用法（95% 场景 — 零标注）

```python
from typing import Literal
from pydantic import BaseModel, Field


class MyConfig(BaseModel):
    # 数值型: Field(ge/le) 直接定义搜索范围
    threshold: float = Field(default=3.0, ge=1.0, le=10.0)
    n_neighbors: int = Field(default=5, ge=1, le=20)

    # 离散型: Enum 或 Literal 自动提取候选值
    metric: Literal["euclidean", "manhattan"] = "euclidean"

    # 固定参数（无 ge/le/Literal/Enum 约束，不参与搜索）
    name: str = "default"
```

### 2.3 高级用法（5% 场景 — SearchHint）

```python
from typing import Annotated
from pydantic import BaseModel, Field
from tsas.engine.hpo import SearchHint


class AdvancedConfig(BaseModel):
    # log 尺度采样
    learning_rate: Annotated[float, Field(default=0.001, ge=1e-5, le=1e-1),
                             SearchHint(log=True)]
    # 非1步长
    batch_size: Annotated[int, Field(default=32, ge=8, le=256),
                          SearchHint(step=8)]
```

### 2.4 搜索空间类型

| 类型   | Pydantic 声明               | 映射为 Optuna 方法              |
|------|---------------------------|---------------------------|
| 连续型 | `Field(ge=..., le=...)`  | `trial.suggest_float`     |
| 整数型 | `Field(ge=..., le=...)`  | `trial.suggest_int`       |
| 离散型 | `Literal[...]` / `Enum`    | `trial.suggest_categorical` |

**注意**：数值型字段必须有 `ge`/`le`/`gt`/`lt` 中至少一个边界约束才会参与搜索；无约束的字段自动跳过。

### 2.5 搜索空间提取

`extract_search_space` 从 Config 类中自动提取所有可搜索字段：

```python
from tsas.engine.hpo import extract_search_space

space = extract_search_space(MyConfig)
# {"threshold": {"type": "float", "low": 1.0, "high": 10.0, "default": 3.0},
#  "n_neighbors": {"type": "int", "low": 1, "high": 20, "step": 1, "default": 5},
#  "metric": {"type": "cat", "choices": ["euclidean", "manhattan"]}}
```

`extract_search_space_from_operator` 从算子实例递归提取（支持 Composite 嵌套）：

```python
from tsas.engine.hpo import extract_search_space_from_operator

space = extract_search_space_from_operator(composite_detector)
# {"predictor.n_components": {...}, "scorer_0.metric": {...}, "decider.percentile": {...}}
```

### 2.6 Optuna 映射

`config_to_optuna_suggestions` 将搜索空间映射为 Optuna trial 建议调用：

```python
import optuna
from tsas.engine.hpo import config_to_optuna_suggestions

study = optuna.create_study(direction="maximize")
def objective(trial):
    params = config_to_optuna_suggestions(trial, search_space)
    detector = KNNDetector(**params)
    ...
```

---

## 3. HPO 编排器（HPOTrainer）

### 3.1 核心流程

```
1. 从算子的 Config 中提取搜索空间（或使用自定义搜索空间）
2. 对每个 Optuna trial:
   a. 根据搜索空间采样超参数
   b. 用采样的超参数构建算子实例
   c. 在训练数据上 fit
   d. 在验证数据上 run 并计算评估分数
3. 排序并返回最优 TopK 结果
```

### 3.2 使用示例

```python
from tsas.engine.hpo import HPOTrainer
from tsas.engine.operator.detection.zscore import ZScoreDetector
from tsas.engine.operator.evaluation.binary_classification import BinaryClassificationMetric

metric_op = BinaryClassificationMetric()
trainer = HPOTrainer(ZScoreDetector, metric_op, n_trials=50, top_k=3)
result = trainer.fit(train_data, val_labels=val_labels, val_split=0.3)

# 访问结果
print(result.best_params)       # 最优参数
print(result.best_score)        # 最优分数（dict）
print(result.best_score_value)  # 最优主分数值（float）
print(result.best_operator)     # 最优算子实例
```

### 3.3 HPOTrainer 参数

| 参数             | 默认值          | 说明                                          |
|----------------|--------------|---------------------------------------------|
| `operator`     | 必填           | 算子类或算子实例                                    |
| `metric_op`    | 必填           | 评估指标算子实例（BaseMetricOperator 子类）             |
| `search_space` | `None`       | 自定义搜索空间，None 时自动从 Config 提取               |
| `directions`   | `"maximize"` | 优化方向，支持单目标字符串或多目标列表                       |
| `sampler`      | `"tpe"`      | Optuna 采样器：`"tpe"` / `"random"` / `"grid"` |
| `n_trials`     | `100`        | 搜索试验次数                                      |
| `time_limit`   | `None`       | 时间限制（秒），None 表示不限                          |
| `pruning`      | `False`      | 是否启用 Optuna MedianPruner 剪枝               |
| `top_k`        | `None`       | 返回最优 TopK 结果，None 时单目标返回1个               |
| `random_seed`  | `42`         | 随机种子                                        |

### 3.4 支持的采样器

| 采样器       | 参数值       | 说明                |
|-----------|-----------|-------------------|
| TPE       | `"tpe"`   | Tree-structured Parzen Estimator，适合大多数场景 |
| 随机搜索     | `"random"` | 随机搜索，适合基线对比        |
| 网格搜索     | `"grid"`  | 网格搜索，适合小搜索空间       |

### 3.5 验证策略

支持三种验证方式：
- 独立验证集: `val_data` 参数
- 训练集内切分: `val_split` 参数（如 `0.3` 表示 30% 验证）
- K-Fold 交叉验证: `cv_folds` 参数

### 3.6 结果访问

| 属性             | 类型          | 说明                     |
|----------------|-------------|------------------------|
| `best_params`  | dict        | 最优试验的超参数组合              |
| `best_score`   | dict[str, float] | 最优试验的各指标分数           |
| `best_score_value` | float   | 最优试验的主分数值               |
| `best_operator` | BaseOperator | None | 最优试验对应的已训练算子实例     |
| `all_trials`   | list[TrialInfo] | 全量试验记录               |
| `best_trials`  | list[TrialInfo] | TopK 最优试验列表          |

---

## 4. 关键注意事项

### 4.1 搜索空间声明原则

- 优先使用 Pydantic 原生 `Field(ge/le)` 和 `Enum`/`Literal` 表达搜索范围
- 仅在需要 log 采样或非1步长时使用 `SearchHint`
- 数值型字段无边界约束时不参与搜索（自动跳过）

### 4.2 Optuna 延迟导入

HPO 模块对 Optuna 采用延迟导入策略，避免非 HPO 场景下的依赖加载。

### 4.3 Composite 算子支持

HPOTrainer 支持 Composite 算子的递归搜索空间提取和重建，子算子参数自动添加层级前缀：
- Predictor: `predictor.`
- Scorer: `scorer_0.`, `scorer_1.`, ...
- Decider: `decider.`

### 4.4 覆盖率工具

```bash
coverage run -m pytest tests/test_engine_hpo/ --tb=short
coverage report --include="src/tsas/engine/hpo/*" --show-missing
```
