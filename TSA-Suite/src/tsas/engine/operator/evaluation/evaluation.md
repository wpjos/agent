# 评价指标算子开发指南

本模块实现了**评价指标算子基础类型**（BaseMetricOperator），基于 BaseOperator 扩展，新增 `scores()` 方法供 HPO 单目标/多目标优化统一调用。

本文档旨在指导开发者基于 `base.py` 框架开发新的评价指标算子，并提供已实现算子的使用参考。

**目录结构**：

```
src/tsas/engine/operator/evaluation/
├── __init__.py             # 包导出（MR, MC, BaseMetricConfig, BaseMetricOperator）
├── base.py                 # 基础类型定义（开发核心）
├── binary_classification.py # 二分类离散标签指标
├── binary_curve.py         # 二分类曲线指标（连续分数输入）
├── multi_classification.py # 多分类指标
├── point_adjust.py         # 点调整指标（PA-F1）
├── self_evaluation.py      # 无标签自评估指标
└── evaluation.md           # 本文档
```

---

## 0. 已实现算子概览

本模块已提供以下评价指标算子，可直接使用：

| 算子类                      | 输入类型                                  | MR 类型                        | 功能描述                      |
|---------------------------|----------------------------------------|------------------------------|-----------------------------|
| `BinaryClassificationMetric` | `tuple[np.ndarray, np.ndarray]`      | `BinaryClassificationResult`  | 二分类离散标签指标（F1, FAR, MCC） |
| `BinaryClassificationCurve` | `tuple[np.ndarray, np.ndarray]`      | `BinaryClassificationCurveResult` | 二分类曲线指标（AUC-ROC, AUC-PR, Best-F1） |
| `MultipleClassificationMetric` | `tuple[np.ndarray, np.ndarray]`   | `MultiClassificationMetricResult` | 多分类指标（Macro 平均 + Per-Class） |
| `PointAdjust`              | `tuple[np.ndarray, np.ndarray]`      | `PointAdjustResult`           | 点调整指标（PA-F1，时序异常检测）   |
| `SelfEvaluation`           | `np.ndarray`                         | `float`                       | 无标签自评估（变异系数 + Sigmoid） |

### 0.1 BinaryClassificationMetric 使用示例

```python
from tsas.engine.operator.evaluation import BinaryClassificationMetric

# 实例化（默认 positive_label 为非 0/False/None/"")
op = BinaryClassificationMetric()

# 计算
y_truth = np.array([0, 1, 0, 1, 1])
y_predict = np.array([0, 1, 1, 1, 0])
result = op.run((y_truth, y_predict))

# 访问指标
print(result.f1)      # F1 值
print(result.far)     # 故障误报率（FPR）
print(result.mcc)     # Matthews 相关系数
print(result.confusion_matrix)  # 混淆矩阵 [[TN, FP], [FN, TP]]

# HPO 集成：提取优化目标
from tsas.engine.operator.evaluation.binary_classification import BinaryClassificationConfig

op = BinaryClassificationMetric(config=BinaryClassificationConfig(main_scores={"f1": "f1", "far": "far"}))
scores = op.scores((y_truth, y_predict))  # -> {"f1": 0.75, "far": 0.5}
```

### 0.2 BinaryClassificationCurve 使用示例

```python
from tsas.engine.operator.evaluation import BinaryClassificationCurve

# 输入：真实标签 + 异常分数（连续）
labels = np.array([0, 0, 1, 1, 1])
scores = np.array([0.1, 0.3, 0.5, 0.7, 0.9])

op = BinaryClassificationCurve()
result = op.run((labels, scores))

print(result.auc_roc)       # ROC 曲线面积
print(result.auc_pr)        # PR 曲线面积
print(result.best_f1)       # 最优 F1 值
print(result.best_f1_threshold)  # 最优 F1 对应阈值

# HPO 集成
from tsas.engine.operator.evaluation.binary_curve import BinaryClassificationCurveConfig

op = BinaryClassificationCurve(config=BinaryClassificationCurveConfig(main_scores={"auc": "auc_roc", "f1": "best_f1"}))
scores = op.scores((labels, scores))  # -> {"auc": 0.87, "f1": 0.67}
```

### 0.3 MultipleClassificationMetric 使用示例

```python
from tsas.engine.operator.evaluation import MultipleClassificationMetric

y_truth = np.array([0, 1, 2, 0, 1])
y_predict = np.array([0, 1, 1, 0, 2])

op = MultipleClassificationMetric()
result = op.run((y_truth, y_predict))

print(result.accuracy)     # 总准确率（Macro）
print(result.precision)    # Macro 精确率
print(result.per_label_metrics[0].f1)  # 第 0 类的 F1
print(result.confusion_matrix)     # k×k 混淆矩阵
```

### 0.4 PointAdjust 使用示例

```python
from tsas.engine.operator.evaluation import PointAdjust

# 输入：真实标签 + 预测标签（离散 0/1）
y_truth = np.array([0, 0, 1, 1, 1, 0, 0])  # 异常段 [2:5]
y_predict = np.array([0, 0, 0, 1, 0, 0, 0])  # 仅第 3 点检测到

op = PointAdjust()
result = op.run((y_truth, y_predict))

# PA 算法：若异常段内任一点被检测 → 整段为 TP
print(result.pa_f1)        # PA-F1（应高于普通 F1）
print(result.pa_recall)    # PA-Recall
print(result.pa_precision) # PA-Precision
```

### 0.5 SelfEvaluation 使用示例

```python
from tsas.engine.operator.evaluation import SelfEvaluation

# 输入：异常分数（无标签场景）
scores = np.array([0.01, 0.02, 0.95, 0.98])

op = SelfEvaluation()
result = op.run(scores)  # -> float（变异系数经 Sigmoid 映射）

# HPO 集成
scores_dict = op.scores(scores)  # -> {"self_eval": 0.73}
```

---

## 1. 核心设计理念

### 1.1 架构定位

评价指标算子位于检测/预测管线的末端，负责对算法输出进行量化评估。它是一个**无状态纯函数**——不需要训练能力，不使用 `LearnableOperatorMixin`。

```
输入数据（真实值 + 预测值）
        ↓
BaseMetricOperator._run()
        ↓
指标结果（MR: float 或 BaseModel）
        ↓  （可选）
BaseMetricOperator.scores()  ← HPO 优化目标提取
        ↓
dict[str, float]  ← 供 Optuna 等优化框架使用
```

### 1.2 两种 MR 形态

指标结果类型（MR）支持两种形态：

| MR 形态          | 适用场景                 | `main_scores` 路径 | 示例                                       |
|----------------|----------------------|------------------|------------------------------------------|
| `float`        | 单一标量指标（F1、AUC、变异系数等） | `"_"` 占位符        | `main_scores={"f1": "_"}`                |
| `BaseModel` 子类 | 结构化指标（含多字段的完整评估结果）   | 属性路径（支持点分嵌套）     | `main_scores={"f1": "f1", "far": "far"}` |

> **CLI Help 自动渲染**：当 MR 是 `BaseModel` 子类时，算子的 `_output_type`
> 类属性会自动填充为该类型（由 `BaseOperator.__init_subclass__` 通过多层泛型
> 追踪从 MR 等价于 `BaseOperator` 的 O 位置提取，无需 `BaseMetricOperator`
> 定制）。CLI Help 会渲染 `### 主输出 ({MR 类名})` 标题，并在其后追加
> `**结构**：` 字段表，开发者无需在 docstring 中重复字段定义。
> 详见 `base.md` 的 "5.1.2 主输入/输出类型推断" 小节。

### 1.3 泛型参数

算子使用四个泛型参数 **`[I, MR, MC, RP]`**：

```python
class MyMetricOp(BaseMetricOperator[I, MR, MC, RP]):
    ...
```

| 参数   | 含义             | 约束                                    | 常见填写                                           |
|------|----------------|---------------------------------------|------------------------------------------------|
| `I`  | 输入类型           | 无约束                                   | `tuple[np.ndarray, np.ndarray]` 或 `np.ndarray` |
| `MR` | 指标结果类型         | bound `Union[float, BaseModel]`       | `float` 或 Pydantic `BaseModel` 子类              |
| `MC` | 实例参数类型（Config） | bound `Union[BaseMetricConfig, None]` | `BaseMetricConfig` 子类                          |
| `RP` | 运行参数类型         | 无约束                                   | `None`                                         |

### 1.4 Config 体系

所有评价指标算子的 Config 必须继承 `BaseMetricConfig`：

```python
from tsas.engine.operator.evaluation import BaseMetricConfig


class MyConfig(BaseMetricConfig):
    # 子类特有参数
    positive_label: int = 1
    decimals: int = 6
    
    # 重写 main_scores 默认值
    main_scores: dict[str, str] | None = {"f1": "f1", "far": "far"}
```

**`BaseMetricConfig` 关键字段**：

| 字段            | 类型                       | 默认值    | 说明                                                       |
|---------------|--------------------------|--------|----------------------------------------------------------|
| `main_scores` | `dict[str, str] \| None` | `None` | 主评分路径映射。`None` 时 `scores()` 返回 None；非 None 时按路径从 MR 提取标量 |

**配置实例为 frozen 模式**（`ConfigDict(frozen=True)`），创建后不可修改。

### 1.5 `scores()` 方法与 HPO 集成

`scores()` 方法是评价指标算子的核心创新，用于从完整指标结果中提取 HPO 所需的标量字典：

```python
def scores(self, x, *, params=None, **kwargs) -> dict[str, float] | None:
    if self.config is None or self.config.main_scores is None:
        return None
    result = self.run(x, params=params, **kwargs)
    return self._extract_scores(result, self.config.main_scores)
```

**关键行为**：

- `config` 为 `None` 或 `config.main_scores` 为 `None` → 返回 `None`
- `config.main_scores` 非 `None` → 调用 `run()` 后按映射提取，返回 `dict[str, float]`

**路径提取规则**（`_resolve_path`）：

- `"_"` → 直接返回结果对象本身（适用于 `float` MR）
- `"f1"` → 返回 `obj.f1`（单层属性）
- `"macro.f1"` → 返回 `obj.macro.f1`（点分嵌套属性）

---

## 2. 算子开发

### 2.1 新增简单标量指标算子（MR=float）

**适用场景**：指标结果为单一标量值（如变异系数、均值误差）。

```python
import numpy as np
from tsas.engine.operator.evaluation import BaseMetricConfig, BaseMetricOperator


class CVConfig(BaseMetricConfig):
    """变异系数指标配置"""
    # main_scores 路径为 "_"，因为 MR=float
    main_scores: dict[str, str] | None = {"cv": "_"}


class CVMetricOp(BaseMetricOperator[np.ndarray, float, CVConfig, None]):
    """变异系数指标算子
    
    输入: 一维时序数据 np.ndarray
    输出: 变异系数值 float（标准差 / 均值）
    """

    @classmethod
    def name(cls) -> str:
        return "cv_metric"

    @classmethod
    def version(cls) -> tuple[int, ...]:
        return (1, 0, 0)

    def _run(self, x: np.ndarray, *, params) -> float:
        return float(np.std(x) / np.mean(x))
```

**使用方式**：

```python
op = CVMetricOp()
result = op.run(np.array([1.0, 2.0, 3.0]))  # -> 0.4082...
scores = op.scores(np.array([1.0, 2.0, 3.0]))  # -> {"cv": 0.4082...}
```

### 2.2 新增结构化指标算子（MR=BaseModel）

**适用场景**：指标结果包含多个字段（如二分类指标含 TP、FP、TN、FN、F1、FAR 等）。

```python
from pydantic import BaseModel
import numpy as np
from tsas.engine.operator.evaluation import BaseMetricConfig, BaseMetricOperator


class BinaryResult(BaseModel):
    """二分类指标结果"""
    tp: int
    fp: int
    tn: int
    fn: int
    accuracy: float
    precision: float
    recall: float
    f1: float
    far: float


class BinaryConfig(BaseMetricConfig):
    """二分类指标配置"""
    positive_label: int = 1
    decimals: int = 6
    # 指定 HPO 优化目标
    main_scores: dict[str, str] | None = {"f1": "f1", "far": "far"}


class BinaryMetricOp(BaseMetricOperator[tuple[np.ndarray, np.ndarray], BinaryResult, BinaryConfig, None]):
    """二分类评价指标算子
    
    输入: (y_truth, y_predict) 标签数组对
    输出: BinaryResult 结构化指标
    """

    @classmethod
    def name(cls) -> str:
        return "binary_metric"

    @classmethod
    def version(cls) -> tuple[int, ...]:
        return (1, 0, 0)

    def _run(self, x: tuple[np.ndarray, np.ndarray], *, params) -> BinaryResult:
        y_truth, y_predict = x
        # ... 计算逻辑 ...
        return BinaryResult(
            tp=tp, fp=fp, tn=tn, fn=fn,
            accuracy=accuracy, precision=precision,
            recall=recall, f1=f1, far=far,
        )
```

**使用方式**：

```python
op = BinaryMetricOp()
result = op.run((y_truth, y_predict))  # -> BinaryResult(...)
scores = op.scores((y_truth, y_predict))  # -> {"f1": 0.85, "far": 0.12}
```

### 2.3 新增无 Config 的指标算子（MC=None）

**适用场景**：算子不需要配置参数。

```python
from tsas.engine.operator.evaluation import BaseMetricOperator


class SimpleMetricOp(BaseMetricOperator[list[float], float, None, None]):
    """无 Config 的简单均值指标算子"""

    @classmethod
    def name(cls) -> str:
        return "simple_metric"

    @classmethod
    def version(cls) -> tuple[int, ...]:
        return (1, 0, 0)

    def _run(self, x: list[float], *, params) -> float:
        return float(sum(x) / len(x))
```

**注意**：`MC=None` 时 `scores()` 始终返回 `None`，因为不存在 `config.main_scores`。

### 2.4 新增多层嵌套 MR 的指标算子

**适用场景**：指标结果有层级结构（如 macro/micro 聚合、多级指标嵌套）。

```python
from pydantic import BaseModel
from tsas.engine.operator.evaluation import BaseMetricConfig, BaseMetricOperator


class SubMetrics(BaseModel):
    f1: float
    auc: float


class NestedResult(BaseModel):
    macro: SubMetrics
    micro: SubMetrics


class NestedConfig(BaseMetricConfig):
    # 使用点分路径提取嵌套属性
    main_scores: dict[str, str] | None = {
        "macro_f1": "macro.f1",
        "micro_auc": "micro.auc",
    }


class NestedMetricOp(BaseMetricOperator[list[float], NestedResult, NestedConfig, None]):

    @classmethod
    def name(cls) -> str:
        return "nested_metric"

    @classmethod
    def version(cls) -> tuple[int, ...]:
        return (1, 0, 0)

    def _run(self, x, *, params) -> NestedResult:
        return NestedResult(
            macro=SubMetrics(f1=0.9, auc=0.95),
            micro=SubMetrics(f1=0.85, auc=0.88),
        )
```

**使用方式**：

```python
op = NestedMetricOp()
scores = op.scores([1.0, 2.0, 3.0])
# -> {"macro_f1": 0.9, "micro_auc": 0.88}
```

---

## 3. HPO 集成

### 3.1 单目标优化

```python
from tsas.engine.operator.evaluation import BaseMetricConfig, BaseMetricOperator


class F1Config(BaseMetricConfig):
    main_scores: dict[str, str] | None = {"f1": "f1"}


class F1Op(BaseMetricOperator[tuple, float, F1Config, None]):

    @classmethod
    def name(cls) -> str:
        return "f1_metric"

    @classmethod
    def version(cls) -> tuple[int, ...]:
        return (1, 0, 0)

    def _run(self, x, *, params):
        return 0.85  # 实际计算

op = F1Op()
scores = op.scores(x)
objective = scores["f1"]  # 供 Optuna maximize 使用
```

### 3.2 多目标优化（Pareto 前沿）

```python
class BinaryConfig(BaseMetricConfig):
    main_scores: dict[str, str] | None = {"f1": "f1", "far": "far"}


class BinaryOp(BaseMetricOperator[tuple, BinaryResult, BinaryConfig, None]):

    @classmethod
    def name(cls) -> str:
        return "binary_op"

    @classmethod
    def version(cls) -> tuple[int, ...]:
        return (1, 0, 0)

    def _run(self, x, *, params):
        return BinaryResult(f1=0.85, far=0.12, ...)

op = BinaryOp()
scores = op.scores(x)
# Optuna directions=["maximize", "minimize"]
objectives = (scores["f1"], -scores["far"])
```

### 3.3 自定义 main_scores 覆盖默认

`main_scores` 可在实例化时覆盖 Config 的默认值：

```python
# Config 默认提取 f1 + far，但用户只想优化 precision
op = BinaryMetricOp(main_scores={"precision": "precision"})
scores = op.scores(x)  # -> {"precision": 0.75}
```

---

## 4. 继承与多层派生

### 4.1 通过中间抽象类派生

支持通过中间抽象类进行多层继承，Config 类型会自动从具体类的泛型参数中提取：

```python
from typing import TypeVar, Generic
from abc import ABC

from tsas.engine.operator.evaluation import BaseMetricOperator, BaseMetricConfig

C = TypeVar("C", bound=BaseMetricConfig)
RP = TypeVar("RP")


class AbstractBinaryOp(
    BaseMetricOperator[tuple[list, list], BinaryResult, C, RP],
    Generic[C, RP],
    ABC,
):
    """中间抽象类，延迟绑定 Config 和 RP"""
    pass


class ConcreteBinaryOp(AbstractBinaryOp[BinaryConfig, None]):
    """具体实现类"""

    @classmethod
    def name(cls) -> str:
        return "concrete_binary_op"

    @classmethod
    def version(cls) -> tuple[int, ...]:
        return (1, 0, 0)

    def _run(self, x, *, params):
        return BinaryResult(f1=0.9, far=0.1, ...)


# ConcreteBinaryOp._config_type 自动提取为 BinaryConfig
```

### 4.2 Mixin 组合

评价指标算子支持与普通 Mixin 类组合，不影响 Config 提取：

```python
class LoggingMixin:
    """日志 Mixin"""
    pass


class LoggedOp(
    LoggingMixin,
    BaseMetricOperator[list[float], float, MyConfig, None],
):

    @classmethod
    def name(cls) -> str:
        return "logged_op"

    @classmethod
    def version(cls) -> tuple[int, ...]:
        return (1, 0, 0)

    def _run(self, x, *, params):
        return float(sum(x))
```

---

## 5. 关键注意事项

### 5.1 Config 类型约束

- Config 必须继承 `BaseMetricConfig`（运行时由 `__init_subclass__` 验证）
- Config 为 `None` 时不验证（`MC=None` 合法）
- 使用非 `BaseMetricConfig` 子类的 Pydantic `BaseModel` 会导致 `TypeError`

### 5.2 `scores()` 返回 None 的场景

以下场景 `scores()` 返回 `None`，HPO 编排层应做相应处理：

1. `MC=None`（算子无 Config）
2. `config.main_scores=None`（未配置提取路径）
3. 实例化时显式传入 `main_scores=None`

此时应直接使用 `run()` 获取完整指标结果。

### 5.3 `_run` 方法签名

子类必须实现 `_run` 方法，签名为：

```python
def _run(self, x, *, params) -> MR:
    ...
```

- `x` 为输入数据（类型由 `I` 泛型参数决定）
- `params` 为运行参数（类型由 `RP` 泛型参数决定，通常为 `None`）
- 返回值为指标结果（类型由 `MR` 泛型参数决定）

### 5.4 `main_scores` 路径的正确性

- `float` MR 只能使用 `"_"` 路径
- `BaseModel` MR 的路径必须与结果类属性对应
- 无效路径会在运行时抛出 `AttributeError`
- 建议在单元测试中覆盖 `scores()` 的正常和异常路径

### 5.5 算子实例化方式

评价指标算子的实例化方式继承自 `BaseOperator`：

```python
# 使用默认 Config（自动实例化）
op = MyMetricOp()

# 覆盖 Config 默认值
op = MyMetricOp(config=MyConfig(main_scores={"f1": "f1"}))

# 传入自定义 Config 实例
op = MyMetricOp(config=MyConfig(main_scores={"f1": "f1"}, positive_label=1))

# 显式指定算子实例标识后缀
op = MyMetricOp(oid="my_metric", config=MyConfig(main_scores={"f1": "f1"}))
```

### 5.6 测试要求

评价指标算子的单元测试需满足：

- 测试通过率 100%
- 代码覆盖率 > 80%
- 测试文件与源码文件一一映射

运行覆盖率命令：

```bash
python -m pytest tests/test_engine_operator/test_evaluation/ --cov=tsas.engine.operator.evaluation.base --cov-report=term-missing -q
```

---

## 6. API 参考

### 6.1 `BaseMetricConfig`

```python
class BaseMetricConfig(BaseModel):
    model_config = ConfigDict(frozen=True)
    main_scores: dict[str, str] | None = None
```

### 6.2 `BaseMetricOperator[I, MR, MC, RP]`

| 方法                  | 签名                                                          | 说明                             |
|---------------------|-------------------------------------------------------------|--------------------------------|
| `scores()`          | `(x, *, params=None, **kwargs) -> dict[str, float] \| None` | 按 main_scores 提取命名标量字典         |
| `_run()`            | `(x, *, params) -> MR`                                      | **子类实现**，完成指标计算                |
| `_extract_scores()` | `(result, main_scores) -> dict[str, float]`                 | 从 MR 中按映射提取标量字典                |
| `_resolve_path()`   | `(obj, path: str) -> Any`                                   | 静态方法，按路径从对象中提取属性值（`"_"` 或点分路径） |

### 6.3 泛型类型变量

| 名称   | 定义                                                   | 用途             |
|------|------------------------------------------------------|----------------|
| `MR` | `TypeVar("MR", bound=Union[float, BaseModel])`       | 指标结果类型         |
| `MC` | `TypeVar("MC", bound=Union[BaseMetricConfig, None])` | 实例参数（Config）类型 |
