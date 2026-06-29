# 时序异常检测算子开发指南

本模块实现了**四层异常检测算子架构**（Predictor → Scorer → Decider → Detector），采用"组合优于继承"的 Mixin 组合设计原则，支持 `DataFrame` + `ndarray` 双类型输入输出。

HPO（超参数优化）能力已独立为 `tsas.engine.hpo` 模块，详见 `hpo/hpo.md`。

本文档旨在指导开发者基于四层基类框架开发新的异常检测算法算子。

**目录结构**：

```
src/tsas/engine/operator/detection/
├── base.py                    # 四层基类定义 + Mixin（开发核心）
├── composite.py             # 组合算子（CompositeScorer / CompositeDetector）
├── mean_predictor.py        # 均值预测器（Predictor 示例）
├── pca.py                   # PCA 预测器 + 评分器 + 检测器
├── residual_scorer.py       # 残差评分器（BiNumericOperator Scorer 示例）
├── mean_scorer.py           # 均值评分器（NumericOperator Scorer 示例）
├── threshold_decider.py     # 固定阈值决策器（Decider 示例）
├── percentile_decider.py    # 百分位阈值决策器（Decider 示例）
├── zscore.py                # Z-Score 评分器 + 检测器
├── knn.py                   # KNN 评分器 + 检测器
├── xihe.py                  # 羲和异常检测评分器（预训练模型示例）
├── cicada.py                # CICADA 预测器（Mixture-of-Experts + MAML 重构型 Predictor）
└── detection.md             # 本文档
```

HPO 模块目录结构：

```
src/tsas/engine/hpo/
├── __init__.py          # 模块入口，导出公共 API
├── search_hint.py      # SearchHint 标记 + 搜索空间提取
├── result.py           # HPOResult + TrialInfo 结果容器
├── trainer.py          # HPOTrainer 编排器
└── hpo.md              # HPO 模块开发指南
```

---

## 1. 核心设计理念

### 1.1 四层架构与数据流

```
第1层  Predictor（预测器）
        ↓ 预测值
第2层  Scorer（评分器）←── SingleScorerMixin / MultiScorerMixin
        ↓ 异常分数（越大越异常）
第3层  Decider（决策器）
        ↓ 二分类标签（0/1）
第4层  Detector（检测器）── 组合 Scorer + Decider
```

开发者需要理解的关键分层：

| 层级 | 基类 / Mixin                                                   | 是否可训练 | 你需要做什么                               |
|----|--------------------------------------------------------------|-------|--------------------------------------|
| 1  | `BasePredictorMixin` + `NumericOperator`                       | ✅     | 实现 `_fit_data` / `_run_data`，输出预测值   |
| 2  | `SingleScorerMixin` / `MultiScorerMixin` + `NumericOperator` | ✅     | 实现 `_fit_data` / `_run_data`，输出异常分数  |
| 3  | `BaseDeciderMixin` + `NumericOperator`                       | 可选    | 实现 `_fit_data` / `_run_data`，分数 → 标签 |
| 4  | Detector                                                     | ✅     | 组合 Scorer + Decider（内部持有子组件实例）       |

### 1.2 Mixin 组合模式

新架构采用 Mixin 组合代替深层继承。每个算子通过选择需要的 Mixin 来组合能力：

| Mixin                              | 提供的能力                  |
|------------------------------------|------------------------|
| `UnsupervisedNumericOperatorMixin` | 无监督训练（`fit(x)`）        |
| `SupervisedNumericOperatorMixin`   | 有监督训练（`fit(x, y)`）     |
| `SingleScorerMixin`                | 单分数输出（列名 `["score"]`）  |
| `MultiScorerMixin`                 | 多变量分数输出（列名继承输入）        |
| `NumericBatchRunMixin`             | 批量推理生成器能力（`batch_run`） |
| `BasePredictorMixin`              | 预测器输出（列名继承输入）        |
| `BaseDeciderMixin`                 | 决策器输出（列名 `["label"]`）  |
| `NumericOperator`                  | 单输入算子基类                |
| `BiNumericOperator`                | 双输入算子基类（tuple 输入）      |

**组合示例**：

```python
# 直接评分器：单分数 + 无监督训练 + 数值算子
class ZScoreScorer(SingleScorerMixin[None],
                   UnsupervisedNumericOperatorMixin[None],
                   NumericOperator[None, ZScoreScorerConfig, None]):


# 检测器：无监督训练 + 决策能力 + 数值算子
class ZScoreDetector(UnsupervisedNumericOperatorMixin[None],
                     BaseDeciderMixin[None],
                     NumericOperator[None, ZScoreDetectorConfig, None]):
```

### 1.3 泛型参数

算子使用三个泛型参数 **`[EO, C, RP]`**（无训练参数 FP）：

```python
class MyScorer(SingleScorerMixin[RP], UnsupervisedNumericOperatorMixin[FP],
               NumericOperator[EO, C, RP]):
    ...
```

| 参数   | 含义               | 常见填写                    |
|------|------------------|-------------------------|
| `EO` | 扩展输出类型，承载结构化附加信息 | `None` 或 `BaseModel` 子类 |
| `C`  | 实例参数类型（Config）   | 必须为 `BaseModel` 子类      |
| `RP` | 运行参数类型           | `None`                  |

### 1.4 DataFrame + ndarray 双类型支持

**核心约定：子类的 `_fit_data` / `_run_data` 始终接收 `np.ndarray`，无需关心 DataFrame。**

模板方法管线自动完成：

1. `_validate_input` → 验证输入类型
2. `_filter_data` → 列名筛选
3. `_unwrap_data` → DataFrame → ndarray + DataFrameMeta
4. `_adjust_data` → 数据预处理
5. `_run_data` → 核心计算（子类实现）
6. `_wrap_data` → ndarray → DataFrame（如输入为 DataFrame）

**开发者只需实现纯 ndarray 逻辑即可。**

### 1.5 Config 声明与可搜索超参

每个算子需要一个 Pydantic `BaseModel` 作为 Config。搜索空间声明基于 Pydantic 原生 `Field(ge/le/gt/lt)` 和 `Enum`/`Literal` 类型注解，零侵入。绝大多数场景下无需额外标注：

```python
from enum import Enum
from pydantic import BaseModel, Field


class DistanceMetric(str, Enum):
    """距离度量方式枚举"""
    EUCLIDEAN = "euclidean"
    MANHATTAN = "manhattan"


class MyConfig(BaseModel):
    # 数值型可搜索超参 → Field(ge/le) 直接定义搜索范围，HPO 时自动搜索
    threshold: float = Field(default=3.0, ge=1.0, le=10.0)
    n_neighbors: int = Field(default=5, ge=1, le=20)

    # 离散型可搜索超参 → str, Enum 自动提取候选值，HPO 时自动搜索
    metric: DistanceMetric = Field(default=DistanceMetric.EUCLIDEAN, description="距离度量方式")

    # 固定参数（无 ge/le/Enum 约束，不参与搜索）
    name: str = "default"
```

#### 离散值类型选择原则（`str, Enum` vs `Literal`）

对于取值有限的离散型参数，**必须**使用 `str, Enum` 或 `Literal` 进行约束，**绝对禁止**使用裸 `str` 配合 `description` 文本说明合法值。

**优先推荐**使用 `str, Enum`，仅在满足以下全部条件时可使用 `Literal`：

- 选项 ≤ 3 个
- 无需为每个选项提供行内注释
- 无跨 Config 复用需求

| 方式          | 适用场景                            | 示例                                   |
|-------------|---------------------------------|--------------------------------------|
| `str, Enum` | **优先推荐**。选项较多、需行内注释、跨 Config 共享 | `KNNDistanceMetric(str, Enum)`       |
| `Literal`   | 仅 ≤3 个选项且无复用需求                  | `Literal["offline", "online"]`       |
| 裸 `str`     | **绝对禁止**                        | ~~`str = Field(description="...")`~~ |

> **HPO 兼容性**：`str, Enum` 和 `Literal` 在 HPO 搜索空间提取中行为完全一致，均自动提取为 `choices` 列表，映射为 Optuna `suggest_categorical`。详见 `tsas/engine/hpo/hpo.md`。

仅在需要 log 尺度采样、非1步长等特殊语义时，通过 `Annotated` 注入 `SearchHint`：

```python
from typing import Annotated
from tsas.engine.hpo import SearchHint


class AdvancedConfig(BaseModel):
    # log 尺度采样
    learning_rate: Annotated[float, Field(default=0.001, ge=1e-5, le=1e-1),
    SearchHint(log=True)]
    # 非1步长
    batch_size: Annotated[int, Field(default=32, ge=8, le=256),
    SearchHint(step=8)]
```

> **注意**：数值型字段必须有 `ge`/`le`/`gt`/`lt` 中至少一个边界约束才会参与 HPO 搜索；无约束的字段自动跳过。详细说明请参阅 `tsas/engine/hpo/hpo.md`。

### 1.6 输出格式

`_run_data` 返回值有两种形式：

- **仅主输出**：`np.ndarray`（纯数组，无扩展输出）
- **主输出 + 扩展输出**：`tuple[np.ndarray, EO]`

模板方法会自动判断返回类型并做相应处理。

---

## 2. 算子开发

### 2.1 新增 DirectScorer 算子（如 IForest）

**适用场景**：算法自身即可直接计算异常分数。

**步骤**：创建文件 → 定义 Config → Mixin 组合定义 Scorer → 组合定义 Detector

```python
# iforest.py
import numpy as np
from enum import Enum
from pydantic import BaseModel, Field

from tsas.engine.operator.detection.base import (
    SingleScorerMixin,
    UnsupervisedNumericOperatorMixin,
    BaseDeciderMixin,
    NumericOperator,
)
from tsas.engine.operator.detection.percentile_decider import PercentileDecider


class IForestScorerConfig(BaseModel):
    """IForest 评分器配置"""
    n_estimators: int = Field(
        default=100, ge=10, le=500,
        description="树的数量"
    )
    contamination: float = Field(
        default=0.1, ge=0.01, le=0.5,
        description="异常比例"
    )


class IForestScorer(SingleScorerMixin[None],
                    UnsupervisedNumericOperatorMixin[None],
                    NumericOperator[None, IForestScorerConfig, None]):
    """Isolation Forest 直接评分器"""

    @classmethod
    def name(cls) -> str:
        return "iforest_scorer"

    @classmethod
    def version(cls) -> tuple[int, ...]:
        return (1, 0, 0)

    def __init__(self, *, oid=None, config=None, **kwargs):
        super().__init__(oid=oid, config=config, **kwargs)
        self._model = None

    def _fit_data(self, x: np.ndarray, params: None) -> None:
        from sklearn.ensemble import IsolationForest
        self._model = IsolationForest(
            n_estimators=self.config.n_estimators,
            contamination=self.config.contamination,
            random_state=42,
        )
        self._model.fit(x)

    def _run_data(self, x: np.ndarray, params: None) -> np.ndarray:
        # decision_function: 越小越异常 → 取反使其越大越异常
        raw_scores = self._model.decision_function(x)
        return -raw_scores

    def _save_fit_state(self, path):
        """持久化 sklearn 模型"""
        import pickle
        super()._save_fit_state(path)
        with open(path / '_model.pkl', 'wb') as f:
            pickle.dump(self._model, f)

    def _load_fit_state(self, path):
        """恢复 sklearn 模型"""
        import pickle
        super()._load_fit_state(path)
        with open(path / '_model.pkl', 'rb') as f:
            self._model = pickle.load(f)
        self._fitted = True


class IForestDetectorConfig(BaseModel):
    """IForest 检测器配置"""
    n_estimators: int = Field(default=100, ge=10, le=500)
    contamination: float = Field(default=0.1, ge=0.01, le=0.5)
    percentile: float = Field(default=95.0, ge=50.0, le=99.9)


class IForestDetector(UnsupervisedNumericOperatorMixin[None],
                      BaseDeciderMixin[None],
                      NumericOperator[None, IForestDetectorConfig, None]):
    """IForest 检测器 — 组合 IForestScorer + PercentileDecider"""

    @classmethod
    def name(cls) -> str:
        return "iforest_detector"

    @classmethod
    def version(cls) -> tuple[int, ...]:
        return (1, 0, 0)

    def __init__(self, *, oid=None, config=None, **kwargs):
        super().__init__(oid=oid, config=config, **kwargs)
        scorer_config = IForestScorerConfig(
            n_estimators=self.config.n_estimators,
            contamination=self.config.contamination,
        )
        self._scorer = IForestScorer(config=scorer_config)
        self._decider = PercentileDecider(config=PercentileDeciderConfig(percentile=self.config.percentile))

    def _fit_data(self, x: np.ndarray, params: None) -> None:
        self._scorer.fit(x)
        scores, _ = self._scorer.run(x)
        self._decider.fit(scores)

    def _run_data(self, x: np.ndarray, params: None) -> np.ndarray:
        scores, _ = self._scorer.run(x)
        labels, _ = self._decider.run(scores)
        return labels
```

### 2.2 新增 Predictor 算子（如 AutoEncoder）

**适用场景**：算法需要先预测再评分。

创建自定义 Predictor，再用 `BiNumericOperator` 创建 Scorer 来比较预测值和真实值：

```python
# Step 1: 创建 Predictor
class AEPredictor(UnsupervisedNumericOperatorMixin[None],
                  BasePredictorMixin[None, AEPredictorConfig, None],
                  NumericOperator[None, AEPredictorConfig, None]):

    @classmethod
    def name(cls) -> str:
        return "ae_predictor"

    @classmethod
    def version(cls) -> tuple[int, ...]:
        return (1, 0, 0)

    def _fit_data(self, x: np.ndarray, params: None) -> None:
        # 训练 AutoEncoder...
        pass

    def _run_data(self, x: np.ndarray, params: None) -> np.ndarray:
        reconstructed = self._decoder(self._encoder(x))
        return reconstructed

# Step 2: 使用 BiNumericOperator 的 ResidualScorer 来计算残差
```

### 2.3 新增中间层组件（Decider）

**Decider 示例**：

```python
class TopKDeciderConfig(BaseModel):
    k: int = Field(default=10, gt=0)


class TopKDecider(BaseDeciderMixin[None],
                  UnsupervisedNumericOperatorMixin[None],
                  NumericOperator[None, TopKDeciderConfig, None]):

    @classmethod
    def name(cls) -> str:
        return "topk_decider"

    @classmethod
    def version(cls) -> tuple[int, ...]:
        return (1, 0, 0)

    def _fit_data(self, x: np.ndarray, params: None) -> None:
        pass  # 无需训练

    def _run_data(self, x: np.ndarray, params: None) -> np.ndarray:
        labels = np.zeros(len(x), dtype=int)
        topk_indices = np.argsort(x.ravel())[-self.config.k:]
        labels[topk_indices] = 1
        return labels
```

---

## 3. 组合算子

组合算子提供了将多个子算子按顺序串联成一个完整检测流程的能力，是构建复杂异常检测管线的关键组件。

### 3.1 概述

组合算子包括：

- **CompositeScorer**：将 0 或 1 个 Predictor + 1 或 K 个 Scorer 串联成一个评分器
- **CompositeDetector**：将 0 或 1 个 Predictor + 1 或 K 个 Scorer + 1 个 Decider 串联成一个检测器

```
CompositeScorer 数据流:
    [Predictor] → Scorer_1 → Scorer_2 → ... → Scorer_K → 异常分数

CompositeDetector 数据流:
    [Predictor] → Scorer_1 → ... → Scorer_K → Decider → 二分类标签
```

### 3.2 算子类型与数据流规则

组合算子支持两类输入算子：

| 算子类型              | 基类                | 数据输入                        | 适用场景           |
|-------------------|-------------------|-----------------------------|----------------|
| NumericOperator   | NumericOperator   | `prev_output`               | 单输入算子，如 ZScore |
| BiNumericOperator | BiNumericOperator | `(prev_input, prev_output)` | 双输入算子，如残差计算    |

**数据流规则**：

1. **Predictor 位置**：如果存在，必须在算子列表第 0 位
2. **Decider 位置**：如果存在，必须在算子列表最后一位
3. **BiNumericOperator 约束**：第一个算子若是 BiNumericOperator，则必须有 Predictor 提供 `x_pred`
4. **内部数据格式**：全程 ndarray 传递，1D 分数自动 reshape 为 2D

### 3.3 使用示例

#### 3.3.1 CompositeScorer 示例

```python
from tsas.engine.operator.detection.composite import CompositeScorer
from tsas.engine.operator.detection.pca import PCAPredictor, PCAPredictorConfig
from tsas.engine.operator.detection.residual_scorer import ResidualScorer, ResidualScorerConfig
from tsas.engine.operator.detection.zscore import ZScoreScorer
from tsas.engine.operator.detection.knn import KNNScorer

# 示例1: Predictor + Scorer（重构型评分）
scorer = CompositeScorer(operators=[
    PCAPredictor(config=PCAPredictorConfig(n_components=5)),
    ResidualScorer(config=ResidualScorerConfig(metric="mse")),
])
scorer.fit(train_data)
scores, eo = scorer.run(test_data)  # 输出 1D 分数

# 示例2: 多 Scorer 串行（无 Predictor）
scorer = CompositeScorer(operators=[
    ZScoreScorer(),
    KNNScorer(),
])
scorer.fit(train_data)
scores, eo = scorer.run(test_data)

# 示例3: Predictor + 多 Scorer 串行
scorer = CompositeScorer(operators=[
    PCAPredictor(config=PCAPredictorConfig(n_components=5)),
    ResidualScorer(config=ResidualScorerConfig(metric="mse")),  # BiNumericOperator
    ZScoreScorer(),  # NumericOperator
])
scorer.fit(train_data)
scores, eo = scorer.run(test_data)
```

#### 3.3.2 CompositeDetector 示例

```python
from tsas.engine.operator.detection.composite import CompositeDetector
from tsas.engine.operator.detection.pca import PCAPredictor, PCAPredictorConfig
from tsas.engine.operator.detection.residual_scorer import ResidualScorer, ResidualScorerConfig
from tsas.engine.operator.detection.percentile_decider import PercentileDecider, PercentileDeciderConfig
from tsas.engine.operator.detection.threshold_decider import ThresholdDecider, ThresholdDeciderConfig
from tsas.engine.operator.detection.zscore import ZScoreScorer

# 示例1: 完整检测管线（Predictor + Scorer + Decider）
detector = CompositeDetector(operators=[
    PCAPredictor(config=PCAPredictorConfig(n_components=5)),
    ResidualScorer(config=ResidualScorerConfig(metric="mse")),
    PercentileDecider(config=PercentileDeciderConfig(percentile=95.0)),
])
detector.fit(train_data)
labels, eo = detector.run(test_data)  # 输出二分类标签

# 示例2: Scorer + Decider（无 Predictor）
detector = CompositeDetector(operators=[
    ZScoreScorer(),
    PercentileDecider(config=PercentileDeciderConfig(percentile=95.0)),
])
detector.fit(train_data)
labels, eo = detector.run(test_data)

# 示例3: Predictor + Decider（无 Scorer）
detector = CompositeDetector(operators=[
    PCAPredictor(config=PCAPredictorConfig(n_components=2)),
    ThresholdDecider(config=ThresholdDeciderConfig(threshold=0.5)),
])
detector.fit(train_data)
labels, eo = detector.run(test_data)
```

### 3.4 配置校验规则

#### CompositeScorer 校验规则

| 配置                               | 校验结果 | 原因                             |
|----------------------------------|------|--------------------------------|
| `[]`                             | ❌    | 算子列表不能为空                       |
| `[Scorer]`（单个）                   | ❌    | 无组合意义，请直接使用该 Scorer            |
| `[Predictor]`                    | ❌    | 无法产生异常分数                       |
| `[BiNumericScorer]`（无 Predictor） | ❌    | BiNumericOperator 需要 x_pred 来源 |
| `[Predictor, Scorer]`            | ✅    | 合法配置                           |
| `[Scorer1, Scorer2]`             | ✅    | 多 Scorer 串行                    |
| `[Predictor, Scorer1, Scorer2]`  | ✅    | 合法配置                           |

#### CompositeDetector 校验规则

| 配置                             | 校验结果 | 原因                   |
|--------------------------------|------|----------------------|
| `[]`                           | ❌    | 算子列表不能为空             |
| `[Scorer]`（无 Decider）          | ❌    | 必须有 1 个 Decider      |
| `[Decider]`（单个）                | ❌    | 无组合意义，请直接使用该 Decider |
| `[Decider, Scorer]`            | ❌    | Decider 必须在最后一位      |
| `[Predictor, Scorer, Decider]` | ✅    | 合法配置                 |
| `[Scorer, Decider]`            | ✅    | 合法配置                 |
| `[Predictor, Decider]`         | ✅    | 合法配置                 |

### 3.5 扩展输出（EO）聚合

组合算子会自动聚合所有子算子的扩展输出：

```python
# CompositeScorer 返回 CompositeScorerExtraOutput
scores, eo = scorer.run(test_data)
print(eo.outputs)  # list[BaseModel | None]，长度等于算子数量

# CompositeDetector 返回 CompositeDetectorExtraOutput
labels, eo = detector.run(test_data)
print(eo.outputs)  # list[BaseModel | None]，长度等于算子数量
```

无 EO 的子算子对应位置为 `None`。

### 3.6 持久化

组合算子的 `save` 方法会为每个子算子创建独立子目录：

```python
detector = CompositeDetector(operators=[
    PCAPredictor(config=PCAPredictorConfig(n_components=5)),
    ResidualScorer(config=ResidualScorerConfig(metric="mse")),
    PercentileDecider(config=PercentileDeciderConfig(percentile=95.0)),
])
detector.fit(train_data)
detector.save("./model_dir")

# 目录结构：
# ./model_dir/
# ├── predictor/
# ├── scorer_0/
# └── decider/
```

**注意**：`load` 方法需要在子类中覆写以实现完整加载逻辑，因为基类无法知道具体的算子类型。`load` 的签名中参数为 `oid`（非 `name`），与 `__init__` 的 `oid` 参数一致，用于自定义算子实例唯一标识后缀。

### 3.7 已实现算子列表

| 算子                | 类型        | 输入类型              | 是否可训练  | 有 EO |
|-------------------|-----------|-------------------|--------|------|
| PCAPredictor      | Predictor | NumericOperator   | ✅      | ✅    |
| MeanPredictor     | Predictor | NumericOperator   | ✅      | ❌    |
| CICADAPredictor   | Predictor | NumericOperator   | ✅      | ❌    |
| PCAScorer         | Scorer    | NumericOperator   | ✅      | ✅    |
| ZScoreScorer      | Scorer    | NumericOperator   | ✅      | ❌    |
| KNNScorer         | Scorer    | NumericOperator   | ✅      | ✅    |
| MeanScorer        | Scorer    | NumericOperator   | ❌      | ❌    |
| ResidualScorer    | Scorer    | BiNumericOperator | ❌      | ✅    |
| XiHeGammaScorer   | Scorer    | NumericOperator   | ❌(预训练) | ✅    |
| PCADetector       | Detector  | NumericOperator   | ✅      | ✅    |
| PercentileDecider | Decider   | NumericOperator   | ✅      | ✅    |
| ThresholdDecider  | Decider   | NumericOperator   | ❌      | ❌    |

---

## 4. HPO 使用

### 4.1 运行 HPO

```python
from tsas.engine.hpo import HPOTrainer
from tsas.engine.operator.detection.zscore import ZScoreDetector
from tsas.engine.operator.evaluation.binary_classification import BinaryClassificationMetric

metric_op = BinaryClassificationMetric()
trainer = HPOTrainer(ZScoreDetector, metric_op, n_trials=50, top_k=3)
result = trainer.fit(train_data, val_labels=val_labels, val_split=0.3)

# 访问结果
print(result.best_params)  # 最优参数
print(result.best_score)  # 最优分数（dict）
print(result.best_score_value)  # 最优主分数值（float）
print(result.best_operator)  # 最优算子实例
```

> **注意**：HPO 模块的详细说明请参阅 `tsas/engine/hpo/hpo.md`。

---

## 5. 关键注意事项

### 5.1 Detector 的训练流程

自定义 Detector 通过组合模式在 `_fit_data` 中实现训练——通常需要：

1. 训练 Scorer
2. 用训练数据计算训练分数
3. 用训练分数训练 Decider

### 5.2 `_fitted` 标志

Mixin 的 `_fit` 模板方法会在调用 `_fit_data` 后自动设置 `self._fitted = True`。**子类的 `_fit_data` 不需要手动设置 `_fitted`。**

### 5.3 扩展输出 EO

- 需要附加信息 → 定义 `BaseModel` 子类作为 EO，在 `_run_data` 中构造并返回 `tuple[np.ndarray, EO]`
- 不需要 → 直接返回 `np.ndarray`

### 5.3.1 预判附加输出

调用方可在实例化前通过 `has_extra_output()` 类方法预判 `run()` 的返回形态：

```python
if scorer.has_extra_output():
    result, eo = scorer.run(data)  # tuple[NumericData, EO]
else:
    result = scorer.run(data)  # NumericData
```

### 5.4 save / load

可训练算子支持持久化。**推荐覆写 `_save_fit_state(path)` 和 `_load_fit_state(path)` 钩子方法**以保存额外状态（如模型权重、训练统计量等），而非直接覆写 `save` / `load`。

**叶节点算子**（如 ZScoreScorer、KNNScorer、PCAPredictor、PercentileDecider）：覆写钩子方法直接保存自有状态（如 `np.savez` 保存 numpy 数组，`pickle` 保存 sklearn 对象）。

**组合算子**（如 KNNDetector、PCADetector）：覆写钩子方法，通过子目录分发调用子组件的 `save()` / `load()`：
```python
def _save_fit_state(self, path):
    super()._save_fit_state(path)
    self._scorer.save(path / '_scorer')
    self._decider.save(path / '_decider')

def _load_fit_state(self, path):
    super()._load_fit_state(path)
    self._scorer = KNNScorer.load(path / '_scorer')
    self._decider = PercentileDecider.load(path / '_decider')
    self._fitted = True
```

> **重要**：`_fitted` 状态不会自动恢复，`_load_fit_state` 中**必须**手动设置 `self._fitted = True`。

### 5.5 覆盖率工具

由于 numpy 双版本冲突，使用以下方式运行覆盖率：

```bash
coverage run -m pytest tests/test_engine_operator/test_detection/ --tb=short
coverage report --include="src/tsas/engine/operator/detection/*" --show-missing
```
