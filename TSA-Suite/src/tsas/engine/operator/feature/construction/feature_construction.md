# 特征构造算子模块开发指南 (Feature Construction)

本文档旨在指导开发者基于 `feature/construction` 模块开发和使用新的特征构造算子。

## 1. 架构概览

特征构造模块基于统一的算子接口（`BaseOperator` 和 `LearnableOperatorMixin`），采用了**多维度正交组合混入类型（Mixin）+ 模板方法模式**的设计。整体包含两个核心组件：

1. **特征算子基类体系 (`base.py`)**：通过组合“列关系”、“行关系”和“是否可训练”这三个维度，提供了 8 个供开发者直接继承的编排基类。基类自动处理了参数校验、列遍历、滑动窗口切片、填充（Padding）与对齐（Alignment）等底层逻辑。
2. **统一命令行 (`cli/`)**：提供基于包扫描的自动算子注册发现机制和统一命令行调用入口，通过声明式配置批量协调并执行特征算子。

---

## 2. 核心概念与基类选择

在开发新的特征算子时，你需要根据算子的计算特性，从以下 8 个编排基类中选择最合适的一个进行继承。

### 2.1 维度解析

* **列关系（Column Strategy）**:
    * **独立单列 (`Independent`)**：表示输出中每列只与输入中的一列相关的语义约定。框架将 `input_columns` 对应的**全部列作为一个完整 ndarray** 传入 `compute`（不会按列拆分逐个调用），列间独立性由 `compute` 实现保证（推荐利用 NumPy 的 `axis` 参数沿列方向进行独立计算）。框架负责根据输入列数和输出列数自动分组命名（输出列数量必须为输入列数量的整数倍）。
    * **多列联合 (`Joint`)**：计算依赖于多列的联合信息（如协方差、距离等）。`compute` 接收多列的 NumPy 数组，输出列名通过 `_name_output_columns` 方法统一命名。
* **行关系（Row Strategy）**:
    * **逐行映射 (`Map`)**：一行输入对应一行输出，行与行之间不产生时序依赖。
    * **滑动窗口 (`Window`)**：计算基于时序滑动窗口。基类自动根据配置的 `window_size`、`padding` 和 `alignment` 进行滑动切窗。
* **是否可训练（Learnability）**:
    * **不可训练 (`BaseFeature`)**：纯计算算子（如求均值、对数、差分等）。
  * **可训练 (`LearnableFeature`)**：需要根据训练数据进行拟合学习（Fit）的算子（如 PCA、Scaler 等）。

### 2.2 八个核心编排基类

开发者可以直接继承以下基类（泛型参数中的 `C` 代表你的 Config 类；`FS` 代表特征状态类型，无状态算子传 `None`）：

1. **不可训练类**（泛型参数仅为 `[C]`）：
    * `IndependentMapFeature[C]`
    * `IndependentWindowFeature[C]`
    * `JointMapFeature[C]`
    * `JointWindowFeature[C]`
2. **可训练类**（泛型参数为 `[C, FS]`）：
    * `LearnableIndependentMapFeature[C, FS]`
    * `LearnableIndependentWindowFeature[C, FS]`
    * `LearnableJointMapFeature[C, FS]`
    * `LearnableJointWindowFeature[C, FS]`

---

## 3. 开发新的特征算子

### 3.1 步骤一：定义 Config 类

特征的配置必须继承 `BaseFeatureConfig`，若是 `Window` 类型的特征，则必须继承 `WindowFeatureConfig`。

```python
from pydantic import Field
from tsas.engine.operator.feature.construction.base import BaseFeatureConfig, WindowFeatureConfig


# 针对 Map 类型
class SquareConfig(BaseFeatureConfig):
    pass  # 基类已经包含了 input_columns 字段


# 针对 Window 类型
class RollingMeanConfig(WindowFeatureConfig):
    # 基类已经包含了 input_columns, window_size, padding, alignment
    # 可在子类添加特有参数
    min_periods: int = Field(default=1)
```

**关于 Window 模式的参数说明**:

* `padding`: 支持 `None` (不填充，输出长度减少)、`Padding.EDGE` (边界值)、`Padding.NAN`、`Padding.REFLECT` (镜像)、`Padding.RING` (首尾相接)，以及直接传入数值（如 `0`、`3.14`）。
* `alignment`: 支持 `Alignment.RIGHT` (右对齐，当前时刻与历史时刻构成窗口) 和 `Alignment.LEFT` (左对齐)。

### 3.2 步骤二：继承编排基类并实现 `compute` 和命名方法

所有的特征算子都必须实现静态方法 `compute(x: np.ndarray, *, state=None, **params)` 和输出列命名方法。

**不可训练的独立映射算子示例 (SquareFeature)**：

```python
import numpy as np
from tsas.engine.operator.feature.construction.base import (
    BaseFeatureConfig, IndependentMapFeature
)


class SquareConfig(BaseFeatureConfig):
    pass


class SquareFeature(IndependentMapFeature[SquareConfig]):
    @classmethod
    def name(cls) -> str:
        return "square_feature"

    @classmethod
    def version(cls) -> tuple[int, ...]:
        return (1, 0, 0)

    @staticmethod
    def compute(x: np.ndarray, *, state=None, **params) -> np.ndarray:
        # x 是 input_columns 对应列的完整 NumPy 数组（全部行，可能多列）
        # Independent 模式下列间独立性由 compute 自行保证（本例利用广播机制）
        return x ** 2

    def _name_output_column(self, input_col: str, output_val) -> str:
        # 生成标准化的输出列名 "{源列名}_{特征名}"
        return self._make_output_column_name(input_col, "square")
```

**不可训练的独立窗口算子示例 (RollingMeanFeature)**：

```python
from tsas.engine.operator.feature.construction.base import (
    WindowFeatureConfig, IndependentWindowFeature
)


class RollingMeanConfig(WindowFeatureConfig):
    pass  # 基类已包含 window_size, padding, alignment


class RollingMeanFeature(IndependentWindowFeature[RollingMeanConfig]):
    @classmethod
    def name(cls) -> str:
        return "rolling_mean_feature"

    @classmethod
    def version(cls) -> tuple[int, ...]:
        return (1, 0, 0)

    @staticmethod
    def compute(x: np.ndarray, *, state=None, **params) -> float:
        # x 是大小为 window_size 的切片（由 Window 机制保证）
        # Independent 模式下 x 可能包含多列，列间独立性由 compute 自行保证
        # 窗口计算返回单个数值
        return float(np.mean(x))

    def _name_output_column(self, input_col: str, output_val) -> str:
        return self._make_output_column_name(input_col, "rolling_mean")
```

### 3.3 针对可训练特征算子 (Learnable Feature)

如果你继承的是 `Learnable...` 相关的基类，需要实现以下内容：

1. **`compute` 静态方法**：接收 `state` 参数用于推理
2. **`train` 静态方法**：覆盖默认训练逻辑，返回状态对象（Pydantic BaseModel 子类）
3. **`_get_train_params` 方法**（可选）：向 `train` 传递额外参数
4. **`_name_output_columns` 方法**：定义输出列名
5. **`save` / `load` 方法**：持久化训练状态

可训练特征算子的训练流程由基类 `LearnableFeature._fit` 模板方法自动编排：输入校验 → 列筛选 → 数据解包 → 调用 `train` → 保存状态到 `_state` → 标记已训练。开发者无需覆写 `_fit`，只需实现 `train` 静态方法即可。

```python
from typing import Self
from pathlib import Path

import numpy as np
from pydantic import BaseModel, ConfigDict, Field

from tsas.engine.operator.base import DataFrameMeta
from tsas.engine.operator.feature.construction.base import (
    BaseFeatureConfig, LearnableJointMapFeature
)


class PCAConfig(BaseFeatureConfig):
    """PCA 降维特征的 Config"""
    n_components: int = Field(gt=0)


class PCAState(BaseModel):
    """PCA 训练状态

    使用 ``arbitrary_types_allowed`` 以支持 numpy 数组类型。
    """
    model_config = ConfigDict(arbitrary_types_allowed=True)

    mean: np.ndarray
    components: np.ndarray


class PCAFeature(LearnableJointMapFeature[PCAConfig, PCAState]):

    @classmethod
    def name(cls) -> str:
        return "pca_feature"

    @classmethod
    def version(cls) -> tuple[int, ...]:
        return (1, 0, 0)

    @staticmethod
    def train(x: np.ndarray, **params) -> PCAState:
        """基于训练数据学习 PCA 状态

        由基类 ``LearnableFeature._fit`` 自动调用，
        无需开发者手动调用或覆写 ``_fit``。
        """
        n_components = params.get("n_components", 2)
        mean = x.mean(axis=0)
        centered = x - mean
        cov = np.cov(centered, rowvar=False)
        eigenvalues, eigenvectors = np.linalg.eigh(cov)
        idx = np.argsort(eigenvalues)[::-1][:n_components]
        components = eigenvectors[:, idx]
        return PCAState(mean=mean, components=components)

    def _get_train_params(self):
        """向 ``train`` 传递额外参数"""
        return {"n_components": self.config.n_components}

    @staticmethod
    def compute(x: np.ndarray, *, state=None, **params) -> np.ndarray:
        # state 由基类自动传入 PCAState 对象
        if state is None:
            raise ValueError("PCA 需要先训练")
        centered = x - state.mean
        return centered @ state.components

    def _name_output_columns(self, output_data: np.ndarray, meta: DataFrameMeta | None, params: None) -> list[str]:
        n_components = output_data.shape[1] if output_data.ndim > 1 else 1
        return [f"pca_{i}" for i in range(n_components)]

    def save(self, path: str | Path):
        """持久化 PCA 算子"""
        super().save(path)  # 保存 config 等基类信息
        path = Path(path)
        if self._state is not None:
            np.save(path / "pca_mean.npy", self._state.mean)
            np.save(path / "pca_components.npy", self._state.components)

    @classmethod
    def load(cls, path: str | Path, *, name: str | None = None) -> Self:
        """从指定目录加载 PCA 算子"""
        instance = super().load(path, name=name)
        path = Path(path)
        mean_file = path / "pca_mean.npy"
        components_file = path / "pca_components.npy"
        if mean_file.exists() and components_file.exists():
            instance._state = PCAState(
                mean=np.load(mean_file),
                components=np.load(components_file)
            )
            instance._fitted = True
        return instance
```

---

## 4. 算子注册与编排引擎使用

基于最新的架构设计，开发者无需手动进行算子注册（旧版 `FeatureRegistry` 已被移除）。

**自动注册机制**：
只要你的特征算子类放置在 `tsas.engine.operator.feature.construction` 模块或其子包下，并且正确实现了 `name()` 类方法，算子模块的 CLI 工具底层的 `OperatorRegistry` 就会在启动时自动扫描并注册该算子。

**编排与调用**：
对于多个特征构造算子的批量调用和对齐合并（取代旧版的 `FeatureConstructor`），现在统一由 CLI 命令提供声明式调度。请参考 `src/tsas/engine/operator/cli/README.md` 文档获取更多配置详情和使用指南。