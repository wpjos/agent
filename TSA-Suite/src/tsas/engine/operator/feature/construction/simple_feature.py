# -*- coding: utf-8 -*-

"""
简单特征构造算子实现模块

提供基于特征构造基类的具体算子实现，覆盖列关系和行关系的各种组合：

- SquareFeature: 逐元素平方（Independent + Map）
- PolynomialFeature: 多项式展开（Independent + Map，1:N）
- RollingMeanFeature: 滑动均值（Independent + Window）
- ColumnMedianFeature: 多列取中位数（Joint + Map，N:1）
- PCAFeature: PCA 降维（Learnable + Joint + Map，N:N）
"""

from pathlib import Path
from typing import Self

import numpy as np
from pydantic import BaseModel, ConfigDict, Field

from tsas.engine.operator.base import DataFrameMeta
from tsas.engine.operator.feature.construction.base import (
    BaseFeatureConfig,
    WindowFeatureConfig,
    IndependentMapFeature,
    IndependentWindowFeature,
    JointMapFeature,
    LearnableJointMapFeature,
)

__all__ = [
    'SquareConfig',
    'SquareFeature',
    'PolynomialConfig',
    'PolynomialFeature',
    'RollingMeanConfig',
    'RollingMeanFeature',
    'ColumnMedianConfig',
    'ColumnMedianFeature',
    'PCAConfig',
    'PCAState',
    'PCAFeature',
]


# ============================================================================
# SquareFeature: Independent + Map（1:1 列，1:1 行）
# ============================================================================

class SquareConfig(BaseFeatureConfig):
    """逐元素平方特征的 Config"""
    pass


class SquareFeature(IndependentMapFeature[SquareConfig]):
    """
    逐元素平方特征

    对每个输入列独立计算平方值。一列输入产出一列输出。

    输出列名格式: ``{源列名}_square``

    Input:
        x: 特征矩阵，形状 (n_samples, n_features)，由 Config 的 ``input_columns`` 选取

    Output:
        平方变换后的特征矩阵，形状 (n_samples, n_features)，列数与输入相同
    """

    @classmethod
    def name(cls) -> str:
        return "square_feature"

    @classmethod
    def version(cls) -> tuple[int, ...]:
        """返回算子版本号

        Returns:
            tuple[int, ...]: 版本号三元组 ``(1, 0, 0)``
        """
        return (1, 0, 0)

    @staticmethod
    def compute(x: np.ndarray, *, state=None, **params) -> np.ndarray:
        return x ** 2

    def _name_output_column(self, input_col: str, output_val) -> str:
        return self._make_output_column_name(input_col, "square")


# ============================================================================
# PolynomialFeature: Independent + Map（1:N 列，1:1 行）
# ============================================================================

class PolynomialConfig(BaseFeatureConfig):
    """多项式展开特征的 Config"""

    degrees: list[int] = Field(default=[2, 3], min_length=1)
    """多项式阶数列表，默认为 [2, 3]（即 x² 和 x³）"""


class PolynomialFeature(IndependentMapFeature[PolynomialConfig]):
    """
    多项式展开特征

    对每个输入列独立计算多阶多项式。一列输入产出多列输出（1:N）。

    输出列名格式: ``{源列名}_poly_{阶数}``

    Input:
        x: 特征矩阵，形状 (n_samples, n_features)，由 Config 的 ``input_columns`` 选取

    Output:
        多项式展开后的特征矩阵，形状 (n_samples, n_features * len(degrees))，
        每列按输入列和阶数组合排列
    """

    @classmethod
    def name(cls) -> str:
        return "polynomial_feature"

    @classmethod
    def version(cls) -> tuple[int, ...]:
        """返回算子版本号

        Returns:
            tuple[int, ...]: 版本号三元组 ``(1, 0, 0)``
        """
        return (1, 0, 0)

    @staticmethod
    def compute(x: np.ndarray, *, state=None, **params) -> np.ndarray:
        degrees = params.get("degrees", [2, 3])
        return np.column_stack([x ** d for d in degrees])

    def _get_compute_params(self):
        return {"degrees": self.config.degrees}

    def _name_output_column(self, input_col: str, output_val) -> str:
        # output_val 是多列输出的列名组，这里需要根据 degrees 生成
        # 由于 IndependentFeatureMixin._name_output_columns 会为每列调用此方法
        # 我们需要知道当前是第几个 degree
        # 但新接口中 output_val 是整个输出组，这里需要特殊处理
        # 改为在 _name_output_columns 中处理
        return self._make_output_column_name(input_col, "poly")

    def _name_output_columns(self, output_data: np.ndarray, meta: DataFrameMeta | None, params: None) -> list[str]:
        n_inputs = len(self.config.input_columns)
        n_degrees = len(self.config.degrees)
        if output_data.shape[1] != n_inputs * n_degrees:
            raise ValueError(f"输出列数({output_data.shape[1]})与预期({n_inputs * n_degrees})不符")
        result = []
        for i, col in enumerate(self.config.input_columns):
            for degree in self.config.degrees:
                result.append(self._make_output_column_name(col, "poly", str(degree)))
        return result


# ============================================================================
# RollingMeanFeature: Independent + Window（1:1 列，N:1 行）
# ============================================================================

class RollingMeanConfig(WindowFeatureConfig):
    """滑动均值特征的 Config"""
    pass


class RollingMeanFeature(IndependentWindowFeature[RollingMeanConfig]):
    """
    滑动均值特征

    对每个输入列独立计算滑动窗口内的均值。一列输入产出一列输出。
    每个窗口切片（window_size 行）通过 ``compute`` 计算均值，返回每列的均值。

    输出列名格式: ``{源列名}_rolling_mean``

    Input:
        x: 特征矩阵，形状 (n_samples, n_features)，由 Config 的 ``input_columns`` 选取

    Output:
        滑动均值特征矩阵，列数与输入相同；行数取决于 ``padding`` 配置
        （无填充时为 ``n_samples - window_size + 1``，有填充时与输入相同）
    """

    @classmethod
    def name(cls) -> str:
        return "rolling_mean_feature"

    @classmethod
    def version(cls) -> tuple[int, ...]:
        """返回算子版本号

        Returns:
            tuple[int, ...]: 版本号三元组 ``(1, 0, 0)``
        """
        return (1, 0, 0)

    @staticmethod
    def compute(x: np.ndarray, *, state=None, **params) -> np.ndarray:
        # x 形状: (window_size, n_cols)
        # 返回每列的均值，形状: (n_cols,)
        return np.mean(x, axis=0)

    def _name_output_column(self, input_col: str, output_val) -> str:
        return self._make_output_column_name(input_col, "rolling_mean")


# ============================================================================
# ColumnMedianFeature: Joint + Map（N:1 列，1:1 行）
# ============================================================================

class ColumnMedianConfig(BaseFeatureConfig):
    """多列取中位数特征的 Config"""

    output_column: str = Field(default="")
    """输出列名，为空时自动生成"""


class ColumnMedianFeature(JointMapFeature[ColumnMedianConfig]):
    """
    多列取中位数特征

    对多个输入列逐行计算中位数。多列输入产出一列输出（N:1）。

    输出列名格式: ``{所有源列名连接}_median``（或自定义 output_column）

    Input:
        x: 特征矩阵，形状 (n_samples, n_features)，由 Config 的 ``input_columns`` 选取

    Output:
        逐行中位数特征，形状 (n_samples, 1)，每行为对应行的中位数
    """

    @classmethod
    def name(cls) -> str:
        return "column_median_feature"

    @classmethod
    def version(cls) -> tuple[int, ...]:
        """返回算子版本号

        Returns:
            tuple[int, ...]: 版本号三元组 ``(1, 0, 0)``
        """
        return (1, 0, 0)

    @staticmethod
    def compute(x: np.ndarray, *, state=None, **params) -> np.ndarray:
        # x 是 (n_rows, n_cols)，逐行计算中位数
        return np.median(x, axis=1)

    def _name_output_columns(self, output_data: np.ndarray, meta: DataFrameMeta | None, params: None) -> list[str]:
        if self.config.output_column:
            return [self.config.output_column]
        source = "_".join(self.config.input_columns)
        return [self._make_output_column_name(source, "median")]


# ============================================================================
# PCAFeature: Learnable + Joint + Map（N:N 列，1:1 行）
# ============================================================================

class PCAConfig(BaseFeatureConfig):
    """PCA 降维特征的 Config"""

    n_components: int = Field(ge=1)
    """降维后的目标维度数"""


class PCAState(BaseModel):
    """PCA 训练状态

    存储 PCA 训练后的均值向量和主成分变换矩阵。
    使用 ``arbitrary_types_allowed`` 以支持 numpy 数组类型。

    Attributes:
        mean (np.ndarray): 各列的均值向量，形状为 (n_features,)。
        components (np.ndarray): 主成分变换矩阵，形状为 (n_features, n_components)。
    """
    model_config = ConfigDict(arbitrary_types_allowed=True)

    mean: np.ndarray
    """各列的均值向量"""

    components: np.ndarray
    """主成分变换矩阵"""


class PCAFeature(LearnableJointMapFeature[PCAConfig, PCAState]):
    """PCA 降维特征

    先通过 ``train`` 学习主成分变换矩阵（结果保存为 ``PCAState``），
    再通过 ``run`` 将多列输入降维为指定维度的输出。多列输入产出多列输出（N:N）。

    输出列名格式: ``pca_{序号}``

    Input:
        x: 特征矩阵，形状 (n_samples, n_features)，由 Config 的 ``input_columns`` 选取

    Output:
        PCA 降维后的特征矩阵，形状 (n_samples, n_components)，列名为 ``pca_0, pca_1, ...``
    """

    @classmethod
    def name(cls) -> str:
        return "pca_feature"

    @classmethod
    def version(cls) -> tuple[int, ...]:
        """返回算子版本号

        Returns:
            tuple[int, ...]: 版本号三元组 ``(1, 0, 0)``
        """
        return (1, 0, 0)

    @staticmethod
    def train(x: np.ndarray, **params) -> PCAState:
        """基于训练数据学习 PCA 状态

        计算均值向量和协方差矩阵的特征向量，选取前 n_components 个主成分。

        Args:
            x (np.ndarray): 训练数据，形状为 (n_samples, n_features)。
            **params: 由 ``_get_train_params`` 返回的额外训练参数，
                包含 ``n_components`` (int) 目标维度数。

        Returns:
            PCAState: 包含均值向量和主成分变换矩阵的训练状态。
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
        """获取传递给 ``train`` 的额外参数

        Returns:
            dict: 包含 ``n_components`` 目标维度数。
        """
        return {"n_components": self.config.n_components}

    @staticmethod
    def compute(x: np.ndarray, *, state=None, **params) -> np.ndarray:
        """PCA 降维计算

        使用训练好的均值和主成分变换矩阵对输入数据进行降维。

        Args:
            x (np.ndarray): 输入数据，形状为 (n_samples, n_features)。
            state (PCAState | None): 训练状态，包含 mean 和 components。
            **params: 额外计算参数（未使用）。

        Returns:
            np.ndarray: 降维后的数据，形状为 (n_samples, n_components)。

        Raises:
            ValueError: state 为 None 时（未训练）。
        """
        if state is None:
            raise ValueError("PCA 需要先训练")
        centered = x - state.mean
        return centered @ state.components

    def _name_output_columns(self, output_data: np.ndarray, meta: DataFrameMeta | None, params: None) -> list[str]:
        n_components = output_data.shape[1] if output_data.ndim > 1 else 1
        return [f"pca_{i}" for i in range(n_components)]

    def save(self, path: str | Path) -> None:
        """持久化 PCA 算子到指定目录

        在父类 save 的基础上，额外将均值向量和主成分变换矩阵保存为 .npy 文件。

        Args:
            path (str | Path): 目标目录路径。
        """
        super().save(path)
        path = Path(path)
        if self._state is not None:
            np.save(path / "pca_mean.npy", self._state.mean)
            np.save(path / "pca_components.npy", self._state.components)

    @classmethod
    def load(cls, path: str | Path, *, oid: str | None = None) -> Self:
        """从指定目录加载 PCA 算子

        在父类 load 的基础上，额外从 .npy 文件恢复均值向量和主成分变换矩阵，
        重建 PCAState 并标记为已训练。

        Args:
            path (str | Path): 源目录路径。
            oid (str | None): 算子标识。

        Returns:
            加载后的 PCAFeature 实例。
        """
        instance = super().load(path, oid=oid)
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
