# -*- coding: utf-8 -*-

"""
时序异常检测算子基类定义模块

基于 base.py 提供四层检测算子架构，
采用组合优于继承的设计原则，支持 DataFrame + ndarray 双类型输入输出。

四层类型架构::

    第1层：Predictor（预测器）
      BasePredictorMixin[EO, C, RP]    ← 混入类，需配合 NumericOperator
        职责：对输入数据给出预测值（重构型/预测型）

    第2层：Scorer（评分器）
      BaseScorerMixin
        职责：对输入数据给出异常分数
        SingleScorerMixin
        MultiScorerMixin

    第3层：Decider（决策器）
      BaseDeciderMixin
        职责：对输入数据给出二分类标签（0=正常/1=异常）

    第4层：Detector（检测器）
      BaseDetector[EO, C, RP]     ← BaseDeciderMixin + NumericOperator
        职责：由原始变量数据，输出完整检测结果二分类标签（0=正常/1=异常）
"""

from abc import ABCMeta
from typing import Generic

import numpy as np

from tsas.engine.operator.base import (
    C,
    RP,
    EO,
    DataFrameMeta,
    NumericOperator,
)

__all__ = [
    'BasePredictorMixin',
    'BaseScorerMixin',
    'SingleScorerMixin',
    'MultiScorerMixin',
    'BaseDeciderMixin',
    'BaseDetector',
]


# ============================================================================
# 第1层：Predictor（预测器）
# ============================================================================

class BasePredictorMixin(Generic[EO, C, RP], metaclass=ABCMeta):
    """
    预测器混入基类（第1层）

    所有预测器类型的抽象基类，定义预测器层的公共接口。
    预测器的职责是对输入数据产出预测值，支持两种预测模式:
        - 重构型: 输出与输入同维度（如 PCA 重构）
        - 预测型: 输出未来值（如时序预测）

    作为混入类使用，需配合 ``NumericOperator`` 一起定义具体的预测器算子类。
    与 ``BaseScorerMixin``、``BaseDeciderMixin`` 保持统一的设计范式。

    使用示例::

        class MyPredictor(UnsupervisedNumericOperatorMixin[None],
                          BasePredictorMixin[None, MyConfig, None],
                          NumericOperator[None, MyConfig, None]):
            ...

    泛型参数:
        - EO: 附加输出类型
        - C: 实例参数类型
        - RP: 运行参数类型
    """

    def _name_output_columns(self, output_data: np.ndarray, meta: DataFrameMeta | None, params: RP | None) -> list[str]:
        """
        推断输出列名

        预测器的输出与输入同维度，因此直接沿用输入列名作为输出列名。

        Args:
            output_data (np.ndarray): 计算结果 ndarray
            meta (DataFrameMeta | None): 输入 DataFrame 的元信息快照
            params (RP | None): 运行参数

        Returns:
            list[str]: 输入列名列表
        """
        return meta.column_names


# ============================================================================
# 第2层：Scorer（评分器）
# ============================================================================

class BaseScorerMixin(metaclass=ABCMeta):
    """
    评分器基类混入（第2层）

    所有评分器类型的抽象基类，定义评分器层的公共接口。
    评分器的职责是对输入数据产出异常分数，用于量化每个样本（或每个特征）的异常程度。

    子类包括:
        - SingleScorerMixin: 单输出评分器，产出全局异常分数（一维）
        - MultiScorerMixin: 多输出评分器，产出逐特征异常分数（多维）
    """
    pass


class SingleScorerMixin(BaseScorerMixin, Generic[RP], metaclass=ABCMeta):
    """
    单输出评分器混入

    产出单一全局异常分数，输出为一维数组 (n_samples,)，
    输出列名固定为 ``"score"``。

    适用于需要将多维输入聚合为一个整体异常程度的场景，
    例如基于残差的总体异常评分。
    """

    def _name_output_columns(self, output_data: np.ndarray, meta: DataFrameMeta | None, params: RP | None) -> list[str]:
        """
        推断输出列名

        单输出评分器的列名固定为 ``["score"]``。

        Args:
            output_data(np.ndarray): 计算结果 ndarray
            meta(DataFrameMeta | None): 输入 DataFrame 的元信息快照
            params(RP | None): 运行参数

        Returns:
            list[str]: 固定列名 ``["score"]``
        """
        return ["score"]


class MultiScorerMixin(BaseScorerMixin, Generic[RP], metaclass=ABCMeta):
    """
    多输出评分器混入

    产出的异常分数与输入特征一一对应，输出为多维数组 (n_samples, n_features)，
    输出列名沿用输入列名。

    适用于需要逐特征分析异常程度的场景，
    例如 PCA 各主成分的独立异常分数。
    """

    def _name_output_columns(self, output_data: np.ndarray, meta: DataFrameMeta | None, params: RP | None) -> list[str]:
        """
        推断输出列名

        多输出评分器的列名沿用输入列名，保持特征维度的对应关系。

        Args:
            output_data(np.ndarray): 计算结果 ndarray
            meta(DataFrameMeta | None): 输入 DataFrame 的元信息快照
            params(RP | None): 运行参数

        Returns:
            list[str]: 输入列名列表
        """
        return meta.column_names


# ============================================================================
# 第3层：Decider（决策器）
# ============================================================================

class BaseDeciderMixin(Generic[RP], metaclass=ABCMeta):

    def _name_output_columns(self, output_data: np.ndarray, meta: DataFrameMeta | None, params: RP | None) -> list[str]:
        """
        推断输出列名

        默认策略:
            - 1D 或单列 2D → ``["label"]``
            - 多列 2D → None

        Args:
            output_data(np.ndarray): 计算结果 ndarray
            meta(DataFrameMeta): 输入 DataFrame 的元信息快照

        Returns:
            list[str] | None: 列名列表，或 None
        """
        if output_data.ndim == 1 or (output_data.ndim == 2 and output_data.shape[1] == 1):
            return ["label"]
        return None


# ============================================================================
# 第4层：Detector（检测器）
# ============================================================================


class BaseDetector(BaseDeciderMixin[RP], NumericOperator[EO, C, RP], Generic[EO, C, RP], metaclass=ABCMeta):
    """
    检测器基类（第4层）

    组合 ``BaseDeciderMixin`` 与 ``NumericOperator``，提供端到端的异常检测能力：
    原始数据 → 二分类标签（0=正常，1=异常）。

    检测器是检测管线中最顶层的算子，直接面向用户使用，
    内部封装了预测、评分、决策等完整流程。

    泛型参数:
        - EO: 附加输出类型
        - C: 实例参数类型
        - RP: 运行参数类型
    """
    pass
