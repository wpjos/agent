# -*- coding: utf-8 -*-

"""
残差评分算子模块

基于真实值与预测值之间的残差（误差）计算异常分数，属于检测算子第2层（Scorer）。

本模块实现了 BiNumericOperator（双输入算子），接收 x_real 和 x_pred 两组数据，
根据配置的度量方式（MSE 或 MAE）计算残差作为异常分数。残差越大，表示预测偏差越大，
异常可能性越高。

支持两种评分模式:
    - ResidualScorer: 沿特征轴聚合为 1D 异常分数，适合整体异常判定
    - ResidualMapScorer: 保留逐变量残差分数（2D），适合定位具体异常变量

示例用法::

    # ResidualScorer — 聚合残差评分
    scorer = ResidualScorer(config=ResidualScorerConfig(metric="mse"))
    scores, extra = scorer.run(x_real, x_pred)
    # scores: 形状 (n_samples,) 的 1D 异常分数
    # extra.scores: 形状 (n_samples, n_features) 的逐变量分数

    # ResidualMapScorer — 逐变量残差评分
    map_scorer = ResidualMapScorer(metric="mae")
    scores = map_scorer.run(x_real, x_pred)
    # scores: 形状 (n_samples, n_features) 的 2D 残差矩阵

主要组件:
    - ResidualScorerConfig: 残差评分器配置
    - ResidualScorerExtraOutput: 残差评分器附加输出
    - ResidualScorer: 聚合残差评分器
    - ResidualMapScorer: 逐变量残差评分器
"""

from enum import Enum

import numpy as np
import pandas as pd
from pydantic import BaseModel, Field, ConfigDict

from tsas.engine.operator.base import BiNumericOperator
from tsas.engine.operator.detection.base import SingleScorerMixin, MultiScorerMixin

__all__ = [
    'ResidualMetric',
    "ResidualScorerConfig",
    "ResidualScorerExtraOutput",
    "ResidualScorer",
]


class ResidualMetric(str, Enum):
    """残差度量方式枚举

    定义残差评分器中真实值与预测值之间的误差计算方式。

    Attributes:
        MSE: 均方误差（平方残差），对大偏差更敏感
        MAE: 平均绝对误差（绝对残差），对异常值更鲁棒
    """
    MSE = "mse"
    """均方误差（平方残差），对大偏差更敏感"""
    MAE = "mae"
    """平均绝对误差（绝对残差），对异常值更鲁棒"""


class ResidualScorerConfig(BaseModel):
    """
    残差评分器配置

    Attributes:
        metric (ResidualMetric): 残差计算公式。``MSE`` 为均方误差（平方残差），
            ``MAE`` 为平均绝对误差（绝对残差）。默认 ``MSE``。
    """
    metric: ResidualMetric = Field(default=ResidualMetric.MSE,
                                   description="计算公式: 'mse' 为均方误差, 'mae' 为平均绝对误差")


class ResidualScorerExtraOutput(BaseModel):
    """
    残差评分器附加输出

    包含各变量的单独异常分数，用于细粒度分析具体哪个变量偏差较大。

    Attributes:
        scores (np.ndarray): 各变量的单独异常分数，形状 (n_samples, n_features)，
            值越大表示该变量偏差越大。
    """
    scores: np.ndarray = Field(default=None,
                               description="各变量的单独异常分数，形状 (n_samples, n_features)，值越大表示该变量偏差越大")

    model_config = ConfigDict(arbitrary_types_allowed=True)


class ResidualScorer(SingleScorerMixin[None],
                     BiNumericOperator[ResidualScorerExtraOutput, ResidualScorerConfig, None]):
    """
    残差评分器 — 检测算子第2层（Scorer）

    继承 SingleScorerMixin 与 BiNumericOperator（双输入算子），
    接收真实值 x_real 和预测值 x_pred 两组输入，计算二者之间的残差作为异常分数。

    计算流程:
        1. 计算残差 ``residual = x_real - x_pred``
        2. 根据配置的度量方式（MSE/MAE）计算逐变量误差
        3. 沿特征轴取均值，聚合为 1D 异常分数

    Input:
        x_real: 真实值矩阵，形状 (n_samples, n_features)
        x_pred: 预测值矩阵，形状与 x_real 相同

    Output:
        异常分数，形状 (n_samples,)，值越大越异常。
        分数由 x_real 与 x_pred 的逐元素残差（MSE 或 MAE）沿特征轴聚合得到

    泛型参数:
        - EO: ResidualScorerExtraOutput（附加输出由 ``_eo_type`` 自动渲染，含逐变量残差分数）
        - C: ResidualScorerConfig
        - RP: None（无运行参数）
    """

    @classmethod
    def name(cls) -> str:
        """
        返回算子名称

        Returns:
            str: 算子名称 ``"residual_scorer"``
        """
        return "residual_scorer"

    @classmethod
    def version(cls) -> tuple[int, ...]:
        """返回算子版本号

        Returns:
            tuple[int, ...]: 版本号三元组 ``(1, 0, 0)``
        """
        return (1, 0, 0)

    def _run_data(self, x_real: np.ndarray, x_pred: np.ndarray, params: None,
                  real_idx: pd.Index | None = None, pred_idx: pd.Index | None = None) -> (
            np.ndarray | tuple[np.ndarray, ResidualScorerExtraOutput]):
        """
        计算真实值与预测值之间的残差异常分数

        根据配置的度量方式（MSE 或 MAE）计算逐变量残差，再沿特征轴取均值
        聚合为 1D 异常分数，同时返回逐变量分数作为附加输出。

        Args:
            x_real(np.ndarray): 真实值，形状 (n_samples, n_features)
            x_pred(np.ndarray): 预测值，形状 (n_samples, n_features)
            params(None): 无运行参数

        Returns:
            tuple[np.ndarray, ResidualScorerExtraOutput]:
                - total_scores: 1D 异常分数，形状 (n_samples,)
                - extra: 附加输出，包含逐变量分数 scores (n_samples, n_features)
        """
        residual = x_real - x_pred
        if self.config.metric == ResidualMetric.MAE:
            # 逐样本绝对误差
            scores = np.abs(residual)
        else:
            # 逐样本平方误差
            scores = residual ** 2
        total_scores = np.mean(scores, axis=1).ravel()
        return total_scores, ResidualScorerExtraOutput(scores=scores)


class ResidualMapScorer(MultiScorerMixin[None],
                        BiNumericOperator[None, ResidualScorerConfig, None]):
    """
    残差映射评分器 — 检测算子第2层（Scorer）

    继承 MultiScorerMixin 与 BiNumericOperator（双输入算子），
    接收真实值 x_real 和预测值 x_pred 两组输入，计算逐变量残差分数。

    与 ResidualScorer 的区别:
        - ResidualScorer: 沿特征轴聚合为 1D 分数 (n_samples,)
        - ResidualMapScorer: 保留 2D 残差矩阵 (n_samples, n_features)，
          不做聚合，适合定位具体异常变量

    Input:
        x_real: 真实值矩阵，形状 (n_samples, n_features)
        x_pred: 预测值矩阵，形状与 x_real 相同

    Output:
        逐变量异常分数，形状 (n_samples, n_features)，值越大越异常。
        分数为 x_real 与 x_pred 的逐元素残差（MSE 或 MAE），不做特征轴聚合

    泛型参数:
        - EO: None（无附加输出）
        - C: ResidualScorerConfig
        - RP: None（无运行参数）
    """

    @classmethod
    def name(cls) -> str:
        """
        返回算子名称

        Returns:
            str: 算子名称 ``"residual_map_scorer"``
        """
        return "residual_map_scorer"

    @classmethod
    def version(cls) -> tuple[int, ...]:
        """返回算子版本号

        Returns:
            tuple[int, ...]: 版本号三元组 ``(1, 0, 0)``
        """
        return (1, 0, 0)

    def _run_data(self, x_real: np.ndarray, x_pred: np.ndarray, params: None,
                  real_idx: pd.Index | None = None, pred_idx: pd.Index | None = None) -> np.ndarray:
        """
        计算逐样本残差

        根据配置的度量方式（MSE 或 MAE），计算 y_pred 与 y_real 之间的
        逐样本残差，保留 2D 形状不做聚合。

        Args:
            x_real(np.ndarray): 真实值，形状 (n_samples, n_features)
            x_pred(np.ndarray): 预测值，形状 (n_samples, n_features)
            params(None): 无运行参数

        Returns:
            np.ndarray: 残差 ndarray，形状 (n_samples, n_features)，越大表示偏差越大
        """
        residual = x_real - x_pred
        if self.config.metric == ResidualMetric.MAE:
            # 逐样本绝对误差
            scores = np.abs(residual)
        else:
            # 逐样本平方误差
            scores = residual ** 2
        return scores
