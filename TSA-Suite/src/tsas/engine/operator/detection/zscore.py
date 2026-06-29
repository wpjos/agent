# -*- coding: utf-8 -*-

"""
Z-Score 异常检测算子

基于 Z-Score 的异常检测，属于 DirectScorer 路径。
训练阶段学习各特征的均值和标准差，推理阶段计算每个样本的最大绝对 Z-Score 作为异常分数。

包含:
    - ZScoreScorer: 直接评分器，计算 max(|z-score|)
    - ZScoreDetector: 端到端检测器，组合 ZScoreScorer + ThresholdDecider

示例用法::

    # 直接评分
    scorer = ZScoreScorer(config=ZScoreScorerConfig(threshold=3.0))
    scorer.fit(train_data)
    scores, eo = scorer.run(test_data)

    # 端到端检测
    detector = ZScoreDetector(config=ZScoreDetectorConfig(threshold=3.0))
    detector.fit(train_data)
    labels, eo = detector.run(test_data)
"""

from pathlib import Path

import numpy as np
import pandas as pd
from pydantic import BaseModel, Field

from tsas.engine.operator.base import NumericOperator, UnsupervisedNumericOperatorMixin
from tsas.engine.operator.detection.base import (
    SingleScorerMixin,
    BaseDeciderMixin,
)
from tsas.engine.operator.detection.threshold_decider import (
    ThresholdDecider,
    ThresholdDeciderConfig,
)

__all__ = ['ZScoreScorerConfig', 'ZScoreScorer', 'ZScoreDetectorConfig', 'ZScoreDetector']


class ZScoreScorerConfig(BaseModel):
    """
    Z-Score 评分器实例参数

    Attributes:
        threshold (float): Z-Score 阈值（可搜索，范围 [1.0, 10.0]）
    """
    threshold: float = Field(
        default=3.0, ge=1.0, le=10.0,
        description="Z-Score 阈值（可搜索）"
    )


class ZScoreScorer(SingleScorerMixin[None],
                   UnsupervisedNumericOperatorMixin[None],
                   NumericOperator[None, ZScoreScorerConfig, None]):
    """
    Z-Score 直接评分器

    基于标准分数（Z-Score）的异常检测评分器。

    训练阶段:
        - 学习各特征的均值向量 ``_mean`` 和标准差向量 ``_std``
        - 对零标准差的特征做保护处理（替换为 1.0），避免除零

    推理阶段:
        - 计算每个样本各特征的绝对 Z-Score: ``|x - mean| / std``
        - 取每个样本的最大绝对 Z-Score 作为异常分数: ``max(|z|, axis=1)``

    参考实现:
        算法核心等价于 sklearn.preprocessing.StandardScaler + max(|z|)，
        但不引入 sklearn 依赖，纯 NumPy 实现。

    Input:
        x: 特征矩阵，形状 (n_samples, n_features)

    Output:
        异常分数，形状 (n_samples,)，值越大越异常

    泛型参数:
        - EO: None（无附加输出）
        - C: ZScoreScorerConfig
        - RP: None（无运行参数）
    """

    # 训练状态文件名
    _LEARNED_STATE_FILE = '_learned_state.npz'

    @classmethod
    def name(cls) -> str:
        return "zscore_scorer"

    @classmethod
    def version(cls) -> tuple[int, ...]:
        """返回算子版本号

        Returns:
            tuple[int, ...]: 版本号三元组 ``(1, 0, 0)``
        """
        return (1, 0, 0)

    def __init__(self, *, oid: str | None = None, config: ZScoreScorerConfig | None = None, **kwargs):
        """
        初始化 Z-Score 评分器

        Args:
            oid (str | None): 算子实例唯一标识后缀
            config (ZScoreScorerConfig | None): 评分器配置
            **kwargs: 透传给基类的参数，支持:
                - threshold (float): Z-Score 阈值，默认 3.0
        """
        super().__init__(oid=oid, config=config, **kwargs)
        self._mean: np.ndarray | None = None
        """训练数据的列均值向量"""
        self._std: np.ndarray | None = None
        """训练数据的列标准差向量"""

    def _fit_data(self, x: np.ndarray, params: None) -> None:
        """
        学习均值和标准差

        计算训练数据各特征的均值和标准差。
        对零标准差的特征做保护处理（替换为 1.0），避免推理时除零。

        Args:
            x (np.ndarray): 训练数据，形状 (n_samples, n_features)
            params (None): 无训练参数
        """
        # 沿样本轴计算均值和标准差
        self._mean = x.mean(axis=0)
        self._std = x.std(axis=0)
        # 零标准差保护: 常数特征的 z-score 应为 0，将 std=0 替换为 1 使 (x-mean)/std = 0
        self._std[self._std == 0] = 1.0

    def _run_data(self, x: np.ndarray, params: None, idx: pd.Index | None = None) -> np.ndarray:
        """
        计算最大绝对 Z-Score

        对每个样本计算各特征的绝对 Z-Score，取最大值作为异常分数。

        Args:
            x (np.ndarray): 输入数据，形状 (n_samples, n_features)
            params (None): 无运行参数
            idx (pd.Index | None): 输入数据的行索引

        Returns:
            np.ndarray: 异常分数 ndarray，形状 (n_samples,)，每个值为该样本的最大 |z-score|
        """
        # 计算绝对 Z-Score: |x - mean| / std
        z = np.abs((x - self._mean) / self._std)
        # 每个样本取最大 Z-Score
        scores = z.max(axis=1)
        return scores

    def _save_fit_state(self, path: Path) -> None:
        """
        保存训练状态：训练参数 + 学习到的均值/标准差

        在基类保存 last_fit_params 的基础上，额外将 _mean 和 _std 持久化到 npz 文件。

        Args:
            path (Path): 目标目录路径
        """
        super()._save_fit_state(path)
        # 持久化学习到的统计量
        np.savez(path / self._LEARNED_STATE_FILE, mean=self._mean, std=self._std)

    def _load_fit_state(self, path: Path) -> None:
        """
        恢复训练状态：训练参数 + 学习到的均值/标准差

        从 npz 文件恢复 _mean 和 _std，并将 _fitted 标记为 True。

        Args:
            path (Path): 源目录路径
        """
        super()._load_fit_state(path)
        # 恢复学习到的统计量
        data = np.load(path / self._LEARNED_STATE_FILE)
        self._mean = data['mean']
        self._std = data['std']
        # 恢复训练完成标记
        self._fitted = True


class ZScoreDetectorConfig(BaseModel):
    """
    Z-Score 检测器实例参数

    Attributes:
        threshold (float): 异常判定阈值（可搜索，范围 [1.0, 10.0]）
    """
    threshold: float = Field(
        default=3.0, ge=1.0, le=10.0,
        description="异常判定阈值，Z-Score > threshold 判定为异常"
    )


class ZScoreDetector(UnsupervisedNumericOperatorMixin[None],
                     BaseDeciderMixin[None],
                     NumericOperator[None, ZScoreDetectorConfig, None]):
    """
    Z-Score 检测器 — 组合 ZScoreScorer + ThresholdDecider

    端到端的 Z-Score 异常检测器:

    ::

        ZScoreDetector
          ├── ZScoreScorer (Direct)
          │     _fit_data: 学习均值/标准差
          │     _run_data: 计算每个样本的 max(|z-score|)
          └── ThresholdDecider
                _run_data: scores > threshold → labels

    使用示例::

        detector = ZScoreDetector(config=ZScoreDetectorConfig(threshold=3.0))
        detector.fit(train_data)
        labels, eo = detector.run(test_data)

    Input:
        x: 特征矩阵，形状 (n_samples, n_features)

    Output:
        0/1 异常标签，形状 (n_samples,)，1 表示异常。
        标签由内部 ZScoreScorer 输出的 max(|z-score|) 经 ThresholdDecider 的固定阈值决策得到

    泛型参数:
        - EO: None（无附加输出）
        - C: ZScoreDetectorConfig
        - RP: None（无运行参数）
    """

    @classmethod
    def name(cls) -> str:
        return "zscore_detector"

    @classmethod
    def version(cls) -> tuple[int, ...]:
        """返回算子版本号

        Returns:
            tuple[int, ...]: 版本号三元组 ``(1, 0, 0)``
        """
        return (1, 0, 0)

    def __init__(self, *, oid: str | None = None, config: ZScoreDetectorConfig | None = None, **kwargs):
        """
        初始化 Z-Score 检测器

        自动创建 ZScoreScorer 和 ThresholdDecider 子组件，
        使用相同的 threshold 参数保持一致性。

        Args:
            oid (str | None): 算子实例唯一标识后缀
            config (ZScoreDetectorConfig | None): 检测器配置
            **kwargs: 透传给基类的参数，支持:
                - threshold (float): 异常判定阈值，默认 3.0
        """
        super().__init__(oid=oid, config=config, **kwargs)
        # 提取子组件配置参数
        scorer_config = ZScoreScorerConfig(threshold=self.config.threshold)
        self._scorer = ZScoreScorer(config=scorer_config)
        self._decider = ThresholdDecider(config=ThresholdDeciderConfig(threshold=self.config.threshold))

    def _fit_data(self, x: np.ndarray, params: None) -> None:
        """
        训练检测器 — 训练 ZScoreScorer

        Args:
            x (np.ndarray): 训练数据
            params (None): 无训练参数
        """
        # 训练评分器（ThresholdDecider 不需要训练）
        self._scorer.fit(x)

    def _run_data(self, x: np.ndarray, params: None, idx: pd.Index | None = None) -> np.ndarray:
        """
        检测推理: ZScoreScorer → ThresholdDecider

        Args:
            x (np.ndarray): 输入数据
            params (None): 无运行参数
            idx (pd.Index | None): 输入数据的行索引

        Returns:
            np.ndarray: 二分类标签，1=异常/0=正常
        """
        # 步骤1: 评分器推理
        scores = self._scorer.run(x)

        # 步骤2: 决策器推理
        labels = self._decider.run(scores)

        return labels

    def _save_fit_state(self, path: Path) -> None:
        """
        保存训练状态：训练参数 + 子组件状态

        将 ZScoreScorer 保存到子目录 _scorer。
        ThresholdDecider 为无状态算子，无需保存。

        Args:
            path (Path): 目标目录路径
        """
        super()._save_fit_state(path)
        # 保存评分器（ThresholdDecider 无状态，无需保存）
        self._scorer.save(path / '_scorer')

    def _load_fit_state(self, path: Path) -> None:
        """
        恢复训练状态：训练参数 + 子组件状态

        从子目录 _scorer 恢复 ZScoreScorer，并将 _fitted 标记为 True。
        ThresholdDecider 在 __init__ 中重建，无需从磁盘恢复。

        Args:
            path (Path): 源目录路径
        """
        super()._load_fit_state(path)
        # 恢复评分器
        self._scorer = ZScoreScorer.load(path / '_scorer')
        # 恢复训练完成标记
        self._fitted = True
