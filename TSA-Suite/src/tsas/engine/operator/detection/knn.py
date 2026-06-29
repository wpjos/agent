# -*- coding: utf-8 -*-

"""
KNN 异常检测算子

基于 K 近邻距离的异常检测，属于 DirectScorer 路径。
核心思想: 异常点与其邻居之间的距离会显著大于正常点。

计算逻辑参考:
    - tsas.engine.operator.model.knn._scoring.KNNDiscordDetectionModel
    - tsas.engine.operator.model.knn._classification

包含:
    - KNNScorer: 直接评分器，基于 KNN 距离计算异常分数
    - KNNDetector: 端到端检测器，组合 KNNScorer + PercentileDecider

示例用法::

    # 直接评分
    scorer = KNNScorer(n_neighbors=5)
    scorer.fit(train_data)
    scores, eo = scorer.run(test_data)

    # 端到端检测
    detector = KNNDetector(config=KNNDetectorConfig(n_neighbors=5, percentile=95.0))
    detector.fit(train_data)
    labels, eo = detector.run(test_data)
"""

import pickle
from enum import Enum
from pathlib import Path

import numpy as np
import pandas as pd
from pydantic import BaseModel, Field

from tsas.engine.operator.base import NumericOperator, UnsupervisedNumericOperatorMixin
from tsas.engine.operator.detection.base import (
    SingleScorerMixin,
    BaseDeciderMixin,
)
from tsas.engine.operator.detection.percentile_decider import (
    PercentileDecider,
    PercentileDeciderConfig,
)

__all__ = [
    'KNNDistanceMetric',
    'KNNScoreMethod',
    'KNNScorerConfig',
    'KNNScorerExtraOutput',
    'KNNScorer',
    'KNNDetectorConfig',
    'KNNDetector',
]


class KNNDistanceMetric(str, Enum):
    """
    KNN 距离度量方法枚举

    参考: tsas.engine.operator.model.knn._scoring.DistanceMetricMethod
    """
    EUCLIDEAN = "euclidean"
    """欧氏距离（L2 范数）"""
    MANHATTAN = "manhattan"
    """曼哈顿距离（L1 范数）"""


class KNNScoreMethod(str, Enum):
    """
    KNN 异常分数合并方法枚举

    决定如何从 K 个近邻距离合并为单一的异常分数。
    参考: tsas.engine.operator.model.knn._scoring.ScoreMergeMethod
    """
    MAXIMUM = "maximum"
    """取 K 个距离的最大值 — 最保守策略，任一邻居距离大即判异常"""
    MEAN = "mean"
    """取 K 个距离的平均值 — 平滑策略，抗噪声"""
    MEDIAN = "median"
    """取 K 个距离的中位数 — 鲁棒策略，抗极端邻居"""


class KNNScorerConfig(BaseModel):
    """
    KNN 评分器实例参数

    Attributes:
        n_neighbors (int): 近邻数量 K，必须为正整数
        distance_metric (KNNDistanceMetric): 距离度量方式，euclidean 或 manhattan
        score_method (KNNScoreMethod): 分数合并方式，maximum/mean/median
    """
    n_neighbors: int = Field(
        default=5, ge=1, le=20,
        description="近邻数量 K"
    )
    distance_metric: KNNDistanceMetric = Field(
        default=KNNDistanceMetric.EUCLIDEAN,
        description="距离度量方式: 'euclidean' 或 'manhattan'"
    )
    score_method: KNNScoreMethod = Field(
        default=KNNScoreMethod.MAXIMUM,
        description="分数合并方式: 'maximum', 'mean' 或 'median'"
    )


class KNNScorerExtraOutput(BaseModel):
    """
    KNN 评分器附加输出

    Attributes:
        n_neighbors (int): 使用的近邻数量
        distance_metric (str): 使用的距离度量方式
        score_method (str): 使用的分数合并方式
    """
    n_neighbors: int = Field(default=5, description="使用的近邻数量")
    distance_metric: str = Field(default="euclidean", description="使用的距离度量方式")
    score_method: str = Field(default="maximum", description="使用的分数合并方式")


class KNNScorer(SingleScorerMixin[None],
                UnsupervisedNumericOperatorMixin[None],
                NumericOperator[KNNScorerExtraOutput, KNNScorerConfig, None]):
    """
    KNN 直接评分器

    基于 K 近邻距离的异常检测评分器。

    核心思想（参考 _scoring.KNNDiscordDetectionModel）:
        异常点与其邻居之间的距离会显著大于正常点与其邻居之间的距离。
        通过计算每个数据点到其 K 个最近邻的距离，将距离转化为异常评分。

    训练阶段:
        - 构建近邻索引（基于 sklearn.neighbors.NearestNeighbors）
        - 存储训练数据以供推理时查询

    推理阶段:
        - 查询每个样本的 K 个近邻距离
        - 根据 score_method 合并距离为异常分数:
            - maximum: max(distances) — 最保守，参考 Ramaswamy et al. 2000
            - mean: mean(distances) — 平滑策略
            - median: median(distances) — 鲁棒策略

    注意:
        当训练样本数不足 K 个时，自动调整 K 为训练样本数。

    Input:
        x: 特征矩阵，形状 (n_samples, n_features)

    Output:
        异常分数，形状 (n_samples,)，值越大越异常

    泛型参数:
        - EO: KNNScorerExtraOutput（附加输出由 ``_eo_type`` 自动渲染到 CLI Help 的"附加输出"段）
        - C: KNNScorerConfig
        - RP: None（无运行参数）
    """

    # 训练状态文件名
    _INDEX_FILE = '_nn_index.pkl'
    _TRAIN_DATA_FILE = '_train_data.npy'

    @classmethod
    def name(cls) -> str:
        """返回算子标识名称"""
        return "knn_scorer"

    @classmethod
    def version(cls) -> tuple[int, ...]:
        """返回算子版本号

        Returns:
            tuple[int, ...]: 版本号三元组 ``(1, 0, 0)``
        """
        return (1, 0, 0)

    def __init__(self, *, oid: str | None = None, config: KNNScorerConfig | None = None, **kwargs):
        """
        初始化 KNN 评分器

        Args:
            oid (str | None): 算子标识
            config (KNNScorerConfig | None): 评分器配置
            **kwargs: 透传给基类的参数，支持:
                - n_neighbors (int): 近邻数量，默认 5
                - distance_metric (str): 距离度量，默认 "euclidean"
                - score_method (str): 分数合并方式，默认 "maximum"
        """
        super().__init__(oid=oid, config=config, **kwargs)
        self._index = None
        """近邻索引实例"""
        self._train_data: np.ndarray | None = None
        """训练数据引用（供推理查询）"""

    def _fit_data(self, x: np.ndarray, params: None) -> None:
        """
        构建 KNN 近邻索引

        使用 sklearn.neighbors.NearestNeighbors 构建索引。
        当训练样本数不足 K 个时，自动调整 K。

        Args:
            x (np.ndarray): 训练数据，形状 (n_samples, n_features)
            params (None): 无训练参数
        """
        from sklearn.neighbors import NearestNeighbors

        # 当训练样本数不足 K 时，自动调整
        effective_k = min(self.config.n_neighbors, len(x))

        # 构建近邻索引
        self._index = NearestNeighbors(
            n_neighbors=effective_k,
            metric=self.config.distance_metric.value,
        )
        self._index.fit(x)
        self._train_data = x

    def _run_data(self, x: np.ndarray, params: None, idx: pd.Index | None = None) -> (
            np.ndarray | tuple[np.ndarray, KNNScorerExtraOutput]):
        """
        计算 KNN 异常分数

        查询每个样本的 K 个近邻距离，根据 score_method 合并为异常分数。

        Args:
            x (np.ndarray): 输入数据，形状 (n_samples, n_features)
            params (None): 无运行参数
            idx (pd.Index | None): 输入数据的行索引

        Returns:
            tuple[np.ndarray, KNNScorerExtraOutput]:
                - scores: 异常分数 ndarray，形状 (n_samples,)
                - eo: 附加输出，包含使用的参数元信息
        """
        # 查询 K 个近邻距离（不包含自身）
        distances, _ = self._index.kneighbors(x)

        # 根据 score_method 合并 K 个距离
        if self.config.score_method == KNNScoreMethod.MEDIAN:
            scores = np.median(distances, axis=1)
        elif self.config.score_method == KNNScoreMethod.MEAN:
            scores = np.mean(distances, axis=1)
        else:
            # 默认 "maximum"
            scores = np.max(distances, axis=1)

        eo = KNNScorerExtraOutput(
            n_neighbors=self.config.n_neighbors,
            distance_metric=self.config.distance_metric,
            score_method=self.config.score_method,
        )
        return scores.ravel(), eo

    def _save_fit_state(self, path: Path) -> None:
        """
        保存训练状态：训练参数 + KNN 索引 + 训练数据

        在基类保存 last_fit_params 的基础上，额外将 sklearn 近邻索引和训练数据持久化。

        Args:
            path (Path): 目标目录路径
        """
        super()._save_fit_state(path)
        # 持久化近邻索引（sklearn 对象使用 pickle 序列化）
        with open(path / self._INDEX_FILE, 'wb') as f:
            pickle.dump(self._index, f)
        # 持久化训练数据
        np.save(path / self._TRAIN_DATA_FILE, self._train_data)

    def _load_fit_state(self, path: Path) -> None:
        """
        恢复训练状态：训练参数 + KNN 索引 + 训练数据

        从 pkl 文件恢复近邻索引，从 npy 文件恢复训练数据，并将 _fitted 标记为 True。

        Args:
            path (Path): 源目录路径
        """
        super()._load_fit_state(path)
        # 恢复近邻索引
        with open(path / self._INDEX_FILE, 'rb') as f:
            self._index = pickle.load(f)
        # 恢复训练数据
        self._train_data = np.load(path / self._TRAIN_DATA_FILE)
        # 恢复训练完成标记
        self._fitted = True


class KNNDetectorConfig(BaseModel):
    """
    KNN 检测器实例参数

    Attributes:
        n_neighbors (int): 近邻数量 K
        distance_metric (KNNDistanceMetric): 距离度量方式
        score_method (KNNScoreMethod): 分数合并方式
        percentile (float): 百分位阈值
    """
    n_neighbors: int = Field(
        default=5, ge=1, le=20,
        description="近邻数量 K"
    )
    distance_metric: KNNDistanceMetric = Field(
        default=KNNDistanceMetric.EUCLIDEAN,
        description="距离度量方式"
    )
    score_method: KNNScoreMethod = Field(
        default=KNNScoreMethod.MAXIMUM,
        description="分数合并方式"
    )
    percentile: float = Field(
        default=95.0, ge=50.0, le=99.9,
        description="百分位阈值"
    )


class KNNDetector(UnsupervisedNumericOperatorMixin[None],
                  BaseDeciderMixin[None],
                  NumericOperator[None, KNNDetectorConfig, None]):
    """
    KNN 检测器 — 组合 KNNScorer + PercentileDecider

    端到端的 KNN 异常检测器:

    ::

        KNNDetector
          ├── KNNScorer (Direct)
          │     _fit_data: 构建 KNN 索引
          │     _run_data: 计算 KNN 距离分数
          └── PercentileDecider
                _fit_data: 学习训练分数的百分位阈值
                _run_data: scores > percentile_threshold → labels

    使用示例::

        detector = KNNDetector(config=KNNDetectorConfig(n_neighbors=5, percentile=95.0))
        detector.fit(train_data)
        labels, eo = detector.run(test_data)

    Input:
        x: 特征矩阵，形状 (n_samples, n_features)

    Output:
        0/1 异常标签，形状 (n_samples,)，1 表示异常。
        标签由内部 KNNScorer 输出的异常分数经 PercentileDecider 的百分位阈值决策得到

    泛型参数:
        - EO: None（无附加输出）
        - C: KNNDetectorConfig
        - RP: None（无运行参数）
    """

    @classmethod
    def name(cls) -> str:
        """返回算子标识名称"""
        return "knn_detector"

    @classmethod
    def version(cls) -> tuple[int, ...]:
        """返回算子版本号

        Returns:
            tuple[int, ...]: 版本号三元组 ``(1, 0, 0)``
        """
        return (1, 0, 0)

    def __init__(self, *, oid: str | None = None, config: KNNDetectorConfig | None = None, **kwargs):
        """
        初始化 KNN 检测器

        自动创建 KNNScorer 和 PercentileDecider 子组件。

        Args:
            oid (str | None): 算子标识
            config (KNNDetectorConfig | None): 检测器配置
            **kwargs: 透传给基类的参数，支持:
                - n_neighbors (int): 近邻数量，默认 5
                - distance_metric (str): 距离度量，默认 "euclidean"
                - score_method (str): 分数合并方式，默认 "maximum"
                - percentile (float): 百分位阈值，默认 95.0
        """
        super().__init__(oid=oid, config=config, **kwargs)
        # 提取子组件配置参数
        scorer_config = KNNScorerConfig(
            n_neighbors=self.config.n_neighbors,
            distance_metric=self.config.distance_metric,
            score_method=self.config.score_method,
        )
        self._scorer = KNNScorer(config=scorer_config)
        self._decider = PercentileDecider(config=PercentileDeciderConfig(percentile=self.config.percentile))

    def _fit_data(self, x: np.ndarray, params: None) -> None:
        """
        训练检测器 — 先训练 Scorer，再用训练分数训练 Decider

        Args:
            x (np.ndarray): 训练数据
            params (None): 无训练参数
        """
        # 步骤1: 训练评分器
        self._scorer.fit(x)

        # 步骤2: 计算训练数据的异常分数
        scores, _ = self._scorer.run(x)

        # 步骤3: 训练决策器（使用训练分数）
        self._decider.fit(scores)

    def _run_data(self, x: np.ndarray, params: None, idx: pd.Index | None = None) -> np.ndarray:
        """
        检测推理: Scorer → Decider

        Args:
            x (np.ndarray): 输入数据
            params (None): 无运行参数
            idx (pd.Index | None): 输入数据的行索引

        Returns:
            np.ndarray: 二分类标签，1=异常/0=正常
        """
        # 步骤1: 评分器推理
        scores, _ = self._scorer.run(x)

        # 步骤2: 决策器推理
        labels, _ = self._decider.run(scores)

        return labels

    def _save_fit_state(self, path: Path) -> None:
        """
        保存训练状态：训练参数 + 子组件状态

        将 KNNScorer 和 PercentileDecider 分别保存到子目录。

        Args:
            path (Path): 目标目录路径
        """
        super()._save_fit_state(path)
        # 保存评分器和决策器到子目录
        self._scorer.save(path / '_scorer')
        self._decider.save(path / '_decider')

    def _load_fit_state(self, path: Path) -> None:
        """
        恢复训练状态：训练参数 + 子组件状态

        从子目录恢复 KNNScorer 和 PercentileDecider，并将 _fitted 标记为 True。

        Args:
            path (Path): 源目录路径
        """
        super()._load_fit_state(path)
        # 恢复子组件
        self._scorer = KNNScorer.load(path / '_scorer')
        self._decider = PercentileDecider.load(path / '_decider')
        # 恢复训练完成标记
        self._fitted = True
