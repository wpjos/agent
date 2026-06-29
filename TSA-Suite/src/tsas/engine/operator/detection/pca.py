# -*- coding: utf-8 -*-

"""
PCA 异常检测算子

基于主成分分析（PCA）重构误差的异常检测。
核心思想: 正常数据可以被低维主成分有效重构，异常数据的重构误差显著偏大。

包含:
    - PCAPredictor: PCA 预测器，学习主成分并输出重构值
    - PCAScorer: PCA 评分器，组合 PCAPredictor + ResidualScorer，输出异常分数
    - PCADetector: 端到端检测器，组合 PCAScorer + PercentileDecider，输出二分类标签

示例用法::

    # 直接评分
    scorer = PCAScorer(n_components=3)
    scorer.fit(train_data)
    scores, eo = scorer.run(test_data)

    # 端到端检测
    detector = PCADetector(config=PCADetectorConfig(n_components=3, percentile=95.0))
    detector.fit(train_data)
    labels, eo = detector.run(test_data)
"""

from pathlib import Path

import numpy as np
import pandas as pd
from pydantic import BaseModel, ConfigDict, Field

from tsas.engine.operator.base import NumericOperator, UnsupervisedNumericOperatorMixin
from tsas.engine.operator.detection.base import BaseDeciderMixin, BasePredictorMixin, SingleScorerMixin
from tsas.engine.operator.detection.percentile_decider import (
    PercentileDecider,
    PercentileDeciderConfig,
    PercentileDeciderExtraOutput,
)
from tsas.engine.operator.detection.residual_scorer import (
    ResidualMetric,
    ResidualScorer,
    ResidualScorerConfig,
    ResidualScorerExtraOutput,
)

__all__ = [
    'PCAPredictorConfig',
    'PCAPredictorExtraOutput',
    'PCAPredictor',
    'PCAScorerConfig',
    'PCAScorerExtraOutput',
    'PCAScorer',
    'PCADetectorConfig',
    'PCADetectorExtraOutput',
    'PCADetector',
]


class PCAPredictorConfig(BaseModel):
    """
    PCA 预测器实例参数

    Attributes:
        n_components (int): 保留的主成分数量，必须为正整数
    """
    n_components: int = Field(default=2, ge=1, description="保留的主成分数量，必须为正整数")


class PCAPredictorExtraOutput(BaseModel):
    """
    PCA 预测器附加输出

    Attributes:
        explained_variance_ratio (list[float]): 各主成分的解释方差比
        n_components (int): 保留的主成分数量
    """
    explained_variance_ratio: list[float] = Field(default=[], description="各主成分的解释方差比")
    n_components: int = Field(default=2, description="保留的主成分数量")


class PCAPredictor(UnsupervisedNumericOperatorMixin[None],
                   BasePredictorMixin[PCAPredictorExtraOutput, PCAPredictorConfig, None],
                   NumericOperator[PCAPredictorExtraOutput, PCAPredictorConfig, None]):
    """
    PCA 预测器 — 重构型预测器

    基于主成分分析的重构型预测器。训练阶段学习数据的主成分方向，
    推理阶段将输入投影到主成分空间再重构回原始空间。

    核心逻辑:
        - ``_fit``: 使用 SVD 分解学习主成分矩阵 ``_components`` 和均值 ``_mean``
        - ``_run``: 投影 → 重构，输出与输入同维度的重构值

    数学原理:
        - 投影: ``z = (x - mean) @ components.T``（降维）
        - 重构: ``x_hat = z @ components + mean``（升维）
        - 重构值即为预测值，后续与真实值的残差反映异常程度

    注意:
        当 n_components 大于特征数时，自动调整为特征数。

    Input:
        x: 特征矩阵，形状 (n_samples, n_features)

    Output:
        重构值矩阵，与输入同形状

    泛型参数:
        - EO: PCAPredictorExtraOutput（附加输出由 ``_eo_type`` 自动渲染）
        - C: PCAPredictorConfig
        - RP: None（无运行参数）
        - FP: None（无训练参数）
    """

    # 训练状态文件名
    _LEARNED_STATE_FILE = '_learned_state.npz'

    @classmethod
    def name(cls) -> str:
        return "pca_predictor"

    @classmethod
    def version(cls) -> tuple[int, ...]:
        """返回算子版本号

        Returns:
            tuple[int, ...]: 版本号三元组 ``(1, 0, 0)``
        """
        return (1, 0, 0)

    def __init__(self, *, oid: str | None = None, config: PCAPredictorConfig | None = None, **kwargs):
        """
        初始化 PCA 预测器

        Args:
            oid (str | None): 算子标识
            config (PCAPredictorConfig | None): PCA 预测器配置
            **kwargs: 透传给基类的参数，支持:
                - n_components (int): 保留的主成分数量，默认 2
        """
        super().__init__(oid=oid, config=config, **kwargs)
        self._components: np.ndarray | None = None
        """主成分矩阵，形状 (n_components, n_features)"""
        self._mean: np.ndarray | None = None
        """训练数据的列均值向量"""
        self._explained_variance_ratio: np.ndarray | None = None
        """各主成分的解释方差比"""

    def _fit_data(self, x: np.ndarray, params: None) -> None:
        """
        学习主成分

        对去中心化的训练数据进行 SVD 分解，提取前 n_components 个主成分。
        当 n_components 大于特征数时，自动调整。

        Args:
            x (np.ndarray): 训练数据，形状 (n_samples, n_features)
            params (None): 无训练参数
        """
        # 去中心化
        self._mean = x.mean(axis=0)
        x_centered = x - self._mean

        # 自动调整 n_components（不超过特征数）
        effective_k = min(self.config.n_components, x.shape[1])

        # SVD 分解: Vh 的前 k 行为主成分方向
        # full_matrices=False 使 Vh 形状为 (min(n,p), p)，节省计算
        _, s, vh = np.linalg.svd(x_centered, full_matrices=False)

        # 提取前 effective_k 个主成分
        self._components = vh[:effective_k]

        # 计算解释方差比
        total_var = np.sum(s ** 2)
        if total_var > 0:
            self._explained_variance_ratio = (s[:effective_k] ** 2) / total_var
        else:
            # 退化情况: 所有特征为常数
            self._explained_variance_ratio = np.zeros(effective_k)

    def _run_data(self, x: np.ndarray, params: None, idx: pd.Index | None = None) -> np.ndarray | tuple[
        np.ndarray, PCAPredictorExtraOutput]:

        """
        PCA 重构预测

        将输入投影到主成分空间再重构回原始空间，
        输出与输入同维度的重构值。

        Args:
            x (np.ndarray): 输入数据，形状 (n_samples, n_features)
            params (None): 无运行参数
            idx (pd.Index | None): 输入数据的行索引

        Returns:
            tuple[np.ndarray, PCAPredictorExtraOutput]:
                - 重构值 ndarray，形状 (n_samples, n_features)
                - 附加输出，包含解释方差比和主成分数
        """
        # 投影: (x - mean) @ components.T → (n_samples, n_components)
        z = (x - self._mean) @ self._components.T
        # 重构: z @ components + mean → (n_samples, n_features)
        pred = z @ self._components + self._mean

        eo = PCAPredictorExtraOutput(
            explained_variance_ratio=self._explained_variance_ratio.tolist(),
            n_components=self._components.shape[0],
        )
        return pred, eo

    def _save_fit_state(self, path: Path) -> None:
        """
        保存训练状态：训练参数 + PCA 主成分和统计量

        在基类保存 last_fit_params 的基础上，额外将 _mean、_components 和
        _explained_variance_ratio 持久化到 npz 文件。

        Args:
            path (Path): 目标目录路径
        """
        super()._save_fit_state(path)
        # 持久化学习到的 PCA 参数
        np.savez(
            path / self._LEARNED_STATE_FILE,
            mean=self._mean,
            components=self._components,
            explained_variance_ratio=self._explained_variance_ratio,
        )

    def _load_fit_state(self, path: Path) -> None:
        """
        恢复训练状态：训练参数 + PCA 主成分和统计量

        从 npz 文件恢复 _mean、_components 和 _explained_variance_ratio，
        并将 _fitted 标记为 True。

        Args:
            path (Path): 源目录路径
        """
        super()._load_fit_state(path)
        # 恢复学习到的 PCA 参数
        data = np.load(path / self._LEARNED_STATE_FILE)
        self._mean = data['mean']
        self._components = data['components']
        self._explained_variance_ratio = data['explained_variance_ratio']
        # 恢复训练完成标记
        self._fitted = True


# ============================================================================
# PCA 评分器
# ============================================================================


class PCAScorerConfig(BaseModel):
    """
    PCA 评分器实例参数

    Attributes:
        n_components (int): PCA 保留的主成分数量
        metric (ResidualMetric): 残差计算方式
    """
    n_components: int = Field(default=3, ge=1, description="PCA 保留的主成分数量")
    metric: ResidualMetric = Field(default=ResidualMetric.MSE, description="残差计算方式: 'mse' 或 'mae'")


class PCAScorerExtraOutput(BaseModel):
    """
    PCA 评分器附加输出

    聚合子组件 PCAPredictor 和 ResidualScorer 的附加输出。

    Attributes:
        pca_eo (PCAPredictorExtraOutput | None): PCAPredictor 的附加输出（解释方差比等）
        residual_eo (ResidualScorerExtraOutput | None): ResidualScorer 的附加输出（逐变量分数）
    """
    pca_eo: PCAPredictorExtraOutput | None = Field(default=None, description="PCAPredictor 的附加输出（解释方差比等）")
    residual_eo: ResidualScorerExtraOutput | None = Field(default=None,
                                                          description="ResidualScorer 的附加输出（逐变量分数）")

    model_config = ConfigDict(arbitrary_types_allowed=True)


class PCAScorer(SingleScorerMixin[None],
                UnsupervisedNumericOperatorMixin[None],
                NumericOperator[PCAScorerExtraOutput, PCAScorerConfig, None]):
    """
    PCA 评分器 — 组合 PCAPredictor + ResidualScorer

    基于 PCA 重构误差的异常评分器。

    内部数据流::

        输入 x → PCAPredictor.run(x) → (x_pred, pca_eo)
               → ResidualScorer.run((x, x_pred)) → (scores, residual_eo)
               → 输出: (1D 异常分数, PCAScorerExtraOutput)

    训练阶段仅训练 PCAPredictor（ResidualScorer 为 BiNumericOperator，无需训练）。

    Input:
        x: 特征矩阵，形状 (n_samples, n_features)

    Output:
        异常分数，形状 (n_samples,)，值越大越异常。
        分数由内部 PCAPredictor 的重构值与原始输入的残差经 ResidualScorer 聚合得到

    泛型参数:
        - EO: PCAScorerExtraOutput（附加输出由 ``_eo_type`` 自动渲染）
        - C: PCAScorerConfig
        - RP: None（无运行参数）
    """

    @classmethod
    def name(cls) -> str:
        return "pca_scorer"

    @classmethod
    def version(cls) -> tuple[int, ...]:
        """返回算子版本号

        Returns:
            tuple[int, ...]: 版本号三元组 ``(1, 0, 0)``
        """
        return (1, 0, 0)

    def __init__(self, *, oid: str | None = None, config: PCAScorerConfig | None = None, **kwargs):
        """
        初始化 PCA 评分器

        自动创建 PCAPredictor 和 ResidualScorer 子组件。

        Args:
            oid (str | None): 算子标识
            config (PCAScorerConfig | None): PCA 评分器配置
            **kwargs: 透传给基类的参数，支持:
                - n_components (int): PCA 主成分数，默认 3
                - metric (str): 残差计算方式，默认 "mse"
        """
        super().__init__(oid=oid, config=config, **kwargs)
        self._predictor = PCAPredictor(config=PCAPredictorConfig(n_components=self.config.n_components))
        self._scorer = ResidualScorer(config=ResidualScorerConfig(metric=self.config.metric))

    def _fit_data(self, x: np.ndarray, params: None) -> None:
        """
        训练 PCA 评分器

        仅训练 PCAPredictor（学习主成分），ResidualScorer 无需训练。

        Args:
            x (np.ndarray): 训练数据，形状 (n_samples, n_features)
            params (None): 无训练参数
        """
        self._predictor.fit(x)

    def _run_data(self, x: np.ndarray, params: None, idx: pd.Index | None = None) -> tuple[
        np.ndarray, PCAScorerExtraOutput]:
        """
        计算 PCA 重构误差异常分数

        Args:
            x (np.ndarray): 输入数据，形状 (n_samples, n_features)
            params (None): 无运行参数
            idx (pd.Index | None): 输入数据的行索引

        Returns:
            tuple[np.ndarray, PCAScorerExtraOutput]:
                - scores: 1D 异常分数，形状 (n_samples,)
                - eo: 聚合子组件的附加输出
        """
        # 步骤1: PCA 重构
        x_pred, pca_eo = self._predictor.run(x)

        # 步骤2: 残差评分
        scores, residual_eo = self._scorer.run((x, x_pred))

        return scores.ravel(), PCAScorerExtraOutput(pca_eo=pca_eo, residual_eo=residual_eo)

    def _save_fit_state(self, path: Path) -> None:
        """
        保存训练状态：训练参数 + 子组件状态

        将 PCAPredictor 保存到子目录 _predictor。
        ResidualScorer 为无状态算子，无需保存。

        Args:
            path (Path): 目标目录路径
        """
        super()._save_fit_state(path)
        # 保存预测器（ResidualScorer 无状态，无需保存）
        self._predictor.save(path / '_predictor')

    def _load_fit_state(self, path: Path) -> None:
        """
        恢复训练状态：训练参数 + 子组件状态

        从子目录 _predictor 恢复 PCAPredictor，并将 _fitted 标记为 True。
        ResidualScorer 在 __init__ 中重建，无需从磁盘恢复。

        Args:
            path (Path): 源目录路径
        """
        super()._load_fit_state(path)
        # 恢复预测器
        self._predictor = PCAPredictor.load(path / '_predictor')
        # 恢复训练完成标记
        self._fitted = True


# ============================================================================
# PCA 检测器
# ============================================================================


class PCADetectorConfig(BaseModel):
    """
    PCA 检测器实例参数

    Attributes:
        n_components (int): PCA 保留的主成分数量
        metric (ResidualMetric): 残差计算方式
        percentile (float): 百分位阈值
    """
    n_components: int = Field(default=3, ge=1, description="PCA 保留的主成分数量")
    metric: ResidualMetric = Field(default=ResidualMetric.MSE, description="残差计算方式: 'mse' 或 'mae'")
    percentile: float = Field(default=95.0, ge=50.0, le=99.9, description="百分位阈值")


class PCADetectorExtraOutput(BaseModel):
    """
    PCA 检测器附加输出

    聚合子组件 PCAScorer 和 PercentileDecider 的附加输出。

    Attributes:
        scorer_eo (PCAScorerExtraOutput | None): PCAScorer 的附加输出
        decider_eo (PercentileDeciderExtraOutput | None): PercentileDecider 的附加输出
    """
    scorer_eo: PCAScorerExtraOutput | None = Field(default=None, description="PCAScorer 的附加输出")
    decider_eo: PercentileDeciderExtraOutput | None = Field(default=None, description="PercentileDecider 的附加输出")

    model_config = ConfigDict(arbitrary_types_allowed=True)


class PCADetector(UnsupervisedNumericOperatorMixin[None],
                  BaseDeciderMixin[None],
                  NumericOperator[PCADetectorExtraOutput, PCADetectorConfig, None]):
    """
    PCA 检测器 — 组合 PCAScorer + PercentileDecider

    端到端的 PCA 异常检测器:

    ::

        PCADetector
          ├── PCAScorer
          │     ├── PCAPredictor
          │     │     _fit: 学习主成分
          │     │     _run: 输出重构值
          │     └── ResidualScorer
          │           _run: 计算重构残差分数
          └── PercentileDecider
                _fit: 学习训练分数的百分位阈值
                _run: scores > threshold → labels

    使用示例::

        detector = PCADetector(config=PCADetectorConfig(n_components=3, percentile=95.0))
        detector.fit(train_data)
        labels, eo = detector.run(test_data)

    Input:
        x: 特征矩阵，形状 (n_samples, n_features)

    Output:
        0/1 异常标签，形状 (n_samples,)，1 表示异常。
        标签由内部 PCAScorer 输出的重构残差分数经 PercentileDecider 的百分位阈值决策得到

    泛型参数:
        - EO: PCADetectorExtraOutput（附加输出由 ``_eo_type`` 自动渲染）
        - C: PCADetectorConfig
        - RP: None（无运行参数）
    """

    @classmethod
    def name(cls) -> str:
        return "pca_detector"

    @classmethod
    def version(cls) -> tuple[int, ...]:
        """返回算子版本号

        Returns:
            tuple[int, ...]: 版本号三元组 ``(1, 0, 0)``
        """
        return (1, 0, 0)

    def __init__(self, *, oid: str | None = None, config: PCADetectorConfig | None = None, **kwargs):
        """
        初始化 PCA 检测器

        自动创建 PCAScorer 和 PercentileDecider 子组件。

        Args:
            oid (str | None): 算子标识
            config (PCADetectorConfig | None): PCA 检测器配置
            **kwargs: 透传给基类的参数，支持:
                - n_components (int): PCA 主成分数，默认 3
                - metric (str): 残差计算方式，默认 "mse"
                - percentile (float): 百分位阈值，默认 95.0
        """
        super().__init__(oid=oid, config=config, **kwargs)
        scorer_config = PCAScorerConfig(
            n_components=self.config.n_components,
            metric=self.config.metric,
        )
        self._scorer = PCAScorer(config=scorer_config)
        self._decider = PercentileDecider(config=PercentileDeciderConfig(percentile=self.config.percentile))

    def _fit_data(self, x: np.ndarray, params: None) -> None:
        """
        训练 PCA 检测器

        1. 训练 PCAScorer（学习 PCA 主成分）
        2. 用训练数据计算训练分数
        3. 用训练分数训练 PercentileDecider

        Args:
            x (np.ndarray): 训练数据，形状 (n_samples, n_features)
            params (None): 无训练参数
        """
        # 步骤1: 训练评分器
        self._scorer.fit(x)
        # 步骤2: 计算训练分数
        scores, _ = self._scorer.run(x)
        # 步骤3: 训练决策器（学习百分位阈值）
        self._decider.fit(scores)

    def _run_data(self, x: np.ndarray, params: None, idx: pd.Index | None = None) -> tuple[
        np.ndarray, PCADetectorExtraOutput]:
        """
        检测推理: PCAScorer → PercentileDecider

        Args:
            x (np.ndarray): 输入数据，形状 (n_samples, n_features)
            params (None): 无运行参数
            idx (pd.Index | None): 输入数据的行索引

        Returns:
            tuple[np.ndarray, PCADetectorExtraOutput]:
                - labels: 二分类标签，1=异常/0=正常
                - eo: 聚合子组件的附加输出
        """
        # 步骤1: 评分器推理
        scores, scorer_eo = self._scorer.run(x)
        # 步骤2: 决策器推理
        labels, decider_eo = self._decider.run(scores)

        return labels, PCADetectorExtraOutput(scorer_eo=scorer_eo, decider_eo=decider_eo)

    def _save_fit_state(self, path: Path) -> None:
        """
        保存训练状态：训练参数 + 子组件状态

        将 PCAScorer 和 PercentileDecider 分别保存到子目录。

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

        从子目录恢复 PCAScorer 和 PercentileDecider，并将 _fitted 标记为 True。

        Args:
            path (Path): 源目录路径
        """
        super()._load_fit_state(path)
        # 恢复子组件
        self._scorer = PCAScorer.load(path / '_scorer')
        self._decider = PercentileDecider.load(path / '_decider')
        # 恢复训练完成标记
        self._fitted = True
