# -*- coding: utf-8 -*-

"""内置特征选择器实现。"""

from pathlib import Path
from typing import cast

import numpy as np
import pandas as pd
from loguru import logger
from pydantic import Field

from tsas.engine.operator.feature.selection.base import (
    BaseFeatureSelector,
    BaseFeatureSelectorConfig,
    FeatureSelectorExtraOutput,
    UnsupervisedFeatureSelector,
)

__all__ = [
    'ColumnSelectorConfig',
    'ColumnSelectorExtraOutput',
    'ColumnSelector',
    'VarianceThresholdSelectorConfig',
    'VarianceThresholdSelectorExtraOutput',
    'VarianceThresholdSelector',
]


class ColumnSelectorConfig(BaseFeatureSelectorConfig):
    """静态列选择器实例参数。

    Attributes:
        input_columns (list[str] | list[int] | None): 需要保留的列；``None`` 表示保留全部列。
    """


class ColumnSelectorExtraOutput(FeatureSelectorExtraOutput):
    """静态列选择器附加输出。

    Attributes:
        selected_indices (list[int]): 输出列到完整输入列位置的映射。
    """


class ColumnSelector(BaseFeatureSelector[ColumnSelectorExtraOutput, ColumnSelectorConfig]):
    """静态按列名或列索引选择特征。

    ``ColumnSelector`` 不需要训练，直接将候选列作为最终输出列。

    Input:
        x: 候选特征矩阵，形状 (n_samples, n_features)

    Output:
        选择后的特征矩阵，形状 (n_samples, n_selected)，列由 Config 的 ``input_columns`` 指定
    """

    _eo_type = ColumnSelectorExtraOutput
    _config_type = ColumnSelectorConfig

    @classmethod
    def name(cls) -> str:
        """返回算子注册名称。

        Returns:
            str: 算子名称。
        """
        return 'column_selector'

    @classmethod
    def version(cls) -> tuple[int, ...]:
        """返回算子版本号

        Returns:
            tuple[int, ...]: 版本号三元组 ``(1, 0, 0)``
        """
        return (1, 0, 0)

    def _run_data(self, x: np.ndarray, params: None, idx: pd.Index | None = None) -> tuple[
        np.ndarray, ColumnSelectorExtraOutput]:
        """执行静态列选择。

        Args:
            x (np.ndarray): 候选列矩阵。
            params (None): 无运行参数。
            idx (pd.Index | None): 输入行索引。

        Returns:
            tuple[np.ndarray, ColumnSelectorExtraOutput]: 选择后的矩阵与附加输出。
        """
        local_indices = list(range(x.shape[1]))
        selected_indices = self._to_global_indices(local_indices)
        eo = ColumnSelectorExtraOutput(selected_indices=selected_indices)
        return self._select_columns(x, local_indices, eo)


class VarianceThresholdSelectorConfig(BaseFeatureSelectorConfig):
    """方差阈值选择器实例参数。

    Attributes:
        input_columns (list[str] | list[int] | None): 候选特征列。
        threshold (float): 方差阈值，保留方差严格大于该阈值的特征。
    """

    threshold: float = Field(default=0.0, ge=0.0, description='方差阈值')


class VarianceThresholdSelectorExtraOutput(FeatureSelectorExtraOutput):
    """方差阈值选择器附加输出。

    Attributes:
        selected_indices (list[int]): 输出列到完整输入列位置的映射。
        variances (list[float]): 训练阶段候选特征的方差，顺序与候选列一致。
    """

    variances: list[float] = Field(description='训练阶段候选特征方差')


class VarianceThresholdSelector(
    UnsupervisedFeatureSelector[VarianceThresholdSelectorExtraOutput, VarianceThresholdSelectorConfig, None]
):
    """根据训练集方差阈值保留特征。

    训练阶段计算候选特征的方差，保留方差严格大于 ``threshold`` 的特征列。
    推理阶段直接按训练结果选择列。

    Input:
        x: 候选特征矩阵，形状 (n_samples, n_features)

    Output:
        选择后的特征矩阵，形状 (n_samples, n_selected)，保留方差大于阈值的特征列
    """

    _eo_type = VarianceThresholdSelectorExtraOutput
    _config_type = VarianceThresholdSelectorConfig

    _STATE_FILE = 'variance_threshold_state.npz'

    @classmethod
    def name(cls) -> str:
        """返回算子注册名称。

        Returns:
            str: 算子名称。
        """
        return 'variance_threshold_selector'

    @classmethod
    def version(cls) -> tuple[int, ...]:
        """返回算子版本号

        Returns:
            tuple[int, ...]: 版本号三元组 ``(1, 0, 0)``
        """
        return (1, 0, 0)

    def __init__(self, *, oid: str | None = None, config: VarianceThresholdSelectorConfig | None = None, **kwargs):
        """初始化方差阈值选择器。

        Args:
            oid (str | None): 算子实例唯一标识后缀。
            config (VarianceThresholdSelectorConfig | None): 选择器配置。
            **kwargs: 透传给配置模型的键值参数。
        """
        super().__init__(oid=oid, config=config, **kwargs)
        self._variances: np.ndarray | None = None
        """训练阶段候选特征方差。"""
        self._selected_local_indices: list[int] | None = None
        """训练后保留特征相对于候选列矩阵的局部索引。"""
        self._selected_global_indices: list[int] | None = None
        """训练后保留特征相对于完整输入的全局索引。"""

    def _fit_data(self, x: np.ndarray, params: None) -> None:
        """学习方差阈值选择结果。

        Args:
            x (np.ndarray): 候选列训练矩阵。
            params (None): 无训练参数。

        Returns:
            None: 本方法无返回值。
        """
        self._variances = np.var(x, axis=0)
        variances = cast(np.ndarray, self._variances)
        selected_local_indices = [idx for idx, value in enumerate(variances.tolist()) if value > self.config.threshold]
        self._selected_local_indices = selected_local_indices
        self._selected_global_indices = self._to_global_indices(selected_local_indices)
        if not self._selected_global_indices:
            logger.warning('VarianceThresholdSelector 未保留任何特征，将在 run 时返回空特征数据')

    def _run_data(self, x: np.ndarray, params: None, idx: pd.Index | None = None) -> tuple[
        np.ndarray, VarianceThresholdSelectorExtraOutput]:
        """按训练得到的方差选择结果执行推理。

        Args:
            x (np.ndarray): 候选列输入矩阵。
            params (None): 无运行参数。
            idx (pd.Index | None): 输入行索引。

        Returns:
            tuple[np.ndarray, VarianceThresholdSelectorExtraOutput]: 选择后的矩阵与附加输出。

        Raises:
            RuntimeError: 训练状态不完整时抛出。
        """
        if self._variances is None or self._selected_local_indices is None or self._selected_global_indices is None:
            raise RuntimeError('VarianceThresholdSelector 训练状态不完整')
        if not self._selected_global_indices:
            logger.warning('VarianceThresholdSelector 未保留任何特征，返回空特征数据')
        eo = VarianceThresholdSelectorExtraOutput(
            selected_indices=list(self._selected_global_indices),
            variances=[float(v) for v in self._variances],
        )
        return self._select_columns(x, list(self._selected_local_indices), eo)

    def _save_fit_state(self, path: Path) -> None:
        """保存训练状态。

        Args:
            path (Path): 目标目录路径。

        Returns:
            None: 本方法无返回值。
        """
        super()._save_fit_state(path)
        np.savez(
            path / self._STATE_FILE,
            variances=self._variances,
            selected_local_indices=np.array(self._selected_local_indices or [], dtype=int),
            selected_global_indices=np.array(self._selected_global_indices or [], dtype=int),
        )

    def _load_fit_state(self, path: Path) -> None:
        """加载训练状态。

        Args:
            path (Path): 源目录路径。

        Returns:
            None: 本方法无返回值。
        """
        super()._load_fit_state(path)
        data = np.load(path / self._STATE_FILE)
        self._variances = data['variances']
        self._selected_local_indices = data['selected_local_indices'].astype(int).tolist()
        self._selected_global_indices = data['selected_global_indices'].astype(int).tolist()
        self._fitted = True
