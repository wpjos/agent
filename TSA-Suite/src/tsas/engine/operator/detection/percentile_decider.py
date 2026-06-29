# -*- coding: utf-8 -*-
"""
百分位阈值决策器模块

训练阶段学习训练分数的指定百分位数作为阈值，
推理阶段将异常分数与该阈值比较，严格大于阈值则判定为异常。

示例用法::

    from tsas.engine.operator.detection.percentile_decider import PercentileDecider

    # 创建百分位决策器，使用默认第 95 百分位
    decider = PercentileDecider(oid="my_decider", config=PercentileDeciderConfig(percentile=95.0))

    # 训练阶段：从训练分数中学习百分位阈值
    train_scores = np.array([0.1, 0.3, 0.5, 0.7, 0.9, 1.1, 1.3])
    decider.fit(train_scores)

    # 推理阶段：比较分数 > 阈值，输出异常标签
    test_scores = np.array([0.4, 0.8, 1.0, 1.5])
    labels, eo = decider.run(test_scores)

主要组件:
    - PercentileDeciderConfig: 百分位决策器配置
    - PercentileDeciderExtraOutput: 百分位决策器额外输出
    - PercentileDecider: 百分位阈值决策器
"""

from pathlib import Path

import numpy as np
import pandas as pd
from pydantic import BaseModel, Field

from tsas.engine.operator.base import NumericOperator, UnsupervisedNumericOperatorMixin
from tsas.engine.operator.detection.base import BaseDeciderMixin

__all__ = [
    'PercentileDeciderConfig',
    'PercentileDeciderExtraOutput',
    'PercentileDecider',
]


class PercentileDeciderConfig(BaseModel):
    """
    百分位阈值决策器配置

    Attributes:
        percentile (float): 百分位数，训练分数的此百分位将作为阈值。
            取值范围 (0, 100)，默认 95.0。
    """
    percentile: float = Field(default=95.0, gt=0, lt=100, description="百分位数，训练分数的此百分位将作为阈值")


class PercentileDeciderExtraOutput(BaseModel):
    """
    百分位阈值决策器额外输出

    Attributes:
        threshold (float): 训练阶段学习到的百分位阈值
    """
    threshold: float = Field(description="训练阶段学习到的百分位阈值")


class PercentileDecider(BaseDeciderMixin[None],
                        UnsupervisedNumericOperatorMixin[None],
                        NumericOperator[PercentileDeciderExtraOutput, PercentileDeciderConfig, None]):
    """
    百分位阈值决策器（第 3 层 Decider）

    继承 BaseDeciderMixin、UnsupervisedNumericOperatorMixin 与 NumericOperator，
    用于在异常检测管线中将连续异常分数转换为离散异常标签。

    训练阶段：计算训练分数展平后的指定百分位数作为阈值。
    推理阶段：将分数严格大于阈值（>）的样本判定为异常（label=1），否则判定为正常（label=0）。

    注意：使用"严格大于"比较，即等于阈值的样本判定为正常。

    Input:
        scores: 异常分数，形状 (n_samples,) 或 (n_samples, 1)

    Output:
        0/1 标签，形状 (n_samples,)，1 表示异常。
        阈值由训练阶段学习得到（训练分数的指定百分位数），严格大于阈值判定为异常

    泛型参数:
        EO: PercentileDeciderExtraOutput（附加输出由 ``_eo_type`` 自动渲染，含学习到的阈值）
        C: PercentileDeciderConfig，百分位决策器配置
        RP: None，无运行参数
    """

    # 训练状态文件名
    _LEARNED_STATE_FILE = '_learned_state.npz'

    @classmethod
    def name(cls) -> str:
        """
        返回算子名称

        Returns:
            str: 固定返回 "percentile_decider"
        """
        return "percentile_decider"

    @classmethod
    def version(cls) -> tuple[int, ...]:
        """返回算子版本号

        Returns:
            tuple[int, ...]: 版本号三元组 ``(1, 0, 0)``
        """
        return (1, 0, 0)

    def __init__(self, *, oid: str | None = None, config: PercentileDeciderConfig | None = None, **kwargs):
        """
        初始化百分位阈值决策器

        Args:
            oid (str | None): 算子标识符，默认为 None
            config (PercentileDeciderConfig | None): 百分位决策器配置，默认为 None（使用默认配置）
            **kwargs: 透传给基类的参数
        """
        super().__init__(oid=oid, config=config, **kwargs)
        self._threshold: float | None = None
        """训练阶段学习到的百分位阈值"""

    def _fit_data(self, x: np.ndarray, params: None) -> None:
        """
        训练阶段：从训练分数中计算百分位阈值

        将训练分数展平为一维数组后，使用 np.percentile 计算指定百分位数作为阈值，
        结果存储在 self._threshold 中。

        Args:
            x (np.ndarray): 训练分数数组，形状任意（内部会展平为一维）
            params (None): 无训练参数
        """
        # 计算百分位阈值
        self._threshold = float(np.percentile(x.ravel(), self.config.percentile))

    def _run_data(self, x: np.ndarray, params: None, idx: pd.Index | None = None) -> np.ndarray | tuple[np.ndarray, PercentileDeciderExtraOutput]:
        """
        推理阶段：比较分数与阈值，输出异常标签

        将输入异常分数严格大于阈值（>）的样本判定为异常（label=1），
        否则判定为正常（label=0）。输出为一维整数数组及包含阈值的额外输出。

        Args:
            x (np.ndarray): 异常分数数组，形状 (n_samples,) 或 (n_samples, 1)
            params (None): 无运行参数
            idx (pd.Index | None): 输入数据的行索引

        Returns:
            tuple[np.ndarray, PercentileDeciderExtraOutput]:
                - labels: 标签数组，形状 (n_samples,)，1=异常 / 0=正常
                - eo: 额外输出，包含学习到的百分位阈值
        """
        # 严格大于阈值判定为异常
        labels = (x > self._threshold).astype(int).ravel()
        eo = PercentileDeciderExtraOutput(threshold=self._threshold)
        return labels, eo

    def _save_fit_state(self, path: Path) -> None:
        """
        保存训练状态：训练参数 + 学习到的阈值

        在基类保存 last_fit_params 的基础上，额外将 _threshold 持久化到 npz 文件。

        Args:
            path (Path): 目标目录路径
        """
        super()._save_fit_state(path)
        # 持久化学习到的阈值
        np.savez(path / self._LEARNED_STATE_FILE, threshold=np.array([self._threshold]))

    def _load_fit_state(self, path: Path) -> None:
        """
        恢复训练状态：训练参数 + 学习到的阈值

        从 npz 文件恢复 _threshold，并将 _fitted 标记为 True。

        Args:
            path (Path): 源目录路径
        """
        super()._load_fit_state(path)
        # 恢复学习到的阈值
        data = np.load(path / self._LEARNED_STATE_FILE)
        self._threshold = float(data['threshold'][0])
        # 恢复训练完成标记
        self._fitted = True
