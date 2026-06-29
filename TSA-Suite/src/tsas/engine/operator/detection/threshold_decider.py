# -*- coding: utf-8 -*-

"""
固定阈值决策器

将异常分数与固定阈值比较，超过阈值则判定为异常。
不需要训练，初始化后即可直接使用。

示例用法::

    decider = ThresholdDecider(config=ThresholdDeciderConfig(threshold=3.0))
    decider.fit(dummy_data)  # 无需训练，但需满足 fit 接口
    labels, eo = decider.run(scores)
"""

import numpy as np
import pandas as pd
from pydantic import BaseModel, Field

from tsas.engine.operator.base import NumericOperator
from tsas.engine.operator.detection.base import BaseDeciderMixin

__all__ = [
    'ThresholdDeciderConfig',
    'ThresholdDecider',
]


class ThresholdDeciderConfig(BaseModel):
    """
    固定阈值决策器实例参数

    Attributes:
        threshold: 异常判定阈值，分数严格大于此值判定为异常
    """
    threshold: float = Field(default=3.0, gt=0, description="异常判定阈值，分数 > threshold 判定为异常")


class ThresholdDecider(BaseDeciderMixin[None], NumericOperator[None, ThresholdDeciderConfig, None]):
    """
    固定阈值决策器 — 无需训练

    将异常分数与配置中的固定阈值比较，严格大于阈值判定为异常（label=1），
    否则判定为正常（label=0）。

    Input:
        scores: 异常分数，形状 (n_samples,) 或 (n_samples, 1)

    Output:
        0/1 标签，形状 (n_samples,)，1 表示异常。
        阈值来自 Config（``threshold``），严格大于阈值判定为异常
    """

    @classmethod
    def name(cls) -> str:
        return "threshold_decider"

    @classmethod
    def version(cls) -> tuple[int, ...]:
        """返回算子版本号

        Returns:
            tuple[int, ...]: 版本号三元组 ``(1, 0, 0)``
        """
        return (1, 0, 0)

    def _run_data(self, x: np.ndarray, params: None, idx: pd.Index | None = None) -> np.ndarray:
        """
        阈值决策

        将异常分数与阈值比较，严格大于阈值判定为异常。
        输出标签 ndarray 为 1D 整数数组。

        Args:
            x(np.ndarray): 异常分数，形状 (n_samples,) 或 (n_samples, 1)
            params(None): 无运行参数

        Returns:
            np.ndarray: 标签 ndarray，形状 (n_samples,)，1=异常/0=正常
        """
        # 严格大于阈值判定为异常
        return (x > self.config.threshold).astype(int).ravel()
