# -*- coding: utf-8 -*-

"""
时序预测算子基类

输入输出约定::

    fit(x, y):
        x: (timesteps, num_features)   DataFrame 或 ndarray
        y: (timesteps, num_targets)    DataFrame 或 ndarray

    run(x):
        x: (seq_len, num_features) 或 (batch, seq_len, num_features)
        返回: (pred_len, num_targets) 或 (batch, pred_len, num_targets)
"""

from abc import ABCMeta, abstractmethod
from typing import Generic, TypeVar

import numpy as np
import pandas as pd
from pydantic import BaseModel

from tsas.engine.operator.base import (
    BaseOperator,
    C,
    DataFrameMeta,
    EO,
    FP,
    I,
    LearnableOperatorMixin,
    NumericData,
    O,
    RP,
)

__all__ = [
    'BaseForecaster',
    'ForecastExtraOutput',
]


class ForecastExtraOutput(BaseModel):
    """预测算子通用附加输出占位模型。

    子类可扩展该模型以返回注意力权重、隐藏状态等额外信息。
    """
    pass


class BaseForecaster(
    LearnableOperatorMixin[NumericData, NumericData, FP],
    BaseOperator[NumericData, NumericData, C, RP],
    Generic[EO, C, RP, FP],
    metaclass=ABCMeta,
):
    """时序预测算子基类。

    参考 detection 算子的 ``NumericOperator`` 设计，支持 DataFrame + ndarray
    双类型输入输出，并通过 ``_name_output_columns`` 让子类定制输出列名。

    泛型参数:
        - EO: 附加输出类型（当前基类不使用，保留给未来扩展）
        - C: 实例参数类型
        - RP: 运行参数类型
        - FP: 训练参数类型
    """

    _eo_type: type[BaseModel] | None = None

    def _to_ndarray(self, x: NumericData) -> np.ndarray:
        """将 DataFrame 或 ndarray 统一转换为 ndarray。"""
        if isinstance(x, pd.DataFrame):
            return x.to_numpy()
        if isinstance(x, np.ndarray):
            return x
        raise TypeError(f"输入数据类型必须是 pd.DataFrame 或 np.ndarray，但当前是 {type(x)}")

    def _name_output_columns(self, output_data: np.ndarray, meta: DataFrameMeta | None, params: RP | None) -> list[str]:
        """推断输出 DataFrame 的列名。

        默认策略按 ``forecast_{i}`` 生成，与输出目标维度一一对应。
        子类可覆写以自定义列名（例如沿用输入列名）。

        Args:
            output_data (np.ndarray): 计算结果 ndarray
            meta (DataFrameMeta | None): 输入 DataFrame 的元信息快照
            params (RP | None): 运行参数

        Returns:
            list[str]: 输出列名列表
        """
        n_targets = output_data.shape[-1] if output_data.ndim >= 2 else 1
        return [f"forecast_{i}" for i in range(n_targets)]

    def _to_dataframe(self, arr: np.ndarray, original: NumericData) -> NumericData:
        """当原始输入为 DataFrame 时，将 ndarray 结果转换回 DataFrame。

        输出索引从 0 开始连续编号；输出列名由 ``_name_output_columns`` 决定。
        子类可覆写 ``_name_output_columns`` 以自定义列名。
        """
        if isinstance(original, pd.DataFrame):
            meta = DataFrameMeta.from_dataframe(original)
            n_steps = arr.shape[-2] if arr.ndim >= 2 else arr.shape[0]
            index = pd.RangeIndex(start=0, stop=n_steps, step=1)
            columns = self._name_output_columns(arr, meta, None)
            return pd.DataFrame(arr, columns=columns, index=index)
        return arr

    def _validate_fit_input(self, x: NumericData, y: NumericData) -> tuple[np.ndarray, np.ndarray]:
        """校验训练输入并返回 ndarray 元组。"""
        x_arr = self._to_ndarray(x)
        y_arr = self._to_ndarray(y)
        if x_arr.ndim != 2:
            raise ValueError(f"训练输入 x 必须是 2-D (timesteps, features)，当前维度 {x_arr.ndim}")
        if y_arr.ndim == 1:
            y_arr = y_arr.reshape(-1, 1)
        elif y_arr.ndim != 2:
            raise ValueError(f"训练目标 y 必须是 1-D 或 2-D，当前维度 {y_arr.ndim}")
        if x_arr.shape[0] != y_arr.shape[0]:
            raise ValueError(
                f"训练输入 x 与目标 y 的时间步数不一致: {x_arr.shape[0]} != {y_arr.shape[0]}"
            )
        return x_arr, y_arr

    def _validate_run_input(self, x: NumericData) -> np.ndarray:
        """校验推理输入并返回 ndarray。"""
        x_arr = self._to_ndarray(x)
        if x_arr.ndim not in (2, 3):
            raise ValueError(f"推理输入 x 必须是 2-D (seq_len, features) 或 3-D (batch, seq_len, features)，"
                             f"当前维度 {x_arr.ndim}")
        return x_arr

    def _fit(self, x: NumericData, y: NumericData, *, params: FP | None) -> None:
        """训练模板方法。

        完成输入校验后调用 ``_fit_data``；训练完成后自动设置 ``_fitted = True``。
        """
        x_arr, y_arr = self._validate_fit_input(x, y)
        self._fit_data(x_arr, y_arr, params=params)
        self._fitted = True

    def _run(self, x: NumericData, *, params: RP | None) -> NumericData:
        """推理模板方法。

        完成输入校验后调用 ``_run_data``，并按原始输入类型回包输出。
        """
        x_arr = self._validate_run_input(x)
        output = self._run_data(x_arr, params=params)
        return self._to_dataframe(output, x)

    @abstractmethod
    def _fit_data(self, x: np.ndarray, y: np.ndarray, *, params: FP | None) -> None:
        """子类实现的核心训练逻辑。

        Args:
            x: 训练输入，形状 ``(timesteps, num_features)``
            y: 训练目标，形状 ``(timesteps, num_targets)``
            params: 验证后的训练参数
        """
        ...

    @abstractmethod
    def _run_data(self, x: np.ndarray, *, params: RP | None) -> np.ndarray:
        """子类实现的核心推理逻辑。

        Args:
            x: 推理输入，形状 ``(seq_len, num_features)`` 或
               ``(batch, seq_len, num_features)``
            params: 验证后的运行参数

        Returns:
            np.ndarray: 预测结果，形状 ``(pred_len, num_targets)`` 或
            ``(batch, pred_len, num_targets)``
        """
        ...
