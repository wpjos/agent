# -*- coding: utf-8 -*-

"""特征选择器基础类型。

本模块定义 ``feature.selection`` 子包的公共基类和接口约定。特征选择器
（Selector）继承数值算子管线，主输出为选择后的特征数据，附加输出 EO
强制包含 ``selected_indices``，用于记录输出列到完整输入列位置的映射。
"""

from abc import ABCMeta
from typing import Generic, TypeVar, cast

import numpy as np
import pandas as pd
from pydantic import BaseModel, Field, model_validator

from tsas.engine.operator.base import (
    FP,
    NumericData,
    NumericOperator,
    RP,
    SupervisedNumericOperatorMixin,
    UnsupervisedNumericOperatorMixin,
)

__all__ = [
    'BaseFeatureSelectorConfig',
    'FeatureSelectorExtraOutput',
    'BaseFeatureSelectorMixin',
    'BaseFeatureSelector',
    'UnsupervisedFeatureSelector',
    'SupervisedFeatureSelector',
    'FSEO',
    'FSC',
]

FSEO = TypeVar('FSEO', bound='FeatureSelectorExtraOutput')
FSC = TypeVar('FSC', bound='BaseFeatureSelectorConfig')


class BaseFeatureSelectorConfig(BaseModel):
    """特征选择器基础实例参数。

    Attributes:
        input_columns (list[str] | list[int] | None): 候选特征列。``None`` 表示完整输入的全部列；
            ``list[str]`` 表示按 ``DataFrame`` 列名选择；``list[int]`` 表示按完整输入列位置选择。
            不允许字符串与整数混用，也不允许重复。
    """

    input_columns: list[str] | list[int] | None = Field(default=None, description='候选特征列名或列位置索引')

    @model_validator(mode='after')
    def _validate_input_columns(self):
        """校验候选列配置。

        Returns:
            BaseFeatureSelectorConfig: 当前配置对象。

        Raises:
            ValueError: 当 ``input_columns`` 混用 ``str`` 和 ``int``，或存在重复项时抛出。
        """
        if self.input_columns is None:
            return self
        columns = cast(list[str] | list[int], self.input_columns)
        if len(columns) != len(set(cast(list, columns))):
            raise ValueError('input_columns 不允许包含重复项')
        has_str = any(isinstance(col, str) for col in columns)
        has_int = any(isinstance(col, int) for col in columns)
        if has_str and has_int:
            raise ValueError('input_columns 不允许混用 str 和 int')
        return self


class FeatureSelectorExtraOutput(BaseModel):
    """特征选择器附加输出基类型。

    Attributes:
        selected_indices (list[int]): 输出特征各列对应完整输入数据中的原始列位置索引，顺序与主输出列顺序一致。
    """

    selected_indices: list[int] = Field(description='输出列到完整输入列位置的映射')


class BaseFeatureSelectorMixin(Generic[FSEO, FSC, RP], metaclass=ABCMeta):
    """特征选择器领域混入。

    该 Mixin 负责解析 ``input_columns``、缓存候选列到完整输入列的映射、恢复
    ``DataFrame`` 输出列名，并提供子类复用的列选择工具方法。

    Attributes:
        _candidate_indices (list[int] | None): 最近一次运行或训练解析出的候选列全局索引。
        _last_selected_indices (list[int] | None): 最近一次运行得到的输出列全局索引。
    """

    def __init__(self, **kwargs):
        """初始化特征选择器混入状态。

        Args:
            **kwargs: 透传给后续基类的初始化参数。
        """
        super().__init__(**kwargs)
        self._candidate_indices: list[int] | None = None
        self._last_selected_indices: list[int] | None = None

    def _resolve_candidate_indices(self, x: NumericData) -> list[int]:
        """解析候选列到完整输入列位置的映射。

        Args:
            x (NumericData): 原始输入数据。

        Returns:
            list[int]: 候选列在完整输入中的列位置索引。

        Raises:
            TypeError: ``ndarray`` 输入使用字符串列名时抛出。
            ValueError: 列名不存在、索引越界或配置非法时抛出。
        """
        column_count = len(x.columns) if isinstance(x, pd.DataFrame) else (1 if x.ndim == 1 else x.shape[1])
        input_columns = self._selector_config().input_columns
        if input_columns is None:
            return list(range(column_count))
        if not input_columns:
            return []
        if isinstance(input_columns[0], str):
            if not isinstance(x, pd.DataFrame):
                raise TypeError('ndarray 输入的 input_columns 只能使用整数列索引')
            missing = [col for col in input_columns if col not in x.columns]
            if missing:
                raise ValueError(f'输入数据缺少以下列: {missing}')
            return [int(x.columns.get_loc(col)) for col in input_columns]
        indices = [int(col) for col in input_columns]
        invalid = [idx for idx in indices if idx < 0 or idx >= column_count]
        if invalid:
            raise ValueError(f'input_columns 存在越界列索引: {invalid}')
        return indices

    def _filter_data(self, x: NumericData, params: RP | None) -> NumericData:
        """按候选列筛选运行输入。

        Args:
            x (NumericData): 原始输入数据。
            params (RP | None): 运行参数。

        Returns:
            NumericData: 只包含候选列的数据。
        """
        indices = self._resolve_candidate_indices(x)
        self._candidate_indices = indices
        if isinstance(x, pd.DataFrame):
            return x.iloc[:, indices]
        data = x.reshape(-1, 1) if x.ndim == 1 else x
        return data[:, indices]

    def _filter_fit_data(self, x: NumericData, *args, **kwargs) -> NumericData | tuple[NumericData, NumericData]:
        """按候选列筛选训练输入。

        Args:
            x (NumericData): 原始训练输入数据。
            *args: 兼容有监督 Mixin 的标签参数。
            **kwargs: 兼容训练模板传入的命名参数。

        Returns:
            NumericData | tuple[NumericData, NumericData]: 筛选后的训练输入；有监督场景下同时返回标签。
        """
        indices = self._resolve_candidate_indices(x)
        self._candidate_indices = indices
        filtered = x.iloc[:, indices] if isinstance(x, pd.DataFrame) else (x.reshape(-1, 1) if x.ndim == 1 else x)[
            :, indices]
        if args and isinstance(args[0], (pd.DataFrame, np.ndarray)):
            return filtered, args[0]
        return filtered

    def _selector_config(self) -> FSC:
        """获取已校验的选择器配置。

        Returns:
            FSC: 特征选择器配置对象。

        Raises:
            RuntimeError: 当前算子配置缺失时抛出。
        """
        config = getattr(self, 'config', None)
        if config is None:
            raise RuntimeError('特征选择器缺少 config 配置')
        return cast(FSC, config)

    def _to_global_indices(self, local_indices: list[int]) -> list[int]:
        """将候选列局部索引转换为完整输入全局索引。

        Args:
            local_indices (list[int]): 相对于候选列矩阵的位置索引。

        Returns:
            list[int]: 相对于完整输入的位置索引。

        Raises:
            RuntimeError: 尚未解析候选列映射时抛出。
        """
        if self._candidate_indices is None:
            raise RuntimeError('尚未解析候选列映射，无法生成 selected_indices')
        return [self._candidate_indices[i] for i in local_indices]

    def _select_columns(self, x: np.ndarray, local_indices: list[int], eo: FSEO) -> tuple[np.ndarray, FSEO]:
        """根据局部索引生成主输出和 EO。

        Args:
            x (np.ndarray): 候选列矩阵。
            local_indices (list[int]): 需要保留的候选列局部索引。
            eo (FSEO): 已构造的附加输出对象。

        Returns:
            tuple[np.ndarray, FSEO]: 选择后的矩阵与附加输出。
        """
        self._last_selected_indices = eo.selected_indices
        return x[:, local_indices], eo

    def _name_output_columns(self, output_data: np.ndarray, meta, params: RP | None) -> list[str]:
        """恢复 ``DataFrame`` 输出列名。

        Args:
            output_data (np.ndarray): 输出数据。
            meta: 候选列数据的元信息。
            params (RP | None): 运行参数。

        Returns:
            list[str]: 输出列名列表。
        """
        if self._last_selected_indices is None:
            return list(meta.column_names[:output_data.shape[1]])
        # meta 来自候选列过滤后的 DataFrame，因此需要先把全局索引映射回候选列位置。
        selected_names = []
        for global_idx in self._last_selected_indices:
            local_idx = self._candidate_indices.index(global_idx)
            selected_names.append(meta.column_names[local_idx])
        return selected_names


class BaseFeatureSelector(BaseFeatureSelectorMixin[FSEO, FSC, None], NumericOperator[FSEO, FSC, None],
                          Generic[FSEO, FSC], metaclass=ABCMeta):
    """非训练型特征选择器基类。

    泛型参数:
        FSEO: 附加输出类型，必须继承 ``FeatureSelectorExtraOutput``。
        FSC: 实例参数类型，必须继承 ``BaseFeatureSelectorConfig``。
    """


class UnsupervisedFeatureSelector(
    UnsupervisedNumericOperatorMixin[FP],
    BaseFeatureSelectorMixin[FSEO, FSC, None],
    NumericOperator[FSEO, FSC, None],
    Generic[FSEO, FSC, FP],
    metaclass=ABCMeta,
):
    """无监督训练型特征选择器基类。

    泛型参数:
        FSEO: 附加输出类型，必须继承 ``FeatureSelectorExtraOutput``。
        FSC: 实例参数类型，必须继承 ``BaseFeatureSelectorConfig``。
        FP: 训练参数类型。
    """

    def _filter_fit_data(self, x: NumericData, params: FP | None) -> NumericData:
        """按候选列筛选无监督训练输入。

        Args:
            x (NumericData): 原始训练输入数据。
            params (FP | None): 训练参数。

        Returns:
            NumericData: 筛选后的训练输入数据。
        """
        filtered = BaseFeatureSelectorMixin._filter_fit_data(self, x, params=params)
        return cast(NumericData, filtered)


class SupervisedFeatureSelector(
    SupervisedNumericOperatorMixin[FP],
    BaseFeatureSelectorMixin[FSEO, FSC, None],
    NumericOperator[FSEO, FSC, None],
    Generic[FSEO, FSC, FP],
    metaclass=ABCMeta,
):
    """有监督训练型特征选择器基类。

    泛型参数:
        FSEO: 附加输出类型，必须继承 ``FeatureSelectorExtraOutput``。
        FSC: 实例参数类型，必须继承 ``BaseFeatureSelectorConfig``。
        FP: 训练参数类型。
    """

    def _filter_fit_data(self, x: NumericData, y: NumericData, params: FP | None) -> tuple[NumericData, NumericData]:
        """按候选列筛选有监督训练输入。

        Args:
            x (NumericData): 原始训练输入数据。
            y (NumericData): 标签或目标数据。
            params (FP | None): 训练参数。

        Returns:
            tuple[NumericData, NumericData]: 筛选后的输入数据与原始标签数据。
        """
        filtered = BaseFeatureSelectorMixin._filter_fit_data(self, x, y, params=params)
        return cast(tuple[NumericData, NumericData], filtered)
