# -*- coding: utf-8 -*-

"""特征选择器单元测试。"""

import numpy as np
import pandas as pd
import pytest
from pydantic import ValidationError

from tsas.engine.operator.feature.selection.base import BaseFeatureSelectorConfig
from tsas.engine.operator.feature.selection.simple_selector import (
    ColumnSelector,
    ColumnSelectorConfig,
    VarianceThresholdSelector,
    VarianceThresholdSelectorConfig,
)


def test_column_selector_dataframe_by_names_keeps_global_indices() -> None:
    """测试目的：验证 DataFrame 按列名静态选择时，输出列名与全局索引映射正确。"""
    data = pd.DataFrame({'a': [1, 2], 'b': [3, 4], 'c': [5, 6]})
    selector = ColumnSelector(config=ColumnSelectorConfig(input_columns=['c', 'a']))

    output, eo = selector.run(data)

    assert list(output.columns) == ['c', 'a']
    assert output.values.tolist() == [[5, 1], [6, 2]]
    assert eo.selected_indices == [2, 0]


def test_column_selector_ndarray_by_indices() -> None:
    """测试目的：验证 ndarray 按列索引静态选择时，主输出和 EO 映射正确。"""
    data = np.array([[1, 2, 3], [4, 5, 6]])
    selector = ColumnSelector(config=ColumnSelectorConfig(input_columns=[2, 0]))

    output, eo = selector.run(data)

    assert output.tolist() == [[3, 1], [6, 4]]
    assert eo.selected_indices == [2, 0]


def test_input_columns_rejects_mixed_and_duplicate_values() -> None:
    """测试目的：验证配置层禁止混用字符串/整数，也禁止重复列配置。"""
    with pytest.raises(ValidationError):
        BaseFeatureSelectorConfig(input_columns=['a', 1])
    with pytest.raises(ValidationError):
        BaseFeatureSelectorConfig(input_columns=[1, 1])


def test_column_selector_rejects_string_columns_for_ndarray() -> None:
    """测试目的：验证 ndarray 输入不能使用字符串列名配置。"""
    selector = ColumnSelector(config=ColumnSelectorConfig(input_columns=['a']))

    with pytest.raises(TypeError):
        selector.run(np.array([[1, 2]]))


def test_variance_threshold_selector_fit_run_and_save_load(tmp_path) -> None:
    """测试目的：验证方差阈值选择器训练、推理、持久化加载后的结果一致。"""
    train = pd.DataFrame({'a': [1, 1, 1], 'b': [1, 2, 3], 'c': [3, 3, 3], 'd': [1, 3, 5]})
    test = pd.DataFrame({'a': [9], 'b': [8], 'c': [7], 'd': [6]})
    selector = VarianceThresholdSelector(
        config=VarianceThresholdSelectorConfig(input_columns=['a', 'b', 'd'], threshold=1.0))

    selector.fit(train)
    output, eo = selector.run(test)

    assert list(output.columns) == ['d']
    assert output.values.tolist() == [[6]]
    assert eo.selected_indices == [3]
    assert len(eo.variances) == 3

    selector.save(tmp_path)
    loaded = VarianceThresholdSelector.load(tmp_path)
    loaded_output, loaded_eo = loaded.run(test)
    assert loaded_output.equals(output)
    assert loaded_eo == eo


def test_variance_threshold_selector_empty_output_warns(caplog) -> None:
    """测试目的：验证方差阈值过高时返回零列数据，并通过 EO 给出空映射。"""
    data = pd.DataFrame({'a': [1, 1], 'b': [2, 2]})
    selector = VarianceThresholdSelector(config=VarianceThresholdSelectorConfig(threshold=10.0))

    selector.fit(data)
    output, eo = selector.run(data)

    assert output.shape == (2, 0)
    assert eo.selected_indices == []


def test_variance_threshold_selector_requires_fit_before_run() -> None:
    """测试目的：验证训练型选择器不允许未训练直接运行。"""
    selector = VarianceThresholdSelector(config=VarianceThresholdSelectorConfig())

    with pytest.raises(RuntimeError):
        selector.run(np.array([[1, 2]]))
