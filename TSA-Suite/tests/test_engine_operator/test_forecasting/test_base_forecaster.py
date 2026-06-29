# -*- coding: utf-8 -*-

"""
BaseForecaster 时序预测基类单元测试

对应源文件：
- forecasting/base.py: BaseForecaster

测试范围：
- IO 辅助方法（DataFrame/ndarray 转换）
- 输入校验（维度、长度）
- 抽象方法约束
- DataFrame 输出回包
"""

import numpy as np
import pandas as pd
import pytest
from pandas import DataFrame

from tsas.engine.operator.forecasting.base import BaseForecaster, ForecastExtraOutput


# ============================================================================
# 最小化具体子类 fixture
# ============================================================================

class DummyForecaster(BaseForecaster[ForecastExtraOutput, None, None, None]):
    """仅用于测试 BaseForecaster 模板方法的最小化实现"""

    @classmethod
    def name(cls):
        return "dummy_forecaster"

    @classmethod
    def version(cls):
        return (1, 0, 0)

    def _fit_data(self, x, y, *, params=None):
        self._learned = True

    def _run_data(self, x, *, params=None):
        # 简单返回未来 5 步的最后一个特征值
        pred_len = 5
        num_targets = 1
        if x.ndim == 3:
            return np.tile(x[:, -1:, -1:], (1, pred_len, num_targets))
        return np.tile(x[-1:, -1:], (pred_len, num_targets))


@pytest.fixture
def dummy_forecaster():
    return DummyForecaster()


@pytest.fixture
def train_data():
    np.random.seed(42)
    return np.cumsum(np.random.randn(100, 3), axis=0)


@pytest.fixture
def train_df(train_data):
    return DataFrame(train_data, columns=["a", "b", "c"])


@pytest.fixture
def test_window():
    np.random.seed(123)
    return np.cumsum(np.random.randn(20, 3), axis=0)


@pytest.fixture
def test_window_df(test_window):
    return DataFrame(test_window, columns=["a", "b", "c"])


# ============================================================================
# IO 辅助方法测试
# ============================================================================

class TestBaseForecasterIO:
    """测试基类 IO 转换逻辑"""

    def test_to_ndarray_from_ndarray(self, dummy_forecaster, train_data):
        """目的：验证 ndarray 输入直接返回"""
        arr = dummy_forecaster._to_ndarray(train_data)
        assert isinstance(arr, np.ndarray)
        np.testing.assert_array_equal(arr, train_data)

    def test_to_ndarray_from_dataframe(self, dummy_forecaster, train_df):
        """目的：验证 DataFrame 输入转换为 ndarray"""
        arr = dummy_forecaster._to_ndarray(train_df)
        assert isinstance(arr, np.ndarray)
        np.testing.assert_array_equal(arr, train_df.values)

    def test_to_ndarray_invalid_type(self, dummy_forecaster):
        """目的：验证非法输入类型报错"""
        with pytest.raises(TypeError):
            dummy_forecaster._to_ndarray([1, 2, 3])

    def test_to_dataframe_from_dataframe(self, dummy_forecaster, test_window_df):
        """目的：验证 DataFrame 输入时输出为 DataFrame"""
        arr = dummy_forecaster._run_data(test_window_df.values)
        df = dummy_forecaster._to_dataframe(arr, test_window_df)
        assert isinstance(df, DataFrame)
        assert df.shape == (5, 1)
        assert list(df.columns) == ["forecast_0"]

    def test_to_dataframe_from_ndarray(self, dummy_forecaster, test_window):
        """目的：验证 ndarray 输入时输出保持 ndarray"""
        arr = dummy_forecaster._run_data(test_window)
        out = dummy_forecaster._to_dataframe(arr, test_window)
        assert isinstance(out, np.ndarray)


# ============================================================================
# 输入校验测试
# ============================================================================

class TestBaseForecasterValidation:
    """测试基类输入校验"""

    def test_validate_fit_input_ok(self, dummy_forecaster, train_data):
        """目的：验证合法训练输入通过校验"""
        y = train_data[:, [-1]]
        x_arr, y_arr = dummy_forecaster._validate_fit_input(train_data, y)
        assert x_arr.ndim == 2
        assert y_arr.ndim == 2
        assert x_arr.shape[0] == y_arr.shape[0]

    def test_validate_fit_input_1d_target(self, dummy_forecaster, train_data):
        """目的：验证 1-D 目标自动 reshape 为 2-D"""
        y = train_data[:, -1]
        _, y_arr = dummy_forecaster._validate_fit_input(train_data, y)
        assert y_arr.ndim == 2
        assert y_arr.shape == (100, 1)

    def test_validate_fit_input_dimension_mismatch(self, dummy_forecaster):
        """目的：验证 x 与 y 长度不一致时报错"""
        x = np.random.randn(100, 3)
        y = np.random.randn(50, 1)
        with pytest.raises(ValueError, match="时间步数不一致"):
            dummy_forecaster._validate_fit_input(x, y)

    def test_validate_fit_input_x_not_2d(self, dummy_forecaster):
        """目的：验证 x 不是 2-D 时报错"""
        x = np.random.randn(100, 3, 2)
        y = np.random.randn(100, 1)
        with pytest.raises(ValueError, match="2-D"):
            dummy_forecaster._validate_fit_input(x, y)

    def test_validate_run_input_2d(self, dummy_forecaster, test_window):
        """目的：验证 2-D 推理输入通过"""
        arr = dummy_forecaster._validate_run_input(test_window)
        assert arr.ndim == 2

    def test_validate_run_input_3d(self, dummy_forecaster):
        """目的：验证 3-D 批量推理输入通过"""
        x = np.random.randn(4, 20, 3)
        arr = dummy_forecaster._validate_run_input(x)
        assert arr.ndim == 3

    def test_validate_run_input_invalid_dim(self, dummy_forecaster):
        """目的：验证非 2-D/3-D 推理输入报错"""
        x = np.random.randn(20, 3, 2, 1)
        with pytest.raises(ValueError, match="2-D .* 3-D"):
            dummy_forecaster._validate_run_input(x)


# ============================================================================
# 模板方法流程测试
# ============================================================================

class TestBaseForecasterPipeline:
    """测试基类 fit/run 模板流程"""

    def test_fit_sets_fitted(self, dummy_forecaster, train_data):
        """目的：验证 fit 后 _fitted 为 True"""
        y = train_data[:, [-1]]
        dummy_forecaster.fit(train_data, y)
        assert dummy_forecaster.is_fitted
        assert dummy_forecaster._learned is True

    def test_run_output_shape_2d(self, dummy_forecaster, train_data, test_window):
        """目的：验证 2-D 输入输出形状"""
        y = train_data[:, [-1]]
        dummy_forecaster.fit(train_data, y)
        pred = dummy_forecaster.run(test_window)
        assert pred.shape == (5, 1)

    def test_run_output_shape_3d(self, dummy_forecaster, train_data):
        """目的：验证 3-D 批量输入输出形状"""
        y = train_data[:, [-1]]
        dummy_forecaster.fit(train_data, y)
        x_batch = np.stack([train_data[:20], train_data[20:40]])
        pred = dummy_forecaster.run(x_batch)
        assert pred.shape == (2, 5, 1)

    def test_run_with_dataframe(self, dummy_forecaster, train_df, test_window_df):
        """目的：验证 DataFrame 端到端流程"""
        y = train_df[["c"]]
        dummy_forecaster.fit(train_df, y)
        pred = dummy_forecaster.run(test_window_df)
        assert isinstance(pred, DataFrame)
        assert pred.shape == (5, 1)

    def test_before_fit_raises(self, dummy_forecaster, test_window):
        """目的：验证未训练时 run 抛出 RuntimeError"""
        with pytest.raises(RuntimeError):
            dummy_forecaster.run(test_window)


# ============================================================================
# 抽象类约束测试
# ============================================================================

class TestBaseForecasterAbstract:
    """测试抽象方法约束"""

    def test_cannot_instantiate_base(self):
        """目的：验证 BaseForecaster 不能直接实例化"""
        with pytest.raises(TypeError):
            BaseForecaster()

    def test_missing_abstract_methods(self):
        """目的：验证未实现抽象方法的子类不能实例化"""
        class IncompleteForecaster(BaseForecaster[ForecastExtraOutput, None, None, None]):
            @classmethod
            def name(cls):
                return "incomplete"

        with pytest.raises(TypeError):
            IncompleteForecaster()
