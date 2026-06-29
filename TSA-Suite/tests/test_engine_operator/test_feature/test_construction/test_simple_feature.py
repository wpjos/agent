# -*- coding: utf-8 -*-

"""
简单特征构造算子测试

测试覆盖:
    1. SquareFeature: 逐元素平方
    2. PolynomialFeature: 多项式展开 (1:N)
    3. RollingMeanFeature: 滑动均值
    4. ColumnMedianFeature: 多列取中位数
    5. PCAFeature: PCA 降维 (fit -> run, state, save/load)

每种特征测试:
    - 基本计算正确性
    - DataFrame 输入/输出
    - ndarray 输入/输出
    - Config 校验
    - name() 类方法
    - 边界情况
"""

import numpy as np
import pandas as pd
import pytest

from tsas.engine.operator.feature.construction.base import (
    Alignment,
    Padding,
)
from tsas.engine.operator.feature.construction.simple_feature import (
    SquareConfig,
    SquareFeature,
    PolynomialConfig,
    PolynomialFeature,
    RollingMeanConfig,
    RollingMeanFeature,
    ColumnMedianConfig,
    ColumnMedianFeature,
    PCAConfig,
    PCAState,
    PCAFeature,
)


# ============================================================================
# SquareFeature 测试
# ============================================================================

class TestSquareFeature:
    """逐元素平方特征测试"""

    def test_name(self):
        assert SquareFeature.name() == "square_feature"

    def test_compute_correctness(self):
        """验证 compute 直接调用"""
        x = np.array([1.0, 2.0, 3.0, -1.0])
        result = SquareFeature.compute(x)
        np.testing.assert_array_equal(result, [1.0, 4.0, 9.0, 1.0])

    def test_run_dataframe_single_column(self):
        config = SquareConfig(input_columns=["val"])
        feat = SquareFeature(config=config)
        df = pd.DataFrame({"val": [1.0, 2.0, 3.0]})
        result = feat.run(df)
        assert isinstance(result, pd.DataFrame)
        assert "val_square" in result.columns
        np.testing.assert_array_equal(result["val_square"].values, [1.0, 4.0, 9.0])

    def test_run_dataframe_multi_column(self):
        config = SquareConfig(input_columns=["a", "b"])
        feat = SquareFeature(config=config)
        df = pd.DataFrame({"a": [1.0, 2.0], "b": [3.0, 4.0]})
        result = feat.run(df)
        assert "a_square" in result.columns
        assert "b_square" in result.columns
        np.testing.assert_array_equal(result["a_square"].values, [1.0, 4.0])
        np.testing.assert_array_equal(result["b_square"].values, [9.0, 16.0])

    def test_run_ndarray(self):
        config = SquareConfig(input_columns=["a"])
        feat = SquareFeature(config=config)
        arr = np.array([2.0, -3.0, 0.0])
        result = feat.run(arr)
        assert isinstance(result, np.ndarray)
        np.testing.assert_array_equal(result, [4.0, 9.0, 0.0])

    def test_run_preserves_index(self):
        config = SquareConfig(input_columns=["a"])
        feat = SquareFeature(config=config)
        idx = pd.Index([10, 20, 30])
        df = pd.DataFrame({"a": [1.0, 2.0, 3.0]}, index=idx)
        result = feat.run(df)
        assert list(result.index) == [10, 20, 30]

    def test_negative_values(self):
        config = SquareConfig(input_columns=["a"])
        feat = SquareFeature(config=config)
        df = pd.DataFrame({"a": [-1.0, -2.0, -3.0]})
        result = feat.run(df)
        np.testing.assert_array_equal(result["a_square"].values, [1.0, 4.0, 9.0])

    def test_zero_values(self):
        config = SquareConfig(input_columns=["a"])
        feat = SquareFeature(config=config)
        df = pd.DataFrame({"a": [0.0]})
        result = feat.run(df)
        np.testing.assert_array_equal(result["a_square"].values, [0.0])

    def test_config_validation_no_columns(self):
        with pytest.raises(Exception):
            SquareConfig(input_columns=[])

    def test_output_column_naming(self):
        config = SquareConfig(input_columns=["temperature"])
        feat = SquareFeature(config=config)
        df = pd.DataFrame({"temperature": [1.0]})
        result = feat.run(df)
        assert list(result.columns) == ["temperature_square"]


# ============================================================================
# PolynomialFeature 测试
# ============================================================================

class TestPolynomialFeature:
    """多项式展开特征测试"""

    def test_name(self):
        assert PolynomialFeature.name() == "polynomial_feature"

    def test_default_degrees(self):
        config = PolynomialConfig(input_columns=["a"])
        assert config.degrees == [2, 3]

    def test_custom_degrees(self):
        config = PolynomialConfig(input_columns=["a"], degrees=[2, 4, 6])
        assert config.degrees == [2, 4, 6]

    def test_config_validation_empty_degrees(self):
        with pytest.raises(Exception):
            PolynomialConfig(input_columns=["a"], degrees=[])

    def test_run_single_column(self):
        """单列: 输入 [2.0] -> [4.0, 8.0] (degrees=[2,3])"""
        config = PolynomialConfig(input_columns=["a"], degrees=[2, 3])
        feat = PolynomialFeature(config=config)
        df = pd.DataFrame({"a": [2.0]})
        result = feat.run(df)
        assert isinstance(result, pd.DataFrame)
        assert "a_poly_2" in result.columns
        assert "a_poly_3" in result.columns
        np.testing.assert_array_equal(result["a_poly_2"].values, [4.0])
        np.testing.assert_array_equal(result["a_poly_3"].values, [8.0])

    def test_run_multi_column(self):
        """多列: 每列展开为多列"""
        config = PolynomialConfig(input_columns=["x", "y"], degrees=[2])
        feat = PolynomialFeature(config=config)
        df = pd.DataFrame({"x": [3.0], "y": [4.0]})
        result = feat.run(df)
        assert "x_poly_2" in result.columns
        assert "y_poly_2" in result.columns
        np.testing.assert_array_equal(result["x_poly_2"].values, [9.0])
        np.testing.assert_array_equal(result["y_poly_2"].values, [16.0])

    def test_run_ndarray(self):
        config = PolynomialConfig(input_columns=["a"], degrees=[2, 3])
        feat = PolynomialFeature(config=config)
        arr = np.array([2.0, 3.0])
        result = feat.run(arr)
        assert isinstance(result, np.ndarray)
        assert result.shape == (2, 2)
        np.testing.assert_array_equal(result[:, 0], [4.0, 9.0])  # x^2
        np.testing.assert_array_equal(result[:, 1], [8.0, 27.0])  # x^3

    def test_multiple_degrees(self):
        """多个阶数展开"""
        config = PolynomialConfig(input_columns=["a"], degrees=[1, 2, 3])
        feat = PolynomialFeature(config=config)
        df = pd.DataFrame({"a": [2.0]})
        result = feat.run(df)
        assert result.shape[1] == 3
        np.testing.assert_array_equal(result["a_poly_1"].values, [2.0])
        np.testing.assert_array_equal(result["a_poly_2"].values, [4.0])
        np.testing.assert_array_equal(result["a_poly_3"].values, [8.0])

    def test_output_column_count(self):
        """输出列数 = 输入列数 * 阶数"""
        config = PolynomialConfig(input_columns=["a", "b"], degrees=[2, 3])
        feat = PolynomialFeature(config=config)
        df = pd.DataFrame({"a": [1.0], "b": [2.0]})
        result = feat.run(df)
        assert result.shape[1] == 4  # 2 columns * 2 degrees


# ============================================================================
# RollingMeanFeature 测试
# ============================================================================

class TestRollingMeanFeature:
    """滑动均值特征测试"""

    def test_name(self):
        assert RollingMeanFeature.name() == "rolling_mean_feature"

    def test_basic_no_padding(self):
        """无填充滑动均值"""
        config = RollingMeanConfig(
            input_columns=["a"],
            window_size=3,
            padding=None,
            alignment=Alignment.RIGHT,
        )
        feat = RollingMeanFeature(config=config)
        df = pd.DataFrame({"a": [1.0, 2.0, 3.0, 4.0, 5.0]})
        result = feat.run(df)
        assert len(result) == 3  # 5 - 3 + 1
        np.testing.assert_array_almost_equal(
            result["a_rolling_mean"].values, [2.0, 3.0, 4.0]
        )

    def test_edge_padding_right_align(self):
        config = RollingMeanConfig(
            input_columns=["a"],
            window_size=3,
            padding=Padding.EDGE,
            alignment=Alignment.RIGHT,
        )
        feat = RollingMeanFeature(config=config)
        df = pd.DataFrame({"a": [2.0, 4.0, 6.0]})
        result = feat.run(df)
        assert len(result) == 3
        # 填充后: [2, 2, 2, 4, 6] -> 窗口均值: [2.0, 8/3, 4.0]
        np.testing.assert_array_almost_equal(
            result["a_rolling_mean"].values, [2.0, 8.0 / 3.0, 4.0]
        )

    def test_nan_padding(self):
        config = RollingMeanConfig(
            input_columns=["a"],
            window_size=3,
            padding=Padding.NAN,
            alignment=Alignment.RIGHT,
        )
        feat = RollingMeanFeature(config=config)
        df = pd.DataFrame({"a": [1.0, 2.0, 3.0]})
        result = feat.run(df)
        assert len(result) == 3
        assert np.isnan(result["a_rolling_mean"].values[0])
        assert np.isnan(result["a_rolling_mean"].values[1])
        np.testing.assert_almost_equal(result["a_rolling_mean"].values[2], 2.0)

    def test_left_align_no_padding(self):
        config = RollingMeanConfig(
            input_columns=["a"],
            window_size=2,
            padding=None,
            alignment=Alignment.LEFT,
        )
        feat = RollingMeanFeature(config=config)
        idx = pd.Index([0, 1, 2, 3])
        df = pd.DataFrame({"a": [10.0, 20.0, 30.0, 40.0]}, index=idx)
        result = feat.run(df)
        assert len(result) == 3
        np.testing.assert_array_almost_equal(
            result["a_rolling_mean"].values, [15.0, 25.0, 35.0]
        )
        assert list(result.index) == [0, 1, 2]

    def test_multi_column(self):
        """多列滑动均值"""
        config = RollingMeanConfig(
            input_columns=["a", "b"],
            window_size=2,
            padding=None,
        )
        feat = RollingMeanFeature(config=config)
        df = pd.DataFrame({"a": [1.0, 2.0, 3.0], "b": [10.0, 20.0, 30.0]})
        result = feat.run(df)
        assert "a_rolling_mean" in result.columns
        assert "b_rolling_mean" in result.columns
        np.testing.assert_array_almost_equal(
            result["a_rolling_mean"].values, [1.5, 2.5]
        )
        np.testing.assert_array_almost_equal(
            result["b_rolling_mean"].values, [15.0, 25.0]
        )

    def test_ndarray_input(self):
        config = RollingMeanConfig(input_columns=["a"], window_size=2)
        feat = RollingMeanFeature(config=config)
        arr = np.array([1.0, 3.0, 5.0])
        result = feat.run(arr)
        assert isinstance(result, np.ndarray)
        np.testing.assert_array_almost_equal(result, [2.0, 4.0])

    def test_window_size_1(self):
        """窗口大小为 1: 恒等映射"""
        config = RollingMeanConfig(input_columns=["a"], window_size=1)
        feat = RollingMeanFeature(config=config)
        df = pd.DataFrame({"a": [1.0, 2.0, 3.0]})
        result = feat.run(df)
        assert len(result) == 3
        np.testing.assert_array_almost_equal(
            result["a_rolling_mean"].values, [1.0, 2.0, 3.0]
        )

    def test_config_validation_window_size_zero(self):
        with pytest.raises(Exception):
            RollingMeanConfig(input_columns=["a"], window_size=0)


# ============================================================================
# ColumnMedianFeature 测试
# ============================================================================

class TestColumnMedianFeature:
    """多列取中位数特征测试"""

    def test_name(self):
        assert ColumnMedianFeature.name() == "column_median_feature"

    def test_basic_median(self):
        config = ColumnMedianConfig(input_columns=["a", "b", "c"])
        feat = ColumnMedianFeature(config=config)
        df = pd.DataFrame({"a": [1.0, 2.0], "b": [3.0, 4.0], "c": [5.0, 6.0]})
        result = feat.run(df)
        assert isinstance(result, pd.DataFrame)
        assert "a_b_c_median" in result.columns
        np.testing.assert_array_equal(result["a_b_c_median"].values, [3.0, 4.0])

    def test_even_columns(self):
        """偶数列中位数"""
        config = ColumnMedianConfig(input_columns=["a", "b"])
        feat = ColumnMedianFeature(config=config)
        df = pd.DataFrame({"a": [1.0], "b": [3.0]})
        result = feat.run(df)
        np.testing.assert_array_equal(result["a_b_median"].values, [2.0])

    def test_custom_output_column(self):
        config = ColumnMedianConfig(input_columns=["a", "b"], output_column="my_median")
        feat = ColumnMedianFeature(config=config)
        df = pd.DataFrame({"a": [1.0], "b": [5.0]})
        result = feat.run(df)
        assert "my_median" in result.columns

    def test_ndarray_input(self):
        config = ColumnMedianConfig(input_columns=["a", "b"])
        feat = ColumnMedianFeature(config=config)
        arr = np.array([[1.0, 3.0], [2.0, 4.0]])
        result = feat.run(arr)
        assert isinstance(result, np.ndarray)
        np.testing.assert_array_equal(result, [2.0, 3.0])

    def test_three_columns_known_values(self):
        config = ColumnMedianConfig(input_columns=["x", "y", "z"])
        feat = ColumnMedianFeature(config=config)
        df = pd.DataFrame({"x": [10.0, 1.0], "y": [5.0, 2.0], "z": [1.0, 3.0]})
        result = feat.run(df)
        np.testing.assert_array_equal(result["x_y_z_median"].values, [5.0, 2.0])

    def test_preserves_index(self):
        config = ColumnMedianConfig(input_columns=["a", "b"])
        feat = ColumnMedianFeature(config=config)
        idx = pd.Index([100, 200])
        df = pd.DataFrame({"a": [1.0, 2.0], "b": [3.0, 4.0]}, index=idx)
        result = feat.run(df)
        assert list(result.index) == [100, 200]


# ============================================================================
# PCAFeature 测试
# ============================================================================

class TestPCAFeature:
    """PCA 降维特征测试"""

    def test_name(self):
        assert PCAFeature.name() == "pca_feature"

    def test_config_validation_n_components_positive(self):
        with pytest.raises(Exception):
            PCAConfig(input_columns=["a", "b"], n_components=0)

    def test_config_validation_n_components_negative(self):
        with pytest.raises(Exception):
            PCAConfig(input_columns=["a", "b"], n_components=-1)

    def test_fit_and_run_basic(self):
        """基本 fit -> run 流程"""
        config = PCAConfig(input_columns=["a", "b"], n_components=1)
        feat = PCAFeature(config=config)

        np.random.seed(42)
        train_df = pd.DataFrame({
            "a": np.random.randn(100),
            "b": np.random.randn(100),
        })
        feat.fit(train_df)
        assert feat.is_fitted is True
        assert feat.state is not None
        assert isinstance(feat.state, PCAState)

        test_df = pd.DataFrame({"a": [1.0, 2.0], "b": [3.0, 4.0]})
        result = feat.run(test_df)
        assert isinstance(result, pd.DataFrame)
        assert "pca_0" in result.columns
        assert result.shape[1] == 1
        assert result.shape[0] == 2

    def test_multi_component(self):
        """多维降维"""
        config = PCAConfig(input_columns=["a", "b", "c"], n_components=2)
        feat = PCAFeature(config=config)

        np.random.seed(42)
        data = np.random.randn(50, 3)
        train_df = pd.DataFrame(data, columns=["a", "b", "c"])
        feat.fit(train_df)

        result = feat.run(train_df)
        assert result.shape == (50, 2)
        assert "pca_0" in result.columns
        assert "pca_1" in result.columns

    def test_ndarray_fit_and_run(self):
        """ndarray 输入"""
        config = PCAConfig(input_columns=["a", "b"], n_components=1)
        feat = PCAFeature(config=config)

        np.random.seed(0)
        train_arr = np.random.randn(30, 2)
        feat.fit(train_arr)
        assert feat.is_fitted

        test_arr = np.random.randn(10, 2)
        result = feat.run(test_arr)
        assert isinstance(result, np.ndarray)
        assert result.shape == (10, 1)

    def test_state_has_mean_and_components(self):
        """训练后 state 包含 mean 和 components"""
        config = PCAConfig(input_columns=["a", "b"], n_components=1)
        feat = PCAFeature(config=config)
        df = pd.DataFrame({"a": [1.0, 2.0, 3.0], "b": [4.0, 5.0, 6.0]})
        feat.fit(df)

        assert feat.state.mean.shape == (2,)
        assert feat.state.components.shape == (2, 1)

    def test_run_before_fit_raises(self):
        config = PCAConfig(input_columns=["a", "b"], n_components=1)
        feat = PCAFeature(config=config)
        df = pd.DataFrame({"a": [1.0], "b": [2.0]})
        with pytest.raises(RuntimeError):
            feat.run(df)

    def test_save_load_roundtrip(self, tmp_path):
        """save/load 回环一致性"""
        config = PCAConfig(input_columns=["a", "b"], n_components=1)
        feat = PCAFeature(config=config)

        np.random.seed(42)
        df = pd.DataFrame({"a": np.random.randn(50), "b": np.random.randn(50)})
        feat.fit(df)

        save_dir = tmp_path / "pca"
        feat.save(save_dir)

        loaded = PCAFeature.load(save_dir)
        assert loaded.is_fitted is True
        assert loaded.state is not None
        np.testing.assert_array_almost_equal(loaded.state.mean, feat.state.mean)
        np.testing.assert_array_almost_equal(loaded.state.components, feat.state.components)

        # 运行结果一致
        test_df = pd.DataFrame({"a": [1.0, 2.0], "b": [3.0, 4.0]})
        original_result = feat.run(test_df)
        loaded_result = loaded.run(test_df)
        np.testing.assert_array_almost_equal(
            original_result.values, loaded_result.values
        )

    def test_pca_variance_capture(self):
        """验证 PCA 捕获最大方差方向"""
        # 构造数据: 沿 y=x 方向有大方差，沿 y=-x 方向有小方差
        np.random.seed(0)
        n = 200
        t = np.random.randn(n) * 10  # 大方差
        noise = np.random.randn(n) * 0.1  # 小方差
        data = np.column_stack([t + noise, t - noise])
        df = pd.DataFrame(data, columns=["a", "b"])

        config = PCAConfig(input_columns=["a", "b"], n_components=1)
        feat = PCAFeature(config=config)
        feat.fit(df)

        # 第一主成分应接近 [1/sqrt(2), 1/sqrt(2)] 或 [-1/sqrt(2), -1/sqrt(2)]
        component = feat.state.components[:, 0]
        component_normalized = np.abs(component) / np.linalg.norm(component)
        np.testing.assert_array_almost_equal(
            component_normalized, [1.0 / np.sqrt(2), 1.0 / np.sqrt(2)], decimal=1
        )

    def test_column_naming(self):
        """输出列名格式: pca_{序号}"""
        config = PCAConfig(input_columns=["a", "b"], n_components=2)
        feat = PCAFeature(config=config)
        np.random.seed(42)
        df = pd.DataFrame({"a": np.random.randn(30), "b": np.random.randn(30)})
        feat.fit(df)
        result = feat.run(df)
        assert list(result.columns) == ["pca_0", "pca_1"]

    def test_compute_with_none_state_raises(self):
        """compute 不带 state 应抛 ValueError"""
        with pytest.raises(ValueError, match="PCA 需要先训练"):
            PCAFeature.compute(np.array([[1.0, 2.0]]), state=None)
