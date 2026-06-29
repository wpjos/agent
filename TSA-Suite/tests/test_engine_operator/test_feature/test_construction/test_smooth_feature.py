# -*- coding: utf-8 -*-

"""
低频特征增强算子测试

测试覆盖 9 个滑窗特征类，按分组组织：
    Group H: 滑窗基础统计 (5) — smooth_mean, smooth_min, smooth_max, smooth_std, smooth_slope
    Group I: 滑窗梯度特征 (3) — smooth_grad_mean, smooth_grad_std, smooth_grad2_mean
    Group J: 滑窗频域特征 (1) — smooth_fft_energy

每种特征测试：
    - name() 类方法
    - Config 校验
    - 手工参考值
    - 输出长度与输入一致
    - 边界情况（空信号、短信号、常数信号）
    - 超参数（win/stride/padding_mode）
"""

import numpy as np
import pandas as pd
import pytest

from tsas.engine.operator.feature.construction.smooth_feature import (
    SmoothFeatureConfig,
    SmoothMeanFeature,
    SmoothMinFeature,
    SmoothMaxFeature,
    SmoothStdFeature,
    SmoothSlopeFeature,
    SmoothGradMeanFeature,
    SmoothGradStdFeature,
    SmoothGrad2MeanFeature,
    SmoothFftEnergyFeature,
)


# ============================================================================
# 共享 Fixtures
# ============================================================================

@pytest.fixture
def random_data_1d():
    """seed=42 可复现随机信号。"""
    np.random.seed(42)
    return np.random.randn(100).tolist()


@pytest.fixture
def signal_df(random_data_1d):
    """单列 DataFrame，每格一个信号段。"""
    return pd.DataFrame({"sig": [random_data_1d]})


def _make_signal_df(signals, col="sig"):
    """将多个信号段构造成 DataFrame。"""
    return pd.DataFrame({col: signals})


def _make_config(win=30, stride=1, padding_mode='backward', col="sig"):
    """构造 SmoothFeatureConfig。"""
    return SmoothFeatureConfig(
        input_columns=[col],
        win=win,
        stride=stride,
        padding_mode=padding_mode,
    )


# ============================================================================
# Group H: 滑窗基础统计 (5)
# ============================================================================

# --- SmoothMeanFeature ---

class TestSmoothMeanFeature:

    def test_name(self):
        assert SmoothMeanFeature.name() == "smooth_mean_feature"

    def test_handcrafted_constant(self):
        """常数信号的 smooth_mean 应等于常数本身。"""
        config = _make_config(win=5)
        feat = SmoothMeanFeature(config=config)
        df = _make_signal_df([[3.0, 3.0, 3.0, 3.0, 3.0, 3.0, 3.0, 3.0, 3.0, 3.0]])
        result = feat.run(df)
        out = result["sig_smooth_mean_win5"].iloc[0]
        out = np.asarray(out)
        assert np.allclose(out, 3.0)

    def test_output_length_matches_input(self, random_data_1d):
        config = _make_config(win=30)
        feat = SmoothMeanFeature(config=config)
        df = _make_signal_df([random_data_1d])
        result = feat.run(df)
        out = np.asarray(result["sig_smooth_mean_win30"].iloc[0])
        assert len(out) == len(random_data_1d)

    def test_ramp_signal(self):
        """线性递增信号的 smooth_mean 应近似局部均值。"""
        sig = list(range(1, 11))  # [1, 2, ..., 10]
        config = _make_config(win=3)
        feat = SmoothMeanFeature(config=config)
        df = _make_signal_df([sig])
        result = feat.run(df)
        out = np.asarray(result["sig_smooth_mean_win3"].iloc[0])
        # 验证中心点: 对于 [1,2,3,4,5,6,7,8,9,10], win=3
        # backward padding: 前两个用尾部填充 [9,10, 1,2,3,...,10]
        assert len(out) == 10

    def test_empty_signal(self):
        config = _make_config(win=5)
        feat = SmoothMeanFeature(config=config)
        df = _make_signal_df([[]])
        result = feat.run(df)
        out = np.asarray(result["sig_smooth_mean_win5"].iloc[0])
        assert len(out) == 0

    def test_multi_row(self):
        config = _make_config(win=3)
        feat = SmoothMeanFeature(config=config)
        df = _make_signal_df([[1.0, 2.0, 3.0, 4.0, 5.0], [10.0, 10.0, 10.0, 10.0, 10.0]])
        result = feat.run(df)
        col = result["sig_smooth_mean_win3"]
        out0 = np.asarray(col.iloc[0])
        out1 = np.asarray(col.iloc[1])
        assert len(out0) == 5
        assert np.allclose(out1, 10.0)


# --- SmoothMinFeature ---

class TestSmoothMinFeature:

    def test_name(self):
        assert SmoothMinFeature.name() == "smooth_min_feature"

    def test_constant_signal(self):
        config = _make_config(win=5)
        feat = SmoothMinFeature(config=config)
        df = _make_signal_df([[7.0] * 10])
        result = feat.run(df)
        out = np.asarray(result["sig_smooth_min_win5"].iloc[0])
        assert np.allclose(out, 7.0)

    def test_output_length(self, random_data_1d):
        config = _make_config(win=30)
        feat = SmoothMinFeature(config=config)
        df = _make_signal_df([random_data_1d])
        result = feat.run(df)
        out = np.asarray(result["sig_smooth_min_win30"].iloc[0])
        assert len(out) == len(random_data_1d)

    def test_min_captures_lowest(self):
        """smooth_min 应捕获窗口内的最小值。"""
        sig = [1.0, 5.0, 3.0, -2.0, 4.0, 6.0, 2.0, 8.0, 0.0, 7.0]
        config = _make_config(win=3)
        feat = SmoothMinFeature(config=config)
        df = _make_signal_df([sig])
        result = feat.run(df)
        out = np.asarray(result["sig_smooth_min_win3"].iloc[0])
        assert len(out) == 10

    def test_empty_signal(self):
        config = _make_config(win=5)
        feat = SmoothMinFeature(config=config)
        df = _make_signal_df([[]])
        result = feat.run(df)
        out = np.asarray(result["sig_smooth_min_win5"].iloc[0])
        assert len(out) == 0


# --- SmoothMaxFeature ---

class TestSmoothMaxFeature:

    def test_name(self):
        assert SmoothMaxFeature.name() == "smooth_max_feature"

    def test_constant_signal(self):
        config = _make_config(win=5)
        feat = SmoothMaxFeature(config=config)
        df = _make_signal_df([[7.0] * 10])
        result = feat.run(df)
        out = np.asarray(result["sig_smooth_max_win5"].iloc[0])
        assert np.allclose(out, 7.0)

    def test_output_length(self, random_data_1d):
        config = _make_config(win=30)
        feat = SmoothMaxFeature(config=config)
        df = _make_signal_df([random_data_1d])
        result = feat.run(df)
        out = np.asarray(result["sig_smooth_max_win30"].iloc[0])
        assert len(out) == len(random_data_1d)

    def test_max_captures_highest(self):
        sig = [1.0, 5.0, 3.0, -2.0, 4.0, 6.0, 2.0, 8.0, 0.0, 7.0]
        config = _make_config(win=3)
        feat = SmoothMaxFeature(config=config)
        df = _make_signal_df([sig])
        result = feat.run(df)
        out = np.asarray(result["sig_smooth_max_win3"].iloc[0])
        assert len(out) == 10


# --- SmoothStdFeature ---

class TestSmoothStdFeature:

    def test_name(self):
        assert SmoothStdFeature.name() == "smooth_std_feature"

    def test_constant_signal_zero_std(self):
        config = _make_config(win=5)
        feat = SmoothStdFeature(config=config)
        df = _make_signal_df([[5.0] * 10])
        result = feat.run(df)
        out = np.asarray(result["sig_smooth_std_win5"].iloc[0])
        assert np.allclose(out, 0.0)

    def test_output_length(self, random_data_1d):
        config = _make_config(win=30)
        feat = SmoothStdFeature(config=config)
        df = _make_signal_df([random_data_1d])
        result = feat.run(df)
        out = np.asarray(result["sig_smooth_std_win30"].iloc[0])
        assert len(out) == len(random_data_1d)

    def test_alternating_signal_high_std(self):
        """交替信号应有较大局部标准差。"""
        sig = [1.0, -1.0] * 50
        config = _make_config(win=4)
        feat = SmoothStdFeature(config=config)
        df = _make_signal_df([sig])
        result = feat.run(df)
        out = np.asarray(result["sig_smooth_std_win4"].iloc[0])
        assert np.all(out > 0.5)

    def test_empty_signal(self):
        config = _make_config(win=5)
        feat = SmoothStdFeature(config=config)
        df = _make_signal_df([[]])
        result = feat.run(df)
        out = np.asarray(result["sig_smooth_std_win5"].iloc[0])
        assert len(out) == 0


# --- SmoothSlopeFeature ---

class TestSmoothSlopeFeature:

    def test_name(self):
        assert SmoothSlopeFeature.name() == "smooth_slope_feature"

    def test_linear_signal_constant_slope(self):
        """线性信号在非边界区域的 smooth_slope 应为恒定斜率（边界因循环填充有偏移）。"""
        sig = list(range(100))
        config = _make_config(win=10)
        feat = SmoothSlopeFeature(config=config)
        df = _make_signal_df([sig])
        result = feat.run(df)
        out = np.asarray(result["sig_smooth_slope_win10"].iloc[0])
        # 跳过 backward padding 影响的前 win 个位置，中心区域斜率应为 1.0
        assert np.allclose(out[20:], 1.0)

    def test_constant_signal_zero_slope(self):
        config = _make_config(win=5)
        feat = SmoothSlopeFeature(config=config)
        df = _make_signal_df([[3.0] * 20])
        result = feat.run(df)
        out = np.asarray(result["sig_smooth_slope_win5"].iloc[0])
        assert np.allclose(out, 0.0)

    def test_output_length(self, random_data_1d):
        config = _make_config(win=30)
        feat = SmoothSlopeFeature(config=config)
        df = _make_signal_df([random_data_1d])
        result = feat.run(df)
        out = np.asarray(result["sig_smooth_slope_win30"].iloc[0])
        assert len(out) == len(random_data_1d)

    def test_win_less_than_2_returns_zeros(self):
        config = SmoothFeatureConfig(input_columns=["sig"], win=1, stride=1, padding_mode='backward')
        feat = SmoothSlopeFeature(config=config)
        df = _make_signal_df([[1.0, 2.0, 3.0]])
        result = feat.run(df)
        out = np.asarray(result["sig_smooth_slope_win1"].iloc[0])
        assert np.allclose(out, 0.0)


# ============================================================================
# Group I: 滑窗梯度特征 (3)
# ============================================================================

# --- SmoothGradMeanFeature ---

class TestSmoothGradMeanFeature:

    def test_name(self):
        assert SmoothGradMeanFeature.name() == "smooth_grad_mean_feature"

    def test_constant_signal_zero_grad(self):
        config = _make_config(win=5)
        feat = SmoothGradMeanFeature(config=config)
        df = _make_signal_df([[5.0] * 20])
        result = feat.run(df)
        out = np.asarray(result["sig_smooth_grad_mean_win5"].iloc[0])
        assert np.allclose(out, 0.0)

    def test_linear_signal_constant_grad(self):
        """线性信号的 smooth_grad_mean 在非边界区域应近似为恒定值。"""
        sig = list(range(100))
        config = _make_config(win=10)
        feat = SmoothGradMeanFeature(config=config)
        df = _make_signal_df([sig])
        result = feat.run(df)
        out = np.asarray(result["sig_smooth_grad_mean_win10"].iloc[0])
        # 跳过 padding 影响区域，中心区域 gradient 应接近 1.0
        assert np.allclose(out[20:], 1.0, atol=0.1)

    def test_output_length(self, random_data_1d):
        config = _make_config(win=30)
        feat = SmoothGradMeanFeature(config=config)
        df = _make_signal_df([random_data_1d])
        result = feat.run(df)
        out = np.asarray(result["sig_smooth_grad_mean_win30"].iloc[0])
        assert len(out) == len(random_data_1d)

    def test_empty_signal(self):
        config = _make_config(win=5)
        feat = SmoothGradMeanFeature(config=config)
        df = _make_signal_df([[]])
        result = feat.run(df)
        out = np.asarray(result["sig_smooth_grad_mean_win5"].iloc[0])
        assert len(out) == 0


# --- SmoothGradStdFeature ---

class TestSmoothGradStdFeature:

    def test_name(self):
        assert SmoothGradStdFeature.name() == "smooth_grad_std_feature"

    def test_constant_signal_zero_grad(self):
        config = _make_config(win=5)
        feat = SmoothGradStdFeature(config=config)
        df = _make_signal_df([[5.0] * 20])
        result = feat.run(df)
        out = np.asarray(result["sig_smooth_grad_std_win5"].iloc[0])
        assert np.allclose(out, 0.0)

    def test_output_length(self, random_data_1d):
        config = _make_config(win=30)
        feat = SmoothGradStdFeature(config=config)
        df = _make_signal_df([random_data_1d])
        result = feat.run(df)
        out = np.asarray(result["sig_smooth_grad_std_win30"].iloc[0])
        assert len(out) == len(random_data_1d)

    def test_empty_signal(self):
        config = _make_config(win=5)
        feat = SmoothGradStdFeature(config=config)
        df = _make_signal_df([[]])
        result = feat.run(df)
        out = np.asarray(result["sig_smooth_grad_std_win5"].iloc[0])
        assert len(out) == 0


# --- SmoothGrad2MeanFeature ---

class TestSmoothGrad2MeanFeature:

    def test_name(self):
        assert SmoothGrad2MeanFeature.name() == "smooth_grad2_mean_feature"

    def test_linear_signal_zero_curvature(self):
        """线性信号的二阶差分在非边界区域应接近 0。"""
        sig = list(range(100))
        config = _make_config(win=10)
        feat = SmoothGrad2MeanFeature(config=config)
        df = _make_signal_df([sig])
        result = feat.run(df)
        out = np.asarray(result["sig_smooth_grad2_mean_win10"].iloc[0])
        # 跳过 padding 和 diff2 边界效应区域
        assert np.allclose(out[20:-1], 0.0, atol=1e-10)

    def test_constant_signal_zero_curvature(self):
        config = _make_config(win=5)
        feat = SmoothGrad2MeanFeature(config=config)
        df = _make_signal_df([[3.0] * 20])
        result = feat.run(df)
        out = np.asarray(result["sig_smooth_grad2_mean_win5"].iloc[0])
        assert np.allclose(out, 0.0)

    def test_output_length(self, random_data_1d):
        config = _make_config(win=30)
        feat = SmoothGrad2MeanFeature(config=config)
        df = _make_signal_df([random_data_1d])
        result = feat.run(df)
        out = np.asarray(result["sig_smooth_grad2_mean_win30"].iloc[0])
        assert len(out) == len(random_data_1d)

    def test_empty_signal(self):
        config = _make_config(win=5)
        feat = SmoothGrad2MeanFeature(config=config)
        df = _make_signal_df([[]])
        result = feat.run(df)
        out = np.asarray(result["sig_smooth_grad2_mean_win5"].iloc[0])
        assert len(out) == 0


# ============================================================================
# Group J: 滑窗频域特征 (1)
# ============================================================================

# --- SmoothFftEnergyFeature ---

class TestSmoothFftEnergyFeature:

    def test_name(self):
        assert SmoothFftEnergyFeature.name() == "smooth_fft_energy_feature"

    def test_positive_energy(self, random_data_1d):
        config = _make_config(win=30)
        feat = SmoothFftEnergyFeature(config=config)
        df = _make_signal_df([random_data_1d])
        result = feat.run(df)
        out = np.asarray(result["sig_smooth_fft_energy_win30"].iloc[0])
        assert len(out) == len(random_data_1d)
        assert np.all(out > 0)

    def test_zero_signal_zero_energy(self):
        config = _make_config(win=5)
        feat = SmoothFftEnergyFeature(config=config)
        df = _make_signal_df([[0.0] * 20])
        result = feat.run(df)
        out = np.asarray(result["sig_smooth_fft_energy_win5"].iloc[0])
        assert np.allclose(out, 0.0)

    def test_output_length(self, random_data_1d):
        config = _make_config(win=30)
        feat = SmoothFftEnergyFeature(config=config)
        df = _make_signal_df([random_data_1d])
        result = feat.run(df)
        out = np.asarray(result["sig_smooth_fft_energy_win30"].iloc[0])
        assert len(out) == len(random_data_1d)

    def test_empty_signal(self):
        config = _make_config(win=5)
        feat = SmoothFftEnergyFeature(config=config)
        df = _make_signal_df([[]])
        result = feat.run(df)
        out = np.asarray(result["sig_smooth_fft_energy_win5"].iloc[0])
        assert len(out) == 0

    def test_larger_win_smoother_energy(self, random_data_1d):
        """更大的窗口应产生更平滑的能量曲线（变异系数更小）。"""
        df = _make_signal_df([random_data_1d])

        feat_small = SmoothFftEnergyFeature(config=_make_config(win=5))
        result_small = feat_small.run(df)
        out_small = np.asarray(result_small["sig_smooth_fft_energy_win5"].iloc[0])

        feat_large = SmoothFftEnergyFeature(config=_make_config(win=50))
        result_large = feat_large.run(df)
        out_large = np.asarray(result_large["sig_smooth_fft_energy_win50"].iloc[0])

        # 用变异系数 (CV=std/mean) 比较平滑度，消除绝对值差异
        cv_small = np.std(out_small) / np.mean(out_small)
        cv_large = np.std(out_large) / np.mean(out_large)
        assert cv_large < cv_small


# ============================================================================
# Config 校验测试
# ============================================================================

class TestSmoothFeatureConfig:

    def test_default_values(self):
        config = SmoothFeatureConfig(input_columns=["sig"])
        assert config.win == 30
        assert config.stride == 1
        assert config.padding_mode == 'backward'

    def test_custom_values(self):
        config = SmoothFeatureConfig(input_columns=["sig"], win=50, stride=2, padding_mode='forward')
        assert config.win == 50
        assert config.stride == 2
        assert config.padding_mode == 'forward'

    def test_invalid_win_zero(self):
        with pytest.raises(Exception):
            SmoothFeatureConfig(input_columns=["sig"], win=0)

    def test_invalid_win_negative(self):
        with pytest.raises(Exception):
            SmoothFeatureConfig(input_columns=["sig"], win=-1)

    def test_invalid_stride_zero(self):
        with pytest.raises(Exception):
            SmoothFeatureConfig(input_columns=["sig"], stride=0)

    def test_invalid_padding_mode(self):
        with pytest.raises(Exception):
            SmoothFeatureConfig(input_columns=["sig"], padding_mode='invalid')


# ============================================================================
# 超参数组合测试
# ============================================================================

class TestSmoothHyperparams:

    def test_stride_2_output_length(self, random_data_1d):
        """stride=2 时输出长度仍应与输入一致。"""
        config = SmoothFeatureConfig(input_columns=["sig"], win=30, stride=2, padding_mode='backward')
        feat = SmoothMeanFeature(config=config)
        df = _make_signal_df([random_data_1d])
        result = feat.run(df)
        out = np.asarray(result["sig_smooth_mean_win30"].iloc[0])
        assert len(out) == len(random_data_1d)

    def test_forward_padding(self, random_data_1d):
        config = SmoothFeatureConfig(input_columns=["sig"], win=30, stride=1, padding_mode='forward')
        feat = SmoothMeanFeature(config=config)
        df = _make_signal_df([random_data_1d])
        result = feat.run(df)
        out = np.asarray(result["sig_smooth_mean_win30"].iloc[0])
        assert len(out) == len(random_data_1d)

    def test_bidirectional_padding(self, random_data_1d):
        config = SmoothFeatureConfig(input_columns=["sig"], win=30, stride=1, padding_mode='bidirectional')
        feat = SmoothMeanFeature(config=config)
        df = _make_signal_df([random_data_1d])
        result = feat.run(df)
        out = np.asarray(result["sig_smooth_mean_win30"].iloc[0])
        assert len(out) == len(random_data_1d)

    def test_win_equals_signal_length(self):
        """win 等于信号长度时，每个位置的窗口都包含完整信号（循环填充）。"""
        sig = [1.0, 2.0, 3.0, 4.0, 5.0]
        config = SmoothFeatureConfig(input_columns=["sig"], win=5, stride=1, padding_mode='backward')
        feat = SmoothMeanFeature(config=config)
        df = _make_signal_df([sig])
        result = feat.run(df)
        out = np.asarray(result["sig_smooth_mean_win5"].iloc[0])
        assert len(out) == 5
        # 每个窗口覆盖完整信号（循环填充后），均值应接近 3.0
        assert np.allclose(out, 3.0, atol=0.5)
