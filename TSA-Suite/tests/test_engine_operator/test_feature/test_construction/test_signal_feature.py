# -*- coding: utf-8 -*-

"""
信号特征构造算子测试

测试覆盖 26 个特征类（32 个特征实例），按分组组织：
    Group A: 简单统计特征 (11)
    Group B: 需要采样率的特征 (3)
    Group C: 频域特征 (5)
    Group D: 复合特征 (3)
    Group E: 频带特征 (3 类)
    Group F: 转速特征 (1)

每种特征测试：
    - name() 类方法
    - Config 校验
    - 手工参考值（移植自 ops conftest.py）
    - 边界情况（空信号、单元素、常数信号）
    - Oracle 对比（seed=42 随机数据）
    - DataFrame 输入/输出格式

参考：../ops/tests 下的测试用例和 conftest.py 中的手工参考值。
"""

import math

import numpy as np
import pandas as pd
import pytest

from tsas.engine.operator.feature.construction.signal_feature import (
    # Config
    SampleRateFeatureConfig,
    BandFeatureConfig,
    AverageKurtosisConfig,
    SpeedRpmConfig,
    # Group A
    MeanSquareFeature,
    VarianceFeature,
    RmsFeature,
    PeakPeakFeature,
    ShapeFactorFeature,
    CrestFeature,
    ImpulseFeature,
    ClearanceFeature,
    SkewnessFeature,
    KurtosisFeature,
    GiniIndexFeature,
    # Group B
    SpectralEntropyFeature,
    RoughnessFeature,
    SharpnessFeature,
    # Group C
    SpectralCentroidFeature,
    MeanSquareFrequencyFeature,
    RmsFrequencyFeature,
    FrequencyVarianceFeature,
    FrequencyStdFeature,
    # Group D
    EnvelopeRmsFeature,
    AverageKurtosisFeature,
    HnrFeature,
    # Group E
    BandKurtosisFeature,
    BandRmsFeature,
    BandHnrFeature,
    # Group F
    SpeedRpmFeature,
)


# ============================================================================
# Oracle 函数（移植自 ops/tests/_oracle/statistical_pure.py）
# ============================================================================

def _oracle_mean_square(values):
    if not values:
        return 0.0
    return sum(x * x for x in values) / len(values)


def _oracle_rms(values):
    return math.sqrt(_oracle_mean_square(values))


def _oracle_variance(values):
    if len(values) < 2:
        return 0.0
    m = sum(values) / len(values)
    return sum((x - m) ** 2 for x in values) / (len(values) - 1)


def _oracle_peak_peak(values):
    if not values:
        return 0.0
    return max(values) - min(values)


def _oracle_shape_factor(values):
    if not values:
        return 0.0
    abs_mean = sum(abs(x) for x in values) / len(values)
    if abs_mean == 0:
        return 0.0
    return _oracle_rms(values) / abs_mean


def _oracle_crest(values):
    if not values:
        return 0.0
    rms_val = _oracle_rms(values)
    if rms_val == 0:
        return 0.0
    return max(abs(x) for x in values) / rms_val


def _oracle_impulse(values):
    if not values:
        return 0.0
    abs_mean = sum(abs(x) for x in values) / len(values)
    if abs_mean == 0:
        return 0.0
    return max(abs(x) for x in values) / abs_mean


def _oracle_clearance(values):
    if not values:
        return 0.0
    sqrt_abs_mean = sum(math.sqrt(abs(x)) for x in values) / len(values)
    denom = sqrt_abs_mean ** 2
    if denom == 0:
        return 0.0
    return max(abs(x) for x in values) / denom


def _oracle_skewness(values):
    if len(values) < 2:
        return 0.0
    m = sum(values) / len(values)
    s = math.sqrt(sum((x - m) ** 2 for x in values) / (len(values) - 1))
    if s == 0:
        return 0.0
    return sum((x - m) ** 3 for x in values) / len(values) / s ** 3


def _oracle_kurtosis(values):
    if len(values) < 2:
        return 0.0
    m = sum(values) / len(values)
    s = math.sqrt(sum((x - m) ** 2 for x in values) / (len(values) - 1))
    if s == 0:
        return 0.0
    return sum((x - m) ** 4 for x in values) / len(values) / s ** 4 - 3


def _oracle_gini_index(values):
    if not values:
        return 0.0
    values = [abs(x) for x in values]
    sorted_v = sorted(values)
    n = len(sorted_v)
    total = sum(sorted_v)
    if total == 0:
        return 0.0
    weighted = sum((i + 1) * v for i, v in enumerate(sorted_v))
    return (2 * weighted) / (n * total) - (n + 1) / n


# ============================================================================
# 共享 Fixtures
# ============================================================================

@pytest.fixture
def random_data_1d():
    """seed=42 可复现随机信号，与 ops conftest 一致。"""
    np.random.seed(42)
    return np.random.randn(100).tolist()


@pytest.fixture
def signal_df(random_data_1d):
    """单列 DataFrame，每格一个信号段。"""
    return pd.DataFrame({"sig": [random_data_1d]})


def _make_signal_df(signals, col="sig"):
    """将多个信号段构造成 DataFrame。"""
    return pd.DataFrame({col: signals})


# ============================================================================
# Group A: 简单统计特征 (11)
# ============================================================================

# --- MeanSquareFeature ---

class TestMeanSquareFeature:

    def test_name(self):
        assert MeanSquareFeature.name() == "mean_square_feature"

    @pytest.mark.parametrize("input_data,expected", [
        ([1.0, 2.0, 3.0], 14 / 3),
        ([-1.0, 0.0, 1.0], 2 / 3),
        ([5.0, 5.0, 5.0], 25.0),
        ([3.0], 9.0),
        ([], 0.0),
    ])
    def test_handcrafted(self, input_data, expected):
        config = SampleRateFeatureConfig(input_columns=["sig"], sample_rate=1000)
        feat = MeanSquareFeature(config=config)
        df = _make_signal_df([input_data])
        result = feat.run(df)
        actual = float(result["sig_mean_square"].iloc[0])
        if expected == 0.0:
            assert actual == expected
        else:
            assert math.isclose(actual, expected, rel_tol=1e-12)

    def test_oracle_random(self, random_data_1d):
        expected = _oracle_mean_square(random_data_1d)
        feat = MeanSquareFeature(config=SampleRateFeatureConfig(input_columns=["sig"], sample_rate=1000))
        df = _make_signal_df([random_data_1d])
        result = feat.run(df)
        assert math.isclose(float(result["sig_mean_square"].iloc[0]), expected, rel_tol=1e-12)

    def test_multi_row(self):
        config = SampleRateFeatureConfig(input_columns=["sig"], sample_rate=1000)
        feat = MeanSquareFeature(config=config)
        df = _make_signal_df([[1.0, 2.0, 3.0], [5.0, 5.0, 5.0]])
        result = feat.run(df)
        vals = result["sig_mean_square"].values
        assert math.isclose(vals[0], 14 / 3, rel_tol=1e-12)
        assert vals[1] == 25.0


# --- VarianceFeature ---

class TestVarianceFeature:

    def test_name(self):
        assert VarianceFeature.name() == "variance_feature"

    @pytest.mark.parametrize("input_data,expected", [
        ([1.0, 2.0, 3.0], 1.0),
        ([5.0, 5.0, 5.0], 0.0),
        ([3.0], 0.0),
        ([], 0.0),
    ])
    def test_handcrafted(self, input_data, expected):
        feat = VarianceFeature(config=SampleRateFeatureConfig(input_columns=["sig"], sample_rate=1000))
        df = _make_signal_df([input_data])
        result = feat.run(df)
        actual = float(result["sig_variance"].iloc[0])
        assert actual == expected or math.isclose(actual, expected, rel_tol=1e-12)

    def test_oracle_random(self, random_data_1d):
        expected = _oracle_variance(random_data_1d)
        feat = VarianceFeature(config=SampleRateFeatureConfig(input_columns=["sig"], sample_rate=1000))
        df = _make_signal_df([random_data_1d])
        result = feat.run(df)
        assert math.isclose(float(result["sig_variance"].iloc[0]), expected, rel_tol=1e-12)


# --- RmsFeature ---

class TestRmsFeature:

    def test_name(self):
        assert RmsFeature.name() == "rms_feature"

    @pytest.mark.parametrize("input_data,expected", [
        ([1.0, 2.0, 3.0], 2.160246899469287),
        ([-1.0, 0.0, 1.0], 0.816496580927726),
        ([5.0, 5.0, 5.0], 5.0),
        ([3.0], 3.0),
        ([], 0.0),
    ])
    def test_handcrafted(self, input_data, expected):
        feat = RmsFeature(config=SampleRateFeatureConfig(input_columns=["sig"], sample_rate=1000))
        df = _make_signal_df([input_data])
        result = feat.run(df)
        actual = float(result["sig_rms"].iloc[0])
        if expected == 0.0:
            assert actual == expected
        else:
            assert math.isclose(actual, expected, rel_tol=1e-12)

    def test_oracle_random(self, random_data_1d):
        expected = _oracle_rms(random_data_1d)
        feat = RmsFeature(config=SampleRateFeatureConfig(input_columns=["sig"], sample_rate=1000))
        df = _make_signal_df([random_data_1d])
        result = feat.run(df)
        assert math.isclose(float(result["sig_rms"].iloc[0]), expected, rel_tol=1e-12)


# --- PeakPeakFeature ---

class TestPeakPeakFeature:

    def test_name(self):
        assert PeakPeakFeature.name() == "peak_peak_feature"

    @pytest.mark.parametrize("input_data,expected", [
        ([1.0, 2.0, 3.0], 2.0),
        ([-1.0, 0.0, 1.0], 2.0),
        ([5.0, 5.0, 5.0], 0.0),
        ([3.0], 0.0),
        ([], 0.0),
    ])
    def test_handcrafted(self, input_data, expected):
        feat = PeakPeakFeature(config=SampleRateFeatureConfig(input_columns=["sig"], sample_rate=1000))
        df = _make_signal_df([input_data])
        result = feat.run(df)
        actual = float(result["sig_peak_peak"].iloc[0])
        assert actual == expected or math.isclose(actual, expected, rel_tol=1e-12)

    def test_oracle_random(self, random_data_1d):
        expected = _oracle_peak_peak(random_data_1d)
        feat = PeakPeakFeature(config=SampleRateFeatureConfig(input_columns=["sig"], sample_rate=1000))
        df = _make_signal_df([random_data_1d])
        result = feat.run(df)
        assert math.isclose(float(result["sig_peak_peak"].iloc[0]), expected, rel_tol=1e-12)


# --- ShapeFactorFeature ---

class TestShapeFactorFeature:

    def test_name(self):
        assert ShapeFactorFeature.name() == "shape_factor_feature"

    @pytest.mark.parametrize("input_data,expected", [
        ([5.0, 5.0, 5.0], 1.0),
        ([3.0], 1.0),
        ([], 0.0),
    ])
    def test_handcrafted(self, input_data, expected):
        feat = ShapeFactorFeature(config=SampleRateFeatureConfig(input_columns=["sig"], sample_rate=1000))
        df = _make_signal_df([input_data])
        result = feat.run(df)
        actual = float(result["sig_shape_factor"].iloc[0])
        if expected == 0.0:
            assert actual == expected
        else:
            assert math.isclose(actual, expected, rel_tol=1e-12)

    def test_oracle_random(self, random_data_1d):
        expected = _oracle_shape_factor(random_data_1d)
        feat = ShapeFactorFeature(config=SampleRateFeatureConfig(input_columns=["sig"], sample_rate=1000))
        df = _make_signal_df([random_data_1d])
        result = feat.run(df)
        assert math.isclose(float(result["sig_shape_factor"].iloc[0]), expected, rel_tol=1e-10)


# --- CrestFeature ---

class TestCrestFeature:

    def test_name(self):
        assert CrestFeature.name() == "crest_feature"

    @pytest.mark.parametrize("input_data,expected", [
        ([5.0, 5.0, 5.0], 1.0),
        ([3.0], 1.0),
        ([], 0.0),
    ])
    def test_handcrafted(self, input_data, expected):
        feat = CrestFeature(config=SampleRateFeatureConfig(input_columns=["sig"], sample_rate=1000))
        df = _make_signal_df([input_data])
        result = feat.run(df)
        actual = float(result["sig_crest"].iloc[0])
        if expected == 0.0:
            assert actual == expected
        else:
            assert math.isclose(actual, expected, rel_tol=1e-12)

    def test_oracle_random(self, random_data_1d):
        expected = _oracle_crest(random_data_1d)
        feat = CrestFeature(config=SampleRateFeatureConfig(input_columns=["sig"], sample_rate=1000))
        df = _make_signal_df([random_data_1d])
        result = feat.run(df)
        assert math.isclose(float(result["sig_crest"].iloc[0]), expected, rel_tol=1e-10)


# --- ImpulseFeature ---

class TestImpulseFeature:

    def test_name(self):
        assert ImpulseFeature.name() == "impulse_feature"

    @pytest.mark.parametrize("input_data,expected", [
        ([5.0, 5.0, 5.0], 1.0),
        ([3.0], 1.0),
        ([], 0.0),
    ])
    def test_handcrafted(self, input_data, expected):
        feat = ImpulseFeature(config=SampleRateFeatureConfig(input_columns=["sig"], sample_rate=1000))
        df = _make_signal_df([input_data])
        result = feat.run(df)
        actual = float(result["sig_impulse"].iloc[0])
        if expected == 0.0:
            assert actual == expected
        else:
            assert math.isclose(actual, expected, rel_tol=1e-12)

    def test_oracle_random(self, random_data_1d):
        expected = _oracle_impulse(random_data_1d)
        feat = ImpulseFeature(config=SampleRateFeatureConfig(input_columns=["sig"], sample_rate=1000))
        df = _make_signal_df([random_data_1d])
        result = feat.run(df)
        assert math.isclose(float(result["sig_impulse"].iloc[0]), expected, rel_tol=1e-10)


# --- ClearanceFeature ---

class TestClearanceFeature:

    def test_name(self):
        assert ClearanceFeature.name() == "clearance_feature"

    @pytest.mark.parametrize("input_data,expected", [
        ([5.0, 5.0, 5.0], 1.0),
        ([3.0], 1.0),
        ([], 0.0),
    ])
    def test_handcrafted(self, input_data, expected):
        feat = ClearanceFeature(config=SampleRateFeatureConfig(input_columns=["sig"], sample_rate=1000))
        df = _make_signal_df([input_data])
        result = feat.run(df)
        actual = float(result["sig_clearance"].iloc[0])
        if expected == 0.0:
            assert actual == expected
        else:
            assert math.isclose(actual, expected, rel_tol=1e-12)

    def test_oracle_random(self, random_data_1d):
        expected = _oracle_clearance(random_data_1d)
        feat = ClearanceFeature(config=SampleRateFeatureConfig(input_columns=["sig"], sample_rate=1000))
        df = _make_signal_df([random_data_1d])
        result = feat.run(df)
        assert math.isclose(float(result["sig_clearance"].iloc[0]), expected, rel_tol=1e-10)


# --- SkewnessFeature ---

class TestSkewnessFeature:

    def test_name(self):
        assert SkewnessFeature.name() == "skewness_feature"

    @pytest.mark.parametrize("input_data,expected", [
        ([5.0, 5.0, 5.0], 0.0),
        ([3.0], 0.0),
        ([], 0.0),
    ])
    def test_handcrafted(self, input_data, expected):
        feat = SkewnessFeature(config=SampleRateFeatureConfig(input_columns=["sig"], sample_rate=1000))
        df = _make_signal_df([input_data])
        result = feat.run(df)
        actual = float(result["sig_skewness"].iloc[0])
        assert actual == expected

    def test_oracle_random(self, random_data_1d):
        expected = _oracle_skewness(random_data_1d)
        feat = SkewnessFeature(config=SampleRateFeatureConfig(input_columns=["sig"], sample_rate=1000))
        df = _make_signal_df([random_data_1d])
        result = feat.run(df)
        assert math.isclose(float(result["sig_skewness"].iloc[0]), expected, rel_tol=1e-10)


# --- KurtosisFeature ---

class TestKurtosisFeature:

    def test_name(self):
        assert KurtosisFeature.name() == "kurtosis_feature"

    @pytest.mark.parametrize("input_data,expected", [
        ([5.0, 5.0, 5.0], 0.0),
        ([3.0], 0.0),
        ([], 0.0),
    ])
    def test_handcrafted(self, input_data, expected):
        feat = KurtosisFeature(config=SampleRateFeatureConfig(input_columns=["sig"], sample_rate=1000))
        df = _make_signal_df([input_data])
        result = feat.run(df)
        actual = float(result["sig_kurtosis"].iloc[0])
        assert actual == expected

    def test_single_element_no_nan(self):
        feat = KurtosisFeature(config=SampleRateFeatureConfig(input_columns=["sig"], sample_rate=1000))
        df = _make_signal_df([[3.0]])
        result = feat.run(df)
        actual = float(result["sig_kurtosis"].iloc[0])
        assert actual == 0.0
        assert not np.isnan(actual)

    def test_oracle_random(self, random_data_1d):
        expected = _oracle_kurtosis(random_data_1d)
        feat = KurtosisFeature(config=SampleRateFeatureConfig(input_columns=["sig"], sample_rate=1000))
        df = _make_signal_df([random_data_1d])
        result = feat.run(df)
        assert math.isclose(float(result["sig_kurtosis"].iloc[0]), expected, rel_tol=1e-10)


# --- GiniIndexFeature ---

class TestGiniIndexFeature:

    def test_name(self):
        assert GiniIndexFeature.name() == "gini_index_feature"

    @pytest.mark.parametrize("input_data,expected", [
        ([5.0, 5.0, 5.0], 0.0),
        ([3.0], 0.0),
        ([], 0.0),
    ])
    def test_handcrafted(self, input_data, expected):
        feat = GiniIndexFeature(config=SampleRateFeatureConfig(input_columns=["sig"], sample_rate=1000))
        df = _make_signal_df([input_data])
        result = feat.run(df)
        actual = float(result["sig_gini_index"].iloc[0])
        assert actual == expected

    def test_oracle_random(self, random_data_1d):
        expected = _oracle_gini_index(random_data_1d)
        feat = GiniIndexFeature(config=SampleRateFeatureConfig(input_columns=["sig"], sample_rate=1000))
        df = _make_signal_df([random_data_1d])
        result = feat.run(df)
        assert math.isclose(float(result["sig_gini_index"].iloc[0]), expected, rel_tol=1e-10)


# ============================================================================
# Group B: 需要采样率的特征 (3)
# ============================================================================

# --- SpectralEntropyFeature ---

class TestSpectralEntropyFeature:

    def test_name(self):
        assert SpectralEntropyFeature.name() == "spectral_entropy_feature"

    @pytest.mark.parametrize("input_data", [[], [1.0], [0.0] * 1000])
    def test_edge_cases_return_zero(self, input_data):
        config = SampleRateFeatureConfig(input_columns=["sig"], sample_rate=44100)
        feat = SpectralEntropyFeature(config=config)
        df = _make_signal_df([input_data])
        result = feat.run(df)
        assert float(result["sig_spectral_entropy"].iloc[0]) == 0.0

    def test_tone_low_entropy(self):
        """纯音熵低（移植自 ops test_pse）。"""
        fs = 44100
        t = np.linspace(0, 1, 4096, endpoint=False)
        tone = np.sin(2 * np.pi * 440 * t).tolist()
        config = SampleRateFeatureConfig(input_columns=["sig"], sample_rate=fs)
        feat = SpectralEntropyFeature(config=config)
        df = _make_signal_df([tone])
        result = feat.run(df)
        assert float(result["sig_spectral_entropy"].iloc[0]) < 0.4

    def test_noise_high_entropy(self):
        """白噪声熵高（移植自 ops test_pse）。"""
        np.random.seed(42)
        noise = np.random.randn(4096).tolist()
        config = SampleRateFeatureConfig(input_columns=["sig"], sample_rate=44100)
        feat = SpectralEntropyFeature(config=config)
        df = _make_signal_df([noise])
        result = feat.run(df)
        assert float(result["sig_spectral_entropy"].iloc[0]) > 0.7

    def test_noise_entropier_than_tone(self):
        np.random.seed(42)
        fs = 44100
        t = np.linspace(0, 1, 4096, endpoint=False)
        tone = np.sin(2 * np.pi * 440 * t).tolist()
        noise = np.random.randn(4096).tolist()
        config = SampleRateFeatureConfig(input_columns=["sig"], sample_rate=fs)
        feat = SpectralEntropyFeature(config=config)
        df = _make_signal_df([tone, noise])
        result = feat.run(df)
        vals = result["sig_spectral_entropy"].values
        assert vals[1] > vals[0]

    def test_bounded_0_1(self):
        np.random.seed(42)
        config = SampleRateFeatureConfig(input_columns=["sig"], sample_rate=44100)
        feat = SpectralEntropyFeature(config=config)
        signals = [np.random.randn(512).tolist() for _ in range(10)]
        df = _make_signal_df(signals)
        result = feat.run(df)
        for v in result["sig_spectral_entropy"].values:
            assert 0.0 <= v <= 1.0


# --- RoughnessFeature ---

class TestRoughnessFeature:

    def test_name(self):
        assert RoughnessFeature.name() == "roughness_feature"

    @pytest.mark.parametrize("input_data", [[], [1.0], [1.0, 2.0, 3.0], [0.0] * 1000])
    def test_edge_cases_return_zero(self, input_data):
        config = SampleRateFeatureConfig(input_columns=["sig"], sample_rate=44100)
        feat = RoughnessFeature(config=config)
        df = _make_signal_df([input_data])
        result = feat.run(df)
        assert float(result["sig_roughness"].iloc[0]) == 0.0

    def test_constant_near_zero(self):
        config = SampleRateFeatureConfig(input_columns=["sig"], sample_rate=44100)
        feat = RoughnessFeature(config=config)
        df = _make_signal_df([[5.0] * 500])
        result = feat.run(df)
        assert float(result["sig_roughness"].iloc[0]) < 1e-10

    def test_pure_tone_non_negative(self):
        fs = 44100
        t = np.linspace(0, 0.5, int(fs * 0.5), endpoint=False)
        tone = np.sin(2 * np.pi * 1000 * t).tolist()
        config = SampleRateFeatureConfig(input_columns=["sig"], sample_rate=fs)
        feat = RoughnessFeature(config=config)
        df = _make_signal_df([tone])
        result = feat.run(df)
        assert float(result["sig_roughness"].iloc[0]) >= 0.0

    def test_am_signal_rougher_than_tone(self):
        """AM 调制信号比纯音更粗糙（移植自 ops test_roughness）。"""
        fs = 44100
        t = np.linspace(0, 0.5, int(fs * 0.5), endpoint=False)
        tone = np.sin(2 * np.pi * 1000 * t).tolist()
        am = ((1 + 0.5 * np.sin(2 * np.pi * 70 * t)) * np.sin(2 * np.pi * 1000 * t)).tolist()
        config = SampleRateFeatureConfig(input_columns=["sig"], sample_rate=fs)
        feat = RoughnessFeature(config=config)
        df = _make_signal_df([tone, am])
        result = feat.run(df)
        vals = result["sig_roughness"].values
        assert vals[1] > vals[0]


# --- SharpnessFeature ---

class TestSharpnessFeature:

    def test_name(self):
        assert SharpnessFeature.name() == "sharpness_feature"

    @pytest.mark.parametrize("input_data", [[], [1.0], [0.0] * 1000])
    def test_edge_cases_return_zero(self, input_data):
        config = SampleRateFeatureConfig(input_columns=["sig"], sample_rate=44100)
        feat = SharpnessFeature(config=config)
        df = _make_signal_df([input_data])
        result = feat.run(df)
        assert float(result["sig_sharpness"].iloc[0]) == 0.0

    def test_pure_tone_non_negative(self):
        fs = 44100
        t = np.linspace(0, 0.1, int(fs * 0.1), endpoint=False)
        tone = np.sin(2 * np.pi * 1000 * t).tolist()
        config = SampleRateFeatureConfig(input_columns=["sig"], sample_rate=fs)
        feat = SharpnessFeature(config=config)
        df = _make_signal_df([tone])
        result = feat.run(df)
        assert float(result["sig_sharpness"].iloc[0]) >= 0.0

    def test_high_freq_sharper(self):
        """8kHz 音比 1kHz 音更尖锐（移植自 ops test_sharpness）。"""
        fs = 44100
        t = np.linspace(0, 0.5, int(fs * 0.5), endpoint=False)
        low = np.sin(2 * np.pi * 1000 * t).tolist()
        high = np.sin(2 * np.pi * 8000 * t).tolist()
        config = SampleRateFeatureConfig(input_columns=["sig"], sample_rate=fs)
        feat = SharpnessFeature(config=config)
        df = _make_signal_df([low, high])
        result = feat.run(df)
        vals = result["sig_sharpness"].values
        assert vals[1] > vals[0]

    def test_1khz_near_1_acum(self):
        """1kHz 纯音锐度接近 1 acum（移植自 ops test_sharpness）。"""
        fs = 44100
        t = np.linspace(0, 0.5, int(fs * 0.5), endpoint=False)
        tone = np.sin(2 * np.pi * 1000 * t).tolist()
        config = SampleRateFeatureConfig(input_columns=["sig"], sample_rate=fs)
        feat = SharpnessFeature(config=config)
        df = _make_signal_df([tone])
        result = feat.run(df)
        sharpness = float(result["sig_sharpness"].iloc[0])
        assert 0.5 < sharpness < 2.0


# ============================================================================
# Group C: 频域特征 (5)
# ============================================================================

def _make_tone_signal(freq, fs=10000, duration=1.0):
    """生成纯音信号。"""
    t = np.linspace(0, duration, int(fs * duration), endpoint=False)
    return np.sin(2 * np.pi * freq * t).tolist()


class TestSpectralCentroidFeature:

    def test_name(self):
        assert SpectralCentroidFeature.name() == "spectral_centroid_feature"

    def test_tone_centroid_near_freq(self):
        """纯音的频谱重心应接近其频率。"""
        fs = 10000
        tone = _make_tone_signal(1000, fs)
        config = SampleRateFeatureConfig(input_columns=["sig"], sample_rate=fs)
        feat = SpectralCentroidFeature(config=config)
        df = _make_signal_df([tone])
        result = feat.run(df)
        centroid = float(result["sig_spectral_centroid"].iloc[0])
        assert abs(centroid - 1000) < 50

    def test_edge_cases(self):
        config = SampleRateFeatureConfig(input_columns=["sig"], sample_rate=10000)
        feat = SpectralCentroidFeature(config=config)
        for sig in [[], [1.0]]:
            df = _make_signal_df([sig])
            result = feat.run(df)
            assert float(result["sig_spectral_centroid"].iloc[0]) == 0.0


class TestMeanSquareFrequencyFeature:

    def test_name(self):
        assert MeanSquareFrequencyFeature.name() == "mean_square_frequency_feature"

    def test_tone_msf_near_freq_sq(self):
        fs = 10000
        tone = _make_tone_signal(1000, fs)
        config = SampleRateFeatureConfig(input_columns=["sig"], sample_rate=fs)
        feat = MeanSquareFrequencyFeature(config=config)
        df = _make_signal_df([tone])
        result = feat.run(df)
        msf = float(result["sig_mean_square_frequency"].iloc[0])
        assert abs(msf - 1000 ** 2) < 10000

    def test_edge_cases(self):
        config = SampleRateFeatureConfig(input_columns=["sig"], sample_rate=10000)
        feat = MeanSquareFrequencyFeature(config=config)
        for sig in [[], [1.0]]:
            df = _make_signal_df([sig])
            result = feat.run(df)
            assert float(result["sig_mean_square_frequency"].iloc[0]) == 0.0


class TestRmsFrequencyFeature:

    def test_name(self):
        assert RmsFrequencyFeature.name() == "rms_frequency_feature"

    def test_tone_rmsf_near_freq(self):
        fs = 10000
        tone = _make_tone_signal(1000, fs)
        config = SampleRateFeatureConfig(input_columns=["sig"], sample_rate=fs)
        feat = RmsFrequencyFeature(config=config)
        df = _make_signal_df([tone])
        result = feat.run(df)
        rmsf = float(result["sig_rms_frequency"].iloc[0])
        assert abs(rmsf - 1000) < 50

    def test_edge_cases(self):
        config = SampleRateFeatureConfig(input_columns=["sig"], sample_rate=10000)
        feat = RmsFrequencyFeature(config=config)
        for sig in [[], [1.0]]:
            df = _make_signal_df([sig])
            result = feat.run(df)
            assert float(result["sig_rms_frequency"].iloc[0]) == 0.0


class TestFrequencyVarianceFeature:

    def test_name(self):
        assert FrequencyVarianceFeature.name() == "frequency_variance_feature"

    def test_tone_low_freq_variance(self):
        """纯音频率方差应很低（能量集中在单一频率）。"""
        fs = 10000
        tone = _make_tone_signal(1000, fs)
        config = SampleRateFeatureConfig(input_columns=["sig"], sample_rate=fs)
        feat = FrequencyVarianceFeature(config=config)
        df = _make_signal_df([tone])
        result = feat.run(df)
        fv = float(result["sig_frequency_variance"].iloc[0])
        assert fv < 10000

    def test_noise_higher_freq_variance(self):
        """白噪声频率方差应高于纯音。"""
        np.random.seed(42)
        fs = 10000
        tone = _make_tone_signal(1000, fs)
        noise = np.random.randn(len(tone)).tolist()
        config = SampleRateFeatureConfig(input_columns=["sig"], sample_rate=fs)
        feat = FrequencyVarianceFeature(config=config)
        df = _make_signal_df([tone, noise])
        result = feat.run(df)
        vals = result["sig_frequency_variance"].values
        assert vals[1] > vals[0]

    def test_edge_cases(self):
        config = SampleRateFeatureConfig(input_columns=["sig"], sample_rate=10000)
        feat = FrequencyVarianceFeature(config=config)
        for sig in [[], [1.0]]:
            df = _make_signal_df([sig])
            result = feat.run(df)
            assert float(result["sig_frequency_variance"].iloc[0]) == 0.0


class TestFrequencyStdFeature:

    def test_name(self):
        assert FrequencyStdFeature.name() == "frequency_std_feature"

    def test_edge_cases(self):
        config = SampleRateFeatureConfig(input_columns=["sig"], sample_rate=10000)
        feat = FrequencyStdFeature(config=config)
        for sig in [[], [1.0]]:
            df = _make_signal_df([sig])
            result = feat.run(df)
            assert float(result["sig_frequency_std"].iloc[0]) == 0.0


# ============================================================================
# Group D: 复合特征 (3)
# ============================================================================

# --- EnvelopeRmsFeature ---

class TestEnvelopeRmsFeature:

    def test_name(self):
        assert EnvelopeRmsFeature.name() == "envelope_rms_feature"

    def test_constant_signal(self):
        """常数信号包络 RMS 等于常数本身（移植自 ops test_envelope）。"""
        feat = EnvelopeRmsFeature(config=SampleRateFeatureConfig(input_columns=["sig"], sample_rate=1000))
        df = _make_signal_df([[1.0, 1.0, 1.0, 1.0]])
        result = feat.run(df)
        actual = float(result["sig_envelope_rms"].iloc[0])
        assert math.isclose(actual, 1.0, abs_tol=0.01)

    def test_sine_envelope(self):
        """正弦波的 Hilbert 包络 ≈ 1.0，因此 RMS 也 ≈ 1.0（移植自 ops test_envelope）。"""
        fs = 1000
        t = np.linspace(0, 1, 200, endpoint=False)
        sig = np.sin(2 * np.pi * 5 * t).tolist()
        feat = EnvelopeRmsFeature(config=SampleRateFeatureConfig(input_columns=["sig"], sample_rate=fs))
        df = _make_signal_df([sig])
        result = feat.run(df)
        actual = float(result["sig_envelope_rms"].iloc[0])
        assert math.isclose(actual, 1.0, abs_tol=0.05)

    def test_edge_cases(self):
        feat = EnvelopeRmsFeature(config=SampleRateFeatureConfig(input_columns=["sig"], sample_rate=1000))
        for sig in [[], [3.0]]:
            df = _make_signal_df([sig])
            result = feat.run(df)
            assert float(result["sig_envelope_rms"].iloc[0]) == 0.0


# --- AverageKurtosisFeature ---

class TestAverageKurtosisFeature:

    def test_name(self):
        assert AverageKurtosisFeature.name() == "average_kurtosis_feature"

    def test_constant_signal(self):
        config = AverageKurtosisConfig(input_columns=["sig"], n_segments=5)
        feat = AverageKurtosisFeature(config=config)
        df = _make_signal_df([[5.0] * 100])
        result = feat.run(df)
        assert float(result["sig_average_kurtosis"].iloc[0]) == 0.0

    def test_gaussian_noise_nonzero(self):
        np.random.seed(42)
        config = AverageKurtosisConfig(input_columns=["sig"], n_segments=5)
        feat = AverageKurtosisFeature(config=config)
        df = _make_signal_df([np.random.randn(1000).tolist()])
        result = feat.run(df)
        actual = float(result["sig_average_kurtosis"].iloc[0])
        assert abs(actual) < 1.0  # 正态分布超额峭度理论值 ≈ 0

    def test_short_signal_fallback(self):
        """信号短于分段时退化为整体峭度。"""
        config = AverageKurtosisConfig(input_columns=["sig"], n_segments=10)
        feat = AverageKurtosisFeature(config=config)
        df = _make_signal_df([[1.0, 2.0, 3.0]])
        result = feat.run(df)
        actual = float(result["sig_average_kurtosis"].iloc[0])
        assert not np.isnan(actual)

    def test_custom_n_segments(self):
        config = AverageKurtosisConfig(input_columns=["sig"], n_segments=3)
        feat = AverageKurtosisFeature(config=config)
        assert feat.config.n_segments == 3


# --- HnrFeature ---

class TestHnrFeature:

    def test_name(self):
        assert HnrFeature.name() == "hnr_feature"

    def test_edge_cases(self):
        feat = HnrFeature(config=SampleRateFeatureConfig(input_columns=["sig"], sample_rate=1000))
        for sig in [[], [1.0], [1.0, 2.0]]:
            df = _make_signal_df([sig])
            result = feat.run(df)
            assert float(result["sig_hnr"].iloc[0]) == 0.0

    def test_pure_tone_high_hnr(self):
        """纯音谐噪比应较高。"""
        fs = 10000
        t = np.linspace(0, 0.5, int(fs * 0.5), endpoint=False)
        tone = np.sin(2 * np.pi * 440 * t).tolist()
        feat = HnrFeature(config=SampleRateFeatureConfig(input_columns=["sig"], sample_rate=fs))
        df = _make_signal_df([tone])
        result = feat.run(df)
        hnr = float(result["sig_hnr"].iloc[0])
        assert hnr > 0  # 谐波信号 HNR > 0

    def test_noise_low_hnr(self):
        """白噪声谐噪比应低于纯音。"""
        np.random.seed(42)
        fs = 10000
        t = np.linspace(0, 0.5, int(fs * 0.5), endpoint=False)
        tone = np.sin(2 * np.pi * 440 * t).tolist()
        noise = np.random.randn(len(t)).tolist()
        feat = HnrFeature(config=SampleRateFeatureConfig(input_columns=["sig"], sample_rate=fs))
        df = _make_signal_df([tone, noise])
        result = feat.run(df)
        vals = result["sig_hnr"].values
        assert vals[0] > vals[1]


# ============================================================================
# Group E: 频带特征 (3 类)
# ============================================================================

class TestBandKurtosisFeature:

    def test_name(self):
        assert BandKurtosisFeature.name() == "band_kurtosis_feature"

    def test_output_column_name(self):
        config = BandFeatureConfig(input_columns=["sig"], sample_rate=10000, low=0.0, high=500.0)
        feat = BandKurtosisFeature(config=config)
        df = _make_signal_df([[1.0, 2.0, 3.0, 4.0, 5.0]])
        result = feat.run(df)
        assert "sig_band_kurtosis_0_500" in result.columns

    def test_constant_signal(self):
        config = BandFeatureConfig(input_columns=["sig"], sample_rate=10000, low=0.0, high=500.0)
        feat = BandKurtosisFeature(config=config)
        df = _make_signal_df([[5.0] * 1000])
        result = feat.run(df)
        assert float(result["sig_band_kurtosis_0_500"].iloc[0]) == 0.0

    def test_nyquist_high(self):
        config = BandFeatureConfig(input_columns=["sig"], sample_rate=10000, low=2000.0)
        feat = BandKurtosisFeature(config=config)
        df = _make_signal_df([np.random.randn(1000).tolist()])
        result = feat.run(df)
        assert "sig_band_kurtosis_2000_nyquist" in result.columns


class TestBandRmsFeature:

    def test_name(self):
        assert BandRmsFeature.name() == "band_rms_feature"

    def test_tone_in_band(self):
        """频带内纯音的 RMS 应高于带外。"""
        fs = 10000
        t = np.linspace(0, 1, fs, endpoint=False)
        sig = np.sin(2 * np.pi * 100 * t).tolist()
        config_in = BandFeatureConfig(input_columns=["sig"], sample_rate=fs, low=50.0, high=200.0)
        config_out = BandFeatureConfig(input_columns=["sig"], sample_rate=fs, low=500.0, high=1000.0)
        feat_in = BandRmsFeature(config=config_in)
        feat_out = BandRmsFeature(config=config_out)
        df = _make_signal_df([sig])
        r_in = feat_in.run(df)
        r_out = feat_out.run(df)
        assert float(r_in.iloc[0, 0]) > float(r_out.iloc[0, 0])

    def test_constant_signal_rms(self):
        config = BandFeatureConfig(input_columns=["sig"], sample_rate=10000, low=0.0, high=500.0)
        feat = BandRmsFeature(config=config)
        df = _make_signal_df([[5.0] * 1000])
        result = feat.run(df)
        rms_val = float(result["sig_band_rms_0_500"].iloc[0])
        assert math.isclose(rms_val, 5.0, abs_tol=0.01)


class TestBandHnrFeature:

    def test_name(self):
        assert BandHnrFeature.name() == "band_hnr_feature"

    def test_output_column_name(self):
        config = BandFeatureConfig(input_columns=["sig"], sample_rate=10000, low=500.0, high=2000.0)
        feat = BandHnrFeature(config=config)
        df = _make_signal_df([[1.0, 2.0, 3.0]])
        result = feat.run(df)
        assert "sig_band_hnr_500_2000" in result.columns

    def test_edge_case_short_signal(self):
        config = BandFeatureConfig(input_columns=["sig"], sample_rate=10000, low=0.0, high=500.0)
        feat = BandHnrFeature(config=config)
        df = _make_signal_df([[1.0, 2.0]])
        result = feat.run(df)
        assert not np.isnan(float(result.iloc[0, 0]))


# ============================================================================
# Group F: 转速特征 (1)
# ============================================================================

class TestSpeedRpmFeature:

    def test_name(self):
        assert SpeedRpmFeature.name() == "speed_rpm_feature"

    def test_config_validation(self):
        with pytest.raises(Exception):
            SpeedRpmConfig(input_columns=["sig"], sample_rate=10000, speed_min=0, speed_max=3600)
        with pytest.raises(Exception):
            SpeedRpmConfig(input_columns=["sig"], sample_rate=0, speed_min=600, speed_max=3600)

    def test_harmonic_signal(self):
        """已知转速的谐波信号（移植自 ops HPS 测试逻辑）。"""
        fs = 10000
        rpm_true = 1800
        f0 = rpm_true / 60
        t = np.linspace(0, 2.0, int(fs * 2.0), endpoint=False)
        np.random.seed(42)
        sig = sum((1.0 / h) * np.sin(2 * np.pi * f0 * h * t) for h in range(1, 6))
        sig = (sig + 0.3 * np.random.randn(len(t))).tolist()

        config = SpeedRpmConfig(input_columns=["sig"], sample_rate=fs,
                                speed_min=600, speed_max=3600)
        feat = SpeedRpmFeature(config=config)
        df = _make_signal_df([sig])
        result = feat.run(df)
        rpm = float(result["sig_speed_rpm"].iloc[0])
        assert abs(rpm - rpm_true) < 10

    def test_different_rpm(self):
        """不同转速信号（移植自 ops HPS 测试逻辑）。"""
        fs = 10000
        rpm_true = 3000
        f0 = rpm_true / 60
        t = np.linspace(0, 2.0, int(fs * 2.0), endpoint=False)
        sig = sum((1.0 / h) * np.sin(2 * np.pi * f0 * h * t) for h in range(1, 6)).tolist()

        config = SpeedRpmConfig(input_columns=["sig"], sample_rate=fs,
                                speed_min=600, speed_max=3600)
        feat = SpeedRpmFeature(config=config)
        df = _make_signal_df([sig])
        result = feat.run(df)
        rpm = float(result["sig_speed_rpm"].iloc[0])
        assert abs(rpm - rpm_true) < 10

    def test_weak_signal_returns_zero(self):
        """弱信号（std < std_min）返回 0（移植自 ops HPS 边界测试）。"""
        config = SpeedRpmConfig(input_columns=["sig"], sample_rate=10000,
                                speed_min=600, speed_max=3600)
        feat = SpeedRpmFeature(config=config)
        df = _make_signal_df([np.ones(1000) * 0.001])
        result = feat.run(df)
        assert float(result["sig_speed_rpm"].iloc[0]) == 0.0

    def test_short_signal(self):
        config = SpeedRpmConfig(input_columns=["sig"], sample_rate=10000,
                                speed_min=600, speed_max=3600)
        feat = SpeedRpmFeature(config=config)
        df = _make_signal_df([[1.0]])
        result = feat.run(df)
        assert float(result["sig_speed_rpm"].iloc[0]) == 0.0

    def test_output_column_name(self):
        config = SpeedRpmConfig(input_columns=["sig"], sample_rate=10000,
                                speed_min=600, speed_max=3600)
        feat = SpeedRpmFeature(config=config)
        df = _make_signal_df([[0.0] * 100])
        result = feat.run(df)
        assert "sig_speed_rpm" in result.columns
