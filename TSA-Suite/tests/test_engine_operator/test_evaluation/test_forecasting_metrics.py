# -*- coding: utf-8 -*-

"""
时序预测评价指标算子测试

测试覆盖:
    1. 基本功能: run() 返回完整指标集 (MSE/RMSE/MAE/MAPE/SMAPE/MASE/DTW/R²)
    2. scores() 方法: 按 main_scores 提取命名标量
    3. 配置覆盖: epsilon、naive_error、max_dtw_len
    4. 边界条件: 完美预测、常数预测、零值保护、空样本、长度不一致
    5. 类型兼容: ndarray、list、DataFrame 输入

测试约束:
    - 代码覆盖率 > 90%
    - 测试通过率 100%
    - 中文注释说明测试目的、输入、输出和预期结果
"""

import warnings

import numpy as np
import pandas as pd
import pytest

from tsas.engine.operator.evaluation.forecasting_metrics import (
    ForecastingMetricResult,
    ForecastingMetricConfig,
    ForecastingMetrics,
)


# ============================================================================
# 基本功能测试
# ============================================================================

class TestForecastingMetricsBasic:
    """基本功能测试"""

    def test_run_returns_complete_result(self):
        """
        测试目的: run() 返回完整指标集
        输入: y_true=[1,2,3,4,5], y_pred=[1.1,1.9,3.2,3.8,5.1]
        输出: ForecastingMetricResult
        预期: 返回包含 8 项指标的结构化结果
        """
        y_true = np.array([1.0, 2.0, 3.0, 4.0, 5.0])
        y_pred = np.array([1.1, 1.9, 3.2, 3.8, 5.1])

        op = ForecastingMetrics()
        result = op.run((y_true, y_pred))

        assert isinstance(result, ForecastingMetricResult)
        assert hasattr(result, 'mse')
        assert hasattr(result, 'rmse')
        assert hasattr(result, 'mae')
        assert hasattr(result, 'mape')
        assert hasattr(result, 'smape')
        assert hasattr(result, 'mase')
        assert hasattr(result, 'dtw')
        assert hasattr(result, 'r2')

    def test_run_with_perfect_prediction(self):
        """
        测试目的: 完美预测时误差指标为 0，R² 为 1
        输入: y_true=[1,2,3,4,5], y_pred=[1,2,3,4,5]（完全一致）
        输出: ForecastingMetricResult
        预期: mse=rmse=mae=mape=smape=0, r2=1
        """
        y_true = np.array([1.0, 2.0, 3.0, 4.0, 5.0])
        y_pred = np.array([1.0, 2.0, 3.0, 4.0, 5.0])

        op = ForecastingMetrics()
        result = op.run((y_true, y_pred))

        assert result.mse == pytest.approx(0.0, abs=1e-10)
        assert result.rmse == pytest.approx(0.0, abs=1e-10)
        assert result.mae == pytest.approx(0.0, abs=1e-10)
        assert result.mape == pytest.approx(0.0, abs=1e-10)
        assert result.smape == pytest.approx(0.0, abs=1e-10)
        assert result.r2 == pytest.approx(1.0, abs=1e-10)

    def test_run_with_biased_prediction(self):
        """
        测试目的: 有偏差预测时各指标为合理正值
        输入: y_true=[1,2,3,4,5], y_pred=[1.1,1.9,3.2,3.8,5.1]
        输出: ForecastingMetricResult
        预期: 误差指标 > 0，R² 在 (0, 1) 之间
        """
        y_true = np.array([1.0, 2.0, 3.0, 4.0, 5.0])
        y_pred = np.array([1.1, 1.9, 3.2, 3.8, 5.1])

        cfg = ForecastingMetricConfig(naive_error=1.0)
        op = ForecastingMetrics(config=cfg)
        result = op.run((y_true, y_pred))

        assert result.mse > 0
        assert result.rmse > 0
        assert result.mae > 0
        assert result.mape > 0
        assert result.smape > 0
        assert result.mase > 0
        assert 0.0 < result.r2 < 1.0

    def test_name(self):
        """
        测试目的: 验证 name() 返回正确标识
        输入: 无
        预期: 返回 "forecasting_metrics"
        """
        assert ForecastingMetrics.name() == "forecasting_metrics"


# ============================================================================
# scores() 方法测试
# ============================================================================

class TestForecastingMetricsScores:
    """scores() 方法测试"""

    def test_scores_returns_dict(self):
        """
        测试目的: scores() 返回按 main_scores 映射的字典
        输入: 默认配置 main_scores（8 项指标）
        输出: dict[str, float]
        预期: 返回包含全部 8 项指标的字典
        """
        y_true = np.array([1.0, 2.0, 3.0, 4.0, 5.0])
        y_pred = np.array([1.1, 1.9, 3.2, 3.8, 5.1])

        cfg = ForecastingMetricConfig(naive_error=1.0)
        op = ForecastingMetrics(config=cfg)
        scores = op.scores((y_true, y_pred))

        assert scores is not None
        assert set(scores.keys()) == {"mse", "rmse", "mae", "mape", "smape", "mase", "dtw", "r2"}
        for v in scores.values():
            assert isinstance(v, float)

    def test_scores_matches_run_values(self):
        """
        测试目的: scores() 提取的值与 run() 结果一致
        输入: 默认配置
        输出: dict[str, float]
        预期: scores["rmse"] == result.rmse，scores["mae"] == result.mae
        """
        y_true = np.array([1.0, 2.0, 3.0, 4.0, 5.0])
        y_pred = np.array([1.1, 1.9, 3.2, 3.8, 5.1])

        cfg = ForecastingMetricConfig(naive_error=1.0)
        op = ForecastingMetrics(config=cfg)
        result = op.run((y_true, y_pred))
        scores = op.scores((y_true, y_pred))

        assert scores["rmse"] == pytest.approx(result.rmse, rel=1e-10)
        assert scores["mae"] == pytest.approx(result.mae, rel=1e-10)
        assert scores["mape"] == pytest.approx(result.mape, rel=1e-10)
        assert scores["r2"] == pytest.approx(result.r2, rel=1e-10)

    def test_scores_with_custom_main_scores(self):
        """
        测试目的: 自定义 main_scores 提取不同指标
        输入: main_scores={"rmse": "rmse", "mae": "mae"}
        输出: dict[str, float]
        预期: 返回 {"rmse": float, "mae": float}
        """
        y_true = np.array([1.0, 2.0, 3.0, 4.0, 5.0])
        y_pred = np.array([1.1, 1.9, 3.2, 3.8, 5.1])

        cfg = ForecastingMetricConfig(
            main_scores={"rmse": "rmse", "mae": "mae"},
            naive_error=1.0,
        )
        op = ForecastingMetrics(config=cfg)
        scores = op.scores((y_true, y_pred))

        assert scores is not None
        assert set(scores.keys()) == {"rmse", "mae"}

    def test_scores_returns_none_when_main_scores_is_none(self):
        """
        测试目的: main_scores=None 时 scores() 返回 None
        输入: main_scores=None
        输出: None
        预期: scores() 返回 None
        """
        y_true = np.array([1.0, 2.0, 3.0, 4.0, 5.0])
        y_pred = np.array([1.1, 1.9, 3.2, 3.8, 5.1])

        op = ForecastingMetrics(main_scores=None)
        scores = op.scores((y_true, y_pred))

        assert scores is None


# ============================================================================
# 配置覆盖测试
# ============================================================================

class TestForecastingMetricsConfig:
    """配置覆盖测试"""

    def test_epsilon_zero_guard(self):
        """
        测试目的: epsilon 参数控制零值保护
        输入: y_true 包含 0，epsilon=1e-8
        输出: ForecastingMetricResult
        预期: MAPE 为有限值，不除零
        """
        y_true = np.array([0.0, 0.0, 1.0])
        y_pred = np.array([0.01, 0.01, 1.1])

        cfg = ForecastingMetricConfig(epsilon=1e-8)
        op = ForecastingMetrics(config=cfg)
        result = op.run((y_true, y_pred))

        assert np.isfinite(result.mape)

    def test_naive_error_for_mase(self):
        """
        测试目的: naive_error 参数用于 MASE 计算
        输入: naive_error=1.0
        输出: ForecastingMetricResult
        预期: mase == mae / naive_error
        """
        y_true = np.array([1.0, 2.0, 3.0, 4.0, 5.0])
        y_pred = np.array([1.1, 1.9, 3.2, 3.8, 5.1])

        cfg = ForecastingMetricConfig(naive_error=1.0)
        op = ForecastingMetrics(config=cfg)
        result = op.run((y_true, y_pred))

        assert result.mase == pytest.approx(result.mae / 1.0, rel=1e-10)

    def test_naive_error_zero(self):
        """
        测试目的: naive_error=0 时 MASE 为 nan（避免除零）
        输入: naive_error=0
        输出: ForecastingMetricResult
        预期: mase 为 nan
        """
        y_true = np.array([1.0, 2.0, 3.0, 4.0, 5.0])
        y_pred = np.array([1.1, 1.9, 3.2, 3.8, 5.1])

        cfg = ForecastingMetricConfig(naive_error=0.0)
        op = ForecastingMetrics(config=cfg)
        result = op.run((y_true, y_pred))

        assert np.isnan(result.mase)

    def test_max_dtw_len_config(self):
        """
        测试目的: max_dtw_len 配置生效
        输入: max_dtw_len=100，超过长度的数据会降采样
        输出: ForecastingMetricResult
        预期: dtw 为有限值
        """
        try:
            from fastdtw import fastdtw  # noqa: F401
        except ImportError:
            pytest.skip("需要安装 fastdtw")

        y_true = np.tile(np.array([1.0, 2.0, 3.0, 4.0, 5.0]), 30)
        y_pred = y_true + 0.1

        cfg = ForecastingMetricConfig(naive_error=1.0, max_dtw_len=100)
        op = ForecastingMetrics(config=cfg)
        result = op.run((y_true, y_pred))

        assert np.isfinite(result.dtw)


# ============================================================================
# 边界条件测试
# ============================================================================

class TestForecastingMetricsEdgeCases:
    """边界条件测试"""

    def test_length_mismatch_raises_error(self):
        """
        测试目的: y_true 和 y_pred 长度不一致时抛出 ValueError
        输入: y_true 长度 5, y_pred 长度 3
        输出: ValueError
        预期: 抛出 ValueError，消息包含 "长度不一致"
        """
        y_true = np.array([1.0, 2.0, 3.0, 4.0, 5.0])
        y_pred = np.array([1.1, 1.9, 3.2])

        op = ForecastingMetrics()
        with pytest.raises(ValueError, match="长度不一致"):
            op.run((y_true, y_pred))

    def test_empty_array_raises_error(self):
        """
        测试目的: 空数组输入时抛出 ValueError
        输入: y_true=[], y_pred=[]
        输出: ValueError
        预期: 抛出 ValueError，消息包含 "输入数组为空"
        """
        y_true = np.array([])
        y_pred = np.array([])

        op = ForecastingMetrics()
        with pytest.raises(ValueError, match="输入数组为空"):
            op.run((y_true, y_pred))

    def test_constant_prediction_r2_zero(self):
        """
        测试目的: 常数预测时 R² 接近 0
        输入: y_true=[1,2,3,4,5], y_pred 全为均值
        输出: ForecastingMetricResult
        预期: r2 ≈ 0
        """
        y_true = np.array([1.0, 2.0, 3.0, 4.0, 5.0])
        y_pred = np.full_like(y_true, np.mean(y_true))

        op = ForecastingMetrics()
        result = op.run((y_true, y_pred))

        assert result.r2 == pytest.approx(0.0, abs=1e-10)

    def test_single_sample(self):
        """
        测试目的: 单样本场景
        输入: y_true=[1], y_pred=[1.1]
        输出: ForecastingMetricResult
        预期: 正常计算，各指标为有限值
        """
        y_true = np.array([1.0])
        y_pred = np.array([1.1])

        cfg = ForecastingMetricConfig(naive_error=1.0)
        op = ForecastingMetrics(config=cfg)
        result = op.run((y_true, y_pred))

        assert np.isfinite(result.mse)
        assert np.isfinite(result.mae)
        assert np.isfinite(result.mape)

    def test_all_zero_true(self):
        """
        测试目的: y_true 全为 0 时的零值保护
        输入: y_true=[0,0,0], y_pred=[0.1,0.1,0.1]
        输出: ForecastingMetricResult
        预期: MAPE 为有限值
        """
        y_true = np.array([0.0, 0.0, 0.0])
        y_pred = np.array([0.1, 0.1, 0.1])

        op = ForecastingMetrics()
        result = op.run((y_true, y_pred))

        assert np.isfinite(result.mape)
        assert np.isfinite(result.smape)


# ============================================================================
# 类型兼容测试
# ============================================================================

class TestForecastingMetricsInputTypes:
    """类型兼容测试"""

    def test_ndarray_input(self):
        """
        测试目的: ndarray 输入正常计算
        输入: y_true/y_pred 为 ndarray
        输出: ForecastingMetricResult
        预期: 正常返回结果
        """
        y_true = np.array([1.0, 2.0, 3.0, 4.0, 5.0])
        y_pred = np.array([1.1, 1.9, 3.2, 3.8, 5.1])

        op = ForecastingMetrics()
        result = op.run((y_true, y_pred))
        assert isinstance(result, ForecastingMetricResult)

    def test_list_input(self):
        """
        测试目的: list 输入自动转换为 ndarray
        输入: y_true/y_pred 为 Python list
        输出: ForecastingMetricResult
        预期: 正常计算
        """
        y_true = [1.0, 2.0, 3.0, 4.0, 5.0]
        y_pred = [1.1, 1.9, 3.2, 3.8, 5.1]

        op = ForecastingMetrics()
        result = op.run((y_true, y_pred))
        assert isinstance(result, ForecastingMetricResult)

    def test_dataframe_input(self):
        """
        测试目的: DataFrame 输入自动转换为 ndarray
        输入: y_true/y_pred 为 DataFrame
        输出: ForecastingMetricResult
        预期: 正常计算
        """
        y_true = pd.DataFrame({"value": [1.0, 2.0, 3.0, 4.0, 5.0]})
        y_pred = pd.DataFrame({"value": [1.1, 1.9, 3.2, 3.8, 5.1]})

        op = ForecastingMetrics()
        result = op.run((y_true, y_pred))
        assert isinstance(result, ForecastingMetricResult)

    def test_mixed_input(self):
        """
        测试目的: ndarray + DataFrame 混合输入
        输入: y_true 为 ndarray，y_pred 为 DataFrame
        输出: ForecastingMetricResult
        预期: 正常计算
        """
        y_true = np.array([1.0, 2.0, 3.0, 4.0, 5.0])
        y_pred = pd.DataFrame({"value": [1.1, 1.9, 3.2, 3.8, 5.1]})

        op = ForecastingMetrics()
        result = op.run((y_true, y_pred))
        assert isinstance(result, ForecastingMetricResult)


# ============================================================================
# 指标公式验证测试
# ============================================================================

class TestForecastingMetricsFormula:
    """指标公式验证测试"""

    def test_mse_rmse_formula(self):
        """
        测试目的: 验证 MSE 和 RMSE 公式
        输入: y_true=[1,2,3], y_pred=[1.1,1.9,3.2]
        输出: ForecastingMetricResult
        预期: mse = mean(error^2), rmse = sqrt(mse)
        """
        y_true = np.array([1.0, 2.0, 3.0])
        y_pred = np.array([1.1, 1.9, 3.2])

        op = ForecastingMetrics()
        result = op.run((y_true, y_pred))

        expected_mse = np.mean((y_pred - y_true) ** 2)
        expected_rmse = np.sqrt(expected_mse)

        assert result.mse == pytest.approx(expected_mse, rel=1e-10)
        assert result.rmse == pytest.approx(expected_rmse, rel=1e-10)

    def test_mae_formula(self):
        """
        测试目的: 验证 MAE 公式
        输入: y_true=[1,2,3], y_pred=[1.1,1.9,3.2]
        输出: ForecastingMetricResult
        预期: mae = mean(|error|)
        """
        y_true = np.array([1.0, 2.0, 3.0])
        y_pred = np.array([1.1, 1.9, 3.2])

        op = ForecastingMetrics()
        result = op.run((y_true, y_pred))

        expected_mae = np.mean(np.abs(y_pred - y_true))
        assert result.mae == pytest.approx(expected_mae, rel=1e-10)

    def test_mape_formula(self):
        """
        测试目的: 验证 MAPE 公式
        输入: y_true=[1,2,3], y_pred=[1.1,1.9,3.2]
        输出: ForecastingMetricResult
        预期: mape = mean(|error| / (|y_true| + epsilon)) * 100
        """
        y_true = np.array([1.0, 2.0, 3.0])
        y_pred = np.array([1.1, 1.9, 3.2])

        op = ForecastingMetrics()
        result = op.run((y_true, y_pred))

        epsilon = 1e-8
        expected_mape = np.mean(np.abs((y_true - y_pred) / (np.abs(y_true) + epsilon))) * 100
        assert result.mape == pytest.approx(expected_mape, rel=1e-10)

    def test_r2_formula(self):
        """
        测试目的: 验证 R² 公式
        输入: y_true=[1,2,3,4,5], y_pred=[1.1,1.9,3.2,3.8,5.1]
        输出: ForecastingMetricResult
        预期: r2 = 1 - SS_res / SS_tot
        """
        y_true = np.array([1.0, 2.0, 3.0, 4.0, 5.0])
        y_pred = np.array([1.1, 1.9, 3.2, 3.8, 5.1])

        op = ForecastingMetrics()
        result = op.run((y_true, y_pred))

        ss_res = np.sum((y_pred - y_true) ** 2)
        ss_tot = np.sum((y_true - np.mean(y_true)) ** 2)
        expected_r2 = 1.0 - ss_res / ss_tot

        assert result.r2 == pytest.approx(expected_r2, rel=1e-10)


# ============================================================================
# DTW 可选依赖测试
# ============================================================================

class TestForecastingMetricsDTW:
    """DTW 可选依赖测试"""

    def test_dtw_fallback_when_fastdtw_missing(self, monkeypatch):
        """
        测试目的: fastdtw 未安装时 DTW 回退为 MAE
        输入: 模拟 fastdtw 导入失败
        输出: ForecastingMetricResult
        预期: dtw == mae
        """
        import tsas.engine.operator.evaluation.forecasting_metrics as fm

        original_import = __builtins__.get("__import__") or __builtins__["__import__"]

        def mock_import(name, *args, **kwargs):
            if name == "fastdtw":
                raise ImportError("mock fastdtw missing")
            return original_import(name, *args, **kwargs)

        monkeypatch.setattr("builtins.__import__", mock_import)

        y_true = np.array([1.0, 2.0, 3.0, 4.0, 5.0])
        y_pred = np.array([1.1, 1.9, 3.2, 3.8, 5.1])

        with warnings.catch_warnings(record=True):
            op = ForecastingMetrics()
            result = op.run((y_true, y_pred))

        assert result.dtw == pytest.approx(result.mae, rel=1e-10)


# ============================================================================
# 冻结结果测试
# ============================================================================

class TestForecastingMetricsFrozen:
    """冻结结果测试"""

    def test_result_is_frozen(self):
        """
        测试目的: 结果对象不可修改（frozen）
        输入: 尝试修改 result.rmse = 0.99
        输出: ValidationError
        预期: 抛出 ValidationError
        """
        y_true = np.array([1.0, 2.0, 3.0, 4.0, 5.0])
        y_pred = np.array([1.1, 1.9, 3.2, 3.8, 5.1])

        op = ForecastingMetrics()
        result = op.run((y_true, y_pred))

        with pytest.raises(Exception):
            result.rmse = 0.99
