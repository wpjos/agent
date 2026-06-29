# -*- coding: utf-8 -*-

"""
组合异常检测算子单元测试

对应源文件：
- composite.py: CompositeScorer, CompositeDetector, _ensure_2d, _extract_main_output

测试范围：
- 辅助函数 _ensure_2d / _extract_main_output
- CompositeScorer 配置校验（所有校验分支）
- CompositeScorer 数据流（Predictor+Scorer、多Scorer串行、DataFrame输入输出、EO聚合）
- CompositeScorer 属性访问 / name 参数
- CompositeDetector 配置校验（所有校验分支）
- CompositeDetector 数据流（完整管线、Scorer+Decider、Predictor+Decider、异常检测、DataFrame、EO聚合）
- CompositeDetector 属性访问 / name 参数
- 持久化 save / load
"""

import tempfile
from pathlib import Path

import numpy as np
import pytest
from pandas import DataFrame

from tsas.engine.operator.detection.composite import (
    CompositeScorer,
    CompositeScorerExtraOutput,
    CompositeDetector,
    CompositeDetectorExtraOutput,
    _ensure_2d,
    _extract_main_output,
)
from tsas.engine.operator.detection.knn import KNNScorer
from tsas.engine.operator.detection.mean_predictor import MeanPredictor
from tsas.engine.operator.detection.mean_scorer import MeanScorer
from tsas.engine.operator.detection.pca import PCAPredictor
from tsas.engine.operator.detection.percentile_decider import PercentileDecider
from tsas.engine.operator.detection.residual_scorer import ResidualScorer
from tsas.engine.operator.detection.threshold_decider import ThresholdDecider
from tsas.engine.operator.detection.zscore import ZScoreScorer


# ============================================================================
# 公共测试数据
# ============================================================================

@pytest.fixture
def train_data():
    """训练数据（ndarray, 100x3, 标准正态分布）"""
    np.random.seed(42)
    return np.random.randn(100, 3)


@pytest.fixture
def test_data():
    """测试数据（ndarray, 20x3, 18个正常点 + 2个异常点偏移+10）"""
    np.random.seed(123)
    normal = np.random.randn(18, 3)
    abnormal = np.random.randn(2, 3) + 10
    return np.vstack([normal, abnormal])


@pytest.fixture
def train_df(train_data):
    """训练数据（DataFrame, 列名 a/b/c）"""
    return DataFrame(train_data, columns=["a", "b", "c"])


@pytest.fixture
def test_df(test_data):
    """测试数据（DataFrame, 列名 a/b/c）"""
    return DataFrame(test_data, columns=["a", "b", "c"])


# ============================================================================
# 辅助函数测试
# ============================================================================

class TestEnsure2d:
    """测试 _ensure_2d 辅助函数"""

    def test_1d_to_2d(self):
        """
        目的：验证 1D 数组被 reshape 为 (n, 1)
        输入：1D ndarray [1, 2, 3]
        预期：输出形状 (3, 1)，值不变
        """
        x = np.array([1.0, 2.0, 3.0])
        result = _ensure_2d(x)
        assert result.shape == (3, 1)
        np.testing.assert_array_equal(result.ravel(), x)

    def test_2d_unchanged(self):
        """
        目的：验证 2D 数组原样返回
        输入：2D ndarray (3, 2)
        预期：输出形状 (3, 2)，值不变
        """
        x = np.array([[1.0, 2.0], [3.0, 4.0], [5.0, 6.0]])
        result = _ensure_2d(x)
        assert result.shape == (3, 2)
        np.testing.assert_array_equal(result, x)

    def test_2d_single_column_unchanged(self):
        """
        目的：验证 (n, 1) 形状的 2D 数组不被 reshape
        输入：2D ndarray (4, 1)
        预期：输出形状仍为 (4, 1)
        """
        x = np.array([[1.0], [2.0], [3.0], [4.0]])
        result = _ensure_2d(x)
        assert result.shape == (4, 1)
        np.testing.assert_array_equal(result, x)


class TestExtractMainOutput:
    """测试 _extract_main_output 辅助函数"""

    def test_tuple_with_eo(self):
        """
        目的：验证 tuple(ndarray, eo) 分解
        输入：(ndarray, "eo_value")
        预期：output 为 ndarray，eo 为 "eo_value"
        """
        arr = np.array([1.0, 2.0])
        eo = "eo_value"
        output, extracted_eo = _extract_main_output((arr, eo))
        np.testing.assert_array_equal(output, arr)
        assert extracted_eo == "eo_value"

    def test_plain_ndarray(self):
        """
        目的：验证纯 ndarray 输入（非 tuple）时 eo 为 None
        输入：ndarray
        预期：output 为 ndarray，eo 为 None
        """
        arr = np.array([1.0, 2.0])
        output, extracted_eo = _extract_main_output(arr)
        np.testing.assert_array_equal(output, arr)
        assert extracted_eo is None

    def test_dataframe_input(self):
        """
        目的：验证 DataFrame 输入转换为 ndarray（通过 .to_numpy()）
        输入：DataFrame
        预期：output 为 ndarray，eo 为 None
        """
        df = DataFrame({"a": [1, 2], "b": [3, 4]})
        output, extracted_eo = _extract_main_output(df)
        assert isinstance(output, np.ndarray)
        assert output.shape == (2, 2)
        assert extracted_eo is None

    def test_tuple_with_none_eo(self):
        """
        目的：验证 tuple(ndarray, None) 正确提取
        输入：(ndarray, None)
        预期：output 为 ndarray，eo 为 None
        """
        arr = np.array([1.0])
        output, extracted_eo = _extract_main_output((arr, None))
        np.testing.assert_array_equal(output, arr)
        assert extracted_eo is None


# ============================================================================
# CompositeScorer 配置校验测试
# ============================================================================

class TestCompositeScorerValidation:
    """测试 CompositeScorer 的所有配置校验分支"""

    def test_empty_operators_raises(self):
        """
        目的：验证空算子列表报错
        输入：operators=[]
        预期：抛出 ValueError，消息含 "operators 不能为空"
        """
        with pytest.raises(ValueError, match="operators 不能为空"):
            CompositeScorer(operators=[])

    def test_single_scorer_no_predictor_raises(self):
        """
        目的：验证无 Predictor + 单个 Scorer 没有组合意义
        输入：operators=[ZScoreScorer()]
        预期：抛出 ValueError，消息含 "没有组合意义"
        """
        with pytest.raises(ValueError, match="没有组合意义"):
            CompositeScorer(operators=[ZScoreScorer()])

    def test_predictor_no_scorer_raises(self):
        """
        目的：验证有 Predictor + 无 Scorer 无法产生分数
        输入：operators=[PCAPredictor()]
        预期：抛出 ValueError，消息含 "无法产生异常分数"
        """
        with pytest.raises(ValueError, match="无法产生异常分数"):
            CompositeScorer(operators=[PCAPredictor()])

    def test_bi_numeric_first_without_predictor_raises(self):
        """
        目的：验证无 Predictor 时第一个 Scorer 是 BiNumericOperator 报错
        输入：operators=[ResidualScorer(), ZScoreScorer()]
        预期：抛出 ValueError，消息含 "BiNumericOperator"
        """
        with pytest.raises(ValueError, match="BiNumericOperator"):
            CompositeScorer(operators=[ResidualScorer(), ZScoreScorer()])

    def test_predictor_not_first_raises(self):
        """
        目的：验证 Predictor 不在第一位报错
        输入：operators=[ZScoreScorer(), PCAPredictor()]
        预期：抛出 ValueError，消息含 "第 0 位"
        """
        with pytest.raises(ValueError, match="第 0 位"):
            CompositeScorer(operators=[ZScoreScorer(), PCAPredictor()])

    def test_multiple_predictors_raises(self):
        """
        目的：验证多个 Predictor 报错（第二个 Predictor 不在第0位）
        输入：operators=[PCAPredictor(), PCAPredictor()]
        预期：抛出 ValueError
        """
        with pytest.raises(ValueError):
            CompositeScorer(operators=[PCAPredictor(), PCAPredictor()])

    def test_unsupported_operator_type_raises(self):
        """
        目的：验证不支持的算子类型报错
        输入：operators=["not_an_operator"]
        预期：抛出 ValueError，消息含 "不支持的算子类型"
        """
        with pytest.raises(ValueError, match="不支持的算子类型"):
            CompositeScorer(operators=["not_an_operator"])

    def test_valid_predictor_scorer_config(self, train_data, test_data):
        """
        目的：验证 Predictor + Scorer 合法配置能正常 fit/run
        输入：PCAPredictor + ResidualScorer，训练/测试数据
        预期：输出分数形状为 (n_samples,)
        """
        scorer = CompositeScorer(operators=[
            PCAPredictor(n_components=2),
            ResidualScorer(),
        ])
        scorer.fit(train_data)
        scores, eo = scorer.run(test_data)
        assert scores.shape == (len(test_data),)

    def test_valid_multiple_scorers_config(self, train_data, test_data):
        """
        目的：验证多 Scorer（无 Predictor）合法配置能正常 fit/run
        输入：ZScoreScorer + KNNScorer，训练/测试数据
        预期：输出分数形状为 (n_samples,)
        """
        scorer = CompositeScorer(operators=[
            ZScoreScorer(),
            KNNScorer(),
        ])
        scorer.fit(train_data)
        scores, eo = scorer.run(test_data)
        assert scores.shape == (len(test_data),)


# ============================================================================
# CompositeScorer 数据流测试
# ============================================================================

class TestCompositeScorerDataFlow:
    """测试 CompositeScorer 的数据流和输出正确性"""

    def test_predictor_residual_scorer(self, train_data, test_data):
        """
        目的：验证 Predictor + ResidualScorer 完整数据流
        输入：PCAPredictor(n_components=2) + ResidualScorer(metric="mse")
        预期：输出 1D 分数，形状 (20,)，所有分数 >= 0（MSE 恒非负）
        """
        scorer = CompositeScorer(operators=[
            PCAPredictor(n_components=2),
            ResidualScorer(metric="mse"),
        ])
        scorer.fit(train_data)
        scores, eo = scorer.run(test_data)
        assert scores.shape == (20,)
        assert np.all(scores >= 0)

    def test_scores_higher_for_anomalies(self, train_data, test_data):
        """
        目的：验证异常点分数显著高于正常点
        输入：测试数据中最后 2 个为异常点（偏移+10）
        预期：异常点平均分数 > 正常点平均分数
        """
        scorer = CompositeScorer(operators=[
            PCAPredictor(n_components=2),
            ResidualScorer(metric="mse"),
        ])
        scorer.fit(train_data)
        scores, _ = scorer.run(test_data)
        normal_avg = scores[:18].mean()
        abnormal_avg = scores[18:].mean()
        assert abnormal_avg > normal_avg

    def test_extended_output_list(self, train_data, test_data):
        """
        目的：验证 EO 聚合为列表，长度等于算子数量
        输入：2 个算子（Predictor + Scorer）
        预期：eo 为 CompositeScorerExtraOutput，len(outputs) == 2
        """
        scorer = CompositeScorer(operators=[
            PCAPredictor(n_components=2),
            ResidualScorer(metric="mse"),
        ])
        scorer.fit(train_data)
        _, eo = scorer.run(test_data)
        assert isinstance(eo, CompositeScorerExtraOutput)
        assert len(eo.outputs) == 2

    def test_eo_none_for_no_eo_operator(self, train_data, test_data):
        """
        目的：验证无 EO 的算子在 EO 列表中对应位置为 None
        输入：MeanPredictor（返回 (pred, None)）+ ResidualScorer（有 EO）
        预期：eo.outputs[0] 为 None，eo.outputs[1] 不为 None
        """
        scorer = CompositeScorer(operators=[
            MeanPredictor(),
            ResidualScorer(metric="mse"),
        ])
        scorer.fit(train_data)
        _, eo = scorer.run(test_data)
        assert len(eo.outputs) == 2
        assert eo.outputs[0] is None
        assert eo.outputs[1] is not None

    def test_with_dataframe(self, train_df, test_df):
        """
        目的：验证 DataFrame 输入输出兼容性
        输入：DataFrame 训练/测试数据
        预期：输出为 DataFrame，列名为 "score"
        """
        scorer = CompositeScorer(operators=[
            PCAPredictor(n_components=2),
            ResidualScorer(metric="mse"),
        ])
        scorer.fit(train_df)
        scores, _ = scorer.run(test_df)
        assert isinstance(scores, DataFrame)
        assert list(scores.columns) == ["score"]

    def test_serial_scorers_flow(self, train_data, test_data):
        """
        目的：验证多 Scorer 串行数据流正确
        输入：ZScoreScorer（输出 1D → reshape 2D）→ KNNScorer（接收 2D）
        预期：输出 1D 分数，形状 (20,)
        """
        scorer = CompositeScorer(operators=[
            ZScoreScorer(),
            KNNScorer(),
        ])
        scorer.fit(train_data)
        scores, eo = scorer.run(test_data)
        assert scores.shape == (20,)

    def test_predictor_multi_scorer_flow(self, train_data, test_data):
        """
        目的：验证 Predictor + 双 Scorer 串行数据流
        输入：MeanPredictor + ResidualScorer(BiNumeric) + MeanScorer(Numeric)
        预期：ResidualScorer 接收 (prev_input, prev_output)，MeanScorer 接收 prev_output
        """
        scorer = CompositeScorer(operators=[
            MeanPredictor(),
            ResidualScorer(metric="mse"),
            MeanScorer(),
        ])
        scorer.fit(train_data)
        scores, eo = scorer.run(test_data)
        assert scores.shape == (20,)

    def test_before_fit_raises(self, test_data):
        """
        目的：验证未训练时 run 抛出 RuntimeError
        输入：未 fit 的 CompositeScorer
        预期：抛出 RuntimeError
        """
        scorer = CompositeScorer(operators=[
            PCAPredictor(),
            ResidualScorer(),
        ])
        with pytest.raises(RuntimeError):
            scorer.run(test_data)


# ============================================================================
# CompositeScorer 属性和参数测试
# ============================================================================

class TestCompositeScorerProperties:
    """测试 CompositeScorer 的属性访问和构造参数"""

    def test_predictor_property(self, train_data):
        """
        目的：验证 predictor 属性正确返回内部 Predictor
        输入：含 PCAPredictor 的 CompositeScorer
        预期：predictor 返回 PCAPredictor 实例
        """
        pca = PCAPredictor(n_components=2)
        scorer = CompositeScorer(operators=[pca, ResidualScorer()])
        assert scorer.predictor is pca

    def test_predictor_property_none(self, train_data):
        """
        目的：验证无 Predictor 时 predictor 属性为 None
        输入：不含 Predictor 的 CompositeScorer
        预期：predictor 返回 None
        """
        scorer = CompositeScorer(operators=[ZScoreScorer(), KNNScorer()])
        assert scorer.predictor is None

    def test_scorers_property(self, train_data):
        """
        目的：验证 scorers 属性正确返回内部 Scorer 列表
        输入：2 个 Scorer
        预期：scorers 返回长度为 2 的列表
        """
        s1 = ZScoreScorer()
        s2 = KNNScorer()
        scorer = CompositeScorer(operators=[s1, s2])
        assert len(scorer.scorers) == 2

    def test_operators_property(self, train_data):
        """
        目的：验证 operators 属性返回完整的算子列表
        输入：3 个算子
        预期：operators 返回长度为 3 的列表
        """
        ops = [PCAPredictor(n_components=2), ResidualScorer()]
        scorer = CompositeScorer(operators=ops)
        assert len(scorer.operators) == 2

    def test_name_parameter(self, train_data):
        """
        目的：验证 oid 参数正确传递
        输入：oid="my_scorer"
        预期："my_scorer" in scorer.oid
        """
        scorer = CompositeScorer(
            operators=[ZScoreScorer(), KNNScorer()],
            oid="my_scorer",
        )
        assert "my_scorer" in scorer.oid


# ============================================================================
# CompositeDetector 配置校验测试
# ============================================================================

class TestCompositeDetectorValidation:
    """测试 CompositeDetector 的所有配置校验分支"""

    def test_empty_operators_raises(self):
        """
        目的：验证空算子列表报错
        输入：operators=[]
        预期：抛出 ValueError，消息含 "operators 不能为空"
        """
        with pytest.raises(ValueError, match="operators 不能为空"):
            CompositeDetector(operators=[])

    def test_no_decider_raises(self):
        """
        目的：验证无 Decider 报错
        输入：operators=[ZScoreScorer()]
        预期：抛出 ValueError，消息含 "必须有 1 个 Decider"
        """
        with pytest.raises(ValueError, match="必须有 1 个 Decider"):
            CompositeDetector(operators=[ZScoreScorer()])

    def test_single_decider_raises(self):
        """
        目的：验证单个 Decider 没有组合意义
        输入：operators=[PercentileDecider()]
        预期：抛出 ValueError，消息含 "没有组合意义"
        """
        with pytest.raises(ValueError, match="没有组合意义"):
            CompositeDetector(operators=[PercentileDecider()])

    def test_decider_not_last_raises(self):
        """
        目的：验证 Decider 不在最后一位报错
        输入：operators=[PercentileDecider(), ZScoreScorer()]
        预期：抛出 ValueError，消息含 "最后一位"
        """
        with pytest.raises(ValueError, match="最后一位"):
            CompositeDetector(operators=[PercentileDecider(), ZScoreScorer()])

    def test_multiple_deciders_raises(self):
        """
        目的：验证多个 Decider 报错（第一个不在最后一位）
        输入：operators=[ZScoreScorer(), PercentileDecider(), ThresholdDecider()]
        预期：抛出 ValueError
        """
        with pytest.raises(ValueError):
            CompositeDetector(operators=[
                ZScoreScorer(),
                PercentileDecider(),
                ThresholdDecider(),
            ])

    def test_predictor_not_first_raises(self):
        """
        目的：验证 Predictor 不在第 0 位报错
        输入：operators=[ZScoreScorer(), PCAPredictor(), PercentileDecider()]
        预期：抛出 ValueError，消息含 "第 0 位"
        """
        with pytest.raises(ValueError, match="第 0 位"):
            CompositeDetector(operators=[
                ZScoreScorer(),
                PCAPredictor(),
                PercentileDecider(),
            ])

    def test_multiple_predictors_raises(self):
        """
        目的：验证多个 Predictor 报错
        输入：operators=[PCAPredictor(), PCAPredictor(), PercentileDecider()]
        预期：抛出 ValueError
        """
        with pytest.raises(ValueError):
            CompositeDetector(operators=[
                PCAPredictor(),
                PCAPredictor(),
                PercentileDecider(),
            ])

    def test_bi_numeric_first_without_predictor_raises(self):
        """
        目的：验证无 Predictor 时第一个 Scorer 是 BiNumericOperator 报错
        输入：operators=[ResidualScorer(), PercentileDecider()]
        预期：抛出 ValueError，消息含 "BiNumericOperator"
        """
        with pytest.raises(ValueError, match="BiNumericOperator"):
            CompositeDetector(operators=[
                ResidualScorer(),
                PercentileDecider(),
            ])

    def test_unsupported_operator_type_raises(self):
        """
        目的：验证不支持的算子类型报错
        输入：operators=[ZScoreScorer(), "invalid", PercentileDecider()]
        预期：抛出 ValueError，消息含 "不支持的算子类型"
        """
        with pytest.raises(ValueError, match="不支持的算子类型"):
            CompositeDetector(operators=[
                ZScoreScorer(),
                "invalid",
                PercentileDecider(),
            ])

    def test_valid_predictor_scorer_decider_config(self, train_data, test_data):
        """
        目的：验证 Predictor + Scorer + Decider 合法配置
        输入：PCAPredictor + ResidualScorer + PercentileDecider
        预期：正常 fit/run，输出标签形状 (n_samples,)
        """
        detector = CompositeDetector(operators=[
            PCAPredictor(n_components=2),
            ResidualScorer(),
            PercentileDecider(percentile=95.0),
        ])
        detector.fit(train_data)
        labels, eo = detector.run(test_data)
        assert labels.shape == (len(test_data),)

    def test_valid_scorer_decider_config(self, train_data, test_data):
        """
        目的：验证 Scorer + Decider（无 Predictor）合法配置
        输入：ZScoreScorer + PercentileDecider
        预期：正常 fit/run，输出标签形状 (n_samples,)
        """
        detector = CompositeDetector(operators=[
            ZScoreScorer(),
            PercentileDecider(percentile=95.0),
        ])
        detector.fit(train_data)
        labels, eo = detector.run(test_data)
        assert labels.shape == (len(test_data),)

    def test_valid_predictor_decider_config(self, train_data, test_data):
        """
        目的：验证 Predictor + Decider（无 Scorer）合法配置
        输入：PCAPredictor + ThresholdDecider
        预期：能正常 fit 和 run。
            Predictor 输出 2D (n, m) 且 m>1 → shape[-1]!=1 → 不 ravel，
            2D 数据直接传给 Decider。ThresholdDecider 内部 ravel 输出 (n*m,) 标签。
        """
        detector = CompositeDetector(operators=[
            PCAPredictor(n_components=2),
            ThresholdDecider(threshold=0.5),
        ])
        detector.fit(train_data)
        labels, eo = detector.run(test_data)
        assert labels.shape == (len(test_data) * test_data.shape[1],)


# ============================================================================
# CompositeDetector 数据流测试
# ============================================================================

class TestCompositeDetectorDataFlow:
    """测试 CompositeDetector 的数据流和输出正确性"""

    def test_end_to_end_detection(self, train_data, test_data):
        """
        目的：验证 Predictor+Scorer+Decider 端到端检测流程
        输入：PCAPredictor + ResidualScorer + PercentileDecider
        预期：输出 1D 标签数组，形状 (20,)，值仅为 0 或 1
        """
        detector = CompositeDetector(operators=[
            PCAPredictor(n_components=2),
            ResidualScorer(metric="mse"),
            PercentileDecider(percentile=95.0),
        ])
        detector.fit(train_data)
        labels, eo = detector.run(test_data)
        assert labels.shape == (20,)
        assert set(labels).issubset({0, 1})

    def test_detects_anomalies(self, train_data, test_data):
        """
        目的：验证异常点能被检测出来
        输入：测试数据最后 2 个为异常点，PercentileDecider(percentile=90.0)
        预期：异常点中至少有一个被判为 1
        """
        detector = CompositeDetector(operators=[
            PCAPredictor(n_components=2),
            ResidualScorer(metric="mse"),
            PercentileDecider(percentile=90.0),
        ])
        detector.fit(train_data)
        labels, _ = detector.run(test_data)
        assert labels[18] == 1 or labels[19] == 1

    def test_scorer_decider_flow(self, train_data, test_data):
        """
        目的：验证 Scorer + Decider（无 Predictor）数据流
        输入：ZScoreScorer + PercentileDecider
        预期：输出 1D 标签数组，值仅为 0 或 1
        """
        detector = CompositeDetector(operators=[
            ZScoreScorer(),
            PercentileDecider(percentile=95.0),
        ])
        detector.fit(train_data)
        labels, _ = detector.run(test_data)
        assert labels.shape == (len(test_data),)
        assert set(labels).issubset({0, 1})

    def test_predictor_scorer_decider_with_binumeric(self, train_data, test_data):
        """
        目的：验证含 BiNumericOperator 的完整管线数据流
        输入：MeanPredictor + ResidualScorer(BiNumeric) + PercentileDecider
        预期：ResidualScorer 接收 (prev_input, prev_output)，Decider 接收 1D raveled 分数，
            输出标签形状 (n_samples,)
        """
        detector = CompositeDetector(operators=[
            MeanPredictor(),
            ResidualScorer(metric="mse"),
            PercentileDecider(percentile=95.0),
        ])
        detector.fit(train_data)
        labels, eo = detector.run(test_data)
        assert labels.shape == (len(test_data),)
        assert set(labels).issubset({0, 1})

    def test_with_dataframe(self, train_df, test_df):
        """
        目的：验证 DataFrame 输入输出兼容性
        输入：DataFrame 训练/测试数据
        预期：输出为 DataFrame，列名为 "label"
        """
        detector = CompositeDetector(operators=[
            PCAPredictor(n_components=2),
            ResidualScorer(metric="mse"),
            PercentileDecider(percentile=95.0),
        ])
        detector.fit(train_df)
        labels, _ = detector.run(test_df)
        assert isinstance(labels, DataFrame)
        assert list(labels.columns) == ["label"]

    def test_extended_output_list(self, train_data, test_data):
        """
        目的：验证 EO 聚合为列表，长度等于算子数量
        输入：3 个算子（Predictor + Scorer + Decider）
        预期：eo 为 CompositeDetectorExtraOutput，len(outputs) == 3
        """
        detector = CompositeDetector(operators=[
            PCAPredictor(n_components=2),
            ResidualScorer(metric="mse"),
            PercentileDecider(percentile=95.0),
        ])
        detector.fit(train_data)
        _, eo = detector.run(test_data)
        assert isinstance(eo, CompositeDetectorExtraOutput)
        assert len(eo.outputs) == 3

    def test_eo_none_for_no_eo_operator(self, train_data, test_data):
        """
        目的：验证无 EO 的算子在 EO 列表中对应位置为 None
        输入：MeanPredictor(无EO) + ResidualScorer(有EO) + PercentileDecider(有EO)
        预期：eo.outputs[0] 为 None
        """
        detector = CompositeDetector(operators=[
            MeanPredictor(),
            ResidualScorer(metric="mse"),
            PercentileDecider(percentile=95.0),
        ])
        detector.fit(train_data)
        _, eo = detector.run(test_data)
        assert len(eo.outputs) == 3
        assert eo.outputs[0] is None

    def test_before_fit_raises(self, test_data):
        """
        目的：验证未训练时 run 抛出 RuntimeError
        输入：未 fit 的 CompositeDetector
        预期：抛出 RuntimeError
        """
        detector = CompositeDetector(operators=[
            PCAPredictor(),
            ResidualScorer(),
            PercentileDecider(),
        ])
        with pytest.raises(RuntimeError):
            detector.run(test_data)

    def test_decider_receives_raveled_1d_scores(self, train_data, test_data):
        """
        目的：验证有 Scorer 时 Decider 接收的是 raveled 1D 分数（shape[-1]==1 触发 ravel）
        输入：PCAPredictor + ResidualScorer + PercentileDecider
        预期：Scorer 输出 (n, 1)，ravel 为 (n,) 传给 Decider，
            PercentileDecider 输出 1D 标签 (n,)，标签值为 0 或 1
        """
        detector = CompositeDetector(operators=[
            PCAPredictor(n_components=2),
            ResidualScorer(metric="mse"),
            PercentileDecider(percentile=95.0),
        ])
        detector.fit(train_data)
        labels, _ = detector.run(test_data)
        # Decider 接收 1D raveled 分数，输出 1D 标签
        assert labels.ndim == 1
        assert labels.shape == (len(test_data),)


# ============================================================================
# CompositeDetector 属性和参数测试
# ============================================================================

class TestCompositeDetectorProperties:
    """测试 CompositeDetector 的属性访问和构造参数"""

    def test_predictor_property(self, train_data):
        """
        目的：验证 predictor 属性正确返回内部 Predictor
        输入：含 PCAPredictor 的 CompositeDetector
        预期：predictor 返回 PCAPredictor 实例
        """
        pca = PCAPredictor(n_components=2)
        detector = CompositeDetector(operators=[pca, ResidualScorer(), PercentileDecider()])
        assert detector.predictor is pca

    def test_predictor_property_none(self, train_data):
        """
        目的：验证无 Predictor 时 predictor 属性为 None
        输入：不含 Predictor 的 CompositeDetector
        预期：predictor 返回 None
        """
        detector = CompositeDetector(operators=[ZScoreScorer(), PercentileDecider()])
        assert detector.predictor is None

    def test_scorers_property(self, train_data):
        """
        目的：验证 scorers 属性正确返回内部 Scorer 列表
        输入：1 个 Scorer
        预期：scorers 返回长度为 1 的列表
        """
        detector = CompositeDetector(operators=[
            PCAPredictor(n_components=2),
            ResidualScorer(),
            PercentileDecider(),
        ])
        assert len(detector.scorers) == 1

    def test_decider_property(self, train_data):
        """
        目的：验证 decider 属性正确返回内部 Decider
        输入：含 PercentileDecider 的 CompositeDetector
        预期：decider 返回 PercentileDecider 实例
        """
        pd = PercentileDecider()
        detector = CompositeDetector(operators=[ZScoreScorer(), pd])
        assert detector.decider is pd

    def test_operators_property(self, train_data):
        """
        目的：验证 operators 属性返回完整的算子列表
        输入：3 个算子
        预期：operators 返回长度为 3 的列表
        """
        detector = CompositeDetector(operators=[
            PCAPredictor(n_components=2),
            ResidualScorer(),
            PercentileDecider(),
        ])
        assert len(detector.operators) == 3

    def test_name_parameter(self, train_data):
        """
        目的：验证 oid 参数正确传递
        输入：oid="my_detector"
        预期："my_detector" in detector.oid
        """
        detector = CompositeDetector(
            operators=[ZScoreScorer(), PercentileDecider()],
            oid="my_detector",
        )
        assert "my_detector" in detector.oid


# ============================================================================
# 持久化测试
# ============================================================================

class TestPersistence:
    """测试 save / load 持久化"""

    def test_composite_scorer_save_with_predictor(self, train_data):
        """
        目的：验证 CompositeScorer（含 Predictor）持久化目录结构
        输入：PCAPredictor + ResidualScorer
        预期：生成 predictor/ 和 scorer_0/ 子目录
        """
        scorer = CompositeScorer(operators=[
            PCAPredictor(n_components=2),
            ResidualScorer(metric="mse"),
        ])
        scorer.fit(train_data)

        with tempfile.TemporaryDirectory() as tmpdir:
            path = Path(tmpdir) / "composite_scorer"
            scorer.save(path)
            assert (path / "predictor").exists()
            assert (path / "scorer_0").exists()

    def test_composite_scorer_save_multi_scorers(self, train_data):
        """
        目的：验证 CompositeScorer（多 Scorer 无 Predictor）持久化目录结构
        输入：ZScoreScorer + KNNScorer
        预期：生成 scorer_0/ 和 scorer_1/ 子目录，无 predictor/
        """
        scorer = CompositeScorer(operators=[
            ZScoreScorer(),
            KNNScorer(),
        ])
        scorer.fit(train_data)

        with tempfile.TemporaryDirectory() as tmpdir:
            path = Path(tmpdir) / "composite_scorer"
            scorer.save(path)
            assert (path / "scorer_0").exists()
            assert (path / "scorer_1").exists()
            assert not (path / "predictor").exists()

    def test_composite_detector_save(self, train_data):
        """
        目的：验证 CompositeDetector（含 Predictor）持久化目录结构
        输入：PCAPredictor + ResidualScorer + PercentileDecider
        预期：生成 predictor/、scorer_0/、decider/ 子目录
        """
        detector = CompositeDetector(operators=[
            PCAPredictor(n_components=2),
            ResidualScorer(metric="mse"),
            PercentileDecider(percentile=95.0),
        ])
        detector.fit(train_data)

        with tempfile.TemporaryDirectory() as tmpdir:
            path = Path(tmpdir) / "composite_detector"
            detector.save(path)
            assert (path / "predictor").exists()
            assert (path / "scorer_0").exists()
            assert (path / "decider").exists()

    def test_composite_detector_save_multi_scorers(self, train_data):
        """
        目的：验证 CompositeDetector（多 Scorer）持久化目录结构
        输入：ZScoreScorer + KNNScorer + PercentileDecider
        预期：生成 scorer_0/、scorer_1/、decider/ 子目录
        """
        detector = CompositeDetector(operators=[
            ZScoreScorer(),
            KNNScorer(),
            PercentileDecider(percentile=95.0),
        ])
        detector.fit(train_data)

        with tempfile.TemporaryDirectory() as tmpdir:
            path = Path(tmpdir) / "composite_detector"
            detector.save(path)
            assert (path / "scorer_0").exists()
            assert (path / "scorer_1").exists()
            assert (path / "decider").exists()
            assert not (path / "predictor").exists()

    def test_composite_scorer_load_raises(self):
        """
        目的：验证 CompositeScorer.load 对不存在的路径抛出 FileNotFoundError
        输入：不存在的路径 "/some/path"
        预期：抛出 FileNotFoundError
        """
        with pytest.raises(FileNotFoundError):
            CompositeScorer.load("/some/path")

    def test_composite_detector_load_raises(self):
        """
        目的：验证 CompositeDetector.load 对不存在的路径抛出 FileNotFoundError
        输入：不存在的路径 "/some/path"
        预期：抛出 FileNotFoundError
        """
        with pytest.raises(FileNotFoundError):
            CompositeDetector.load("/some/path")
