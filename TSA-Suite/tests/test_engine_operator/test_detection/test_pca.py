# -*- coding: utf-8 -*-

"""
PCA 异常检测算子单元测试

对应源文件：
- pca.py: PCAPredictor, PCAScorer, PCADetector

测试范围：
- PCAPredictor: fit/run 基本流程、DataFrame 支持、边界条件
- PCAScorer: fit/run 流程、异常分数正确性、metric 选择、EO 透传、DataFrame 支持
- PCADetector: 端到端 fit/run、异常检测能力、EO 透传、DataFrame 支持
"""

import numpy as np
import pytest
from pandas import DataFrame

from tsas.engine.operator.detection.pca import (
    PCAPredictor,
    PCAPredictorExtraOutput,
    PCAScorer,
    PCAScorerConfig,
    PCAScorerExtraOutput,
    PCADetector,
    PCADetectorConfig,
    PCADetectorExtraOutput,
)


# ============================================================================
# 公共测试数据
# ============================================================================

@pytest.fixture
def train_data():
    """测试用训练数据（ndarray, 100x3）"""
    np.random.seed(42)
    return np.random.randn(100, 3)


@pytest.fixture
def test_data():
    """测试用测试数据（ndarray, 60x3，含异常点）"""
    np.random.seed(123)
    normal = np.random.randn(50, 3)
    abnormal = np.random.randn(10, 3) * 5 + 10
    return np.vstack([normal, abnormal])


@pytest.fixture
def train_df(train_data):
    """测试用训练数据（DataFrame）"""
    return DataFrame(train_data, columns=["a", "b", "c"])


@pytest.fixture
def test_df(test_data):
    """测试用测试数据（DataFrame）"""
    return DataFrame(test_data, columns=["a", "b", "c"])


# ============================================================================
# PCAPredictor 测试
# ============================================================================

class TestPCAPredictor:
    """测试 PCA 预测器"""

    def test_fit_learns_components(self, train_data):
        """
        目的：验证 fit 正确学习主成分
        输入：(100, 3) 训练数据，n_components=2
        预期：_components 形状 (2, 3)
        """
        predictor = PCAPredictor(n_components=2)
        predictor.fit(train_data)
        assert predictor._components is not None
        assert predictor._components.shape == (2, 3)
        assert predictor._mean is not None

    def test_run_reconstruction(self, train_data):
        """
        目的：验证推理输出重构值
        输入：训练数据
        预期：重构值形状与输入相同
        """
        predictor = PCAPredictor(n_components=2)
        predictor.fit(train_data)
        pred, eo = predictor.run(train_data)
        assert pred.shape == train_data.shape
        assert isinstance(eo, PCAPredictorExtraOutput)
        assert eo.n_components == 2
        assert len(eo.explained_variance_ratio) == 2

    def test_full_components_perfect_reconstruction(self, train_data):
        """
        目的：验证 n_components 等于特征数时重构完美
        输入：n_components=3（等于特征数）
        预期：重构值 ≈ 原始值（去中心化后）
        """
        predictor = PCAPredictor(n_components=3)
        predictor.fit(train_data)
        pred, _ = predictor.run(train_data)
        # n_components == n_features 时，重构应完美
        np.testing.assert_allclose(pred, train_data, atol=1e-10)

    def test_with_dataframe(self, train_df, test_df):
        """
        目的：验证 DataFrame 输入输出
        输入：DataFrame
        预期：输出为 DataFrame
        """
        predictor = PCAPredictor(n_components=2)
        predictor.fit(train_df)
        pred, _ = predictor.run(test_df)
        assert isinstance(pred, DataFrame)

    def test_explained_variance_ratio(self, train_data):
        """
        目的：验证解释方差比正确
        输入：(100, 3) 训练数据
        预期：解释方差比之和 ≤ 1.0
        """
        predictor = PCAPredictor(n_components=2)
        predictor.fit(train_data)
        total = sum(predictor._explained_variance_ratio)
        assert total <= 1.0 + 1e-10
        assert all(v >= 0 for v in predictor._explained_variance_ratio)

    def test_constant_data_degenerate_case(self):
        """
        目的：验证所有特征为常数时的退化情况
        输入：全部为常数的训练数据
        预期：解释方差比全为 0，不报错
        """
        # 所有特征为常数（方差为 0）
        train = np.ones((50, 3)) * 5.0
        predictor = PCAPredictor(n_components=2)
        predictor.fit(train)
        # 解释方差比应全为 0
        assert all(v == 0 for v in predictor._explained_variance_ratio)

    def test_n_components_exceeds_features(self):
        """
        目的：验证 n_components 超过特征数时自动调整
        输入：2个特征的训练数据，n_components=5
        预期：实际使用 2 个主成分
        """
        train = np.random.randn(50, 2)
        predictor = PCAPredictor(n_components=5)
        predictor.fit(train)
        assert predictor._components.shape[0] == 2  # 自动调整为特征数


# ============================================================================
# PCAScorer 测试
# ============================================================================

class TestPCAScorer:
    """测试 PCA 评分器"""

    def test_fit_and_run(self, train_data, test_data):
        """
        目的：验证训练+推理基本流程
        输入：(100, 3) 训练数据，(60, 3) 测试数据
        预期：输出 1D 异常分数，形状 (60,)
        """
        scorer = PCAScorer(n_components=2)
        scorer.fit(train_data)
        scores, eo = scorer.run(test_data)
        assert scores.shape == (60,)
        assert isinstance(eo, PCAScorerExtraOutput)

    def test_scores_non_negative(self, train_data, test_data):
        """
        目的：验证 MSE 分数恒非负
        输入：metric="mse"
        预期：所有分数 >= 0
        """
        scorer = PCAScorer(n_components=2, metric="mse")
        scorer.fit(train_data)
        scores, _ = scorer.run(test_data)
        assert np.all(scores >= 0)

    def test_anomalous_higher_scores(self, train_data, test_data):
        """
        目的：验证异常点分数高于正常点
        输入：测试数据最后 10 个为异常点（偏移+10，放大5倍）
        预期：异常点平均分数 > 正常点平均分数
        """
        scorer = PCAScorer(n_components=2)
        scorer.fit(train_data)
        scores, _ = scorer.run(test_data)
        normal_avg = scores[:50].mean()
        abnormal_avg = scores[50:].mean()
        assert abnormal_avg > normal_avg

    def test_metric_mae(self, train_data, test_data):
        """
        目的：验证 MAE 模式正常工作
        输入：metric="mae"
        预期：输出 1D 分数，所有分数 >= 0
        """
        scorer = PCAScorer(n_components=2, metric="mae")
        scorer.fit(train_data)
        scores, eo = scorer.run(test_data)
        assert scores.shape == (60,)
        assert np.all(scores >= 0)
        assert isinstance(eo, PCAScorerExtraOutput)

    def test_with_dataframe(self, train_df, test_df):
        """
        目的：验证 DataFrame 输入输出兼容性
        输入：DataFrame 训练/测试数据
        预期：输出为 DataFrame，列名为 "score"
        """
        scorer = PCAScorer(n_components=2)
        scorer.fit(train_df)
        scores, _ = scorer.run(test_df)
        assert isinstance(scores, DataFrame)
        assert list(scores.columns) == ["score"]

    def test_extended_output(self, train_data, test_data):
        """
        目的：验证 EO 聚合子组件的附加输出
        输入：PCAScorer(n_components=2)
        预期：eo.pca_eo 包含解释方差比，eo.residual_eo 包含逐变量分数
        """
        scorer = PCAScorer(n_components=2)
        scorer.fit(train_data)
        _, eo = scorer.run(test_data)
        assert isinstance(eo, PCAScorerExtraOutput)
        # PCAPredictor EO
        assert eo.pca_eo is not None
        assert isinstance(eo.pca_eo, PCAPredictorExtraOutput)
        assert eo.pca_eo.n_components == 2
        assert len(eo.pca_eo.explained_variance_ratio) == 2
        # ResidualScorer EO
        assert eo.residual_eo is not None

    def test_before_fit_raises(self, test_data):
        """
        目的：验证未训练时 run 抛出 RuntimeError
        预期：抛出 RuntimeError
        """
        scorer = PCAScorer(n_components=2)
        with pytest.raises(RuntimeError, match="训练尚未完成"):
            scorer.run(test_data)

    def test_name(self):
        """
        目的：验证 name() 返回正确标识
        预期：返回 "pca_scorer"
        """
        assert PCAScorer.name() == "pca_scorer"


# ============================================================================
# PCADetector 测试
# ============================================================================

class TestPCADetector:
    """测试 PCA 检测器端到端流程"""

    def test_fit_and_run(self, train_data, test_data):
        """
        目的：验证端到端 fit → run 流程
        预期：输出 1D 标签数组，值为 0 或 1
        """
        detector = PCADetector(n_components=2, percentile=95.0)
        detector.fit(train_data)
        labels, eo = detector.run(test_data)
        assert labels.shape == (60,)
        assert set(labels).issubset({0, 1})
        assert isinstance(eo, PCADetectorExtraOutput)

    def test_detects_anomalies(self, train_data, test_data):
        """
        目的：验证异常点能被检测出来
        输入：测试数据最后 10 个为异常点，percentile=90.0
        预期：异常点中至少有一个被判为 1
        """
        detector = PCADetector(n_components=2, percentile=90.0)
        detector.fit(train_data)
        labels, _ = detector.run(test_data)
        assert any(labels[50:] == 1)

    def test_with_dataframe(self, train_df, test_df):
        """
        目的：验证 DataFrame 输入输出兼容性
        预期：输出为 DataFrame，列名为 "label"
        """
        detector = PCADetector(n_components=2, percentile=95.0)
        detector.fit(train_df)
        labels, _ = detector.run(test_df)
        assert isinstance(labels, DataFrame)
        assert list(labels.columns) == ["label"]

    def test_extended_output(self, train_data, test_data):
        """
        目的：验证 EO 聚合子组件的附加输出
        预期：eo.scorer_eo 和 eo.decider_eo 非空
        """
        detector = PCADetector(n_components=2, percentile=95.0)
        detector.fit(train_data)
        _, eo = detector.run(test_data)
        assert isinstance(eo, PCADetectorExtraOutput)
        assert eo.scorer_eo is not None
        assert isinstance(eo.scorer_eo, PCAScorerExtraOutput)
        assert eo.decider_eo is not None

    def test_before_fit_raises(self, test_data):
        """
        目的：验证未训练时 run 抛出 RuntimeError
        预期：抛出 RuntimeError
        """
        detector = PCADetector(n_components=2, percentile=95.0)
        with pytest.raises(RuntimeError, match="训练尚未完成"):
            detector.run(test_data)

    def test_name(self):
        """
        目的：验证 name() 返回正确标识
        预期：返回 "pca_detector"
        """
        assert PCADetector.name() == "pca_detector"


# ============================================================================
# PCAPredictor Save/Load Roundtrip 测试
# ============================================================================

class TestPCAPredictorSaveLoad:
    """测试 PCAPredictor 持久化 roundtrip"""

    def test_save_load_roundtrip(self, train_data, test_data, tmp_path):
        """
        目的：验证 save → load 后推理结果一致
        预期：loaded predictor 的输出与原始 predictor 完全相同
        """
        predictor = PCAPredictor(n_components=2)
        predictor.fit(train_data)
        original_pred, original_eo = predictor.run(test_data)

        save_dir = tmp_path / 'pca_predictor'
        predictor.save(save_dir)
        loaded = PCAPredictor.load(save_dir)

        loaded_pred, loaded_eo = loaded.run(test_data)
        np.testing.assert_allclose(original_pred, loaded_pred)

    def test_loaded_state_restored(self, train_data, tmp_path):
        """
        目的：验证加载后内部状态正确恢复
        预期：_mean、_components、_explained_variance_ratio 已恢复，_fitted 为 True
        """
        predictor = PCAPredictor(n_components=2)
        predictor.fit(train_data)

        save_dir = tmp_path / 'pca_predictor'
        predictor.save(save_dir)
        loaded = PCAPredictor.load(save_dir)

        assert loaded.is_fitted
        np.testing.assert_allclose(loaded._mean, predictor._mean)
        np.testing.assert_allclose(loaded._components, predictor._components)
        np.testing.assert_allclose(
            loaded._explained_variance_ratio, predictor._explained_variance_ratio
        )


# ============================================================================
# PCAScorer Save/Load Roundtrip 测试
# ============================================================================

class TestPCAScorerSaveLoad:
    """测试 PCAScorer 持久化 roundtrip"""

    def test_save_load_roundtrip(self, train_data, test_data, tmp_path):
        """
        目的：验证 save → load 后推理结果一致
        预期：loaded scorer 的输出与原始 scorer 完全相同
        """
        scorer = PCAScorer(n_components=2)
        scorer.fit(train_data)
        original_scores, _ = scorer.run(test_data)

        save_dir = tmp_path / 'pca_scorer'
        scorer.save(save_dir)
        loaded = PCAScorer.load(save_dir)

        loaded_scores, _ = loaded.run(test_data)
        np.testing.assert_allclose(original_scores, loaded_scores)

    def test_loaded_state_restored(self, train_data, tmp_path):
        """
        目的：验证加载后子组件状态正确恢复
        预期：_predictor 已恢复，_fitted 为 True
        """
        scorer = PCAScorer(n_components=2)
        scorer.fit(train_data)

        save_dir = tmp_path / 'pca_scorer'
        scorer.save(save_dir)
        loaded = PCAScorer.load(save_dir)

        assert loaded.is_fitted
        assert loaded._predictor.is_fitted


# ============================================================================
# PCADetector Save/Load Roundtrip 测试
# ============================================================================

class TestPCADetectorSaveLoad:
    """测试 PCADetector 持久化 roundtrip"""

    def test_save_load_roundtrip(self, train_data, test_data, tmp_path):
        """
        目的：验证 save → load 后推理结果一致
        预期：loaded detector 的输出与原始 detector 完全相同
        """
        detector = PCADetector(n_components=2, percentile=95.0)
        detector.fit(train_data)
        original_labels, _ = detector.run(test_data)

        save_dir = tmp_path / 'pca_detector'
        detector.save(save_dir)
        loaded = PCADetector.load(save_dir)

        loaded_labels, _ = loaded.run(test_data)
        np.testing.assert_array_equal(original_labels, loaded_labels)

    def test_loaded_state_restored(self, train_data, tmp_path):
        """
        目的：验证加载后子组件状态正确恢复
        预期：_scorer 和 _decider 已恢复，_fitted 为 True
        """
        detector = PCADetector(n_components=2, percentile=95.0)
        detector.fit(train_data)

        save_dir = tmp_path / 'pca_detector'
        detector.save(save_dir)
        loaded = PCADetector.load(save_dir)

        assert loaded.is_fitted
        assert loaded._scorer.is_fitted
        assert loaded._decider.is_fitted
