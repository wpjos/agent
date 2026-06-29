# -*- coding: utf-8 -*-

"""
KNN 异常检测算子单元测试

对应源文件：
- knn.py: KNNScorer, KNNDetector

测试范围：
- 训练和推理基本流程
- DataFrame/ndarray 双类型支持
- 不同距离度量方式
- 不同分数合并方式
- Detector 端到端流程
- 小样本边界条件（自动调整 K）
"""

import numpy as np
import pytest
from pandas import DataFrame

from tsas.engine.operator.detection.knn import (
    KNNScorer,
    KNNDetector,
    KNNDistanceMetric,
    KNNScoreMethod,
    KNNScorerExtraOutput,
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
    """测试用测试数据（ndarray, 20x3，含异常点）"""
    np.random.seed(123)
    normal = np.random.randn(18, 3)
    # 添加 2 个异常点（偏移 10 个标准差）
    abnormal = np.random.randn(2, 3) + 10
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
# KNNScorer 测试
# ============================================================================

class TestKNNScorer:
    """测试 KNN 评分器"""

    def test_fit_builds_index(self, train_data):
        """
        目的：验证 fit 正确构建近邻索引
        预期：_index 不为 None，_train_data 存储训练数据
        """
        scorer = KNNScorer(n_neighbors=5)
        scorer.fit(train_data)
        assert scorer._index is not None
        assert scorer._train_data is not None
        np.testing.assert_array_equal(scorer._train_data, train_data)

    def test_run_scores_shape(self, train_data, test_data):
        """
        目的：验证推理输出形状
        预期：输出 (n_samples,)
        """
        scorer = KNNScorer(n_neighbors=5)
        scorer.fit(train_data)
        scores, eo = scorer.run(test_data)
        assert scores.shape == (20,)

    def test_scores_positive(self, train_data, test_data):
        """
        目的：验证分数均为非负
        预期：所有分数 >= 0（距离为正值）
        """
        scorer = KNNScorer(n_neighbors=5)
        scorer.fit(train_data)
        scores, _ = scorer.run(test_data)
        assert np.all(scores >= 0)

    def test_anomalous_higher_scores(self, train_data, test_data):
        """
        目的：验证异常点分数高于正常点
        预期：异常点（最后2个）的分数 > 正常点平均分数
        """
        scorer = KNNScorer(n_neighbors=5)
        scorer.fit(train_data)
        scores, _ = scorer.run(test_data)
        normal_avg = scores[:18].mean()
        abnormal_avg = scores[18:].mean()
        assert abnormal_avg > normal_avg

    def test_score_method_maximum(self, train_data, test_data):
        """
        目的：验证 maximum 合并方式
        预期：maximum 给出最高分数
        """
        scorer_max = KNNScorer(n_neighbors=5, score_method=KNNScoreMethod.MAXIMUM)
        scorer_mean = KNNScorer(n_neighbors=5, score_method=KNNScoreMethod.MEAN)
        scorer_max.fit(train_data)
        scorer_mean.fit(train_data)
        scores_max, _ = scorer_max.run(test_data)
        scores_mean, _ = scorer_mean.run(test_data)
        # maximum >= mean
        assert np.all(scores_max >= scores_mean)

    def test_distance_metric_manhattan(self, train_data, test_data):
        """
        目的：验证曼哈顿距离
        预期：能正常计算，分数 >= 0
        """
        scorer = KNNScorer(n_neighbors=5, distance_metric=KNNDistanceMetric.MANHATTAN)
        scorer.fit(train_data)
        scores, _ = scorer.run(test_data)
        assert np.all(scores >= 0)

    def test_small_sample_auto_adjust_k(self, train_data):
        """
        目的：验证训练样本数不足 K 时自动调整
        输入：训练数据 10 个样本，K=20
        预期：K 自动调整为 10
        """
        small_train = train_data[:10]
        scorer = KNNScorer(n_neighbors=20)
        scorer.fit(small_train)
        # 索引应使用 10 个邻居（训练样本数）
        assert scorer._index.n_neighbors == 10

    def test_extended_output(self, train_data, test_data):
        """
        目的：验证附加输出正确
        预期：返回配置信息
        """
        scorer = KNNScorer(n_neighbors=5, distance_metric="euclidean", score_method="maximum")
        scorer.fit(train_data)
        scores, eo = scorer.run(test_data)
        assert isinstance(eo, KNNScorerExtraOutput)
        assert eo.n_neighbors == 5
        assert eo.distance_metric == "euclidean"
        assert eo.score_method == "maximum"

    def test_with_dataframe(self, train_df, test_df):
        """
        目的：验证 DataFrame 输入输出
        预期：输出为 DataFrame
        """
        scorer = KNNScorer(n_neighbors=5)
        scorer.fit(train_df)
        scores, _ = scorer.run(test_df)
        assert isinstance(scores, DataFrame)

    def test_before_fit_raises(self, test_data):
        """
        目的：验证未训练时 run 抛出 RuntimeError
        预期：抛出 RuntimeError
        """
        scorer = KNNScorer(n_neighbors=5)
        with pytest.raises(RuntimeError):
            scorer.run(test_data)


# ============================================================================
# KNNDetector 测试
# ============================================================================

class TestKNNDetector:
    """测试 KNN 检测器端到端流程"""

    def test_fit_and_run(self, train_data, test_data):
        """
        目的：验证端到端 fit → run 流程
        预期：输出 1D 标签数组，值为 0 或 1
        """
        detector = KNNDetector(n_neighbors=5, percentile=95.0)
        detector.fit(train_data)
        labels = detector.run(test_data)
        assert labels.shape == (20,)
        assert set(labels).issubset({0, 1})

    def test_detects_anomalies(self, train_data, test_data):
        """
        目的：验证异常点被检测出来
        预期：异常点（最后2个）判定为 1
        """
        detector = KNNDetector(n_neighbors=5, percentile=95.0)
        detector.fit(train_data)
        labels = detector.run(test_data)
        # 异常点应被检测到
        assert labels[18] == 1 or labels[19] == 1

    def test_with_dataframe(self, train_df, test_df):
        """
        目的：验证 DataFrame 端到端流程
        预期：输出为 DataFrame
        """
        detector = KNNDetector(n_neighbors=5, percentile=95.0)
        detector.fit(train_df)
        labels = detector.run(test_df)
        assert isinstance(labels, DataFrame)

    def test_before_fit_raises(self, test_data):
        """
        目的：验证未训练时 run 抛出 RuntimeError
        预期：抛出 RuntimeError
        """
        detector = KNNDetector(n_neighbors=5, percentile=95.0)
        with pytest.raises(RuntimeError):
            detector.run(test_data)


# ============================================================================
# KNNScorer Save/Load Roundtrip 测试
# ============================================================================

class TestKNNScorerSaveLoad:
    """测试 KNNScorer 持久化 roundtrip"""

    def test_save_load_roundtrip(self, train_data, test_data, tmp_path):
        """
        目的：验证 save → load 后推理结果一致
        预期：loaded scorer 的输出与原始 scorer 完全相同
        """
        scorer = KNNScorer(n_neighbors=5)
        scorer.fit(train_data)
        original_scores, original_eo = scorer.run(test_data)

        save_dir = tmp_path / 'knn_scorer'
        scorer.save(save_dir)
        loaded = KNNScorer.load(save_dir)

        loaded_scores, loaded_eo = loaded.run(test_data)
        np.testing.assert_allclose(original_scores, loaded_scores)

    def test_loaded_state_restored(self, train_data, tmp_path):
        """
        目的：验证加载后内部状态正确恢复
        预期：_index 和 _train_data 已恢复，_fitted 为 True
        """
        scorer = KNNScorer(n_neighbors=5)
        scorer.fit(train_data)

        save_dir = tmp_path / 'knn_scorer'
        scorer.save(save_dir)
        loaded = KNNScorer.load(save_dir)

        assert loaded.is_fitted
        assert loaded._index is not None
        np.testing.assert_array_equal(loaded._train_data, train_data)


# ============================================================================
# KNNDetector Save/Load Roundtrip 测试
# ============================================================================

class TestKNNDetectorSaveLoad:
    """测试 KNNDetector 持久化 roundtrip"""

    def test_save_load_roundtrip(self, train_data, test_data, tmp_path):
        """
        目的：验证 save → load 后推理结果一致
        预期：loaded detector 的输出与原始 detector 完全相同
        """
        detector = KNNDetector(n_neighbors=5, percentile=95.0)
        detector.fit(train_data)
        original_labels = detector.run(test_data)

        save_dir = tmp_path / 'knn_detector'
        detector.save(save_dir)
        loaded = KNNDetector.load(save_dir)

        loaded_labels = loaded.run(test_data)
        np.testing.assert_array_equal(original_labels, loaded_labels)

    def test_loaded_state_restored(self, train_data, tmp_path):
        """
        目的：验证加载后子组件状态正确恢复
        预期：_scorer 和 _decider 已恢复，_fitted 为 True
        """
        detector = KNNDetector(n_neighbors=5, percentile=95.0)
        detector.fit(train_data)

        save_dir = tmp_path / 'knn_detector'
        detector.save(save_dir)
        loaded = KNNDetector.load(save_dir)

        assert loaded.is_fitted
        assert loaded._scorer.is_fitted
        assert loaded._decider.is_fitted
