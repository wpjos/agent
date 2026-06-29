# -*- coding: utf-8 -*-

"""
Z-Score 异常检测算子单元测试

对应源文件：
- zscore.py: ZScoreScorer, ZScoreDetector

测试范围：
- 训练和推理基本流程
- DataFrame/ndarray 双类型支持
- 边界条件（零标准差、异常检测正确性）
- Detector 端到端流程
"""

import numpy as np
import pytest
from pandas import DataFrame

from tsas.engine.operator.detection.zscore import (
    ZScoreScorer,
    ZScoreDetector,
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
    abnormal = np.random.randn(2, 3) * 2 + 10
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
# ZScoreScorer 测试
# ============================================================================

class TestZScoreScorer:
    """测试 Z-Score 评分器"""

    def test_fit_learns_params(self, train_data):
        """
        目的：验证 fit 正确学习均值和标准差
        预期：_mean 和 _std 与手动计算一致
        """
        scorer = ZScoreScorer()
        scorer.fit(train_data)
        np.testing.assert_allclose(scorer._mean, train_data.mean(axis=0))
        np.testing.assert_allclose(scorer._std, train_data.std(axis=0))

    def test_run_scores_shape(self, train_data, test_data):
        """
        目的：验证推理输出形状
        预期：输出 (n_samples,)
        """
        scorer = ZScoreScorer()
        scorer.fit(train_data)
        scores = scorer.run(test_data)
        assert scores.shape == (20,)

    def test_scores_positive(self, train_data, test_data):
        """
        目的：验证分数均为非负
        预期：所有分数 >= 0
        """
        scorer = ZScoreScorer()
        scorer.fit(train_data)
        scores = scorer.run(test_data)
        assert np.all(scores >= 0)

    def test_anomalous_higher_scores(self, train_data, test_data):
        """
        目的：验证异常点分数高于正常点
        预期：异常点（最后2个）的分数 > 正常点平均分数
        """
        scorer = ZScoreScorer()
        scorer.fit(train_data)
        scores = scorer.run(test_data)
        normal_avg = scores[:18].mean()
        abnormal_avg = scores[18:].mean()
        assert abnormal_avg > normal_avg

    def test_zero_std_protection(self):
        """
        目的：验证零标准差特征的保护处理
        输入：第3列全为常数
        预期：不除零，该特征 z-score 为 0
        """
        train = np.column_stack([
            np.random.randn(50),
            np.random.randn(50),
            np.ones(50) * 5.0,  # 常数列
        ])
        scorer = ZScoreScorer()
        scorer.fit(train)
        # std=0 被替换为 1.0
        assert scorer._std[2] == 1.0

    def test_with_dataframe(self, train_df, test_df):
        """
        目的：验证 DataFrame 输入输出
        预期：输出为 DataFrame
        """
        scorer = ZScoreScorer()
        scorer.fit(train_df)
        scores = scorer.run(test_df)
        assert isinstance(scores, DataFrame)

    def test_before_fit_raises(self, test_data):
        """
        目的：验证未训练时 run 抛出 RuntimeError
        预期：抛出 RuntimeError
        """
        scorer = ZScoreScorer()
        with pytest.raises(RuntimeError):
            scorer.run(test_data)


# ============================================================================
# ZScoreDetector 测试
# ============================================================================

class TestZScoreDetector:
    """测试 Z-Score 检测器端到端流程"""

    def test_fit_and_run(self, train_data, test_data):
        """
        目的：验证端到端 fit → run 流程
        预期：输出 1D 标签数组，值为 0 或 1
        """
        detector = ZScoreDetector(threshold=3.0)
        detector.fit(train_data)
        labels = detector.run(test_data)
        assert labels.shape == (20,)
        assert set(labels).issubset({0, 1})

    def test_detects_anomalies(self, train_data, test_data):
        """
        目的：验证异常点被检测出来
        预期：异常点（最后2个）判定为 1
        """
        detector = ZScoreDetector(threshold=3.0)
        detector.fit(train_data)
        labels = detector.run(test_data)
        # 异常点应被检测到
        assert labels[18] == 1 or labels[19] == 1

    def test_with_dataframe(self, train_df, test_df):
        """
        目的：验证 DataFrame 端到端流程
        预期：输出为 DataFrame
        """
        detector = ZScoreDetector(threshold=3.0)
        detector.fit(train_df)
        labels = detector.run(test_df)
        assert isinstance(labels, DataFrame)

    def test_before_fit_raises(self, test_data):
        """
        目的：验证未训练时 run 抛出 RuntimeError
        预期：抛出 RuntimeError
        """
        detector = ZScoreDetector()
        with pytest.raises(RuntimeError):
            detector.run(test_data)


# ============================================================================
# ZScoreScorer Save/Load Roundtrip 测试
# ============================================================================

class TestZScoreScorerSaveLoad:
    """测试 ZScoreScorer 持久化 roundtrip"""

    def test_save_load_roundtrip(self, train_data, test_data, tmp_path):
        """
        目的：验证 save → load 后推理结果一致
        预期：loaded scorer 的输出与原始 scorer 完全相同
        """
        scorer = ZScoreScorer()
        scorer.fit(train_data)
        original_scores = scorer.run(test_data)

        save_dir = tmp_path / 'zscore_scorer'
        scorer.save(save_dir)
        loaded = ZScoreScorer.load(save_dir)

        loaded_scores = loaded.run(test_data)
        np.testing.assert_array_equal(original_scores, loaded_scores)

    def test_loaded_state_restored(self, train_data, tmp_path):
        """
        目的：验证加载后内部状态正确恢复
        预期：_mean、_std 与原始值一致，_fitted 为 True
        """
        scorer = ZScoreScorer()
        scorer.fit(train_data)

        save_dir = tmp_path / 'zscore_scorer'
        scorer.save(save_dir)
        loaded = ZScoreScorer.load(save_dir)

        np.testing.assert_allclose(loaded._mean, scorer._mean)
        np.testing.assert_allclose(loaded._std, scorer._std)
        assert loaded.is_fitted

# ============================================================================
# ZScoreDetector Save/Load Roundtrip 测试
# ============================================================================

class TestZScoreDetectorSaveLoad:
    """测试 ZScoreDetector 持久化 roundtrip"""

    def test_save_load_roundtrip(self, train_data, test_data, tmp_path):
        """
        目的：验证 save → load 后推理结果一致
        预期：loaded detector 的输出与原始 detector 完全相同
        """
        detector = ZScoreDetector(threshold=3.0)
        detector.fit(train_data)
        original_labels = detector.run(test_data)

        save_dir = tmp_path / 'zscore_detector'
        detector.save(save_dir)
        loaded = ZScoreDetector.load(save_dir)

        loaded_labels = loaded.run(test_data)
        np.testing.assert_array_equal(original_labels, loaded_labels)

    def test_loaded_state_restored(self, train_data, tmp_path):
        """
        目的：验证加载后子组件状态正确恢复
        预期：_scorer 已恢复，_fitted 为 True
        """
        detector = ZScoreDetector(threshold=3.0)
        detector.fit(train_data)

        save_dir = tmp_path / 'zscore_detector'
        detector.save(save_dir)
        loaded = ZScoreDetector.load(save_dir)

        assert loaded.is_fitted
        assert loaded._scorer.is_fitted
