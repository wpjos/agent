# -*- coding: utf-8 -*-

"""
固定阈值决策器单元测试

对应源文件：
- threshold_decider.py: ThresholdDecider

测试范围：
- 阈值决策正确性
- DataFrame/ndarray 双类型支持
- 边界条件
"""

import numpy as np
import pytest
from pandas import DataFrame

from tsas.engine.operator.detection.threshold_decider import ThresholdDecider


# ============================================================================
# 公共测试数据
# ============================================================================

@pytest.fixture
def scores():
    """测试用异常分数（ndarray, 10）"""
    return np.array([0.5, 1.0, 1.5, 2.0, 2.5, 3.0, 3.5, 4.0, 4.5, 5.0])


@pytest.fixture
def scores_df(scores):
    """测试用异常分数（DataFrame）"""
    return DataFrame(scores, columns=["score"])


# ============================================================================
# ThresholdDecider 测试
# ============================================================================

class TestThresholdDecider:
    """测试固定阈值决策器"""

    def test_default_threshold(self, scores):
        """
        目的：验证默认阈值 3.0 的决策结果
        输入：10 个分数 [0.5, ..., 5.0]
        预期：> 3.0 的判定为异常（1），否则为正常（0）
        """
        decider = ThresholdDecider()
        labels = decider.run(scores)
        expected = np.array([0, 0, 0, 0, 0, 0, 1, 1, 1, 1])
        np.testing.assert_array_equal(labels, expected)

    def test_custom_threshold(self, scores):
        """
        目的：验证自定义阈值
        输入：threshold=2.5
        预期：> 2.5 的判定为异常
        """
        decider = ThresholdDecider(threshold=2.5)
        labels = decider.run(scores)
        expected = np.array([0, 0, 0, 0, 0, 1, 1, 1, 1, 1])
        np.testing.assert_array_equal(labels, expected)

    def test_strict_greater(self, scores):
        """
        目的：验证严格大于（非 >=）判定
        输入：threshold=3.0，分数含 3.0
        预期：3.0 判定为正常（0），3.5 判定为异常（1）
        """
        decider = ThresholdDecider(threshold=3.0)
        labels = decider.run(scores)
        # scores[5] == 3.0 应为 0，scores[6] == 3.5 应为 1
        assert labels[5] == 0
        assert labels[6] == 1

    def test_with_dataframe(self, scores_df):
        """
        目的：验证 DataFrame 输入输出
        输入：DataFrame 分数
        预期：输出为 DataFrame
        """
        decider = ThresholdDecider(threshold=3.0)
        labels = decider.run(scores_df)
        assert isinstance(labels, DataFrame)

    def test_all_normal(self):
        """
        目的：验证所有分数低于阈值
        预期：全部为 0
        """
        scores = np.array([0.1, 0.2, 0.3])
        decider = ThresholdDecider(threshold=5.0)
        labels = decider.run(scores)
        np.testing.assert_array_equal(labels, [0, 0, 0])

    def test_all_anomalous(self):
        """
        目的：验证所有分数超过阈值
        预期：全部为 1
        """
        scores = np.array([10.0, 20.0, 30.0])
        decider = ThresholdDecider(threshold=5.0)
        labels = decider.run(scores)
        np.testing.assert_array_equal(labels, [1, 1, 1])

    def test_2d_input(self):
        """
        目的：验证 2D 输入（单列）自动展平
        输入：(10, 1) 形状分数
        预期：输出 1D 标签
        """
        scores = np.array([[0.5], [1.0], [3.5], [4.0]])
        decider = ThresholdDecider(threshold=3.0)
        labels = decider.run(scores)
        expected = np.array([0, 0, 1, 1])
        np.testing.assert_array_equal(labels, expected)
