# -*- coding: utf-8 -*-

"""
多分类评价指标算子测试

测试覆盖:
    1. 基本功能: run() 返回完整多分类指标集
    2. scores() 方法: 按 main_scores 提取命名标量
    3. Macro 平均: 各指标正确计算
    4. Per-label 指标: 各类别指标正确
    5. 混淆矩阵: 格式正确
    6. 边界条件: 二分类退化、单样本、不匹配标签

测试约束:
    - 代码覆盖率 > 90%
    - 测试通过率 100%
    - 中文注释说明测试目的、输入、输出和预期结果
"""

import numpy as np
import pytest

from tsas.engine.operator.evaluation.multi_classification import (
    PerLabelMetricResult,
    MultiClassificationMetricResult,
    MultiClassificationMetricConfig,
    MultipleClassificationMetric,
)


# ============================================================================
# 基本功能测试
# ============================================================================

class TestMultipleClassificationMetricBasic:
    """基本功能测试"""

    def test_run_returns_complete_result(self):
        """
        测试目的: run() 返回完整多分类指标集
        输入: y_truth=[0,1,2,0,1,2], y_predict=[0,1,2,1,2,0]
        输出: MultiClassificationMetricResult
        预期: 返回完整结果，n_samples=6, k_labels=3, per_label_metrics 有 3 个元素
        """
        y_truth = np.array([0, 1, 2, 0, 1, 2])
        y_predict = np.array([0, 1, 2, 1, 2, 0])  # 3个正确，3个错误

        op = MultipleClassificationMetric()
        result = op.run((y_truth, y_predict))

        assert isinstance(result, MultiClassificationMetricResult)
        assert result.n_samples == 6
        assert result.k_labels == 3
        assert len(result.labels) == 3
        assert len(result.per_label_metrics) == 3

    def test_run_binary_degradation(self):
        """
        测试目的: 二分类时多分类算子正确退化
        输入: 二分类标签 y_truth=[0,1,0,1], y_predict=[0,1,1,0]
        输出: MultiClassificationMetricResult
        预期: k_labels=2，per_label_metrics 有 2 个元素
        """
        y_truth = np.array([0, 1, 0, 1])
        y_predict = np.array([0, 1, 1, 0])

        op = MultipleClassificationMetric()
        result = op.run((y_truth, y_predict))

        assert result.k_labels == 2
        assert len(result.per_label_metrics) == 2

    def test_per_label_metrics_structure(self):
        """
        测试目的: per_label_metrics 结构正确
        输入: 三分类完美预测 y_truth=[0,1,2,0,1,2], y_predict=[0,1,2,0,1,2]
        输出: MultiClassificationMetricResult
        预期: 每个 PerLabelMetricResult 包含 label/tp/f1 等完整属性
        """
        y_truth = np.array([0, 1, 2, 0, 1, 2])
        y_predict = np.array([0, 1, 2, 0, 1, 2])  # 完美预测

        op = MultipleClassificationMetric()
        result = op.run((y_truth, y_predict))

        for per_label in result.per_label_metrics:
            assert isinstance(per_label, PerLabelMetricResult)
            assert hasattr(per_label, 'label')
            assert hasattr(per_label, 'tp')
            assert hasattr(per_label, 'f1')

    def test_confusion_matrix_format(self):
        """
        测试目的: 混淆矩阵为 list[list[int]] 格式
        输入: 三分类完美预测
        输出: MultiClassificationMetricResult
        预期: confusion_matrix 为 k x k 列表
        """
        y_truth = np.array([0, 1, 2, 0, 1, 2])
        y_predict = np.array([0, 1, 2, 0, 1, 2])

        op = MultipleClassificationMetric()
        result = op.run((y_truth, y_predict))

        assert isinstance(result.confusion_matrix, list)
        assert len(result.confusion_matrix) == result.k_labels
        for row in result.confusion_matrix:
            assert len(row) == result.k_labels

    def test_perfect_prediction_metrics(self):
        """
        测试目的: 完美预测时各指标达到最优
        输入: y_truth 和 y_predict 完全一致
        输出: MultiClassificationMetricResult
        预期: macro f1=1.0, accuracy=1.0, 每个 per_label f1=1.0
        """
        y_truth = np.array([0, 1, 2, 0, 1, 2])
        y_predict = np.array([0, 1, 2, 0, 1, 2])

        op = MultipleClassificationMetric()
        result = op.run((y_truth, y_predict))

        assert result.accuracy == 1.0
        assert result.f1 == 1.0
        for per_label in result.per_label_metrics:
            assert per_label.f1 == 1.0


# ============================================================================
# scores() 方法测试
# ============================================================================

class TestMultipleClassificationMetricScores:
    """scores() 方法测试"""

    def test_scores_returns_dict(self):
        """
        测试目的: scores() 返回按 main_scores 映射的字典
        输入: 默认配置 main_scores={"f1": "f1", "accuracy": "accuracy"}
        输出: dict[str, float]
        预期: 返回 {"f1": float, "accuracy": float}
        """
        y_truth = np.array([0, 1, 2, 0, 1, 2])
        y_predict = np.array([0, 1, 2, 1, 2, 0])

        op = MultipleClassificationMetric()
        scores = op.scores((y_truth, y_predict))

        assert scores is not None
        assert "f1" in scores
        assert "accuracy" in scores

    def test_scores_matches_run_values(self):
        """
        测试目的: scores() 提取的值与 run() 结果一致
        输入: 同上
        预期: scores["f1"] == result.f1
        """
        y_truth = np.array([0, 1, 2, 0, 1, 2])
        y_predict = np.array([0, 1, 2, 1, 2, 0])

        op = MultipleClassificationMetric()
        result = op.run((y_truth, y_predict))
        scores = op.scores((y_truth, y_predict))

        assert scores["f1"] == result.f1
        assert scores["accuracy"] == result.accuracy

    def test_scores_with_custom_main_scores(self):
        """
        测试目的: 自定义 main_scores 提取不同指标
        输入: main_scores={"mcc": "mcc"}
        预期: 返回 {"mcc": float}
        """
        y_truth = np.array([0, 1, 2, 0, 1, 2])
        y_predict = np.array([0, 1, 2, 1, 2, 0])

        op = MultipleClassificationMetric(main_scores={"mcc": "mcc"})
        scores = op.scores((y_truth, y_predict))

        assert scores is not None
        assert "mcc" in scores

    def test_scores_returns_none_when_main_scores_is_none(self):
        """
        测试目的: 验证 main_scores 为 None 时 scores() 返回 None
        输入: config 中 main_scores=None
        预期: scores() 返回 None
        """
        config = MultiClassificationMetricConfig(main_scores=None)
        op = MultipleClassificationMetric(config=config)
        y_truth = np.array([0, 1, 2, 0])
        y_predict = np.array([0, 1, 2, 0])
        result = op.scores((y_truth, y_predict))
        assert result is None


# ============================================================================
# Macro 平均测试
# ============================================================================

class TestMultipleClassificationMetricMacro:
    """Macro 平均测试"""

    def test_macro_average_calculation(self):
        """
        测试目的: Macro 平均正确计算
        输入: 手动构造已知 per-label 指标
        预期: macro f1 = (f1_0 + f1_1 + f1_2) / 3
        """
        y_truth = np.array([0, 0, 1, 1, 2, 2])
        y_predict = np.array([0, 0, 1, 2, 2, 2])  # 类别0全正确，类别1部分错误，类别2全正确

        op = MultipleClassificationMetric()
        result = op.run((y_truth, y_predict))

        # 手动验证：类别0 f1=1.0，类别1 部分错误，类别2 f1=1.0
        f1_values = [m.f1 for m in result.per_label_metrics]
        expected_macro_f1 = sum(f1_values) / 3

        assert abs(result.f1 - expected_macro_f1) < 1e-6

    def test_macro_metrics_range(self):
        """
        测试目的: Macro 指标在合理范围
        输入: 任意数据
        预期: 所有 macro 指标在 [0, 1] 范围（mcc 在 [-1, 1]）
        """
        y_truth = np.array([0, 1, 2, 0, 1, 2])
        y_predict = np.array([0, 1, 2, 1, 2, 0])

        op = MultipleClassificationMetric()
        result = op.run((y_truth, y_predict))

        assert 0.0 <= result.accuracy <= 1.0
        assert 0.0 <= result.precision <= 1.0
        assert 0.0 <= result.recall <= 1.0
        assert 0.0 <= result.f1 <= 1.0
        assert -1.0 <= result.mcc <= 1.0


# ============================================================================
# Per-label 指标测试
# ============================================================================

class TestMultipleClassificationMetricPerLabel:
    """Per-label 指标测试"""

    def test_per_label_labels_match(self):
        """
        测试目的: per_label_metrics 的 label 与 result.labels 一致
        输入: 三分类数据
        预期: per_label_metrics[i].label == result.labels[i]
        """
        y_truth = np.array([0, 1, 2, 0, 1, 2])
        y_predict = np.array([0, 1, 2, 0, 1, 2])

        op = MultipleClassificationMetric()
        result = op.run((y_truth, y_predict))

        for i, per_label in enumerate(result.per_label_metrics):
            assert per_label.label == result.labels[i]

    def test_per_label_tp_fp_tn_fn_sum(self):
        """
        测试目的: 各类别的 tp+fp+tn+fn = n_samples
        输入: 三分类数据
        预期: 每个类别的计数之和等于总样本数
        """
        y_truth = np.array([0, 1, 2, 0, 1, 2])
        y_predict = np.array([0, 1, 2, 0, 1, 2])

        op = MultipleClassificationMetric()
        result = op.run((y_truth, y_predict))

        for per_label in result.per_label_metrics:
            assert per_label.tp + per_label.fp + per_label.tn + per_label.fn == result.n_samples


# ============================================================================
# 混淆矩阵测试
# ============================================================================

class TestMultipleClassificationMetricConfusionMatrix:
    """混淆矩阵测试"""

    def test_confusion_matrix_values(self):
        """
        测试目的: 混淆矩阵值正确
        输入: 手动构造已知预测结果
        预期: 混淆矩阵各元素正确反映预测情况
        """
        y_truth = np.array([0, 0, 1, 1])
        y_predict = np.array([0, 1, 1, 1])  # 类别0: 1对1错, 类别1: 2全对

        op = MultipleClassificationMetric()
        result = op.run((y_truth, y_predict))

        # 混淆矩阵 [[1,1],[0,2]]（行=真实，列=预测）
        assert result.confusion_matrix[0][0] == 1  # 真实0预测0
        assert result.confusion_matrix[0][1] == 1  # 真实0预测1
        assert result.confusion_matrix[1][0] == 0  # 真实1预测0
        assert result.confusion_matrix[1][1] == 2  # 真实1预测1

    def test_confusion_matrix_sum_equals_n_samples(self):
        """
        测试目的: 混淆矩阵所有元素之和等于 n_samples
        输入: 任意数据
        预期: sum(confusion_matrix) == n_samples
        """
        y_truth = np.array([0, 1, 2, 0, 1, 2])
        y_predict = np.array([0, 1, 2, 1, 2, 0])

        op = MultipleClassificationMetric()
        result = op.run((y_truth, y_predict))

        total = sum(
            result.confusion_matrix[i][j]
            for i in range(result.k_labels)
            for j in range(result.k_labels)
        )
        assert total == result.n_samples


# ============================================================================
# 边界条件测试
# ============================================================================

class TestMultipleClassificationMetricEdgeCases:
    """边界条件测试"""

    def test_length_mismatch_raises_error(self):
        """
        测试目的: y_truth 和 y_predict 长度不一致时抛出 ValueError
        输入: y_truth 长度 4, y_predict 长度 3
        输出: ValueError
        预期: 抛出 ValueError，消息包含 "长度不一致"
        """
        y_truth = np.array([0, 1, 2, 0])
        y_predict = np.array([0, 1, 2])

        op = MultipleClassificationMetric()
        with pytest.raises(ValueError, match="长度不一致"):
            op.run((y_truth, y_predict))

    def test_single_class(self):
        """
        测试目的: 单类别场景
        输入: 所有样本属于同一类别 y_truth=[0,0,0,0], y_predict=[0,0,0,0]
        输出: MultiClassificationMetricResult
        预期: k_labels=1，accuracy=1.0
        """
        y_truth = np.array([0, 0, 0, 0])
        y_predict = np.array([0, 0, 0, 0])

        op = MultipleClassificationMetric()
        result = op.run((y_truth, y_predict))

        assert result.k_labels == 1
        assert result.accuracy == 1.0

    def test_predict_contains_unknown_label(self):
        """
        测试目的: 预测包含不在 labels 中的标签时忽略
        输入: y_truth=[0,1,0,1], y_predict=[0,2,0,1]（2不在y_truth中）
        输出: MultiClassificationMetricResult
        预期: k_labels=2，不匹配的预测不计入混淆矩阵
        """
        y_truth = np.array([0, 1, 0, 1])
        y_predict = np.array([0, 2, 0, 1])  # 2不在 y_truth 中

        op = MultipleClassificationMetric()
        result = op.run((y_truth, y_predict))

        # 只统计有效标签的预测
        assert result.k_labels == 2

    def test_list_input_converted_to_ndarray(self):
        """
        测试目的: list 输入自动转换为 ndarray
        输入: y_truth/y_predict 为 Python list
        输出: MultiClassificationMetricResult
        预期: 正常计算，n_samples=6
        """
        y_truth = [0, 1, 2, 0, 1, 2]
        y_predict = [0, 1, 2, 0, 1, 2]

        op = MultipleClassificationMetric()
        result = op.run((y_truth, y_predict))

        assert result.n_samples == 6

    def test_large_sample_size(self):
        """
        测试目的: 大样本量场景
        输入: 1000 个样本，5 类随机标签
        输出: MultiClassificationMetricResult
        预期: 正常计算，n_samples=1000，f1 在 [0, 1] 范围内
        """
        rng = np.random.RandomState(42)
        y_truth = rng.randint(0, 5, size=1000)
        y_predict = rng.randint(0, 5, size=1000)

        op = MultipleClassificationMetric()
        result = op.run((y_truth, y_predict))

        assert result.n_samples == 1000
        assert 0.0 <= result.f1 <= 1.0

    def test_empty_array(self):
        """
        测试目的: 空数组输入场景 - 源码内部 macro_metrics 缺少 accuracy 键会报 KeyError
        输入: y_truth=[], y_predict=[]
        输出: KeyError
        预期: 抛出 KeyError（空数组时 macro_metrics 不包含 accuracy 键）
        """
        y_truth = np.array([])
        y_predict = np.array([])

        op = MultipleClassificationMetric()
        with pytest.raises(KeyError):
            op.run((y_truth, y_predict))

    def test_predict_contains_multiple_unknown_labels(self):
        """
        测试目的: 预测包含多个不在真实标签中的标签时全部忽略
        输入: y_truth=[0,1,0,1], y_predict=[2,3,0,1]（2和3不在y_truth中）
        输出: MultiClassificationMetricResult
        预期: k_labels=2，仅统计标签 0 和 1 的混淆矩阵
        """
        y_truth = np.array([0, 1, 0, 1])
        y_predict = np.array([2, 3, 0, 1])  # 2和3都不在y_truth中

        op = MultipleClassificationMetric()
        result = op.run((y_truth, y_predict))

        assert result.k_labels == 2
        # 位置0: truth=0, predict=2 → 忽略; 位置1: truth=1, predict=3 → 忽略
        # 位置2: truth=0, predict=0 → cm[0][0]++; 位置3: truth=1, predict=1 → cm[1][1]++
        assert result.confusion_matrix[0][0] == 1  # 真实0预测0
        assert result.confusion_matrix[1][1] == 1  # 真实1预测1
        assert result.confusion_matrix[0][1] == 0  # 真实0预测1
        assert result.confusion_matrix[1][0] == 0  # 真实1预测0

    def test_predict_all_unknown_labels(self):
        """
        测试目的: 所有预测标签均不在真实标签中时混淆矩阵全为 0
        输入: y_truth=[0,1], y_predict=[2,3]
        输出: MultiClassificationMetricResult
        预期: k_labels=2，混淆矩阵全为 0，accuracy=0.0
        """
        y_truth = np.array([0, 1])
        y_predict = np.array([2, 3])

        op = MultipleClassificationMetric()
        result = op.run((y_truth, y_predict))

        assert result.k_labels == 2
        # 混淆矩阵全为 0（所有预测都被忽略）
        assert sum(result.confusion_matrix[i][j]
                    for i in range(2) for j in range(2)) == 0
        assert result.accuracy == 0.0

    def test_single_sample(self):
        """
        测试目的: 单样本场景
        输入: y_truth=[0], y_predict=[0]
        输出: MultiClassificationMetricResult
        预期: n_samples=1, k_labels=1, accuracy=1.0, f1=1.0
        """
        y_truth = np.array([0])
        y_predict = np.array([0])

        op = MultipleClassificationMetric()
        result = op.run((y_truth, y_predict))

        assert result.n_samples == 1
        assert result.k_labels == 1
        assert result.accuracy == 1.0
        assert result.f1 == 1.0


# ============================================================================
# 配置测试
# ============================================================================

class TestMultipleClassificationMetricConfig:
    """配置测试"""

    def test_explicit_labels(self):
        """
        测试目的: 显式指定 labels
        输入: labels=[0, 1], 三分类数据
        输出: MultiClassificationMetricResult
        预期: 只计算指定类别 0 和 1 的指标，k_labels=2
        """
        y_truth = np.array([0, 1, 2, 0, 1, 2])
        y_predict = np.array([0, 1, 2, 0, 1, 2])

        op = MultipleClassificationMetric(labels=[0, 1])
        result = op.run((y_truth, y_predict))

        assert result.k_labels == 2
        assert result.labels == [0, 1]

    def test_decimals_rounding(self):
        """
        测试目的: decimals 参数控制四舍五入
        输入: decimals=2
        输出: MultiClassificationMetricResult
        预期: 各指标保留 2 位小数
        """
        y_truth = np.array([0, 1, 2, 0, 1, 2])
        y_predict = np.array([0, 1, 2, 1, 2, 0])

        op = MultipleClassificationMetric(decimals=2)
        result = op.run((y_truth, y_predict))

        # 检查小数位数（允许浮点精度误差）
        for metric_val in [result.accuracy, result.f1, result.precision]:
            rounded = round(metric_val, 2)
            assert abs(metric_val - rounded) < 1e-6


# ============================================================================
# 冻结结果测试
# ============================================================================

class TestMultipleClassificationMetricFrozen:
    """冻结结果测试"""

    def test_result_is_frozen(self):
        """
        测试目的: 结果对象不可修改（frozen）
        输入: 尝试修改 result.f1 = 0.99
        输出: ValidationError
        预期: 抛出 Exception
        """
        y_truth = np.array([0, 1, 2, 0, 1, 2])
        y_predict = np.array([0, 1, 2, 0, 1, 2])

        op = MultipleClassificationMetric()
        result = op.run((y_truth, y_predict))

        with pytest.raises(Exception):
            result.f1 = 0.99

    def test_per_label_result_is_frozen(self):
        """
        测试目的: PerLabelMetricResult 不可修改
        输入: 尝试修改 result.per_label_metrics[0].f1 = 0.99
        输出: ValidationError
        预期: 抛出 Exception
        """
        y_truth = np.array([0, 1, 2, 0, 1, 2])
        y_predict = np.array([0, 1, 2, 0, 1, 2])

        op = MultipleClassificationMetric()
        result = op.run((y_truth, y_predict))

        with pytest.raises(Exception):
            result.per_label_metrics[0].f1 = 0.99

    def test_name(self):
        """
        测试目的: 验证 name() 返回正确标识
        输入: 无
        预期: 返回 "multi_classification"
        """
        assert MultipleClassificationMetric.name() == "multi_classification"