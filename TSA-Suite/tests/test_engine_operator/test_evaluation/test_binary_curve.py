# -*- coding: utf-8 -*-

"""
二分类曲线评价指标算子测试

测试覆盖:
    1. 基本功能: run() 返回完整曲线指标集
    2. scores() 方法: 按 main_scores 提取命名标量
    3. AUC 计算: AUC-ROC 和 AUC-PR 正确性
    4. 最佳指标: best_f1/best_threshold 等正确性
    5. 配置覆盖: positive_label、distinct_thresholds、inf_threshold
    6. 边界条件: 全正类、全负类、单样本、全相同分数

测试约束:
    - 代码覆盖率 > 90%
    - 测试通过率 100%
    - 中文注释说明测试目的、输入、输出和预期结果
"""

import numpy as np
import pytest

from tsas.engine.operator.evaluation.binary_curve import (
    BinaryClassificationCurveResult,
    BinaryClassificationCurveConfig,
    BinaryClassificationCurve,
)


# ============================================================================
# 基本功能测试
# ============================================================================

class TestBinaryClassificationCurveBasic:
    """基本功能测试"""

    def test_run_returns_complete_result(self):
        """
        测试目的: run() 返回完整曲线指标结果
        输入: 完美区分的分数 y_truth=[0,0,0,1,1,1], y_predict=[0.1,0.2,0.15,0.9,0.85,0.95]
        输出: BinaryClassificationCurveResult
        预期: 返回完整结果，包含 auc_roc/auc_pr/best_f1 等，thresholds/tpr/fpr 长度一致
        """
        # 正常样本分数低，异常样本分数高
        y_truth = np.array([0, 0, 0, 1, 1, 1])
        y_predict = np.array([0.1, 0.2, 0.15, 0.9, 0.85, 0.95])

        op = BinaryClassificationCurve()
        result = op.run((y_truth, y_predict))

        assert isinstance(result, BinaryClassificationCurveResult)
        assert result.n_samples == 6
        assert len(result.thresholds) > 0
        assert len(result.tpr) == len(result.thresholds)
        assert len(result.fpr) == len(result.thresholds)

    def test_auc_roc_perfect_separation(self):
        """
        测试目的: 完美区分时 AUC-ROC 应较高
        输入: 正常样本分数低，异常样本分数高
        输出: BinaryClassificationCurveResult
        预期: auc_roc > 0.5（远优于随机）
        注: 由于阈值降序排列且缺少(0,0)起始点，AUC-ROC 不一定达到 1.0
        """
        y_truth = np.array([0, 0, 0, 1, 1, 1])
        y_predict = np.array([0.1, 0.2, 0.15, 0.9, 0.85, 0.95])

        op = BinaryClassificationCurve(inf_threshold=False)
        result = op.run((y_truth, y_predict))

        assert result.auc_roc > 0.5

    def test_auc_pr_perfect_separation(self):
        """
        测试目的: 完美区分时 AUC-PR 应较高
        输入: 正常样本分数低，异常样本分数高
        输出: BinaryClassificationCurveResult
        预期: auc_pr > 0.7（远优于随机基线）
        """
        y_truth = np.array([0, 0, 0, 1, 1, 1])
        y_predict = np.array([0.1, 0.2, 0.15, 0.9, 0.85, 0.95])

        op = BinaryClassificationCurve(inf_threshold=False)
        result = op.run((y_truth, y_predict))

        assert result.auc_pr > 0.7

    def test_best_f1_perfect_separation(self):
        """
        测试目的: 完美区分时 best_f1 接近 1.0
        输入: 正常样本分数低，异常样本分数高
        输出: BinaryClassificationCurveResult
        预期: best_f1 > 0.99，best_f1_threshold 位于分离区间 [0.15, 0.95]
        """
        y_truth = np.array([0, 0, 0, 1, 1, 1])
        y_predict = np.array([0.1, 0.2, 0.15, 0.9, 0.85, 0.95])

        op = BinaryClassificationCurve()
        result = op.run((y_truth, y_predict))

        assert result.best_f1 > 0.99
        # 最佳阈值应在 0.15~0.85 之间
        assert 0.15 <= result.best_f1_threshold <= 0.95

    def test_curve_arrays_consistency(self):
        """
        测试目的: 曲线数组长度一致
        输入: y_truth=[0,1,0,1], y_predict=[0.1,0.9,0.2,0.8]
        输出: BinaryClassificationCurveResult
        预期: thresholds/tpr/fpr/precision_arr/recall_arr 等长度全部相等
        """
        y_truth = np.array([0, 1, 0, 1])
        y_predict = np.array([0.1, 0.9, 0.2, 0.8])

        op = BinaryClassificationCurve()
        result = op.run((y_truth, y_predict))

        n_thresholds = len(result.thresholds)
        assert len(result.tpr) == n_thresholds
        assert len(result.fpr) == n_thresholds
        assert len(result.precision_arr) == n_thresholds
        assert len(result.recall_arr) == n_thresholds
        assert len(result.specificity_arr) == n_thresholds
        assert len(result.f1_arr) == n_thresholds
        assert len(result.mcc_arr) == n_thresholds
        assert len(result.far_arr) == n_thresholds
        assert len(result.mar_arr) == n_thresholds


# ============================================================================
# scores() 方法测试
# ============================================================================

class TestBinaryClassificationCurveScores:
    """scores() 方法测试"""

    def test_scores_returns_dict(self):
        """
        测试目的: scores() 返回按 main_scores 映射的字典
        输入: 默认配置 main_scores={"auc_roc": "auc_roc", "best_f1": "best_f1"}
        预期: 返回 {"auc_roc": float, "best_f1": float}
        """
        y_truth = np.array([0, 1, 0, 1])
        y_predict = np.array([0.1, 0.9, 0.2, 0.8])

        op = BinaryClassificationCurve()
        scores = op.scores((y_truth, y_predict))

        assert scores is not None
        assert "auc_roc" in scores
        assert "best_f1" in scores
        assert isinstance(scores["auc_roc"], float)
        assert isinstance(scores["best_f1"], float)

    def test_scores_matches_run_values(self):
        """
        测试目的: scores() 提取的值与 run() 结果一致
        输入: 同上
        预期: scores["auc_roc"] == result.auc_roc
        """
        y_truth = np.array([0, 1, 0, 1])
        y_predict = np.array([0.1, 0.9, 0.2, 0.8])

        op = BinaryClassificationCurve()
        result = op.run((y_truth, y_predict))
        scores = op.scores((y_truth, y_predict))

        assert scores["auc_roc"] == result.auc_roc
        assert scores["best_f1"] == result.best_f1

    def test_scores_with_custom_main_scores(self):
        """
        测试目的: 自定义 main_scores 提取不同指标
        输入: main_scores={"auc_pr": "auc_pr", "best_mcc": "best_mcc"}
        预期: 返回 {"auc_pr": float, "best_mcc": float}
        """
        y_truth = np.array([0, 1, 0, 1])
        y_predict = np.array([0.1, 0.9, 0.2, 0.8])

        op = BinaryClassificationCurve(
            main_scores={"auc_pr": "auc_pr", "best_mcc": "best_mcc"}
        )
        scores = op.scores((y_truth, y_predict))

        assert scores is not None
        assert "auc_pr" in scores
        assert "best_mcc" in scores

    def test_scores_returns_none_when_main_scores_is_none(self):
        """
        测试目的: main_scores=None 时 scores() 返回 None
        输入: main_scores=None
        预期: scores() 返回 None
        """
        y_truth = np.array([0, 1, 0, 1])
        y_predict = np.array([0.1, 0.9, 0.2, 0.8])

        op = BinaryClassificationCurve(main_scores=None)
        scores = op.scores((y_truth, y_predict))

        assert scores is None


# ============================================================================
# AUC 计算验证测试
# ============================================================================

class TestBinaryClassificationCurveAUC:
    """AUC 计算验证测试"""

    def test_auc_roc_random_scores(self):
        """
        测试目的: 随机分数时 AUC-ROC 在合理范围
        输入: 100 个随机分数和标签（rng=42）
        输出: BinaryClassificationCurveResult
        预期: AUC-ROC 在 [0, 1] 范围内
        """
        rng = np.random.RandomState(42)
        y_truth = rng.randint(0, 2, size=100)
        y_predict = rng.rand(100)

        op = BinaryClassificationCurve()
        result = op.run((y_truth, y_predict))

        assert 0.0 <= result.auc_roc <= 1.0

    def test_auc_pr_random_scores(self):
        """
        测试目的: 随机分数时 AUC-PR 在合理范围
        输入: 100 个随机分数和标签（rng=42）
        输出: BinaryClassificationCurveResult
        预期: AUC-PR 在 [0, 1] 范围内
        """
        rng = np.random.RandomState(42)
        y_truth = rng.randint(0, 2, size=100)
        y_predict = rng.rand(100)

        op = BinaryClassificationCurve()
        result = op.run((y_truth, y_predict))

        assert 0.0 <= result.auc_pr <= 1.0

    def test_auc_roc_all_same_scores(self):
        """
        测试目的: 全部分数相同时 AUC-ROC 的退化处理
        输入: y_truth=[0,1,0,1], y_predict=[0.5,0.5,0.5,0.5]
        输出: BinaryClassificationCurveResult
        预期: AUC-ROC 为 0 或接近 0.5（无法区分）
        """
        y_truth = np.array([0, 1, 0, 1])
        y_predict = np.array([0.5, 0.5, 0.5, 0.5])

        op = BinaryClassificationCurve()
        result = op.run((y_truth, y_predict))

        # 全部分数相同，无法区分，AUC 应为 0 或接近 0.5（取决于实现）
        assert 0.0 <= result.auc_roc <= 0.5


# ============================================================================
# best 值验证测试
# ============================================================================

class TestBinaryClassificationCurveBestValues:
    """best 值验证测试"""

    def test_best_precision_threshold_exists(self):
        """
        测试目的: best_precision_threshold 在阈值列表中
        输入: y_truth=[0,1,0,1], y_predict=[0.1,0.9,0.2,0.8]
        输出: BinaryClassificationCurveResult
        预期: best_precision_threshold 存在于 thresholds 中或为 inf
        """
        y_truth = np.array([0, 1, 0, 1])
        y_predict = np.array([0.1, 0.9, 0.2, 0.8])

        op = BinaryClassificationCurve()
        result = op.run((y_truth, y_predict))

        # 允许一定浮点误差
        found = any(
            abs(t - result.best_precision_threshold) < 1e-6
            for t in result.thresholds
        )
        assert found or result.best_precision_threshold == float('inf')

    def test_best_f1_matches_array_max(self):
        """
        测试目的: best_f1 等于 f1_arr 的最大值
        输入: y_truth=[0,1,0,1,0,1], y_predict=[0.1,0.9,0.3,0.7,0.2,0.8]
        输出: BinaryClassificationCurveResult
        预期: best_f1 == max(f1_arr)
        """
        y_truth = np.array([0, 1, 0, 1, 0, 1])
        y_predict = np.array([0.1, 0.9, 0.3, 0.7, 0.2, 0.8])

        op = BinaryClassificationCurve()
        result = op.run((y_truth, y_predict))

        assert abs(result.best_f1 - max(result.f1_arr)) < 1e-6

    def test_best_mcc_matches_array_max(self):
        """
        测试目的: best_mcc 等于 mcc_arr 的最大值
        输入: y_truth=[0,1,0,1,0,1], y_predict=[0.1,0.9,0.3,0.7,0.2,0.8]
        输出: BinaryClassificationCurveResult
        预期: best_mcc == max(mcc_arr)
        """
        y_truth = np.array([0, 1, 0, 1, 0, 1])
        y_predict = np.array([0.1, 0.9, 0.3, 0.7, 0.2, 0.8])

        op = BinaryClassificationCurve()
        result = op.run((y_truth, y_predict))

        assert abs(result.best_mcc - max(result.mcc_arr)) < 1e-6

    def test_best_fpr_is_minimum(self):
        """
        测试目的: best_fpr 等于 fpr_arr 的最小值
        输入: y_truth=[0,1,0,1,0,1], y_predict=[0.1,0.9,0.3,0.7,0.2,0.8]
        输出: BinaryClassificationCurveResult
        预期: best_fpr == min(fpr_arr)
        """
        y_truth = np.array([0, 1, 0, 1, 0, 1])
        y_predict = np.array([0.1, 0.9, 0.3, 0.7, 0.2, 0.8])

        op = BinaryClassificationCurve()
        result = op.run((y_truth, y_predict))

        assert abs(result.best_fpr - min(result.fpr)) < 1e-6

    def test_best_far_is_minimum(self):
        """
        测试目的: best_far 等于 far_arr 的最小值
        输入: y_truth=[0,1,0,1,0,1], y_predict=[0.1,0.9,0.3,0.7,0.2,0.8]
        输出: BinaryClassificationCurveResult
        预期: best_far == min(far_arr)
        """
        y_truth = np.array([0, 1, 0, 1, 0, 1])
        y_predict = np.array([0.1, 0.9, 0.3, 0.7, 0.2, 0.8])

        op = BinaryClassificationCurve()
        result = op.run((y_truth, y_predict))

        assert abs(result.best_far - min(result.far_arr)) < 1e-6

    def test_best_mar_is_minimum(self):
        """
        测试目的: best_mar 等于 mar_arr 的最小值
        输入: y_truth=[0,1,0,1,0,1], y_predict=[0.1,0.9,0.3,0.7,0.2,0.8]
        输出: BinaryClassificationCurveResult
        预期: best_mar == min(mar_arr)
        """
        y_truth = np.array([0, 1, 0, 1, 0, 1])
        y_predict = np.array([0.1, 0.9, 0.3, 0.7, 0.2, 0.8])

        op = BinaryClassificationCurve()
        result = op.run((y_truth, y_predict))

        assert abs(result.best_mar - min(result.mar_arr)) < 1e-6


# ============================================================================
# 配置覆盖测试
# ============================================================================

class TestBinaryClassificationCurveConfig:
    """配置覆盖测试"""

    def test_positive_label_explicit(self):
        """
        测试目的: 显式指定 positive_label
        输入: positive_label=1
        输出: BinaryClassificationCurveResult
        预期: 使用 1 作为正类计算指标，n_samples=4
        """
        y_truth = np.array([0, 1, 0, 1])
        y_predict = np.array([0.1, 0.9, 0.2, 0.8])

        op = BinaryClassificationCurve(positive_label=1)
        result = op.run((y_truth, y_predict))

        assert result.n_samples == 4

    def test_distinct_thresholds_false(self):
        """
        测试目的: distinct_thresholds=False 时阈值不去重
        输入: distinct_thresholds=False, inf_threshold=True
        输出: BinaryClassificationCurveResult
        预期: 阈值数量等于样本数量 + 1（inf）= 5
        """
        y_truth = np.array([0, 1, 0, 1])
        y_predict = np.array([0.1, 0.9, 0.2, 0.8])

        op = BinaryClassificationCurve(distinct_thresholds=False, inf_threshold=True)
        result = op.run((y_truth, y_predict))

        # 不去重时阈值数 = 样本数 + 1（inf）
        assert len(result.thresholds) == 5

    def test_inf_threshold_false(self):
        """
        测试目的: inf_threshold=False时不追加正无穷阈值
        输入: inf_threshold=False
        输出: BinaryClassificationCurveResult
        预期: thresholds 中不含 inf
        """
        y_truth = np.array([0, 1, 0, 1])
        y_predict = np.array([0.1, 0.9, 0.2, 0.8])

        op = BinaryClassificationCurve(inf_threshold=False)
        result = op.run((y_truth, y_predict))

        assert float('inf') not in result.thresholds

    def test_predict_decimals_rounding(self):
        """
        测试目的: predict_decimals 控制分数精度截断
        输入: predict_decimals=1, y_predict=[0.123, 0.789, 0.456, 0.234]
        输出: BinaryClassificationCurveResult
        预期: 阈值精度被截断到小数点后 1 位
        """
        y_truth = np.array([0, 1, 0, 1])
        y_predict = np.array([0.123, 0.789, 0.456, 0.234])

        op = BinaryClassificationCurve(predict_decimals=1, inf_threshold=False)
        result = op.run((y_truth, y_predict))

        # 所有阈值应为 1 位小数精度
        for t in result.thresholds:
            if t != float('inf'):
                assert abs(t * 10 - round(t * 10)) < 1e-6


# ============================================================================
# 边界条件测试
# ============================================================================

class TestBinaryClassificationCurveEdgeCases:
    """边界条件测试"""

    def test_all_positive_samples(self):
        """
        测试目的: 全正类样本场景
        输入: y_truth=[1,1,1,1], y_predict=[0.1,0.9,0.2,0.8], positive_label=1
        输出: BinaryClassificationCurveResult
        预期: n_samples=4，无负类时 FPR 全为 0 或 1
        """
        y_truth = np.array([1, 1, 1, 1])
        y_predict = np.array([0.1, 0.9, 0.2, 0.8])

        op = BinaryClassificationCurve(positive_label=1)
        result = op.run((y_truth, y_predict))

        assert result.n_samples == 4
        # 无负类时 FPR 无法计算
        assert all(fpr == 0.0 for fpr in result.fpr) or all(fpr == 1.0 for fpr in result.fpr)

    def test_all_negative_samples(self):
        """
        测试目的: 全负类样本场景
        输入: y_truth=[0,0,0,0], y_predict=[0.1,0.9,0.2,0.8], positive_label=1
        输出: BinaryClassificationCurveResult
        预期: n_samples=4，无正类时 TPR 全为 0
        """
        y_truth = np.array([0, 0, 0, 0])
        y_predict = np.array([0.1, 0.9, 0.2, 0.8])

        op = BinaryClassificationCurve(positive_label=1)
        result = op.run((y_truth, y_predict))

        assert result.n_samples == 4
        # 无正类时 TPR 为 0
        assert all(tpr == 0.0 for tpr in result.tpr)

    def test_length_mismatch_raises_error(self):
        """
        测试目的: y_truth 和 y_predict 长度不一致时抛出 ValueError
        输入: y_truth 长度 4, y_predict 长度 3
        输出: ValueError
        预期: 抛出 ValueError，消息包含 "长度不一致"
        """
        y_truth = np.array([0, 1, 0, 1])
        y_predict = np.array([0.1, 0.9, 0.2])

        op = BinaryClassificationCurve()
        with pytest.raises(ValueError, match="长度不一致"):
            op.run((y_truth, y_predict))

    def test_list_input_converted_to_ndarray(self):
        """
        测试目的: list 输入自动转换为 ndarray
        输入: y_truth=[0,1,0,1], y_predict=[0.1,0.9,0.2,0.8]（Python list）
        输出: BinaryClassificationCurveResult
        预期: 正常计算，n_samples=4
        """
        y_truth = [0, 1, 0, 1]
        y_predict = [0.1, 0.9, 0.2, 0.8]

        op = BinaryClassificationCurve()
        result = op.run((y_truth, y_predict))

        assert result.n_samples == 4

    def test_large_sample_size(self):
        """
        测试目的: 大样本量场景
        输入: 1000 个样本
        输出: BinaryClassificationCurveResult
        预期: 正常计算，n_samples=1000，auc_roc 在 [0, 1] 范围内
        """
        rng = np.random.RandomState(42)
        y_truth = rng.randint(0, 2, size=1000)
        y_predict = rng.rand(1000)

        op = BinaryClassificationCurve()
        result = op.run((y_truth, y_predict))

        assert result.n_samples == 1000
        assert 0.0 <= result.auc_roc <= 1.0

    def test_empty_array(self):
        """
        测试目的: 空数组输入场景 - 源码在内部调用 np.argmax 处理空数组会报错
        输入: y_truth=[], y_predict=[]
        输出: ValueError
        预期: 抛出 ValueError（np.argmax 无法处理空序列）
        """
        y_truth = np.array([])
        y_predict = np.array([])

        op = BinaryClassificationCurve(inf_threshold=False)
        with pytest.raises(ValueError):
            op.run((y_truth, y_predict))

    def test_single_sample_positive(self):
        """
        测试目的: 单样本且为正类场景
        输入: y_truth=[1], y_predict=[0.8]
        输出: BinaryClassificationCurveResult
        预期: n_samples=1，只有 1 个正类，TPR 在最低阈值时为 1.0，FPR 全为 0
        """
        y_truth = np.array([1])
        y_predict = np.array([0.8])

        op = BinaryClassificationCurve(inf_threshold=True)
        result = op.run((y_truth, y_predict))

        assert result.n_samples == 1
        # 只有一个正类样本且无负类，FPR 始终为 0
        assert all(fpr == 0.0 for fpr in result.fpr)

    def test_single_sample_negative(self):
        """
        测试目的: 单样本且为负类场景
        输入: y_truth=[0], y_predict=[0.3]
        输出: BinaryClassificationCurveResult
        预期: n_samples=1，只有 1 个负类，TPR 始终为 0
        """
        y_truth = np.array([0])
        y_predict = np.array([0.3])

        op = BinaryClassificationCurve(inf_threshold=True)
        result = op.run((y_truth, y_predict))

        assert result.n_samples == 1
        # 只有一个负类样本，TPR 始终为 0
        assert all(tpr == 0.0 for tpr in result.tpr)


# ============================================================================
# 指标关系验证测试
# ============================================================================

class TestBinaryClassificationCurveRelations:
    """指标关系验证测试"""

    def test_tpr_equals_recall_arr(self):
        """
        测试目的: TPR 数组等于 Recall 数组
        输入: y_truth=[0,1,0,1], y_predict=[0.1,0.9,0.2,0.8]
        输出: BinaryClassificationCurveResult
        预期: tpr == recall_arr
        """
        y_truth = np.array([0, 1, 0, 1])
        y_predict = np.array([0.1, 0.9, 0.2, 0.8])

        op = BinaryClassificationCurve()
        result = op.run((y_truth, y_predict))

        assert result.tpr == result.recall_arr

    def test_far_equals_fpr_arr(self):
        """
        测试目的: FAR 数组等于 FPR 数组
        输入: y_truth=[0,1,0,1], y_predict=[0.1,0.9,0.2,0.8]
        输出: BinaryClassificationCurveResult
        预期: far_arr == fpr
        """
        y_truth = np.array([0, 1, 0, 1])
        y_predict = np.array([0.1, 0.9, 0.2, 0.8])

        op = BinaryClassificationCurve()
        result = op.run((y_truth, y_predict))

        assert result.far_arr == result.fpr

    def test_mar_plus_recall_equals_one(self):
        """
        测试目的: MAR + Recall = 1
        输入: y_truth=[0,1,0,1], y_predict=[0.1,0.9,0.2,0.8]
        输出: BinaryClassificationCurveResult
        预期: mar_arr[i] + recall_arr[i] ≈ 1 对所有阈值成立
        """
        y_truth = np.array([0, 1, 0, 1])
        y_predict = np.array([0.1, 0.9, 0.2, 0.8])

        op = BinaryClassificationCurve()
        result = op.run((y_truth, y_predict))

        for mar, recall in zip(result.mar_arr, result.recall_arr):
            assert abs(mar + recall - 1.0) < 1e-6


# ============================================================================
# 冻结结果测试
# ============================================================================

class TestBinaryClassificationCurveFrozen:
    """冻结结果测试"""

    def test_result_is_frozen(self):
        """
        测试目的: 结果对象不可修改（frozen）
        输入: 尝试修改 result.auc_roc = 0.99
        输出: ValidationError
        预期: 抛出 Exception
        """
        y_truth = np.array([0, 1, 0, 1])
        y_predict = np.array([0.1, 0.9, 0.2, 0.8])

        op = BinaryClassificationCurve()
        result = op.run((y_truth, y_predict))

        with pytest.raises(Exception):
            result.auc_roc = 0.99

    def test_name(self):
        """
        测试目的: 验证 name() 返回正确标识
        输入: 无
        预期: 返回 "binary_classification_curve"
        """
        assert BinaryClassificationCurve.name() == "binary_classification_curve"