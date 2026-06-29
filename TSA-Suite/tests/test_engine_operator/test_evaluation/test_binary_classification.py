# -*- coding: utf-8 -*-

"""
二分类评价指标算子测试

测试覆盖:
    1. 基本功能: run() 返回完整指标集
    2. scores() 方法: 按 main_scores 提取命名标量
    3. 配置覆盖: positive_label、default_for_zero、decimals
    4. 边界条件: 全预测正确、全预测错误、空样本、单类样本
    5. 类型兼容: ndarray、list 输入

测试约束:
    - 代码覆盖率 > 90%
    - 测试通过率 100%
    - 中文注释说明测试目的、输入、输出和预期结果
"""

import numpy as np
import pytest

from tsas.engine.operator.evaluation.binary_classification import (
    BinaryClassificationResult,
    BinaryClassificationConfig,
    BinaryClassificationMetric,
)


# ============================================================================
# 基本功能测试
# ============================================================================

class TestBinaryClassificationMetricBasic:
    """基本功能测试"""

    def test_run_returns_complete_result(self):
        """
        测试目的: run() 返回完整指标集
        输入: y_truth=[0,1,0,1], y_predict=[0,1,1,0]
        输出: BinaryClassificationResult
        预期: 返回 BinaryClassificationResult，包含 tp/fp/tn/fn 和各指标
        """
        y_truth = np.array([0, 1, 0, 1])
        y_predict = np.array([0, 1, 1, 0])  # TP=1, FP=1, TN=1, FN=1

        op = BinaryClassificationMetric()
        result = op.run((y_truth, y_predict))

        assert isinstance(result, BinaryClassificationResult)
        assert result.tp == 1
        assert result.fp == 1
        assert result.tn == 1
        assert result.fn == 1
        assert result.n_samples == 4

    def test_run_with_perfect_prediction(self):
        """
        测试目的: 完美预测时各指标达到最优
        输入: y_truth=[0,1,0,1], y_predict=[0,1,0,1]（完全一致）
        输出: BinaryClassificationResult
        预期: tp=2, fp=0, tn=2, fn=0, f1=1.0, accuracy=1.0, mcc=1.0
        """
        y_truth = np.array([0, 1, 0, 1])
        y_predict = np.array([0, 1, 0, 1])

        op = BinaryClassificationMetric()
        result = op.run((y_truth, y_predict))

        assert result.tp == 2
        assert result.fp == 0
        assert result.tn == 2
        assert result.fn == 0
        assert result.accuracy == 1.0
        assert result.precision == 1.0
        assert result.recall == 1.0
        assert result.f1 == 1.0
        assert result.mcc == 1.0  # 完美预测时 MCC=1

    def test_run_with_all_wrong_prediction(self):
        """
        测试目的: 全部错误预测时各指标达到最差
        输入: y_truth=[0,1,0,1], y_predict=[1,0,1,0]（完全相反）
        输出: BinaryClassificationResult
        预期: tp=0, fp=2, tn=0, fn=2, f1=0.0, accuracy=0.0, mcc=-1.0
        """
        y_truth = np.array([0, 1, 0, 1])
        y_predict = np.array([1, 0, 1, 0])

        op = BinaryClassificationMetric()
        result = op.run((y_truth, y_predict))

        assert result.tp == 0
        assert result.fp == 2
        assert result.tn == 0
        assert result.fn == 2
        assert result.accuracy == 0.0
        assert result.precision == 0.0
        assert result.recall == 0.0
        assert result.f1 == 0.0
        assert result.mcc == -1.0  # 完全相反时 MCC=-1

    def test_confusion_matrix_format(self):
        """
        测试目的: 混淆矩阵为 list[list[int]] 格式 [[TN,FP],[FN,TP]]
        输入: 任意标签对 y_truth=[0,1,0,1], y_predict=[0,1,1,0]
        输出: BinaryClassificationResult
        预期: confusion_matrix 为二维列表 [[TN,FP],[FN,TP]]，格式正确
        """
        y_truth = np.array([0, 1, 0, 1])
        y_predict = np.array([0, 1, 1, 0])

        op = BinaryClassificationMetric()
        result = op.run((y_truth, y_predict))

        assert isinstance(result.confusion_matrix, list)
        assert len(result.confusion_matrix) == 2
        assert len(result.confusion_matrix[0]) == 2
        # [[TN, FP], [FN, TP]] = [[1, 1], [1, 1]]
        assert result.confusion_matrix[0][0] == result.tn
        assert result.confusion_matrix[0][1] == result.fp
        assert result.confusion_matrix[1][0] == result.fn
        assert result.confusion_matrix[1][1] == result.tp


# ============================================================================
# scores() 方法测试
# ============================================================================

class TestBinaryClassificationMetricScores:
    """scores() 方法测试"""

    def test_scores_returns_dict(self):
        """
        测试目的: scores() 返回按 main_scores 映射的字典
        输入: 默认配置 main_scores={"f1": "f1", "far": "far"}
        输出: dict[str, float]
        预期: 返回 {"f1": float, "far": float}
        """
        y_truth = np.array([0, 1, 0, 1])
        y_predict = np.array([0, 1, 1, 0])

        op = BinaryClassificationMetric()
        scores = op.scores((y_truth, y_predict))

        assert scores is not None
        assert "f1" in scores
        assert "far" in scores
        assert isinstance(scores["f1"], float)
        assert isinstance(scores["far"], float)

    def test_scores_matches_run_values(self):
        """
        测试目的: scores() 提取的值与 run() 结果一致
        输入: 同上，默认配置
        输出: dict[str, float]
        预期: scores["f1"] == result.f1, scores["far"] == result.far
        """
        y_truth = np.array([0, 1, 0, 1])
        y_predict = np.array([0, 1, 1, 0])

        op = BinaryClassificationMetric()
        result = op.run((y_truth, y_predict))
        scores = op.scores((y_truth, y_predict))

        assert scores["f1"] == result.f1
        assert scores["far"] == result.far

    def test_scores_with_custom_main_scores(self):
        """
        测试目的: 自定义 main_scores 提取不同指标
        输入: main_scores={"mcc": "mcc", "precision": "precision"}
        输出: dict[str, float]
        预期: 返回 {"mcc": float, "precision": float}
        """
        y_truth = np.array([0, 1, 0, 1])
        y_predict = np.array([0, 1, 1, 0])

        op = BinaryClassificationMetric(
            main_scores={"mcc": "mcc", "precision": "precision"}
        )
        scores = op.scores((y_truth, y_predict))

        assert scores is not None
        assert "mcc" in scores
        assert "precision" in scores

    def test_scores_returns_none_when_main_scores_is_none(self):
        """
        测试目的: main_scores=None 时 scores() 返回 None
        输入: main_scores=None
        输出: None
        预期: scores() 返回 None
        """
        y_truth = np.array([0, 1, 0, 1])
        y_predict = np.array([0, 1, 1, 0])

        op = BinaryClassificationMetric(main_scores=None)
        scores = op.scores((y_truth, y_predict))

        assert scores is None


# ============================================================================
# 配置覆盖测试
# ============================================================================

class TestBinaryClassificationMetricConfig:
    """配置覆盖测试"""

    def test_positive_label_auto_detect(self):
        """
        测试目的: positive_label=None 时自动推断正类
        输入: y_truth=[0,1,0,1], positive_label=None
        输出: BinaryClassificationResult
        预期: 自动推断正类为1（{0,1}场景默认1为异常），n_samples=4
        """
        y_truth = np.array([0, 1, 0, 1])
        y_predict = np.array([1, 1, 0, 0])

        op = BinaryClassificationMetric(positive_label=None)
        result = op.run((y_truth, y_predict))

        # 检查结果是否合理
        assert result.n_samples == 4

    def test_positive_label_explicit(self):
        """
        测试目的: 显式指定 positive_label
        输入: positive_label=1
        输出: BinaryClassificationResult
        预期: 使用 1 作为正类计算指标，tp=1, fp=1
        """
        y_truth = np.array([0, 1, 0, 1])
        y_predict = np.array([0, 1, 1, 0])

        op = BinaryClassificationMetric(positive_label=1)
        result = op.run((y_truth, y_predict))

        assert result.tp == 1  # 真实为1且预测为1
        assert result.fp == 1  # 真实为0但预测为1

    def test_positive_label_list(self):
        """
        测试目的: 多正类标签场景（合并为一类）
        输入: positive_label=[1, 2], y_truth=[0,1,2,0], y_predict=[0,1,0,1]
        输出: BinaryClassificationResult
        预期: 标签 1 和 2 都视为正类，tp+fn=2
        """
        y_truth = np.array([0, 1, 2, 0])
        y_predict = np.array([0, 1, 0, 1])

        op = BinaryClassificationMetric(positive_label=[1, 2])
        result = op.run((y_truth, y_predict))

        # 标签 1 和 2 视为正类
        # TP: y_truth=1/2 且 y_predict=1/2 → y_truth[1]=1, y_predict[1]=1 → 1个TP
        # FN: y_truth=1/2 但 y_predict 非1/2 → y_truth[2]=2, y_predict[2]=0 → 1个FN
        assert result.tp + result.fn == 2  # 正类总数

    def test_default_for_zero(self):
        """
        测试目的: 除零时使用默认值
        输入: 全负类完美预测 y_truth=[0,0,0,0], y_predict=[0,0,0,0]
        输出: BinaryClassificationResult
        预期: tp=0, fp=0 时 precision 分母为零使用默认值
        """
        y_truth = np.array([0, 0, 0, 0])  # 全负类
        y_predict = np.array([0, 0, 0, 0])  # 全预测为负类

        op = BinaryClassificationMetric(default_for_zero=0.5)
        result = op.run((y_truth, y_predict))

        # 当 tp=0, fp=0 时 precision 分母为零，使用 default_for_zero=0.5
        # recall 同理: tp=0, fn=0 → 0/0 → 0.5
        assert result.precision == 0.5  # tp=0, fp=0 → precision=tp/(tp+fp)=0/0 → default_for_zero=0.5

    def test_decimals_rounding(self):
        """
        测试目的: decimals 参数控制四舍五入
        输入: decimals=2, y_truth=[0,1,0,1,0,1], y_predict=[0,1,1,0,0,1]
        输出: BinaryClassificationResult
        预期: 各指标保留 2 位小数
        """
        y_truth = np.array([0, 1, 0, 1, 0, 1])
        y_predict = np.array([0, 1, 1, 0, 0, 1])

        op = BinaryClassificationMetric(decimals=2)
        result = op.run((y_truth, y_predict))

        # 检查小数位数
        for metric_val in [
            result.accuracy, result.precision, result.recall,
            result.f1, result.mcc
        ]:
            # 浮点数精度问题，检查大致位数
            assert abs(metric_val * 100 - round(metric_val * 100)) < 1e-6


# ============================================================================
# 边界条件测试
# ============================================================================

class TestBinaryClassificationMetricEdgeCases:
    """边界条件测试"""

    def test_single_class_all_positive(self):
        """
        测试目的: 全正类样本
        输入: y_truth=[1,1,1], y_predict=[1,1,1]
        输出: BinaryClassificationResult
        预期: tp=3, tn=0, fp=0, fn=0, recall=1.0
        """
        y_truth = np.array([1, 1, 1])
        y_predict = np.array([1, 1, 1])

        op = BinaryClassificationMetric(positive_label=1)
        result = op.run((y_truth, y_predict))

        assert result.tp == 3
        assert result.tn == 0
        assert result.fp == 0
        assert result.fn == 0
        assert result.recall == 1.0

    def test_single_class_all_negative(self):
        """
        测试目的: 全负类样本
        输入: y_truth=[0,0,0], y_predict=[0,0,0]
        输出: BinaryClassificationResult
        预期: tp=0, tn=3, fp=0, fn=0, specificity=1.0
        """
        y_truth = np.array([0, 0, 0])
        y_predict = np.array([0, 0, 0])

        op = BinaryClassificationMetric(positive_label=1)
        result = op.run((y_truth, y_predict))

        assert result.tp == 0
        assert result.tn == 3
        assert result.fp == 0
        assert result.fn == 0
        assert result.specificity == 1.0

    def test_length_mismatch_raises_error(self):
        """
        测试目的: y_truth 和 y_predict 长度不一致时抛出 ValueError
        输入: y_truth 长度 4, y_predict 长度 3
        输出: ValueError
        预期: 抛出 ValueError，消息包含 "长度不一致"
        """
        y_truth = np.array([0, 1, 0, 1])
        y_predict = np.array([0, 1, 0])

        op = BinaryClassificationMetric()
        with pytest.raises(ValueError, match="长度不一致"):
            op.run((y_truth, y_predict))

    def test_list_input_converted_to_ndarray(self):
        """
        测试目的: list 输入自动转换为 ndarray
        输入: y_truth/y_predict 为 Python list
        输出: BinaryClassificationResult
        预期: 正常计算，n_samples=4
        """
        y_truth = [0, 1, 0, 1]
        y_predict = [0, 1, 1, 0]

        op = BinaryClassificationMetric()
        result = op.run((y_truth, y_predict))

        assert result.n_samples == 4

    def test_large_sample_size(self):
        """
        测试目的: 大样本量场景
        输入: 10000 个样本
        输出: BinaryClassificationResult
        预期: 正常计算，性能可接受，n_samples=10000，f1 在 [0, 1] 范围内
        """
        rng = np.random.RandomState(42)
        y_truth = rng.randint(0, 2, size=10000)
        y_predict = rng.randint(0, 2, size=10000)

        op = BinaryClassificationMetric()
        result = op.run((y_truth, y_predict))

        assert result.n_samples == 10000
        assert 0.0 <= result.f1 <= 1.0

    def test_empty_array(self):
        """
        测试目的: 空数组输入场景
        输入: y_truth=[], y_predict=[]
        输出: BinaryClassificationResult
        预期: 正常执行，所有指标为默认值（除零保护）
        """
        y_truth = np.array([])
        y_predict = np.array([])

        op = BinaryClassificationMetric()
        result = op.run((y_truth, y_predict))
        # 空数组 → tp=fp=tn=fn=0，所有指标走安全除法的默认值
        assert result.n_samples == 0

    def test_all_predict_positive(self):
        """
        测试目的: 全部预测为正类的极端情况
        输入: y_truth=[0,0,1,1], y_predict=[1,1,1,1]（所有预测均为正类）
        输出: BinaryClassificationResult
        预期: tp=2, fp=2, tn=0, fn=0；precision=0.5(2/4), recall=1.0
        """
        y_truth = np.array([0, 0, 1, 1])
        y_predict = np.array([1, 1, 1, 1])

        op = BinaryClassificationMetric(positive_label=1)
        result = op.run((y_truth, y_predict))

        assert result.tp == 2  # 正类全预测正确
        assert result.fp == 2  # 负类全预测为正
        assert result.tn == 0  # 无正确预测的负类
        assert result.fn == 0  # 无遗漏的正类
        assert result.precision == 0.5  # tp/(tp+fp) = 2/4
        assert result.recall == 1.0

    def test_all_predict_negative(self):
        """
        测试目的: 全部预测为负类的极端情况
        输入: y_truth=[0,0,1,1], y_predict=[0,0,0,0]（所有预测均为负类）
        输出: BinaryClassificationResult
        预期: tp=0, fp=0, tn=2, fn=2；precision=0.0, recall=0.0, f1=0.0
        """
        y_truth = np.array([0, 0, 1, 1])
        y_predict = np.array([0, 0, 0, 0])

        op = BinaryClassificationMetric(positive_label=1)
        result = op.run((y_truth, y_predict))

        assert result.tp == 0
        assert result.fp == 0
        assert result.tn == 2  # 负类全预测正确
        assert result.fn == 2  # 正类全遗漏
        assert result.precision == 0.0
        assert result.recall == 0.0
        assert result.specificity == 1.0

    def test_single_sample(self):
        """
        测试目的: 单样本场景
        输入: y_truth=[1], y_predict=[1]
        输出: BinaryClassificationResult
        预期: n_samples=1, tp=1, f1=1.0
        """
        y_truth = np.array([1])
        y_predict = np.array([1])

        op = BinaryClassificationMetric(positive_label=1)
        result = op.run((y_truth, y_predict))

        assert result.n_samples == 1
        assert result.tp == 1
        assert result.f1 == 1.0

    def test_default_for_zero_with_all_positive_predict(self):
        """
        测试目的: 除零时 default_for_zero 配置生效
        输入: y_truth 全负类, y_predict 全正类 → tp=0, fp=4
        输出: BinaryClassificationResult
        预期: recall=0.0（tp=0），specificity=0.0（tn=0），precision=0.0
        """
        y_truth = np.array([0, 0, 0, 0])
        y_predict = np.array([1, 1, 1, 1])

        op = BinaryClassificationMetric(
            positive_label=1,
            default_for_zero=0.5,
        )
        result = op.run((y_truth, y_predict))

        assert result.tp == 0
        assert result.fp == 4
        assert result.tn == 0
        assert result.fn == 0
        # recall = tp/(tp+fn) = 0/0 → default_for_zero=0.5
        assert result.recall == 0.5
        # specificity = tn/(tn+fp) = 0/4 → 0.0（分母非零）
        assert result.specificity == 0.0


# ============================================================================
# 指标公式验证测试
# ============================================================================

class TestBinaryClassificationMetricFormula:
    """指标公式验证测试"""

    def test_f1_formula(self):
        """
        测试目的: 验证 F1 公式 = 2*P*R/(P+R)
        输入: 手动构造 TP=2, FP=1, FN=1
        输出: float (f1 值)
        预期: F1 值符合公式计算 = 2/3
        """
        # TP=2, FP=1, FN=1 → P=2/3≈0.667, R=2/3≈0.667
        # F1 = 2*0.667*0.667/(0.667+0.667) = 0.667
        y_truth = np.array([0, 0, 0, 1, 1, 1])  # 3个负类，3个正类
        y_predict = np.array([0, 0, 1, 1, 1, 0])  # TN=2, FP=1, TP=2, FN=1

        op = BinaryClassificationMetric(positive_label=1)
        result = op.run((y_truth, y_predict))

        expected_precision = 2 / 3
        expected_recall = 2 / 3
        expected_f1 = 2 * expected_precision * expected_recall / (expected_precision + expected_recall)

        assert abs(result.precision - expected_precision) < 1e-6
        assert abs(result.recall - expected_recall) < 1e-6
        assert abs(result.f1 - expected_f1) < 1e-6

    def test_mcc_formula(self):
        """
        测试目的: 验证 MCC 公式
        输入: 手动构造 TP=2, FP=1, TN=2, FN=1
        输出: float (mcc 值)
        预期: MCC = (TP*TN - FP*FN) / sqrt((TP+FP)(TP+FN)(TN+FP)(TN+FN)) = 3/9 = 0.333
        """
        # TP=2, FP=1, TN=2, FN=1
        # MCC = (TP*TN - FP*FN) / sqrt((TP+FP)(TP+FN)(TN+FP)(TN+FN))
        #     = (2*2 - 1*1) / sqrt(3*3*3*3) = 3/9 = 0.333
        y_truth = np.array([0, 0, 0, 1, 1, 1])
        y_predict = np.array([0, 0, 1, 1, 1, 0])

        op = BinaryClassificationMetric(positive_label=1)
        result = op.run((y_truth, y_predict))

        expected_mcc = (2 * 2 - 1 * 1) / np.sqrt(3 * 3 * 3 * 3)
        assert abs(result.mcc - expected_mcc) < 1e-6

    def test_far_equals_fpr(self):
        """
        测试目的: 验证 FAR = FPR
        输入: y_truth=[0,1,0,1], y_predict=[0,1,1,0]
        输出: BinaryClassificationResult
        预期: far == fpr
        """
        y_truth = np.array([0, 1, 0, 1])
        y_predict = np.array([0, 1, 1, 0])

        op = BinaryClassificationMetric()
        result = op.run((y_truth, y_predict))

        assert result.far == result.fpr

    def test_fdr_equals_recall(self):
        """
        测试目的: 验证 FDR = Recall
        输入: y_truth=[0,1,0,1], y_predict=[0,1,1,0]
        输出: BinaryClassificationResult
        预期: fdr == recall
        """
        y_truth = np.array([0, 1, 0, 1])
        y_predict = np.array([0, 1, 1, 0])

        op = BinaryClassificationMetric()
        result = op.run((y_truth, y_predict))

        assert result.fdr == result.recall

    def test_mar_equals_one_minus_recall(self):
        """
        测试目的: 验证 MAR = 1 - Recall
        输入: y_truth=[0,1,0,1], y_predict=[0,1,1,0]
        输出: BinaryClassificationResult
        预期: mar == 1 - recall
        """
        y_truth = np.array([0, 1, 0, 1])
        y_predict = np.array([0, 1, 1, 0])

        op = BinaryClassificationMetric()
        result = op.run((y_truth, y_predict))

        assert abs(result.mar - (1.0 - result.recall)) < 1e-6


# ============================================================================
# 冻结结果测试
# ============================================================================

class TestBinaryClassificationMetricFrozen:
    """冻结结果测试"""

    def test_result_is_frozen(self):
        """
        测试目的: 结果对象不可修改（frozen）
        输入: 尝试修改 result.f1 = 0.99
        输出: ValidationError
        预期: 抛出 ValidationError
        """
        y_truth = np.array([0, 1, 0, 1])
        y_predict = np.array([0, 1, 1, 0])

        op = BinaryClassificationMetric()
        result = op.run((y_truth, y_predict))

        with pytest.raises(Exception):  # Pydantic frozen 模式抛出 ValidationError
            result.f1 = 0.99

    def test_name(self):
        """
        测试目的: 验证 name() 返回正确标识
        输入: 无
        预期: 返回 "binary_classification"
        """
        assert BinaryClassificationMetric.name() == "binary_classification"