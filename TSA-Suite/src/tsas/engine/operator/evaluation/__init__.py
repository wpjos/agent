# -*- coding: utf-8 -*-

"""
评价指标算子包

提供基于 BaseMetricOperator 的评价指标算子实现，支持 HPO 单目标/多目标优化。

模块组成:
    - base: 基础类型（BaseMetricConfig、BaseMetricOperator）
    - binary_classification: 二分类指标（离散标签输入）
    - binary_curve: 二分类曲线指标（连续分数输入）
    - multi_classification: 多分类指标
    - point_adjust: 点调整指标（PA-F1，时序异常检测）
    - self_evaluation: 无标签自评估指标

使用示例::

    from tsas.engine.operator.evaluation import (
        BinaryClassificationMetric,
        BinaryClassificationCurve,
        MultipleClassificationMetric,
        PointAdjust,
        SelfEvaluation,
    )

    # 二分类离散标签
    op = BinaryClassificationMetric()
    result = op.run((y_truth, y_predict))
    print(result.f1, result.far)

    # 二分类连续分数（曲线指标）
    op = BinaryClassificationCurve()
    result = op.run((labels, scores))
    print(result.auc_roc, result.best_f1)

    # HPO 集成
    from tsas.engine.operator.evaluation import BinaryClassificationConfig

    op = BinaryClassificationMetric(config=BinaryClassificationConfig(main_scores={"f1": "f1"}))
    scores = op.scores((y_truth, y_predict))  # -> {"f1": 0.85}
"""

__all__ = []
