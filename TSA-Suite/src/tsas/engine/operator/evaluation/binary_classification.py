# -*- coding: utf-8 -*-

"""
二分类评价指标算子（离散标签输入）

基于混淆矩阵计算完整的二分类评价指标集，适用于离散标签输入场景。
一次计算产出 tp/fp/tn/fn、accuracy/precision/recall/specificity/f1/mcc 等完整指标。

核心组件:
    - BinaryClassificationResult: 二分类指标结果（Pydantic BaseModel）
    - BinaryClassificationConfig: 配置类（继承 BaseMetricConfig）
    - BinaryClassificationMetric: 二分类指标算子

使用示例::

    from tsas.engine.operator.evaluation import BinaryClassificationMetric

    # 基本用法
    op = BinaryClassificationMetric()
    result = op.run((y_truth, y_predict))
    print(result.f1, result.far)

    # 指定正类标签
    op = BinaryClassificationMetric(positive_label=1)
    result = op.run((y_truth, y_predict))

    # HPO 集成：提取主评分
    op = BinaryClassificationMetric(main_scores={"f1": "f1", "far": "far"})
    scores = op.scores((y_truth, y_predict))  # -> {"f1": 0.85, "far": 0.12}
"""

from typing import ClassVar

import numpy as np
from pydantic import BaseModel, ConfigDict, Field

from tsas.engine.operator.evaluation.base import (
    BaseMetricConfig,
    BaseMetricOperator,
)

__all__ = [
    'BinaryClassificationResult',
    'BinaryClassificationConfig',
    'BinaryClassificationMetric',
]


# ============================================================================
# Result 类定义
# ============================================================================

class BinaryClassificationResult(BaseModel):
    """
    二分类评价指标结果

    包含混淆矩阵计数值和衍生指标，全部为标量字段。
    混淆矩阵以 list[list[int]] 形式存储，便于序列化。

    Attributes:
        n_samples (int): 总样本数
        tp (int): True Positive，真正例数
        fp (int): False Positive，假正例数
        tn (int): True Negative，真负例数
        fn (int): False Negative，假负例数
        accuracy (float): 准确率 = (TP+TN) / (TP+TN+FP+FN)
        precision (float): 精确率 = TP / (TP+FP)
        recall (float): 召回率 = TP / (TP+FN)，与 FDR 等价
        specificity (float): 特异度 = TN / (FP+TN)
        f1 (float): F1值 = 2*P*R / (P+R)
        tpr (float): 真阳率，与 Recall 等价
        fpr (float): 假阳率 = FP / (FP+TN)，与 FAR 等价
        far (float): False Alarm Rate，故障误报率，与 FPR 等价
        fdr (float): Fault Detection Rate，故障检出率，与 Recall 等价
        mar (float): Missing Alarm Rate，故障漏报率 = 1 - Recall
        mcc (float): Matthews Correlation Coefficient，Matthews相关系数
        confusion_matrix (list[list[int]]): 混淆矩阵 [[TN,FP],[FN,TP]]
    """
    model_config = ConfigDict(frozen=True)

    # 基础计数
    n_samples: int = Field(description="总样本数")
    tp: int = Field(description="True Positive，真正例数")
    fp: int = Field(description="False Positive，假正例数")
    tn: int = Field(description="True Negative，真负例数")
    fn: int = Field(description="False Negative，假负例数")

    # 标量指标
    accuracy: float = Field(description="准确率 = (TP+TN) / (TP+TN+FP+FN)")
    precision: float = Field(description="精确率 = TP / (TP+FP)")
    recall: float = Field(description="召回率 = TP / (TP+FN)，与 FDR 等价")
    specificity: float = Field(description="特异度 = TN / (FP+TN)")
    f1: float = Field(description="F1值 = 2*P*R / (P+R)")
    tpr: float = Field(description="真阳率，与 Recall 等价")
    fpr: float = Field(description="假阳率 = FP / (FP+TN)，与 FAR 等价")
    far: float = Field(description="False Alarm Rate，故障误报率，与 FPR 等价")
    fdr: float = Field(description="Fault Detection Rate，故障检出率，与 Recall 等价")
    mar: float = Field(description="Missing Alarm Rate，故障漏报率 = 1 - Recall")
    mcc: float = Field(description="Matthews Correlation Coefficient，Matthews相关系数")

    # 混淆矩阵
    confusion_matrix: list[list[int]] = Field(description="混淆矩阵 [[TN,FP],[FN,TP]]")


# ============================================================================
# Config 类定义
# ============================================================================

class BinaryClassificationConfig(BaseMetricConfig):
    """
    二分类评价指标配置

    Attributes:
        positive_label (int | list[int] | None): 正类（异常类）标签值。
            - None: 自动推断 — {-1,1} 场景取 -1，其他场景默认 1
            - int: 单一正类标签值
            - list[int]: 多个正类标签值（合并为一类）
        labels (int | str | list | None): 有效标签列表。
            - None: 自动从 y_truth 中提取唯一值
            - int/str/list: 指定有效标签
        default_for_zero (float): 除零时的默认值，默认 0.0
        decimals (int): 保留小数位数，默认 6
        main_scores (dict[str, str] | None): 主评分路径映射
    """
    positive_label: int | list[int] | None = Field(
        default=None,
        description="正类（异常类）标签值；None 时自动推断（{-1,1} 场景取 -1，其他默认 1）",
    )
    labels: int | str | list | None = Field(
        default=None,
        description="有效标签列表；None 时从 y_truth 自动提取唯一值",
    )
    default_for_zero: float = Field(default=0.0, description="除零时的默认值")
    decimals: int = Field(default=6, ge=0, description="保留小数位数")
    main_scores: dict[str, str] | None = Field(
        default={"f1": "f1", "far": "far"},
        description="主评分路径映射，键为指标名称、值为结果属性路径",
    )


# ============================================================================
# 算子类定义
# ============================================================================

class BinaryClassificationMetric(
    BaseMetricOperator[
        tuple[np.ndarray, np.ndarray],
        BinaryClassificationResult,
        BinaryClassificationConfig,
        None,
    ],
):
    """
    二分类评价指标算子

    输入为离散标签对 (y_truth, y_predict)，计算完整的二分类指标集。
    无需阈值搜索，直接从混淆矩阵计算所有指标。

    Input:
        y_truth: 真实离散标签，一维数组（正值表示正类，负值或其他表示负类）
        y_predict: 预测离散标签，一维数组，与 y_truth 等长

    Output:
        完整的二分类评价指标集（含 accuracy/precision/recall/specificity/f1/tpr/fpr/far/fdr/mar/mcc 共 11 项指标 + 混淆矩阵）。
        可通过 Config 的 ``main_scores`` 配置提取 f1/far 等命名标量用于 HPO。
        字段结构详见下方的"主输出结构"表格。

    泛型参数:
        I: tuple[np.ndarray, np.ndarray] — (y_truth, y_predict) 离散标签对
        MR: BinaryClassificationResult — 结构化指标结果
        MC: BinaryClassificationConfig — 配置类
        RP: None — 无运行参数
    """

    _run_params_type: ClassVar[type | None] = None

    @classmethod
    def name(cls) -> str:
        return "binary_classification"

    @classmethod
    def version(cls) -> tuple[int, ...]:
        """返回算子版本号

        Returns:
            tuple[int, ...]: 版本号三元组 ``(1, 0, 0)``
        """
        return (1, 0, 0)

    def _run(
        self,
        x: tuple[np.ndarray, np.ndarray],
        *,
        params: None,
    ) -> BinaryClassificationResult:
        """
        计算二分类评价指标

        从离散标签对计算完整的二分类指标集。核心流程：
        1. 展平输入数组并校验长度一致
        2. 确定正类标签（显式配置或自动识别）
        3. 计算混淆矩阵（tp, fp, tn, fn）
        4. 从混淆矩阵衍生计算各指标

        Args:
            x (tuple[np.ndarray, np.ndarray]): (y_truth, y_predict) 离散标签对，
                两个数组需长度一致，会被自动展平为一维
            params (None): 无运行参数，仅用于接口统一

        Returns:
            BinaryClassificationResult: 包含完整指标集的结果对象，包含：
                - n_samples (int): 总样本数
                - tp/fp/tn/fn (int): 混淆矩阵计数
                - accuracy/precision/recall/specificity/f1 (float): 核心指标
                - tpr/fpr/far/fdr/mar/mcc (float): 衍生指标
                - confusion_matrix (list[list[int]]): 二维列表 [[TN,FP],[FN,TP]]

        Raises:
            ValueError: y_truth 和 y_predict 长度不一致时抛出
        """
        y_truth, y_predict = x

        # ========== 输入预处理 ==========
        # 转换为 numpy 数组并展平（支持多维输入）
        y_truth = np.asarray(y_truth).ravel()
        y_predict = np.asarray(y_predict).ravel()

        # 校验长度一致（核心前置条件）
        if len(y_truth) != len(y_predict):
            raise ValueError(
                f"y_truth 和 y_predict 长度不一致: {len(y_truth)} vs {len(y_predict)}"
            )

        n_samples = len(y_truth)

        # ========== 确定正类标签 ==========
        # 优先使用配置值，否则自动识别（取较小值）
        positive_label = self._resolve_positive_label(y_truth)

        # ========== 计算混淆矩阵 ==========
        # 根据正类标签统计 TP/FP/TN/FN
        tp, fp, tn, fn = self._compute_confusion_matrix(
            y_truth, y_predict, positive_label
        )

        # ========== 计算各指标 ==========
        # 从混淆矩阵衍生计算所有评估指标
        metrics = self._compute_metrics(tp, fp, tn, fn)

        # ========== 构建结果对象 ==========
        # 混淆矩阵格式: [[TN, FP], [FN, TP]]（行=真实，列=预测）
        confusion_matrix = [[tn, fp], [fn, tp]]

        # 构建并返回完整的指标结果
        return BinaryClassificationResult(
            n_samples=n_samples,
            tp=tp,
            fp=fp,
            tn=tn,
            fn=fn,
            accuracy=metrics["accuracy"],
            precision=metrics["precision"],
            recall=metrics["recall"],
            specificity=metrics["specificity"],
            f1=metrics["f1"],
            tpr=metrics["tpr"],
            fpr=metrics["fpr"],
            far=metrics["far"],
            fdr=metrics["fdr"],
            mar=metrics["mar"],
            mcc=metrics["mcc"],
            confusion_matrix=confusion_matrix,
        )

    def _resolve_positive_label(self, y_truth: np.ndarray) -> int | list[int]:
        """推断正类（异常类）标签。

        推断规则：
            1. 若配置显式指定 positive_label，则直接使用；
            2. 若标签集合恰好为 {-1, 1}，则正类（异常）为 -1；
            3. 其他所有情况（含 {0, 1}），默认正类（异常）为 1。

        Args:
            y_truth (np.ndarray): 真实标签数组（一维）

        Returns:
            int | list[int]: 正类标签值（int）或正类标签列表（list[int])
        """
        config = self.config
        # 显式配置优先
        if config is not None and config.positive_label is not None:
            return config.positive_label
        # 自动推断：基于标签集合特征选择正类（异常类）
        unique_labels = sorted(np.unique(y_truth).tolist())
        if unique_labels == [-1, 1]:
            return -1  # -1 表示异常
        # 默认：1 表示异常（覆盖 {0, 1} 和其他所有情况）
        return 1

    def _compute_confusion_matrix(
        self,
        y_truth: np.ndarray,
        y_predict: np.ndarray,
        positive_label: int | list[int],
    ) -> tuple[int, int, int, int]:
        """
        计算混淆矩阵的四个计数值

        根据正类标签定义，将预测结果与真实标签对比，统计四类计数：
        - TP (True Positive): 真实为正且预测为正
        - FP (False Positive): 真实为负但预测为正（误报）
        - TN (True Negative): 真实为负且预测为负
        - FN (False Negative): 真实为正但预测为负（漏报）

        当 positive_label 为列表时，使用 np.isin 判断多正类归属。

        Args:
            y_truth (np.ndarray): 真实标签数组（一维）
            y_predict (np.ndarray): 预测标签数组（一维）
            positive_label (int | list[int]): 正类标签值或正类标签列表

        Returns:
            tuple[int, int, int, int]: (tp, fp, tn, fn) 混淆矩阵四元组
        """
        # 处理多正类标签情况（如 positive_label=[1, 2] 表示 1 和 2 都为正类）
        if isinstance(positive_label, list):
            is_positive_truth = np.isin(y_truth, positive_label)
            is_positive_predict = np.isin(y_predict, positive_label)
        else:
            # 单一正类标签：直接等值比较
            is_positive_truth = (y_truth == positive_label)
            is_positive_predict = (y_predict == positive_label)

        # TP: 真实为正 AND 预测为正
        tp = int(np.sum(is_positive_truth & is_positive_predict))
        # FP: 真实为负 AND 预测为正（误报）
        fp = int(np.sum(~is_positive_truth & is_positive_predict))
        # TN: 真实为负 AND 预测为负
        tn = int(np.sum(~is_positive_truth & ~is_positive_predict))
        # FN: 真实为正 AND 预测为负（漏报）
        fn = int(np.sum(is_positive_truth & ~is_positive_predict))

        return tp, fp, tn, fn

    def _compute_metrics(
        self,
        tp: int,
        fp: int,
        tn: int,
        fn: int,
    ) -> dict[str, float]:
        """
        从混淆矩阵衍生计算所有二分类指标

        基于混淆矩阵四元组计算以下指标集：
        - accuracy (float): 准确率 = (TP+TN) / N
        - precision (float): 精确率 = TP / (TP+FP)
        - recall (float): 召回率 = TP / (TP+FN)，与 TPR 等价
        - specificity (float): 特异度 = TN / (TN+FP)
        - f1 (float): F1 分数 = 2×P×R / (P+R)
        - tpr (float): 真正率（True Positive Rate），与 Recall 等价
        - fpr (float): 假正率（False Positive Rate）= FP / (FP+TN)
        - far (float): 故障误报率（False Alarm Rate），与 FPR 等价
        - fdr (float): 故障检出率（Fault Detection Rate），与 Recall 等价
        - mar (float): 故障漏报率（Missing Alarm Rate）= 1 - Recall
        - mcc (float): Matthews 相关系数，适用于不平衡数据

        安全方位除法：分母为零时返回 config.default_for_zero，避免除零异常。
        所有指标值经 config.decimals 位四舍五入。

        Args:
            tp (int): True Positive 计数
            fp (int): False Positive 计数
            tn (int): True Negative 计数
            fn (int): False Negative 计数

        Returns:
            dict[str, float]: 指标名到指标值的映射字典，键包含：
                accuracy, precision, recall, specificity, f1,
                tpr, fpr, far, fdr, mar, mcc
        """
        config = self.config
        # 读取配置参数，无配置时使用默认值
        default_for_zero = 0.0 if config is None else config.default_for_zero
        decimals = 6 if config is None else config.decimals

        total = tp + fp + tn + fn  # 总样本数 N

        # ========== 计算核心指标 ==========
        accuracy = self._safe_divide(tp + tn, total, default_for_zero)  # 准确率：正确预测占比
        precision = self._safe_divide(tp, tp + fp, default_for_zero)    # 精确率：预测为正中真正为正的比例
        recall = self._safe_divide(tp, tp + fn, default_for_zero)       # 召回率：真正为正中被正确预测的比例
        specificity = self._safe_divide(tn, tn + fp, default_for_zero)  # 特异度：真正为负中被正确预测的比例

        # F1 分数：精确率与召回率的调和平均
        f1 = self._safe_divide(
            2 * precision * recall,
            precision + recall,
            default_for_zero,
        )

        # ========== 计算衍生指标 ==========
        tpr = recall                                                    # 真正率 = 召回率
        fpr = self._safe_divide(fp, fp + tn, default_for_zero)         # 假正率：负类被误判为正的比例
        far = fpr                                                       # 故障误报率（FAR 等同于 FPR）
        fdr = recall                                                    # 故障检出率（FDR 等同于 Recall）
        mar = 1.0 - recall                                              # 故障漏报率 = 1 - Recall

        # ========== 计算 MCC ==========
        # Matthews 相关系数：适用于不平衡数据集的综合评价指标
        # 取值范围 [-1, 1]：1=完美预测，0=随机预测，-1=完全相反
        # 公式: (TP*TN - FP*FN) / sqrt((TP+FP)(TP+FN)(TN+FP)(TN+FN))
        mcc_denominator = np.sqrt(
            (tp + fp) * (tp + fn) * (tn + fp) * (tn + fn)
        )
        mcc = self._safe_divide(tp * tn - fp * fn, mcc_denominator, default_for_zero)

        # ========== 四舍五入到指定精度 ==========
        return {
            "accuracy": self._round(accuracy, decimals),
            "precision": self._round(precision, decimals),
            "recall": self._round(recall, decimals),
            "specificity": self._round(specificity, decimals),
            "f1": self._round(f1, decimals),
            "tpr": self._round(tpr, decimals),
            "fpr": self._round(fpr, decimals),
            "far": self._round(far, decimals),
            "fdr": self._round(fdr, decimals),
            "mar": self._round(mar, decimals),
            "mcc": self._round(mcc, decimals),
        }

    @staticmethod
    def _safe_divide(
        numerator: float,
        denominator: float,
        default: float,
    ) -> float:
        """
        安全除法，避免除零

        Args:
            numerator (float): 被除数
            denominator (float): 除数
            default (float): 除数为零时的默认值

        Returns:
            float: 除法结果或默认值
        """
        if denominator == 0:
            return default
        return numerator / denominator

    @staticmethod
    def _round(value: float, decimals: int) -> float:
        """
        四舍五入到指定小数位数

        Args:
            value (float): 原始值
            decimals (int): 小数位数

        Returns:
            float: 四舍五入后的值
        """
        return round(value, decimals)