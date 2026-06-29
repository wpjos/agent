# -*- coding: utf-8 -*-

"""
多分类评价指标算子（离散标签输入）

基于混淆矩阵计算完整的多分类评价指标集，支持 macro 平均和 per-label 指标。

核心组件:
    - PerLabelMetricResult: 单类别指标结果（Pydantic BaseModel）
    - MultiClassificationMetricResult: 多分类指标结果（Pydantic BaseModel）
    - MultiClassificationMetricConfig: 配置类（继承 BaseMetricConfig）
    - MultipleClassificationMetric: 多分类指标算子

使用示例::

    from tsas.engine.operator.evaluation import MultipleClassificationMetric

    # 基本用法
    op = MultipleClassificationMetric()
    result = op.run((y_truth, y_predict))
    print(result.f1, result.accuracy)

    # HPO 集成
    op = MultipleClassificationMetric(main_scores={"f1": "f1"})
    scores = op.scores((y_truth, y_predict))  # -> {"f1": 0.75}
"""

from typing import ClassVar

import numpy as np
from pydantic import BaseModel, ConfigDict, Field

from tsas.engine.operator.evaluation.base import (
    BaseMetricConfig,
    BaseMetricOperator,
)

__all__ = [
    'PerLabelMetricResult',
    'MultiClassificationMetricResult',
    'MultiClassificationMetricConfig',
    'MultipleClassificationMetric',
]


# ============================================================================
# Result 类定义
# ============================================================================

class PerLabelMetricResult(BaseModel):
    """
    单类别评价指标结果

    将每个类别视为正类时计算的完整指标集。

    Attributes:
        label (int | str): 类别标签值
        tp (int): True Positive
        fp (int): False Positive
        tn (int): True Negative
        fn (int): False Negative
        accuracy (float): 准确率
        precision (float): 精确率
        recall (float): 召回率
        specificity (float): 特异度
        f1 (float): F1值
        tpr (float): 真阳率
        fpr (float): 假阳率
        far (float): 故障误报率
        fdr (float): 故障检出率
        mar (float): 故障漏报率
        mcc (float): Matthews相关系数
    """
    model_config = ConfigDict(frozen=True)

    label: int | str = Field(description="类别标签值")
    tp: int = Field(description="True Positive")
    fp: int = Field(description="False Positive")
    tn: int = Field(description="True Negative")
    fn: int = Field(description="False Negative")
    accuracy: float = Field(description="准确率")
    precision: float = Field(description="精确率")
    recall: float = Field(description="召回率")
    specificity: float = Field(description="特异度")
    f1: float = Field(description="F1值")
    tpr: float = Field(description="真阳率")
    fpr: float = Field(description="假阳率")
    far: float = Field(description="故障误报率")
    fdr: float = Field(description="故障检出率")
    mar: float = Field(description="故障漏报率")
    mcc: float = Field(description="Matthews相关系数")


class MultiClassificationMetricResult(BaseModel):
    """
    多分类评价指标结果

    包含 macro 平均指标和各类别的 per-label 指标列表。

    Attributes:
        n_samples (int): 总样本数
        k_labels (int): 类别数量
        labels (list[int | str]): 类别标签列表
        accuracy (float): Macro-averaged 准确率
        precision (float): Macro-averaged 精确率
        recall (float): Macro-averaged 召回率
        specificity (float): Macro-averaged 特异度
        f1 (float): Macro-averaged F1值
        tpr (float): Macro-averaged 真阳率
        fpr (float): Macro-averaged 假阳率
        far (float): Macro-averaged 故障误报率
        fdr (float): Macro-averaged 故障检出率
        mar (float): Macro-averaged 故障漏报率
        mcc (float): Macro-averaged Matthews相关系数
        per_label_metrics (list[PerLabelMetricResult]): 各类别指标列表
        confusion_matrix (list[list[int]]): 混淆矩阵 (k x k)
    """
    model_config = ConfigDict(frozen=True)

    n_samples: int = Field(description="总样本数")
    k_labels: int = Field(description="类别数量")
    labels: list[int | str] = Field(description="类别标签列表")

    # Macro 平均指标
    accuracy: float = Field(description="Macro-averaged 准确率")
    precision: float = Field(description="Macro-averaged 精确率")
    recall: float = Field(description="Macro-averaged 召回率")
    specificity: float = Field(description="Macro-averaged 特异度")
    f1: float = Field(description="Macro-averaged F1值")
    tpr: float = Field(description="Macro-averaged 真阳率")
    fpr: float = Field(description="Macro-averaged 假阳率")
    far: float = Field(description="Macro-averaged 故障误报率")
    fdr: float = Field(description="Macro-averaged 故障检出率")
    mar: float = Field(description="Macro-averaged 故障漏报率")
    mcc: float = Field(description="Macro-averaged Matthews相关系数")

    # Per-label 指标
    per_label_metrics: list[PerLabelMetricResult] = Field(description="各类别指标列表")

    # 混淆矩阵
    confusion_matrix: list[list[int]] = Field(description="混淆矩阵 (k x k)")


# ============================================================================
# Config 类定义
# ============================================================================

class MultiClassificationMetricConfig(BaseMetricConfig):
    """
    多分类评价指标配置

    Attributes:
        labels (int | str | list | None): 有效标签列表。
            - None: 自动从 y_truth 中提取唯一值
            - int/str/list: 指定有效标签
        default_for_zero (float): 除零时的默认值，默认 0.0
        decimals (int): 保留小数位数，默认 6
        main_scores (dict[str, str] | None): 主评分路径映射
    """
    labels: int | str | list | None = Field(
        default=None,
        description="有效标签列表；None 时从 y_truth 自动提取唯一值",
    )
    default_for_zero: float = Field(default=0.0, description="除零时的默认值")
    decimals: int = Field(default=6, ge=0, description="保留小数位数")
    main_scores: dict[str, str] | None = Field(
        default={"f1": "f1", "accuracy": "accuracy"},
        description="主评分路径映射，键为指标名称、值为结果属性路径",
    )


# ============================================================================
# 算子类定义
# ============================================================================

class MultipleClassificationMetric(
    BaseMetricOperator[
        tuple[np.ndarray, np.ndarray],
        MultiClassificationMetricResult,
        MultiClassificationMetricConfig,
        None,
    ],
):
    """
    多分类评价指标算子

    输入为离散标签对 (y_truth, y_predict)，计算完整的多分类指标集。
    支持任意类别数量，产出 macro 平均指标和 per-label 指标。

    Input:
        y_truth: 真实离散标签，一维数组（支持任意类别数量）
        y_predict: 预测离散标签，一维数组，与 y_truth 等长

    Output:
        完整的多分类评价指标集（含 macro 平均的 11 项指标 + 各类别的 per-label 指标列表 + k×k 混淆矩阵）。
        可通过 Config 的 ``main_scores`` 配置提取 f1/accuracy 等命名标量用于 HPO。
        字段结构详见下方的"主输出结构"表格。

    泛型参数:
        I: tuple[np.ndarray, np.ndarray] — (y_truth, y_predict) 离散标签对
        MR: MultiClassificationMetricResult — 结构化指标结果
        MC: MultiClassificationMetricConfig — 配置类
        RP: None — 无运行参数
    """

    _run_params_type: ClassVar[type | None] = None

    @classmethod
    def name(cls) -> str:
        return "multi_classification"

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
    ) -> MultiClassificationMetricResult:
        """计算多分类评价指标的核心执行方法。

        该方法是算子的主入口，负责协调整个多分类评价流程。
        输入为离散标签对 (y_truth, y_predict)，依次执行以下步骤：
        1. 输入预处理：将标签转换为 numpy 数组并展平为一维
        2. 长度校验：确保真实标签与预测标签的样本数一致
        3. 标签解析：确定有效类别标签列表（支持配置指定或自动推断）
        4. 混淆矩阵构建：统计各类别的预测情况，生成 k x k 混淆矩阵
        5. 逐类指标计算：将每个类别依次视为正类，计算完整的二分类指标集
        6. 宏平均聚合：对所有类别的指标进行 macro 平均
        7. 结果封装：将所有指标组装为结构化结果对象返回

        Args:
            x (tuple[np.ndarray, np.ndarray]): 离散标签对，格式为
                (y_truth, y_predict)。其中 y_truth 为真实标签数组，
                y_predict 为预测标签数组，两者的长度必须一致。
            params (None): 无运行参数，固定为 None。

        Returns:
            MultiClassificationMetricResult: 包含完整指标集的结果对象，
                包括 macro 平均指标、各类别的 per-label 指标、混淆矩阵等。

        Raises:
            ValueError: 当 y_truth 和 y_predict 长度不一致时抛出。
        """
        y_truth, y_predict = x

        # ======================================================================
        # 第一步：输入预处理 —— 转换为 numpy 数组并展平为一维
        # ======================================================================
        y_truth = np.asarray(y_truth).ravel()
        y_predict = np.asarray(y_predict).ravel()

        # ======================================================================
        # 第二步：长度校验 —— 确保真实标签与预测标签的样本数一致
        # ======================================================================
        if len(y_truth) != len(y_predict):
            raise ValueError(
                f"y_truth 和 y_predict 长度不一致: {len(y_truth)} vs {len(y_predict)}"
            )

        n_samples = len(y_truth)

        # ======================================================================
        # 第三步：标签解析 —— 根据配置或自动推断确定有效类别标签列表
        # ======================================================================
        labels = self._resolve_labels(y_truth)

        k_labels = len(labels)

        # ======================================================================
        # 第四步：构建混淆矩阵 —— 统计各类别的预测情况，生成 k x k 矩阵
        # ======================================================================
        confusion_matrix = self._compute_confusion_matrix(
            y_truth, y_predict, labels
        )

        # ======================================================================
        # 第五步：逐类指标计算 —— 将每个类别依次视为正类，计算完整指标集
        # ======================================================================
        per_label_metrics = []
        for i, label in enumerate(labels):
            # 将第 i 类视为正类，其余类别合并为负类
            # 从混淆矩阵中提取 TP/FP/TN/FN 四个基础统计量
            tp = confusion_matrix[i][i]
            # FP: 预测为第 i 类但实际不是第 i 类的样本数
            fp = sum(confusion_matrix[j][i] for j in range(k_labels) if j != i)
            # TN: 既不是第 i 类且预测也不是第 i 类的样本数（矩阵中排除第 i 行和第 i 列的所有元素之和）
            tn = sum(
                confusion_matrix[j][k]
                for j in range(k_labels)
                for k in range(k_labels)
                if j != i and k != i
            )
            # FN: 实际是第 i 类但预测不是第 i 类的样本数
            fn = sum(confusion_matrix[i][j] for j in range(k_labels) if j != i)

            # 基于四个基础统计量计算该类别的完整评价指标集
            metrics = self._compute_metrics(tp, fp, tn, fn)

            # 将该类别的指标封装为 PerLabelMetricResult 并加入结果列表
            per_label_metrics.append(
                PerLabelMetricResult(
                    label=label,
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
                )
            )

        # ======================================================================
        # 第六步：宏平均聚合 —— 对所有类别的各项指标取算术平均
        # ======================================================================
        macro_metrics = self._compute_macro_average(per_label_metrics)

        # ======================================================================
        # 第七步：结果封装 —— 将 numpy 混淆矩阵转为 list 格式并组装最终结果
        # ======================================================================
        confusion_matrix_list = [
            [int(confusion_matrix[i][j]) for j in range(k_labels)]
            for i in range(k_labels)
        ]

        return MultiClassificationMetricResult(
            n_samples=n_samples,
            k_labels=k_labels,
            labels=list(labels),
            accuracy=macro_metrics["accuracy"],
            precision=macro_metrics["precision"],
            recall=macro_metrics["recall"],
            specificity=macro_metrics["specificity"],
            f1=macro_metrics["f1"],
            tpr=macro_metrics["tpr"],
            fpr=macro_metrics["fpr"],
            far=macro_metrics["far"],
            fdr=macro_metrics["fdr"],
            mar=macro_metrics["mar"],
            mcc=macro_metrics["mcc"],
            per_label_metrics=per_label_metrics,
            confusion_matrix=confusion_matrix_list,
        )

    def _resolve_labels(self, y_truth: np.ndarray) -> list[int | str]:
        """确定用于评价的类别标签列表。

        标签解析策略分为两种模式：
        - **配置指定模式**：当配置中 ``labels`` 字段不为 None 时，直接使用配置值。
          支持单个标签（int/str）或标签列表（list），单个标签会被包装为单元素列表。
        - **自动推断模式**：当配置中 ``labels`` 为 None 时，从 ``y_truth`` 中提取
          所有唯一值作为标签列表，并按升序排列。同时会将 numpy 整数类型转换为
          Python 原生 int 类型，以确保后续序列化（如 Pydantic 模型）的正常工作。

        Args:
            y_truth (np.ndarray): 真实标签的一维数组，用于在自动推断模式下
                提取唯一类别值。

        Returns:
            list[int | str]: 排序后的类别标签列表。当配置指定了标签时
                返回配置值；否则返回从 y_truth 自动推断的唯一值列表。
        """
        config = self.config

        # ======================================================================
        # 配置指定模式 —— 优先使用用户在配置中明确指定的标签列表
        # ======================================================================
        if config is not None and config.labels is not None:
            labels = config.labels
            # 单个标签（int 或 str）包装为列表
            if isinstance(labels, (int, str)):
                return [labels]
            return list(labels)

        # ======================================================================
        # 自动推断模式 —— 从 y_truth 提取唯一值并排序
        # ======================================================================
        unique_labels = np.unique(y_truth)
        # 将 numpy 整数类型转换为 Python 原生 int，确保 Pydantic 序列化兼容
        return [int(l) if isinstance(l, (np.integer, int)) else l for l in unique_labels]

    def _compute_confusion_matrix(
        self,
        y_truth: np.ndarray,
        y_predict: np.ndarray,
        labels: list[int | str],
    ) -> np.ndarray:
        """构建多分类混淆矩阵。

        混淆矩阵是一个 k x k 的方阵（k 为类别数量），其中：
        - 行索引表示真实类别
        - 列索引表示预测类别
        - 矩阵元素 ``C[i][j]`` 表示真实类别为 labels[i] 但被预测为 labels[j] 的样本数

        对角线元素 ``C[i][i]`` 即为第 i 类的 True Positive 数量；
        非对角线元素表示预测错误的情况。

        对于 y_truth 或 y_predict 中出现但不在 labels 列表中的标签值，
        将被静默忽略（不参与计数），以确保矩阵维度始终为 k x k。

        Args:
            y_truth (np.ndarray): 真实标签的一维数组。
            y_predict (np.ndarray): 预测标签的一维数组，长度须与 y_truth 一致。
            labels (list[int | str]): 类别标签列表，决定矩阵的行列顺序和维度。

        Returns:
            np.ndarray: 形状为 (k, k) 的整数矩阵，dtype 为 int，
                其中 k = len(labels)。行对应真实类别，列对应预测类别。
        """
        k = len(labels)

        # 构建标签到矩阵索引的映射字典，用于快速查找行列位置
        label_to_idx = {label: i for i, label in enumerate(labels)}

        # 初始化 k x k 的零矩阵
        confusion_matrix = np.zeros((k, k), dtype=int)

        # ======================================================================
        # 遍历所有样本对，统计真实类别与预测类别的组合频次
        # ======================================================================
        for truth_val, predict_val in zip(y_truth, y_predict):
            # 仅当真实标签和预测标签都在有效标签列表中时才计数
            # 不在 labels 中的标签值将被跳过
            if truth_val in label_to_idx and predict_val in label_to_idx:
                i = label_to_idx[truth_val]
                j = label_to_idx[predict_val]
                confusion_matrix[i][j] += 1

        return confusion_matrix

    def _compute_metrics(
        self,
        tp: int,
        fp: int,
        tn: int,
        fn: int,
    ) -> dict[str, float]:
        """基于混淆矩阵的四个基础统计量计算完整的二分类评价指标集。

        对于多分类场景中的每个类别，将其视为"正类"、其余类别合并为"负类"，
        即可将问题转化为二分类，从而计算以下指标：

        计算指标一览（含公式）：

        - **accuracy** (准确率): (TP + TN) / (TP + FP + TN + FN)
          所有样本中被正确分类的比例。
        - **precision** (精确率): TP / (TP + FP)
          预测为正类的样本中实际为正类的比例，衡量"预测出的正类有多少是准确的"。
        - **recall** (召回率): TP / (TP + FN)
          实际为正类的样本中被正确预测为正类的比例，衡量"正类样本有多少被找出来"。
        - **specificity** (特异度): TN / (TN + FP)
          实际为负类的样本中被正确预测为负类的比例。
        - **f1** (F1 值): 2 * precision * recall / (precision + recall)
          精确率与召回率的调和平均数，综合衡量分类性能。
        - **tpr** (真阳率 / True Positive Rate): TP / (TP + FN)
          等同于召回率，表示正类被正确识别的概率。
        - **fpr** (假阳率 / False Positive Rate): FP / (FP + TN)
          负类被错误预测为正类的概率。
        - **far** (故障误报率 / False Alarm Rate): FP / (FP + TN)
          等同于假阳率 (fpr)，表示正常样本被误报为故障的概率。
        - **fdr** (故障检出率 / Fault Detection Rate): TP / (TP + FN)
          等同于召回率 (recall)，表示故障被正确检出的概率。
        - **mar** (故障漏报率 / Missed Alarm Rate): 1 - recall = FN / (TP + FN)
          故障未被检出而被漏报的概率，等于 1 减去召回率。
        - **mcc** (Matthews 相关系数): (TP*TN - FP*FN) / sqrt((TP+FP)*(TP+FN)*(TN+FP)*(TN+FN))
          取值范围 [-1, 1]，综合考虑 TP/FP/TN/FN 的对称性度量指标。

        所有涉及除法的指标均通过 ``_safe_divide`` 进行除零保护，
        除零时返回配置中指定的默认值（默认为 0.0）。
        所有指标结果均通过 ``_round`` 进行指定小数位数的四舍五入。

        Args:
            tp (int): True Positive，正类被正确预测为正类的样本数。
            fp (int): False Positive，负类被错误预测为正类的样本数。
            tn (int): True Negative，负类被正确预测为负类的样本数。
            fn (int): False Negative，正类被错误预测为负类的样本数。

        Returns:
            dict[str, float]: 包含所有评价指标的字典，键为指标名称，
                值为经四舍五入后的浮点数值。包含 accuracy、precision、
                recall、specificity、f1、tpr、fpr、far、fdr、mar、mcc
                共 11 个指标。
        """
        config = self.config
        default_for_zero = 0.0 if config is None else config.default_for_zero
        decimals = 6 if config is None else config.decimals

        # 总样本数，用于计算准确率
        total = tp + fp + tn + fn

        # ======================================================================
        # 基础指标计算 —— 准确率、精确率、召回率、特异度
        # ======================================================================
        # 准确率: 正确预测的样本（TP + TN）占总样本数的比例
        accuracy = self._safe_divide(tp + tn, total, default_for_zero)
        # 精确率: 预测为正类的样本中实际为正类的比例
        precision = self._safe_divide(tp, tp + fp, default_for_zero)
        # 召回率: 实际为正类的样本中被正确识别的比例
        recall = self._safe_divide(tp, tp + fn, default_for_zero)
        # 特异度: 实际为负类的样本中被正确识别为负类的比例
        specificity = self._safe_divide(tn, tn + fp, default_for_zero)

        # ======================================================================
        # 复合指标计算 —— F1 值（精确率与召回率的调和平均数）
        # ======================================================================
        f1 = self._safe_divide(
            2 * precision * recall,
            precision + recall,
            default_for_zero,
        )

        # ======================================================================
        # 故障诊断相关指标 —— 真阳率、假阳率、误报率、检出率、漏报率
        # ======================================================================
        # 真阳率 (TPR): 等同于召回率，正类被正确识别的概率
        tpr = recall
        # 假阳率 (FPR): 负类被错误预测为正类的概率
        fpr = self._safe_divide(fp, fp + tn, default_for_zero)
        # 故障误报率 (FAR): 等同于假阳率，正常样本被误报为故障的概率
        far = fpr
        # 故障检出率 (FDR): 等同于召回率，故障被正确检出的概率
        fdr = recall
        # 故障漏报率 (MAR): 等于 1 - 召回率，故障未被检出的概率
        mar = 1.0 - recall

        # ======================================================================
        # Matthews 相关系数 (MCC) —— 对称性综合度量指标
        # ======================================================================
        # MCC 的分母为四项条件概率乘积的平方根
        mcc_denominator = np.sqrt(
            float((tp + fp) * (tp + fn) * (tn + fp) * (tn + fn))
        )
        mcc = self._safe_divide(tp * tn - fp * fn, mcc_denominator, default_for_zero)

        # ======================================================================
        # 结果组装 —— 对所有指标进行四舍五入后封装为字典
        # ======================================================================
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

    def _compute_macro_average(
        self,
        per_label_metrics: list[PerLabelMetricResult],
    ) -> dict[str, float]:
        """计算所有类别的 macro 平均指标。

        Macro 平均是一种常用的多分类指标聚合策略：先对每个类别分别计算指标值，
        再对所有类别的同一指标取算术平均。这种方式赋予每个类别相同的权重，
        不受各类别样本数量差异的影响，适用于类别不均衡场景下的综合评价。

        计算公式::

            macro_metric = (1 / K) * sum(metric_k, k=1..K)

        其中 K 为类别数量，metric_k 为第 k 个类别对应的指标值。

        当 ``per_label_metrics`` 为空列表时，返回空字典。

        Args:
            per_label_metrics (list[PerLabelMetricResult]): 各类别的逐标签指标
                结果列表，每个元素包含一个类别的完整评价指标集。列表长度
                即为类别数量 K。

        Returns:
            dict[str, float]: Macro 平均指标字典，键为指标名称（与
                PerLabelMetricResult 中的字段名一致），值为各类别该指标的
                算术平均值经四舍五入后的结果。包含 accuracy、precision、recall、
                specificity、f1、tpr、fpr、far、fdr、mar、mcc 共 11 个指标。
                当输入为空列表时返回空字典 ``{}``。
        """
        config = self.config
        decimals = 6 if config is None else config.decimals

        k = len(per_label_metrics)

        # 空列表保护：没有任何类别时直接返回空字典
        if k == 0:
            return {}

        # ======================================================================
        # 对每个指标名称，从所有类别中收集对应值并计算算术平均
        # ======================================================================
        # 需要求平均的指标名称列表，与 PerLabelMetricResult 中的数值字段一一对应
        metric_names = [
            "accuracy", "precision", "recall", "specificity", "f1",
            "tpr", "fpr", "far", "fdr", "mar", "mcc",
        ]

        macro_metrics = {}
        for name in metric_names:
            # 通过 getattr 从每个 PerLabelMetricResult 对象中提取对应字段的值
            values = [getattr(m, name) for m in per_label_metrics]
            # 计算算术平均值并进行四舍五入
            macro_metrics[name] = self._round(sum(values) / k, decimals)

        return macro_metrics

    @staticmethod
    def _safe_divide(
        numerator: float,
        denominator: float,
        default: float,
    ) -> float:
        """执行安全的浮点除法运算。

        当分母为零时返回预设默认值而非抛出 ZeroDivisionError 异常，
        用于保护评价指标计算中的除零场景（如某类别在预测中从未出现时，
        精确率的分母 TP + FP 可能为零）。

        Args:
            numerator (float): 被除数，即分子。
            denominator (float): 除数，即分母。当为零时触发安全保护。
            default (float): 分母为零时返回的默认值，通常为 0.0。

        Returns:
            float: 除法结果 ``numerator / denominator``；当分母为零时
                返回 ``default``。
        """
        if denominator == 0:
            return default
        return numerator / denominator

    @staticmethod
    def _round(value: float, decimals: int) -> float:
        """对浮点数进行指定小数位数的四舍五入。

        封装 Python 内置 ``round`` 函数，用于统一控制评价指标输出
        的小数位数精度，避免浮点数精度问题导致的显示不一致。

        Args:
            value (float): 需要四舍五入的浮点数值。
            decimals (int): 保留的小数位数，必须为非负整数。

        Returns:
            float: 四舍五入后的浮点数值，保留 ``decimals`` 位小数。
        """
        return round(value, decimals)