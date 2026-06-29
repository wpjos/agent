# -*- coding: utf-8 -*-

"""
二分类曲线评价指标算子（连续分数输入）

基于连续异常分数和离散标签计算完整的二分类曲线指标集。
遍历所有阈值计算 precision/recall/f1/mcc 等指标数组，同时产出
AUC-ROC、AUC-PR、各指标最优值及对应阈值。

核心组件:
    - BinaryClassificationCurveResult: 二分类曲线指标结果（Pydantic BaseModel）
    - BinaryClassificationCurveConfig: 配置类（继承 BaseMetricConfig）
    - BinaryClassificationCurve: 二分类曲线指标算子

使用示例::

    from tsas.engine.operator.evaluation import BinaryClassificationCurve

    # 基本用法
    op = BinaryClassificationCurve()
    result = op.run((labels, scores))
    print(result.auc_roc, result.auc_pr, result.best_f1)

    # HPO 集成
    op = BinaryClassificationCurve(main_scores={"auc_roc": "auc_roc", "best_f1": "best_f1"})
    scores = op.scores((labels, scores))  # -> {"auc_roc": 0.95, "best_f1": 0.88}
"""

from typing import ClassVar

import numpy as np
from pydantic import BaseModel, ConfigDict, Field

from tsas.engine.operator.evaluation.base import (
    BaseMetricConfig,
    BaseMetricOperator,
)

__all__ = [
    'BinaryClassificationCurveResult',
    'BinaryClassificationCurveConfig',
    'BinaryClassificationCurve',
]


# ============================================================================
# Result 类定义
# ============================================================================

class BinaryClassificationCurveResult(BaseModel):
    """
    二分类曲线评价指标结果

    包含阈值无关的标量汇总（AUC-ROC、AUC-PR）、各指标在所有阈值下的最优值及
    对应阈值、以及完整的曲线数组数据。

    Attributes:
        n_samples (int): 总样本数

        auc_roc (float): ROC曲线下面积
        auc_pr (float): PR曲线下面积

        best_precision (float): 最优精确率（越大越好）
        best_precision_threshold (float): 最优精确率对应的阈值
        best_recall (float): 最优召回率（越大越好）
        best_recall_threshold (float): 最优召回率对应的阈值
        best_specificity (float): 最优特异度（越大越好）
        best_specificity_threshold (float): 最优特异度对应的阈值
        best_f1 (float): 最优F1值（越大越好）
        best_f1_threshold (float): 最优F1对应的阈值
        best_mcc (float): 最优MCC值（越大越好）
        best_mcc_threshold (float): 最优MCC对应的阈值
        best_tpr (float): 最优TPR（越大越好）
        best_tpr_threshold (float): 最优TPR对应的阈值
        best_fpr (float): 最优FPR（越小越好）
        best_fpr_threshold (float): 最优FPR对应的阈值
        best_far (float): 最优FAR（越小越好）
        best_far_threshold (float): 最优FAR对应的阈值
        best_mar (float): 最优MAR（越小越好）
        best_mar_threshold (float): 最优MAR对应的阈值

        thresholds (list[float]): 判别阈值数组
        tpr (list[float]): 各阈值下的TPR
        fpr (list[float]): 各阈值下的FPR
        precision_arr (list[float]): 各阈值下的Precision
        recall_arr (list[float]): 各阈值下的Recall
        specificity_arr (list[float]): 各阈值下的Specificity
        f1_arr (list[float]): 各阈值下的F1
        mcc_arr (list[float]): 各阈值下的MCC
        far_arr (list[float]): 各阈值下的FAR
        mar_arr (list[float]): 各阈值下的MAR
    """
    model_config = ConfigDict(frozen=True)

    # 基本统计
    n_samples: int = Field(description="总样本数")

    # 标量汇总
    auc_roc: float = Field(description="ROC曲线下面积")
    auc_pr: float = Field(description="PR曲线下面积")

    # 各指标最优值（越大越好）及对应阈值
    best_precision: float = Field(description="最优精确率（越大越好）")
    best_precision_threshold: float = Field(description="最优精确率对应的阈值")
    best_recall: float = Field(description="最优召回率（越大越好）")
    best_recall_threshold: float = Field(description="最优召回率对应的阈值")
    best_specificity: float = Field(description="最优特异度（越大越好）")
    best_specificity_threshold: float = Field(description="最优特异度对应的阈值")
    best_f1: float = Field(description="最优F1值（越大越好）")
    best_f1_threshold: float = Field(description="最优F1对应的阈值")
    best_mcc: float = Field(description="最优MCC值（越大越好）")
    best_mcc_threshold: float = Field(description="最优MCC对应的阈值")
    best_tpr: float = Field(description="最优TPR（越大越好）")
    best_tpr_threshold: float = Field(description="最优TPR对应的阈值")

    # 各指标最优值（越小越好）及对应阈值
    best_fpr: float = Field(description="最优FPR（越小越好）")
    best_fpr_threshold: float = Field(description="最优FPR对应的阈值")
    best_far: float = Field(description="最优FAR（越小越好）")
    best_far_threshold: float = Field(description="最优FAR对应的阈值")
    best_mar: float = Field(description="最优MAR（越小越好）")
    best_mar_threshold: float = Field(description="最优MAR对应的阈值")

    # 曲线数组
    thresholds: list[float] = Field(description="判别阈值数组")
    tpr: list[float] = Field(description="各阈值下的TPR")
    fpr: list[float] = Field(description="各阈值下的FPR")
    precision_arr: list[float] = Field(description="各阈值下的Precision")
    recall_arr: list[float] = Field(description="各阈值下的Recall")
    specificity_arr: list[float] = Field(description="各阈值下的Specificity")
    f1_arr: list[float] = Field(description="各阈值下的F1")
    mcc_arr: list[float] = Field(description="各阈值下的MCC")
    far_arr: list[float] = Field(description="各阈值下的FAR")
    mar_arr: list[float] = Field(description="各阈值下的MAR")


# ============================================================================
# Config 类定义
# ============================================================================

class BinaryClassificationCurveConfig(BaseMetricConfig):
    """
    二分类曲线评价指标配置

    Attributes:
        positive_label (int): 正类标签值，默认 1
        default_for_zero (float): 除零时的默认值，默认 0.0
        decimals (int): 保留小数位数，默认 6
        predict_decimals (int): 分数精度截断位数，默认 0（不截断）
        distinct_thresholds (bool): 是否使用去重阈值，默认 True
        inf_threshold (bool): 是否在阈值列表末尾追加正无穷阈值，默认 True
        main_scores (dict[str, str] | None): 主评分路径映射
    """
    positive_label: int = Field(default=1, description="正类标签值，默认 1")
    default_for_zero: float = Field(default=0.0, description="除零时的默认值")
    decimals: int = Field(default=6, ge=0, description="保留小数位数")
    predict_decimals: int = Field(default=0, ge=0, description="分数精度截断位数，0 表示不截断")
    distinct_thresholds: bool = Field(default=True, description="是否使用去重阈值")
    inf_threshold: bool = Field(default=True, description="是否在阈值列表末尾追加正无穷阈值")
    main_scores: dict[str, str] | None = Field(
        default={"auc_roc": "auc_roc", "best_f1": "best_f1"},
        description="主评分路径映射，键为指标名称、值为结果属性路径",
    )


# ============================================================================
# 算子类定义
# ============================================================================

class BinaryClassificationCurve(
    BaseMetricOperator[
        tuple[np.ndarray, np.ndarray],
        BinaryClassificationCurveResult,
        BinaryClassificationCurveConfig,
        None,
    ],
):
    """
    二分类曲线评价指标算子

    输入为 (y_truth离散标签, y_predict连续异常分数)，遍历所有阈值
    计算完整的曲线指标集，包括 AUC-ROC、AUC-PR、各指标最优值及对应阈值。

    Input:
        y_truth: 真实离散标签，一维数组（正值表示正类）
        y_predict: 预测连续异常分数，一维数组，与 y_truth 等长，值越大越异常

    Output:
        完整的二分类曲线评价指标集（含 AUC-ROC/AUC-PR 标量、precision/recall/specificity/f1/mcc/tpr/fpr/far/mar 各指标的最优值及对应阈值、完整的阈值-指标曲线数组）。
        可通过 Config 的 ``main_scores`` 配置提取 auc_roc/best_f1 等命名标量用于 HPO。
        字段结构详见下方的"主输出结构"表格。

    泛型参数:
        I: tuple[np.ndarray, np.ndarray] — (y_truth, y_scores)
        MR: BinaryClassificationCurveResult — 结构化曲线结果
        MC: BinaryClassificationCurveConfig — 配置类
        RP: None — 无运行参数
    """

    _run_params_type: ClassVar[type | None] = None

    @classmethod
    def name(cls) -> str:
        return "binary_classification_curve"

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
    ) -> BinaryClassificationCurveResult:
        """计算二分类曲线评价指标

        根据离散标签和连续异常分数，遍历所有候选阈值，计算完整的二分类
        曲线指标集。整体流程如下：

        1. **输入校验**: 将标签和分数转为 numpy 数组，校验长度一致性。
        2. **正负类划分**: 根据 ``positive_label`` 构建正负类掩码。
        3. **阈值构建**: 从预测分数中提取去重阈值（降序），可追加正无穷。
        4. **逐阈值计算**: 对每个阈值，将分数二值化后统计混淆矩阵 (TP/FP/TN/FN)，
           再由 ``_compute_metrics`` 计算全部指标值。
        5. **AUC 积分**: 分别对 (FPR, TPR) 和 (Recall, Precision) 用梯形法
           计算 AUC-ROC 和 AUC-PR。
        6. **最优值提取**: 对每个指标取 argmax/argmin，记录最优值与对应阈值。
        7. **结果组装**: 将所有标量汇总、最优值、曲线数组组装为
           ``BinaryClassificationCurveResult`` 返回。

        Args:
            x (tuple[np.ndarray, np.ndarray]): 输入元组，包含两个元素：
                - ``y_truth``: 离散标签数组，形状为 ``(n_samples,)``
                - ``y_predict``: 连续异常分数数组，形状为 ``(n_samples,)``
            params (None): 无运行参数，此算子不使用运行参数

        Returns:
            BinaryClassificationCurveResult: 包含完整曲线指标的结果对象，
                包含 AUC-ROC、AUC-PR、各指标最优值及对应阈值、以及
                完整的阈值-指标曲线数组。

        Raises:
            ValueError: 当 ``y_truth`` 与 ``y_predict`` 长度不一致时抛出
        """
        y_truth, y_predict = x

        # ========== 1. 输入预处理与校验 ==========
        # 转换为 numpy 数组并展平，确保形状统一为 (n_samples,)
        y_truth = np.asarray(y_truth).ravel()
        y_predict = np.asarray(y_predict).ravel()

        # 校验标签与分数的样本数一致
        if len(y_truth) != len(y_predict):
            raise ValueError(
                f"y_truth 和 y_predict 长度不一致: {len(y_truth)} vs {len(y_predict)}"
            )

        n_samples = len(y_truth)

        # ========== 2. 提取配置参数 ==========
        # 从 config 中读取各项配置，若 config 为 None 则使用默认值
        config = self.config
        positive_label = config.positive_label if config else 1
        default_for_zero = config.default_for_zero if config else 0.0
        decimals = config.decimals if config else 6
        predict_decimals = config.predict_decimals if config else 0
        distinct_thresholds = config.distinct_thresholds if config else True
        inf_threshold = config.inf_threshold if config else True

        # ========== 3. 构建正负类掩码 ==========
        # 根据 positive_label 标记正类样本，统计正负类数量
        is_positive = (y_truth == positive_label)
        n_pos = int(np.sum(is_positive))
        n_neg = int(np.sum(~is_positive))

        # ========== 4. 构建候选阈值列表 ==========
        # 从预测分数中提取候选阈值，按降序排列（首个阈值将所有样本预测为负类）
        thresholds = self._build_thresholds(
            y_predict, predict_decimals, distinct_thresholds, inf_threshold
        )

        # ========== 5. 逐阈值计算混淆矩阵及各指标 ==========
        # 初始化各指标的列表，用于存储每个阈值下的计算结果
        tpr_arr, fpr_arr, precision_arr, recall_arr, specificity_arr = [], [], [], [], []
        f1_arr, mcc_arr, far_arr, mar_arr = [], [], [], []

        for threshold in thresholds:
            # 将连续分数二值化：大于阈值预测为正类（1），否则为负类（0）
            pred = (y_predict > threshold).astype(int)

            # 统计混淆矩阵四个象限
            # TP: 真正例（预测为正类，实际也为正类）
            tp = int(np.sum((pred == 1) & is_positive))
            # FP: 假正例（预测为正类，实际为负类）
            fp = int(np.sum((pred == 1) & ~is_positive))
            # TN: 真负例（预测为负类，实际也为负类）
            tn = int(np.sum((pred == 0) & ~is_positive))
            # FN: 假负例（预测为负类，实际为正类）
            fn = int(np.sum((pred == 0) & is_positive))

            # 基于混淆矩阵计算所有指标
            metrics = self._compute_metrics(
                tp, fp, tn, fn, default_for_zero
            )
            tpr_arr.append(metrics["tpr"])
            fpr_arr.append(metrics["fpr"])
            precision_arr.append(metrics["precision"])
            recall_arr.append(metrics["recall"])
            specificity_arr.append(metrics["specificity"])
            f1_arr.append(metrics["f1"])
            mcc_arr.append(metrics["mcc"])
            far_arr.append(metrics["far"])
            mar_arr.append(metrics["mar"])

        # ========== 6. 转换为 numpy 数组以便后续向量计算 ==========
        tpr_np = np.array(tpr_arr)
        fpr_np = np.array(fpr_arr)
        precision_np = np.array(precision_arr)
        recall_np = np.array(recall_arr)
        specificity_np = np.array(specificity_arr)
        f1_np = np.array(f1_arr)
        mcc_np = np.array(mcc_arr)
        far_np = np.array(far_arr)
        mar_np = np.array(mar_arr)

        # ========== 7. 计算 AUC 指标（梯形法积分） ==========
        # AUC-ROC: 以 FPR 为横轴、TPR 为纵轴的 ROC 曲线下面积
        auc_roc = self._compute_auc_trapezoidal(fpr_np, tpr_np)

        # AUC-PR: 以 Recall 为横轴、Precision 为纵轴的 PR 曲线下面积
        auc_pr = self._compute_auc_trapezoidal(recall_np, precision_np)

        # ========== 8. 计算各指标最优值及对应阈值 ==========
        thresholds_np = np.array(thresholds)

        best = {}

        # 越大越好的指标：通过 argmax 寻找最优索引
        for name, arr in [
            ("precision", precision_np),
            ("recall", recall_np),
            ("specificity", specificity_np),
            ("f1", f1_np),
            ("mcc", mcc_np),
            ("tpr", tpr_np),
        ]:
            idx = int(np.argmax(arr))
            best[f"best_{name}"] = self._round(float(arr[idx]), decimals)
            best[f"best_{name}_threshold"] = self._round(float(thresholds_np[idx]), decimals)

        # 越小越好的指标：通过 argmin 寻找最优索引
        for name, arr in [
            ("fpr", fpr_np),
            ("far", far_np),
            ("mar", mar_np),
        ]:
            idx = int(np.argmin(arr))
            best[f"best_{name}"] = self._round(float(arr[idx]), decimals)
            best[f"best_{name}_threshold"] = self._round(float(thresholds_np[idx]), decimals)

        # ========== 9. 组装并返回结果 ==========
        return BinaryClassificationCurveResult(
            n_samples=n_samples,
            auc_roc=self._round(auc_roc, decimals),
            auc_pr=self._round(auc_pr, decimals),
            **best,
            thresholds=[self._round(float(t), decimals) for t in thresholds],
            tpr=[self._round(float(v), decimals) for v in tpr_arr],
            fpr=[self._round(float(v), decimals) for v in fpr_arr],
            precision_arr=[self._round(float(v), decimals) for v in precision_arr],
            recall_arr=[self._round(float(v), decimals) for v in recall_arr],
            specificity_arr=[self._round(float(v), decimals) for v in specificity_arr],
            f1_arr=[self._round(float(v), decimals) for v in f1_arr],
            mcc_arr=[self._round(float(v), decimals) for v in mcc_arr],
            far_arr=[self._round(float(v), decimals) for v in far_arr],
            mar_arr=[self._round(float(v), decimals) for v in mar_arr],
        )

    def _build_thresholds(
        self,
        y_predict: np.ndarray,
        predict_decimals: int,
        distinct_thresholds: bool,
        inf_threshold: bool,
    ) -> list[float]:
        """构建候选阈值列表

        从连续预测分数中提取候选判别阈值，按降序排列。阈值的语义为：
        当样本分数 **大于** 某阈值时预测为正类（1），否则预测为负类（0）。

        构建流程：
        1. 对预测分数进行精度截断（若 ``predict_decimals > 0``）。
        2. 提取唯一分数值作为候选阈值，或保留全部值。
        3. 按**降序**排列，确保遍历时从最严格阈值开始。
        4. 若 ``inf_threshold`` 为 True，在末尾追加正无穷阈值，
           使得在最高阈值之上的样本仍然能被正确识别。

        Args:
            y_predict (np.ndarray): 连续预测分数数组，形状为 ``(n_samples,)``
            predict_decimals (int): 分数精度截断位数。
                - ``0``（默认）: 不截断，保留原始精度
                - ``> 0``: 使用 ``np.round`` 截断到指定小数位，
                  可减少阈值数量，加速计算
            distinct_thresholds (bool): 是否对阈值去重。
                - ``True``（默认）: 使用 ``np.unique`` 去重，避免
                  相同分数产生冗余计算
                - ``False``: 保留所有值（含重复），适用于需要
                  等权重采样的场景
            inf_threshold (bool): 是否在阈值列表末尾追加正无穷阈值。
                - ``True``（默认）: 追加 ``float('inf')``，确保
                  曲线起始于 (0, 0) 的极端情况
                - ``False``: 不追加，阈值列表以最小分数结尾

        Returns:
            list[float]: 降序排列的候选阈值列表。
                第一个阈值为最大分数，最后一个为 ``float('inf')``（当启用时）。
        """
        # ========== 1. 复制预测分数，避免修改原始数据 ==========
        scores = y_predict.copy()

        # ========== 2. 精度截断 ==========
        # 当 predict_decimals > 0 时，对分数进行四舍五入截断。
        # 这可以合并相近的分数值，从而减少候选阈值数量，
        # 在保证指标精度的同时显著加速计算。
        if predict_decimals > 0:
            scores = np.round(scores, predict_decimals)

        # ========== 3. 提取唯一值或保留全部值 ==========
        if distinct_thresholds:
            # 使用 np.unique 去重并自动排序（升序），消除重复分数带来的冗余计算
            unique_scores = np.unique(scores)
        else:
            # 保留所有分数值（含重复），按升序排列
            unique_scores = np.sort(scores)

        # ========== 4. 转为降序列表 ==========
        # 降序排列使得遍历时从最严格阈值（最大分数）开始，
        # 此时所有样本预测为负类；随着阈值降低，越来越多样本预测为正类。
        thresholds = sorted(unique_scores.tolist(), reverse=True)

        # ========== 5. 追加正无穷阈值 ==========
        # 正无穷阈值确保存在一个阈值使得所有样本被预测为负类（0），
        # 即 FPR=0, TPR=0 的起始点，保证 ROC 曲线从原点开始。
        if inf_threshold:
            thresholds.append(float('inf'))

        return thresholds

    @staticmethod
    def _compute_metrics(
        tp: int,
        fp: int,
        tn: int,
        fn: int,
        default_for_zero: float,
    ) -> dict[str, float]:
        """基于混淆矩阵元素计算全部二分类指标

        根据混淆矩阵的四个象限值 (TP, FP, TN, FN)，计算以下九个
        二分类评价指标，每个指标均通过安全除法处理分母为零的情况。

        计算的指标及其公式如下：

        - **Precision (精确率)**: 在所有预测为正类的样本中，实际为正类的比例。

          ``Precision = TP / (TP + FP)``

          取值范围 [0, 1]，越大表示误报越少。

        - **Recall (召回率)**: 在所有实际为正类的样本中，被正确预测为正类的比例。

          ``Recall = TP / (TP + FN)``

          取值范围 [0, 1]，越大表示漏报越少。

        - **Specificity (特异度)**: 在所有实际为负类的样本中，被正确预测为负类的比例。

          ``Specificity = TN / (TN + FP)``

          取值范围 [0, 1]，越大表示对负类的区分能力越强。

        - **F1 (F1 分数)**: Precision 和 Recall 的调和平均值，衡量两者的平衡。

          ``F1 = 2 * Precision * Recall / (Precision + Recall)``

          取值范围 [0, 1]，越大表示精确率和召回率越均衡。

        - **TPR (真阳性率)**: 与 Recall 相同，即真正例率。

          ``TPR = TP / (TP + FN)``

        - **FPR (假阳性率)**: 在所有实际为负类的样本中，被错误预测为正类的比例。

          ``FPR = FP / (FP + TN)``

          取值范围 [0, 1]，越小表示负类误判越少。

        - **FAR (假警报率)**: 与 FPR 相同，即假警报率。

          ``FAR = FPR = FP / (FP + TN)``

        - **MAR (漏检率)**: 在所有实际为正类的样本中，未被预测为正类的比例。

          ``MAR = 1 - Recall = FN / (TP + FN)``

          取值范围 [0, 1]，越小表示漏检越少。

        - **MCC (Matthews 相关系数)**: 衡量二分类预测质量的综合指标，
          考虑了 TP/FP/TN/FN 四个象限，对类别不平衡具有鲁棒性。

          ``MCC = (TP*TN - FP*FN) / sqrt((TP+FP)(TP+FN)(TN+FP)(TN+FN))``

          取值范围 [-1, 1]，1 表示完美预测，0 表示随机预测，-1 表示完全相反预测。

        Args:
            tp (int): True Positive（真正例）数量，
                即预测为正类且实际为正类的样本数
            fp (int): False Positive（假正例）数量，
                即预测为正类但实际为负类的样本数
            tn (int): True Negative（真负例）数量，
                即预测为负类且实际为负类的样本数
            fn (int): False Negative（假负例）数量，
                即预测为负类但实际为正类的样本数
            default_for_zero (float): 除零时的默认返回值。
                当某指标的分母为零时（例如无正类样本时 Precision 分母为 0），
                返回此默认值而非抛出异常。通常设为 ``0.0``。

        Returns:
            dict[str, float]: 包含九个指标的字典，键名为指标名称，值为 (float)，
                结构如下::

                    {
                        "precision": float,    # 精确率
                        "recall": float,       # 召回率
                        "specificity": float,  # 特异度
                        "f1": float,           # F1 分数
                        "tpr": float,          # 真阳性率
                        "fpr": float,          # 假阳性率
                        "far": float,          # 假警报率
                        "mar": float,          # 漏检率
                        "mcc": float,          # Matthews 相关系数
                    }
        """
        # ========== 安全除法辅助函数 ==========
        # 当分母为零时返回 default_for_zero，避免除零异常。
        # 这是二分类指标计算中的常见模式，因为某些极端阈值下
        # 混淆矩阵的部分象限可能为零。
        def _safe_div(num, den):
            return num / den if den != 0 else default_for_zero

        # ========== 基本比率指标 ==========
        # Precision: 精确率 — 预测为正类中实际为正类的比例
        precision = _safe_div(tp, tp + fp)
        # Recall: 召回率 — 实际为正类中被正确预测的比例
        recall = _safe_div(tp, tp + fn)
        # Specificity: 特异度 — 实际为负类中被正确预测的比例
        specificity = _safe_div(tn, tn + fp)

        # ========== 复合指标 ==========
        # F1: Precision 与 Recall 的调和平均，兼顾两者
        f1 = _safe_div(2 * precision * recall, precision + recall)

        # ========== 等价指标别名 ==========
        # TPR 与 Recall 等价，均描述正类的识别能力
        tpr = recall
        # FPR: 假阳性率，描述负类被误判为正类的比例
        fpr = _safe_div(fp, fp + tn)
        # FAR 与 FPR 等价，在异常检测领域常称为假警报率
        far = fpr
        # MAR: 漏检率，即 1 - Recall，描述正类被遗漏的比例
        mar = 1.0 - recall

        # ========== 综合指标 ==========
        # MCC: Matthews 相关系数，取值范围 [-1, 1]
        # 分母为混淆矩阵四象限行列式乘积的平方根
        mcc_denom = np.sqrt(float((tp + fp) * (tp + fn) * (tn + fp) * (tn + fn)))
        mcc = _safe_div(tp * tn - fp * fn, mcc_denom)

        return {
            "precision": precision,
            "recall": recall,
            "specificity": specificity,
            "f1": f1,
            "tpr": tpr,
            "fpr": fpr,
            "far": far,
            "mar": mar,
            "mcc": mcc,
        }

    @staticmethod
    def _compute_auc_trapezoidal(x: np.ndarray, y: np.ndarray) -> float:
        """梯形法计算曲线下面积（AUC）

        对给定的 (x, y) 数据点，使用梯形法（Trapezoidal Rule）近似计算
        曲线下面积。具体步骤如下：

        1. 按 x 值**升序**排列数据点，确保曲线单调递进。
        2. 对排序后的相邻数据点，以梯形面积公式求和::

            AUC = sum((x[i+1] - x[i]) * (y[i+1] + y[i]) / 2)

        梯形法是数值积分的经典方法，适用于离散采样的曲线数据。
        在二分类评价场景中，本方法被用于计算：
        - **AUC-ROC**: 传入 (FPR, TPR)，得到 ROC 曲线下面积
        - **AUC-PR**: 传入 (Recall, Precision)，得到 PR 曲线下面积

        注意：当数据点数量不足（<= 1）时，无法构成梯形，直接返回 0.0。

        Args:
            x (np.ndarray): 横轴数据数组，通常为 FPR 或 Recall，
                无需预先排序，方法内部会按升序排列
            y (np.ndarray): 纵轴数据数组，通常为 TPR 或 Precision，
                长度必须与 x 相同

        Returns:
            float: 曲线下面积的近似值（非负浮点数）。
                - 数据点数量 <= 1 时返回 ``0.0``
                - 正常情况下返回梯形法积分结果
        """
        # ========== 边界检查 ==========
        # 仅一个或零个数据点时无法构成梯形，直接返回 0.0
        if len(x) <= 1:
            return 0.0

        # ========== 按 x 升序排列数据点 ==========
        # np.argsort 返回升序排列的索引，确保曲线从左到右单调递进
        sorted_indices = np.argsort(x)
        x_sorted = x[sorted_indices]
        y_sorted = y[sorted_indices]

        # ========== 梯形法数值积分 ==========
        # np.trapezoid（numpy >= 2.0）使用梯形法计算积分：
        # AUC = sum((x[i+1] - x[i]) * (y[i+1] + y[i]) / 2)
        # 对于等间距数据退化为梯形公式，非等间距时自动适应。
        try:
            auc = float(np.trapezoid(y_sorted, x_sorted))
        except AttributeError:
            auc = float(np.trapz(y_sorted, x_sorted))
        return auc

    @staticmethod
    def _round(value: float, decimals: int) -> float:
        """对浮点数进行四舍五入截断

        将输入浮点数按照指定的小数位数进行四舍五入，返回截断后的浮点数。
        本方法是对 Python 内置 ``round`` 函数的静态封装，统一用于结果对象
        中所有数值字段的精度控制，确保输出结果的小数位数一致。

        Args:
            value (float): 需要四舍五入的原始浮点数值，
                通常为指标计算结果（如 AUC、F1、Precision 等）
            decimals (int): 保留的小数位数，非负整数。
                - ``0``: 保留整数部分
                - ``6``（默认配置值）: 保留六位小数
                - 负值行为与 Python 内置 round 一致（不推荐使用）

        Returns:
            float: 四舍五入后的浮点数值，小数位数由 ``decimals`` 决定
        """
        return round(value, decimals)