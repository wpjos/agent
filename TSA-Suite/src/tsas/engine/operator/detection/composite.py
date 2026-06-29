# -*- coding: utf-8 -*-

"""
组合异常检测算子

提供组合模式的算子，将多个子算子按顺序串联成一个完整的检测流程。

包含:
    - CompositeScorerExtraOutput: 组合评分器附加输出
    - CompositeScorer: 组合评分器，将 0 或 1 个 Predictor + 1 或 K 个 Scorer 串联
    - CompositeDetectorExtraOutput: 组合检测器附加输出
    - CompositeDetector: 组合检测器，将 0 或 1 个 Predictor + 1 或 K 个 Scorer + 1 个 Decider 串联

数据流规则:
    - 单输入算子（NumericOperator）: 接上一个算子的输出
    - 双输入算子（BiNumericOperator）: 接上一个算子的输入（x_real）+ 输出（x_pred）
    - 内部数据全程 ndarray 传递
    - 隐式适配器: 1D 分数自动 reshape 为 2D (n_samples, 1)

示例用法::

    # 组合评分器: Predictor + Scorer
    scorer = CompositeScorer(
        operators=[
            PCAPredictor(config=PCAPredictorConfig(n_components=5)),
            ResidualScorer(config=ResidualScorerConfig(metric="mse")),
        ]
    )
    scorer.fit(train_data)
    scores, eo = scorer.run(test_data)

    # 组合检测器: Predictor + Scorer + Decider
    detector = CompositeDetector(
        operators=[
            PCAPredictor(config=PCAPredictorConfig(n_components=5)),
            ResidualScorer(config=ResidualScorerConfig(metric="mse")),
            PercentileDecider(config=PercentileDeciderConfig(percentile=95.0)),
        ]
    )
    detector.fit(train_data)
    labels, eo = detector.run(test_data)
"""

import importlib
import json
from pathlib import Path
from typing import Any, Self

import numpy as np
import pandas as pd
from pydantic import BaseModel, ConfigDict, Field

from tsas.engine.operator.base import (
    BiNumericOperator,
    LearnableOperatorMixin,
    NumericData,
    NumericOperator,
    UnsupervisedNumericOperatorMixin,
)
from tsas.engine.operator.detection.base import (
    BasePredictorMixin,
    BaseDeciderMixin,
    SingleScorerMixin,
)

__all__ = [
    'CompositeScorerExtraOutput',
    'CompositeScorer',
    'CompositeDetectorExtraOutput',
    'CompositeDetector',
]

# 持久化文件名
_MANIFEST_FILE_NAME = "manifest.json"


# ============================================================================
# 辅助函数
# ============================================================================

def _ensure_2d(x: np.ndarray) -> np.ndarray:
    """
    确保数组是 2D

    如果数组是 1D，则 reshape 为 (n_samples, 1)。
    如果数组已经是 2D，则原样返回。

    Args:
        x(np.ndarray): 输入数组

    Returns:
        np.ndarray: 2D 数组
    """
    if x.ndim == 1:
        return x.reshape(-1, 1)
    return x


def _extract_main_output(result: NumericData | tuple[NumericData, Any]) -> tuple[np.ndarray, Any]:
    """
    从算子 run() 的返回值中提取主输出和 EO

    Args:
        result: 算子 run() 的返回值，可能是纯 NumericData 或 tuple[NumericData, EO]

    Returns:
        tuple[np.ndarray, Any]: (主输出 ndarray, EO 或 None)
    """
    if isinstance(result, tuple) and len(result) == 2:
        output, eo = result
    else:
        output = result
        eo = None

    # 转换为 ndarray
    if hasattr(output, 'to_numpy'):
        output = output.to_numpy()
    elif hasattr(output, 'values'):
        output = output.values

    return output, eo


# ============================================================================
# 共享管线函数
# ============================================================================

def _run_single_op(op, prev_input: np.ndarray, prev_output: np.ndarray
                   ) -> tuple[np.ndarray, Any]:
    """运行单个算子，返回 (当前输出 ndarray, EO 或 None)。

    根据 op 类型自动选择单输入/双输入调用方式。

    Args:
        op: 子算子实例
        prev_input: 上一个算子的输入
        prev_output: 上一个算子的输出

    Returns:
        tuple[np.ndarray, Any]: (当前输出 ndarray, EO 或 None)
    """
    if isinstance(op, BiNumericOperator):
        result = op.run((prev_input, prev_output))
    else:
        result = op.run(prev_output)
    return _extract_main_output(result)


def _fit_pipeline(operators: list, x: np.ndarray, skip_last: bool = False) -> np.ndarray:
    """串行训练算子列表，返回最后一个算子的输出。

    Args:
        operators: 子算子列表
        x: 初始输入数据 ndarray
        skip_last: 是否跳过最后一个算子的训练（用于 CompositeDetector 跳过 Decider）

    Returns:
        np.ndarray: 最后一个算子的输出
    """
    prev_input = x
    prev_output = x

    for i, op in enumerate(operators):
        if skip_last and i == len(operators) - 1:
            break

        # 训练（如果可训练）
        if isinstance(op, LearnableOperatorMixin):
            if isinstance(op, BiNumericOperator):
                op.fit((prev_input, prev_output))
            else:
                op.fit(prev_output)

        # 计算输出（用于下一个算子的输入）
        curr_output, _ = _run_single_op(op, prev_input, prev_output)
        curr_output = _ensure_2d(curr_output)

        prev_input = prev_output
        prev_output = curr_output

    return prev_output


def _run_pipeline(operators: list, x: np.ndarray, skip_last: bool = False
                  ) -> tuple[np.ndarray, list[Any]]:
    """串行推理算子列表，返回最终输出和 EO 列表。

    Args:
        operators: 子算子列表
        x: 初始输入数据 ndarray
        skip_last: 是否跳过最后一个算子（用于 CompositeDetector 先跑 Predictor+Scorer 再单独跑 Decider）

    Returns:
        tuple[np.ndarray, list[Any]]: (最终输出 ndarray, EO 列表)
    """
    eo_list = []
    prev_input = x
    prev_output = x

    for i, op in enumerate(operators):
        if skip_last and i == len(operators) - 1:
            break

        curr_output, eo = _run_single_op(op, prev_input, prev_output)
        eo_list.append(eo)
        curr_output = _ensure_2d(curr_output)

        prev_input = prev_output
        prev_output = curr_output

    return prev_output, eo_list


def _prepare_decider_input(prev_output: np.ndarray) -> np.ndarray:
    """将 Scorer 输出转换为 Decider 输入格式。

    仅当最后一维为 1（典型 Scorer 输出）时 ravel，否则保留原始维度。

    Args:
        prev_output: 上游算子输出

    Returns:
        np.ndarray: Decider 输入
    """
    if prev_output.ndim >= 2 and prev_output.shape[-1] == 1:
        return prev_output.ravel()
    return prev_output


def _validate_operators(operators: list, *, require_decider: bool = False
                        ) -> tuple[Any, list, Any]:
    """解析和校验子算子列表。

    Args:
        operators: 子算子列表
        require_decider: 是否要求包含 Decider（CompositeDetector 需要）

    Returns:
        tuple[predictor, scorers, decider]: 解析后的子算子组件
            - predictor: Predictor 实例或 None
            - scorers: Scorer 实例列表
            - decider: Decider 实例或 None

    Raises:
        ValueError: 配置不符合校验规则时
    """
    if not operators:
        raise ValueError("operators 不能为空")

    predictor = None
    scorers = []
    decider = None

    for i, op in enumerate(operators):
        if isinstance(op, BasePredictorMixin):
            if i != 0:
                raise ValueError(f"Predictor 必须在算子列表的第 0 位，但当前在第 {i} 位")
            if predictor is not None:
                raise ValueError("最多只能有 1 个 Predictor")
            predictor = op
        elif isinstance(op, BaseDeciderMixin):
            if not require_decider:
                raise ValueError(f"CompositeScorer 中不应包含 Decider，但第 {i} 位是 {type(op).__name__}")
            if i != len(operators) - 1:
                raise ValueError(f"Decider 必须在算子列表的最后一位，但当前在第 {i} 位")
            if decider is not None:
                raise ValueError("最多只能有 1 个 Decider")
            decider = op
        elif isinstance(op, (NumericOperator, BiNumericOperator)):
            scorers.append(op)
        else:
            raise ValueError(f"不支持的算子类型: {type(op)}")

    if require_decider:
        if decider is None:
            raise ValueError("CompositeDetector 必须有 1 个 Decider")
        if predictor is None and len(scorers) == 0:
            raise ValueError("单个 Decider 没有组合意义，请直接使用该 Decider")

    if not require_decider:
        if predictor is None and len(scorers) == 1:
            raise ValueError("无 Predictor + 单个 Scorer 没有组合意义，请直接使用该 Scorer")
        if predictor is not None and len(scorers) == 0:
            raise ValueError("有 Predictor + 无 Scorer 无法产生异常分数")

    if predictor is None and len(scorers) > 0:
        first_scorer = scorers[0]
        if isinstance(first_scorer, BiNumericOperator):
            raise ValueError(
                "无 Predictor 时，第一个 Scorer 不能是 BiNumericOperator "
                "(BiNumericOperator 需要 x_pred，但无 Predictor 时没有 x_pred 来源)"
            )

    return predictor, scorers, decider


# ============================================================================
# 持久化辅助
# ============================================================================

def _save_operators(path: Path, operators: list) -> None:
    """将子算子列表持久化到指定目录。

    同时生成 manifest.json 记录每个子算子的位置和完整类路径，供 load 时重建。

    Args:
        path: 目标目录路径
        operators: 子算子列表
    """
    path.mkdir(parents=True, exist_ok=True)

    manifest = {"operators": []}
    scorer_idx = 0

    for op in operators:
        if isinstance(op, BasePredictorMixin):
            position = "predictor"
        elif isinstance(op, BaseDeciderMixin):
            position = "decider"
        else:
            position = f"scorer_{scorer_idx}"
            scorer_idx += 1

        op_path = path / position
        op.save(op_path)

        # 记录类路径
        cls = type(op)
        class_path = f"{cls.__module__}.{cls.__qualname__}"
        manifest["operators"].append({
            "position": position,
            "class": class_path,
        })

    # 写入 manifest
    manifest_path = path / _MANIFEST_FILE_NAME
    with open(manifest_path, "w", encoding="utf-8") as f:
        json.dump(manifest, f, indent=2, ensure_ascii=False)


def _load_operators(path: str | Path) -> list:
    """从 manifest.json 读取并重建子算子列表。

    递归调用各子算子的 load 方法，天然支持嵌套 Composite 场景。

    Args:
        path: 源目录路径

    Returns:
        list: 重建的子算子列表

    Raises:
        FileNotFoundError: manifest 文件不存在
        RuntimeError: 动态 import 或实例化失败
    """
    path = Path(path)
    manifest_path = path / _MANIFEST_FILE_NAME

    if not manifest_path.exists():
        raise FileNotFoundError(f"未找到 manifest 文件: {manifest_path}")

    with open(manifest_path, "r", encoding="utf-8") as f:
        manifest = json.load(f)

    operators = []
    for entry in manifest["operators"]:
        position = entry["position"]
        class_path = entry["class"]

        # 动态 import 类
        op_cls = _import_class(class_path)

        # 递归调用子算子的 load
        op_path = path / position
        op_instance = op_cls.load(op_path)
        operators.append(op_instance)

    return operators


def _import_class(class_path: str):
    """根据完整类路径动态 import 类。

    Args:
        class_path: 完整类路径，如 'tsas.engine.operator.detection.pca.PCAPredictor'

    Returns:
        type: 导入的类

    Raises:
        RuntimeError: import 失败
    """
    parts = class_path.rsplit(".", 1)
    if len(parts) != 2:
        raise RuntimeError(f"无效的类路径格式: {class_path}")
    module_path, class_name = parts

    try:
        module = importlib.import_module(module_path)
        cls = getattr(module, class_name)
    except (ImportError, AttributeError) as e:
        raise RuntimeError(f"无法导入类 {class_path}: {e}") from e

    return cls


# ============================================================================
# 组合评分器
# ============================================================================

class CompositeScorerExtraOutput(BaseModel):
    """
    组合评分器附加输出

    聚合所有子算子的附加输出，按算子顺序存储在列表中。
    无 EO 的子算子对应位置为 None。

    Attributes:
        outputs (list[BaseModel | None]): 子算子 EO 列表，与传入的 operators 顺序对应
    """
    outputs: list[BaseModel | None] = Field(default=[],
                                            description="子算子 EO 列表，与传入的 operators 顺序对应；无 EO 的子算子对应位置为 None")

    model_config = ConfigDict(arbitrary_types_allowed=True)


class CompositeScorer(
    SingleScorerMixin[None],
    UnsupervisedNumericOperatorMixin[None],
    NumericOperator[CompositeScorerExtraOutput, None, None]
):
    """
    组合评分器

    将 0 或 1 个 Predictor + 1 或 K 个 Scorer 串联成一个评分器。

    校验规则:
        - 无 Predictor + 单个 Scorer → 非法（无组合意义）
        - 有 Predictor + 无 Scorer → 非法（无法产生分数）
        - 无 Predictor + 第一个 Scorer 是 BiNumericOperator → 非法（无 x_pred 来源）
        - Predictor 必须在第 0 位

    数据流:
        - 单输入算子: 接上一个算子的输出
        - 双输入算子: 接上一个算子的输入（x_real）+ 输出（x_pred）
        - 内部数据全程 ndarray
        - 1D 分数自动 reshape 为 2D

    Input:
        x: 特征矩阵，形状 (n_samples, n_features)，由内部编排的算子决定具体语义

    Output:
        异常分数，形状 (n_samples,)，值越大越异常。
        分数由内部编排的最后一个 Scorer 产出

    泛型参数:
        - EO: CompositeScorerExtraOutput（附加输出由 ``_eo_type`` 自动渲染）
        - C: None（无实例参数）
        - RP: None（无运行参数）
    """

    @classmethod
    def name(cls) -> str:
        """算子名称"""
        return "composite_scorer"

    @classmethod
    def version(cls) -> tuple[int, ...]:
        """返回算子版本号

        Returns:
            tuple[int, ...]: 版本号三元组 ``(1, 0, 0)``
        """
        return (1, 0, 0)

    def __init__(self, *, operators: list, oid: str | None = None, **kwargs):
        """
        初始化组合评分器

        Args:
            operators: 子算子列表，按数据流顺序排列
            oid: 算子标识符

        Raises:
            ValueError: 配置不符合校验规则时
        """
        super().__init__(oid=oid, **kwargs)

        predictor, scorers, _ = _validate_operators(operators, require_decider=False)

        self._predictor = predictor
        self._scorers = scorers
        self._operators = operators

    @property
    def predictor(self) -> BasePredictorMixin | None:
        """获取内部 Predictor 实例"""
        return self._predictor

    @property
    def scorers(self) -> list:
        """获取内部 Scorer 列表"""
        return self._scorers

    @property
    def operators(self) -> list:
        """获取所有子算子列表"""
        return self._operators

    def _fit_data(self, x: np.ndarray, params: None) -> None:
        """
        串行训练各子算子

        训练流程:
            1. 如果有 Predictor，训练 Predictor
            2. 计算 Predictor 的输出（作为第一个 Scorer 的输入）
            3. 串行训练各 Scorer，每个 Scorer 用前一个算子的输出作为训练数据

        Args:
            x (np.ndarray): 训练数据 ndarray
            params (None): 无训练参数
        """
        _fit_pipeline(self._operators, x)

    def _run_data(self, x: np.ndarray, params: None, idx: pd.Index | None = None) -> tuple[np.ndarray, CompositeScorerExtraOutput]:
        """
        串行推理各子算子

        推理流程:
            1. 如果有 Predictor，运行 Predictor
            2. 串行运行各 Scorer
            3. 聚合所有子算子的 EO

        Args:
            x: 输入数据 ndarray
            params: 无运行参数

        Returns:
            tuple[np.ndarray, CompositeScorerExtraOutput]:
                - 最终异常分数 ndarray
                - 聚合的附加输出
        """
        prev_output, eo_list = _run_pipeline(self._operators, x)

        # 最终输出应该是 1D 分数
        final_scores = prev_output.ravel()

        return final_scores, CompositeScorerExtraOutput(outputs=eo_list)

    def save(self, path: str | Path) -> None:
        """
        持久化组合评分器到指定目录

        子算子的持久化文件存放在子路径下：
            - predictor/ (如有)
            - scorer_0/, scorer_1/, ...
        同时生成 manifest.json 记录子算子类型信息。

        Args:
            path: 目标目录路径
        """
        _save_operators(Path(path), self._operators)

    @classmethod
    def load(cls, path: str | Path, *, oid: str | None = None) -> Self:
        """
        从指定目录加载组合评分器

        读取 manifest.json 获取子算子类型信息，动态 import 并递归加载各子算子。

        Args:
            path: 源目录路径
            oid: 算子标识符

        Returns:
            加载后的算子实例
        """
        operators = _load_operators(path)
        return cls(operators=operators, oid=oid)


# ============================================================================
# 组合检测器
# ============================================================================

class CompositeDetectorExtraOutput(BaseModel):
    """
    组合检测器附加输出

    聚合所有子算子的附加输出，按算子顺序存储在列表中。
    无 EO 的子算子对应位置为 None。

    Attributes:
        outputs (list[BaseModel | None]): 子算子 EO 列表，与传入的 operators 顺序对应
    """
    outputs: list[BaseModel | None] = Field(default=[],
                                            description="子算子 EO 列表，与传入的 operators 顺序对应；无 EO 的子算子对应位置为 None")

    model_config = ConfigDict(arbitrary_types_allowed=True)


class CompositeDetector(
    BaseDeciderMixin[None],
    UnsupervisedNumericOperatorMixin[None],
    NumericOperator[CompositeDetectorExtraOutput, None, None]
):
    """
    组合检测器

    将 0 或 1 个 Predictor + 1 或 K 个 Scorer + 1 个 Decider 串联成一个检测器。

    校验规则:
        - 单个 Decider → 非法（无组合意义）
        - Decider 必须在最后一位

    数据流:
        - 与 CompositeScorer 相同，最后接 Decider
        - 如果有 Scorers，Decider 接收 Scorers 的最终分数
        - 如果无 Scorer 有 Predictor，Decider 接收 Predictor 的输出

    Input:
        x: 特征矩阵，形状 (n_samples, n_features)，由内部编排的算子决定具体语义

    Output:
        0/1 异常标签，形状 (n_samples,)，1 表示异常。
        标签由内部编排的最后一个 Decider 产出

    泛型参数:
        - EO: CompositeDetectorExtraOutput（附加输出由 ``_eo_type`` 自动渲染）
        - C: None（无实例参数）
        - RP: None（无运行参数）
    """

    @classmethod
    def name(cls) -> str:
        """算子名称"""
        return "composite_detector"

    @classmethod
    def version(cls) -> tuple[int, ...]:
        """返回算子版本号

        Returns:
            tuple[int, ...]: 版本号三元组 ``(1, 0, 0)``
        """
        return (1, 0, 0)

    def __init__(self, *, operators: list, oid: str | None = None, **kwargs):
        """
        初始化组合检测器

        Args:
            operators: 子算子列表，按数据流顺序排列
            oid: 算子标识符

        Raises:
            ValueError: 配置不符合校验规则时
        """
        super().__init__(oid=oid, **kwargs)

        predictor, scorers, decider = _validate_operators(operators, require_decider=True)

        self._predictor = predictor
        self._scorers = scorers
        self._decider = decider
        self._operators = operators

    @property
    def predictor(self) -> BasePredictorMixin | None:
        """获取内部 Predictor 实例"""
        return self._predictor

    @property
    def scorers(self) -> list:
        """获取内部 Scorer 列表"""
        return self._scorers

    @property
    def decider(self):
        """获取内部 Decider 实例"""
        return self._decider

    @property
    def operators(self) -> list:
        """获取所有子算子列表"""
        return self._operators

    def _fit_data(self, x: np.ndarray, params: None) -> None:
        """
        串行训练各子算子

        训练流程:
            1. 训练 Predictor（如有）
            2. 串行训练各 Scorer
            3. 计算训练数据的最终分数
            4. 用训练分数训练 Decider

        Args:
            x: 训练数据 ndarray
            params: 无训练参数
        """
        # 训练 Predictor 和 Scorers（跳过最后一个 Decider）
        prev_output = _fit_pipeline(self._operators, x, skip_last=True)

        # 用训练分数训练 Decider
        decider_input = _prepare_decider_input(prev_output)
        if isinstance(self._decider, LearnableOperatorMixin):
            self._decider.fit(decider_input)

    def _run_data(self, x: np.ndarray, params: None, idx: pd.Index | None = None) -> tuple[np.ndarray, CompositeDetectorExtraOutput]:
        """
        串行推理各子算子

        推理流程:
            1. 运行 Predictor 和 Scorers
            2. 运行 Decider
            3. 聚合所有子算子的 EO

        Args:
            x: 输入数据 ndarray
            params: 无运行参数

        Returns:
            tuple[np.ndarray, CompositeDetectorExtraOutput]:
                - 二分类标签 ndarray
                - 聚合的附加输出
        """
        # 运行 Predictor 和 Scorers（跳过最后一个 Decider）
        prev_output, eo_list = _run_pipeline(self._operators, x, skip_last=True)

        # 运行 Decider
        decider_input = _prepare_decider_input(prev_output)
        result = self._decider.run(decider_input)
        labels, eo = _extract_main_output(result)
        eo_list.append(eo)

        return labels, CompositeDetectorExtraOutput(outputs=eo_list)

    def save(self, path: str | Path) -> None:
        """
        持久化组合检测器到指定目录

        子算子的持久化文件存放在子路径下：
            - predictor/ (如有)
            - scorer_0/, scorer_1/, ... (如有)
            - decider/
        同时生成 manifest.json 记录子算子类型信息。

        Args:
            path: 目标目录路径
        """
        _save_operators(Path(path), self._operators)

    @classmethod
    def load(cls, path: str | Path, *, oid: str | None = None) -> Self:
        """
        从指定目录加载组合检测器

        读取 manifest.json 获取子算子类型信息，动态 import 并递归加载各子算子。

        Args:
            path: 源目录路径
            oid: 算子标识符

        Returns:
            加载后的算子实例
        """
        operators = _load_operators(path)
        return cls(operators=operators, oid=oid)
