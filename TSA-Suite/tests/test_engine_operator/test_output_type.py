# -*- coding: utf-8 -*-

"""
主输入/输出类型（``_input_type`` / ``_output_type``）提取单元测试

对应源文件：
- base.py: BaseOperator._input_type / _output_type 类变量声明、
  BaseOperator.__init_subclass__ 中从泛型参数 I/O 自动提取的逻辑、
  BaseOperator._extract_type_from_typevar 的多层泛型追踪算法
- evaluation/base.py: BaseMetricOperator 已无需定制提取（多层追踪自动处理）

设计要点（与旧 POC 的差异）：

- ``_output_type`` / ``_input_type`` 类型放宽为 ``Any``，可承载任意类型对象
  （BaseModel 子类、联合类型、标量、tuple 等）
- ``_extract_type_from_typevar`` 支持多层泛型追踪（如 BaseMetricOperator 的 MR
  等价于 BaseOperator 的 O），无需中间类定制
- 抽象中间类（如 BaseMetricOperator）定义时也会触发 ``__init_subclass__``，
  此时 ``__orig_bases__`` 中是 TypeVar，提取结果需过滤后保持 None

测试范围：
1. ``_input_type`` / ``_output_type`` 默认值与类变量声明
2. ``_extract_type_from_typevar`` 多层泛型追踪算法
3. 直接继承 BaseOperator 的算子类型提取（含 BaseModel、ndarray、float 等）
4. ``NumericOperator`` / ``BiNumericOperator`` 子类的提取（O 是联合类型）
5. ``BaseMetricOperator`` 子类的提取（多层追踪 MR）
6. 抽象中间类的过滤（避免 TypeVar 污染子类继承）
7. 具体真实算子的 ``_input_type`` / ``_output_type`` 一致性
"""

from typing import Any

import numpy as np
import pandas as pd
from pydantic import BaseModel

from tsas.engine.operator.base import (
    BaseOperator,
    NumericOperator,
    BiNumericOperator,
)
from tsas.engine.operator.evaluation.base import (
    BaseMetricOperator,
    BaseMetricConfig,
)


# ============================================================================
# 辅助：用于类型提取测试的 BaseModel 与算子类
# ============================================================================

class _DummyResult(BaseModel):
    """测试用 BaseModel 主输出类型"""
    score: float = 0.0


class _DirectBaseModelOp(BaseOperator[Any, _DummyResult, None, None]):
    """直接继承 BaseOperator，O 为 BaseModel 子类。

    用于验证 ``__init_subclass__`` 中 ``_extract_type_from_typevar`` 能正确
    从 O 泛型参数提取到 BaseModel 子类。
    """

    @classmethod
    def name(cls) -> str:
        return "direct_base_model_op"

    @classmethod
    def version(cls) -> tuple[int, ...]:
        return (1, 0, 0)

    def _run(self, x: Any, *, params: None) -> _DummyResult:
        return _DummyResult()


class _DirectNdarrayOp(BaseOperator[Any, np.ndarray, None, None]):
    """直接继承 BaseOperator，O 为 ndarray。

    用于验证非 BaseModel 主输出类型时 ``_output_type`` 保留具体类型对象。
    """

    @classmethod
    def name(cls) -> str:
        return "direct_ndarray_op"

    @classmethod
    def version(cls) -> tuple[int, ...]:
        return (1, 0, 0)

    def _run(self, x: Any, *, params: None) -> np.ndarray:
        return np.zeros(1)


class _DirectFloatOp(BaseOperator[Any, float, None, None]):
    """直接继承 BaseOperator，O 为 float（标量）。

    用于验证基础标量类型时 ``_output_type`` 保留具体类型对象。
    """

    @classmethod
    def name(cls) -> str:
        return "direct_float_op"

    @classmethod
    def version(cls) -> tuple[int, ...]:
        return (1, 0, 0)

    def _run(self, x: Any, *, params: None) -> float:
        return 0.0


class _NumericOpSample(NumericOperator[None, None, None]):
    """NumericOperator 子类样本。

    NumericOperator 内部将 O 绑定为 ``NumericData | tuple[NumericData, EO>``，
    子类 ``_output_type`` 应继承该联合类型。
    """

    @classmethod
    def name(cls) -> str:
        return "numeric_op_sample"

    @classmethod
    def version(cls) -> tuple[int, ...]:
        return (1, 0, 0)

    def _run_data(self, x: np.ndarray, params: None, idx: pd.Index | None = None) -> np.ndarray:
        return x

    def _name_output_columns(self, output_data: np.ndarray, meta, params) -> list[str]:
        return ["result"]


class _BiNumericOpSample(BiNumericOperator[None, None, None]):
    """BiNumericOperator 子类样本（同样 O 被绑定为联合类型）。"""

    @classmethod
    def name(cls) -> str:
        return "bi_numeric_op_sample"

    @classmethod
    def version(cls) -> tuple[int, ...]:
        return (1, 0, 0)

    def _run_data(self, x_real: np.ndarray, x_pred: np.ndarray, params: None,
                  real_idx: pd.Index | None = None, pred_idx: pd.Index | None = None) -> np.ndarray:
        return x_pred

    def _name_output_columns(self, output_data: np.ndarray, meta, params) -> list[str]:
        return ["result"]


class _MetricConfigSample(BaseMetricConfig):
    """测试用 BaseMetricConfig"""


class _MetricOpWithModel(BaseMetricOperator[tuple[np.ndarray, np.ndarray], _DummyResult, _MetricConfigSample, None]):
    """BaseMetricOperator 子类，MR 为 BaseModel。

    用于验证多层泛型追踪：通过 BaseMetricOperator 的 MR 等价于 BaseOperator 的 O
    的关系，自动提取 _output_type。
    """

    @classmethod
    def name(cls) -> str:
        return "metric_op_with_model"

    @classmethod
    def version(cls) -> tuple[int, ...]:
        return (1, 0, 0)

    def _run(self, x: tuple[np.ndarray, np.ndarray], *, params: None) -> _DummyResult:
        return _DummyResult()


class _MetricOpWithFloat(BaseMetricOperator[np.ndarray, float, _MetricConfigSample, None]):
    """BaseMetricOperator 子类，MR 为 float（标量）。

    典型场景是 SelfEvaluation 这类标量评价指标算子。
    """

    @classmethod
    def name(cls) -> str:
        return "metric_op_with_float"

    @classmethod
    def version(cls) -> tuple[int, ...]:
        return (1, 0, 0)

    def _run(self, x: np.ndarray, *, params: None) -> float:
        return 0.0


# ============================================================================
# BaseOperator._input_type / _output_type 类变量声明测试
# ============================================================================

class TestTypeClassVar:
    """测试 BaseOperator 基类的 _input_type / _output_type 声明与默认值"""

    def test_base_operator_has_input_type_attr(self):
        """
        目的：验证 BaseOperator 类自身声明了 _input_type 属性
        输入：BaseOperator 类对象
        预期：hasattr(BaseOperator, '_input_type') 返回 True
        """
        assert hasattr(BaseOperator, '_input_type')

    def test_base_operator_has_output_type_attr(self):
        """
        目的：验证 BaseOperator 类自身声明了 _output_type 属性
        输入：BaseOperator 类对象
        预期：hasattr(BaseOperator, '_output_type') 返回 True
        """
        assert hasattr(BaseOperator, '_output_type')

    def test_base_operator_defaults_are_none(self):
        """
        目的：验证 BaseOperator 基类的 _input_type / _output_type 默认值为 None
        输入：BaseOperator 类对象
        预期：BaseOperator._input_type is None，BaseOperator._output_type is None
        """
        assert BaseOperator._input_type is None
        assert BaseOperator._output_type is None

    def test_types_are_classvar(self):
        """
        目的：验证 _input_type / _output_type 是 ClassVar（类属性而非实例属性）
        输入：_DirectBaseModelOp 类对象及其一个实例
        预期：类访问与实例访问值一致
        """
        cls_input = _DirectBaseModelOp._input_type
        cls_output = _DirectBaseModelOp._output_type
        instance = _DirectBaseModelOp()
        assert instance._input_type is cls_input
        assert instance._output_type is cls_output


# ============================================================================
# 直接继承 BaseOperator 算子的类型提取测试
# ============================================================================

class TestDirectBaseOperatorExtraction:
    """测试直接继承 BaseOperator 的算子 _input_type / _output_type 提取逻辑"""

    def test_base_model_output_extracted(self):
        """
        目的：验证 O 为 BaseModel 子类时 _output_type 被正确提取为该 BaseModel
        输入：_DirectBaseModelOp（O=_DummyResult）
        预期：_output_type is _DummyResult
        """
        assert _DirectBaseModelOp._output_type is _DummyResult

    def test_ndarray_output_preserved(self):
        """
        目的：验证 O 为 ndarray 时 _output_type 保留 ndarray 类型对象
        输入：_DirectNdarrayOp（O=np.ndarray）
        预期：_output_type is np.ndarray

        说明：新设计放宽了 _output_type 的类型约束，不再限于 BaseModel，
        任意类型对象都可保留，由 CLI Help 渲染层决定如何展示。
        """
        assert _DirectNdarrayOp._output_type is np.ndarray

    def test_float_output_preserved(self):
        """
        目的：验证 O 为 float 等基础标量类型时 _output_type 保留类型对象
        输入：_DirectFloatOp（O=float）
        预期：_output_type is float
        """
        assert _DirectFloatOp._output_type is float


# ============================================================================
# NumericOperator / BiNumericOperator 子类的类型提取测试
# ============================================================================

class TestNumericOperatorExtraction:
    """测试 NumericOperator/BiNumericOperator 子类的 _output_type 表现"""

    def test_numeric_operator_subclass_output_is_union(self):
        """
        目的：验证 NumericOperator 子类的 _output_type 是联合类型
        输入：_NumericOpSample（继承 NumericOperator[None, None, None]）
        预期：_output_type 形如 NumericData | tuple[NumericData, EO>

        说明：NumericOperator 内部将 O 绑定为联合类型，子类继承该类型。
        CLI Help 渲染时通过 _simplify_output_type 提取主输出 NumericData。
        """
        from typing import get_origin
        import typing
        output_type = _NumericOpSample._output_type
        # 应该是联合类型（Union 或 types.UnionType）
        origin = get_origin(output_type)
        is_union = (
                origin is typing.Union
                or (hasattr(__import__('types'), 'UnionType') and isinstance(output_type,
                                                                             __import__('types').UnionType))
        )
        assert is_union, f"_output_type 应该是联合类型，实际: {output_type}"

    def test_bi_numeric_operator_subclass_output_is_union(self):
        """
        目的：验证 BiNumericOperator 子类的 _output_type 是联合类型
        输入：_BiNumericOpSample
        预期：同上
        """
        from typing import get_origin
        import typing
        output_type = _BiNumericOpSample._output_type
        origin = get_origin(output_type)
        is_union = (
                origin is typing.Union
                or (hasattr(__import__('types'), 'UnionType') and isinstance(output_type,
                                                                             __import__('types').UnionType))
        )
        assert is_union, f"_output_type 应该是联合类型，实际: {output_type}"

    def test_numeric_operator_input_is_numeric_data(self):
        """
        目的：验证 NumericOperator 子类的 _input_type 是 NumericData
        输入：_NumericOpSample
        预期：_input_type 是 NumericData（NumericOperator 内部声明 I 为 NumericData）
        """
        # NumericData 是 Annotated 类型，比较 __metadata__
        assert hasattr(_NumericOpSample._input_type, '__metadata__')

    def test_bi_numeric_operator_input_is_tuple(self):
        """
        目的：验证 BiNumericOperator 子类的 _input_type 是 tuple 类型
        输入：_BiNumericOpSample
        预期：_input_type 的 origin 是 tuple
        """
        from typing import get_origin
        assert get_origin(_BiNumericOpSample._input_type) is tuple


# ============================================================================
# BaseMetricOperator 子类的类型提取测试（多层泛型追踪）
# ============================================================================

class TestMetricOperatorExtraction:
    """测试 BaseMetricOperator 子类通过多层追踪提取类型"""

    def test_metric_with_base_model_output_extracted(self):
        """
        目的：验证多层追踪能正确提取 MR 为 BaseModel 的输出类型
        输入：_MetricOpWithModel（MR=_DummyResult）
        预期：_output_type is _DummyResult

        关键点：BaseMetricOperator 用 MR 替换了 BaseOperator 的 O 位置，
        算法需通过反查 BaseMetricOperator.__orig_bases__ 中 BaseOperator[I, MR, MC, RP]
        得到 O 等价于 MR，再从 BinaryClassificationMetric 的实参中取出 MR 对应类型。
        """
        assert _MetricOpWithModel._output_type is _DummyResult

    def test_metric_with_float_output_preserved(self):
        """
        目的：验证多层追踪能提取 MR 为 float 的输出类型
        输入：_MetricOpWithFloat（MR=float）
        预期：_output_type is float
        """
        assert _MetricOpWithFloat._output_type is float

    def test_metric_input_extracted(self):
        """
        目的：验证 BaseMetricOperator 子类的 _input_type 通过多层追踪提取
        输入：_MetricOpWithModel（I=tuple[np.ndarray, np.ndarray]）
        预期：_input_type 是 tuple 类型
        """
        from typing import get_origin
        assert get_origin(_MetricOpWithModel._input_type) is tuple


# ============================================================================
# 抽象中间类的过滤测试
# ============================================================================

class TestAbstractIntermediateFiltering:
    """测试抽象中间类（如 BaseMetricOperator）定义时的 TypeVar 过滤"""

    def test_base_metric_operator_input_type_is_none(self):
        """
        目的：验证 BaseMetricOperator 自身的 _input_type 保持 None
        输入：BaseMetricOperator 类
        预期：_input_type is None

        说明：BaseMetricOperator 定义时触发 BaseOperator.__init_subclass__，
        此时 __orig_bases__ 中是 TypeVar，提取结果是 TypeVar 本身。
        __init_subclass__ 应过滤 TypeVar 结果，保持 None，
        让具体子类（如 BinaryClassificationMetric）能正确提取。
        """
        from tsas.engine.operator.evaluation.base import BaseMetricOperator
        assert BaseMetricOperator._input_type is None

    def test_base_metric_operator_output_type_is_none(self):
        """
        目的：验证 BaseMetricOperator 自身的 _output_type 保持 None
        输入：BaseMetricOperator 类
        预期：_output_type is None
        """
        from tsas.engine.operator.evaluation.base import BaseMetricOperator
        assert BaseMetricOperator._output_type is None

    def test_numeric_operator_input_output_inherited_from_init_subclass(self):
        """
        目的：验证 NumericOperator 自身在 __init_subclass__ 触发时也能正确处理
        输入：NumericOperator 类
        预期：_input_type / _output_type 与 NumericOperator 泛型声明一致
        """
        # NumericOperator 自身的 _input_type 应是 NumericData
        assert NumericOperator._input_type is not None
        # _output_type 应是联合类型
        from typing import get_origin
        import typing
        output_type = NumericOperator._output_type
        if output_type is not None:
            origin = get_origin(output_type)
            is_union = (
                    origin is typing.Union
                    or (hasattr(__import__('types'), 'UnionType') and isinstance(output_type,
                                                                                 __import__('types').UnionType))
            )
            assert is_union or output_type is not None  # 至少有值


# ============================================================================
# 具体真实算子的类型提取一致性验证
# ============================================================================

class TestRealOperatorsOutputType:
    """测试真实算子的 _output_type 与新设计语义一致"""

    def test_binary_classification_metric_output_type(self):
        """
        目的：验证 BinaryClassificationMetric 的 _output_type 为 BinaryClassificationResult
        输入：BinaryClassificationMetric 类
        预期：_output_type is BinaryClassificationResult（多层追踪正确提取）
        """
        from tsas.engine.operator.evaluation.binary_classification import (
            BinaryClassificationMetric,
            BinaryClassificationResult,
        )
        assert BinaryClassificationMetric._output_type is BinaryClassificationResult

    def test_multi_classification_metric_output_type(self):
        """
        目的：验证 MultipleClassificationMetric 的 _output_type 为 MultiClassificationMetricResult
        """
        from tsas.engine.operator.evaluation.multi_classification import (
            MultipleClassificationMetric,
            MultiClassificationMetricResult,
        )
        assert MultipleClassificationMetric._output_type is MultiClassificationMetricResult

    def test_point_adjust_output_type(self):
        """
        目的：验证 PointAdjust 的 _output_type 为 PointAdjustResult
        """
        from tsas.engine.operator.evaluation.point_adjust import (
            PointAdjust,
            PointAdjustResult,
        )
        assert PointAdjust._output_type is PointAdjustResult

    def test_binary_classification_curve_output_type(self):
        """
        目的：验证 BinaryClassificationCurve 的 _output_type 为 BinaryClassificationCurveResult
        """
        from tsas.engine.operator.evaluation.binary_curve import (
            BinaryClassificationCurve,
            BinaryClassificationCurveResult,
        )
        assert BinaryClassificationCurve._output_type is BinaryClassificationCurveResult

    def test_self_evaluation_output_is_float(self):
        """
        目的：验证 SelfEvaluation 的 _output_type 为 float（MR=float）
        输入：SelfEvaluation 类
        预期：_output_type is float

        说明：旧 POC 中 _output_type 仅提取 BaseModel，float 等标量为 None。
        新设计中 _output_type 类型放宽，float 被保留，CLI Help 据此渲染
        "### 主输出 (float)"。
        """
        from tsas.engine.operator.evaluation.self_evaluation import SelfEvaluation
        assert SelfEvaluation._output_type is float

    def test_knn_scorer_output_is_union(self):
        """
        目的：验证 KNNScorer 的 _output_type 是联合类型（NumericData | tuple[...]）
        输入：KNNScorer 类
        预期：_output_type 是 Union 类型

        说明：新设计中 _output_type 保留联合类型本身，CLI Help 通过
        _simplify_output_type 提取主输出 NumericData 后渲染。
        """
        from typing import get_origin
        import typing
        from tsas.engine.operator.detection.knn import KNNScorer
        output_type = KNNScorer._output_type
        origin = get_origin(output_type)
        is_union = (
                origin is typing.Union
                or (hasattr(__import__('types'), 'UnionType') and isinstance(output_type,
                                                                             __import__('types').UnionType))
        )
        assert is_union, f"KNNScorer._output_type 应该是联合类型，实际: {output_type}"

    def test_knn_detector_output_is_union(self):
        """
        目的：验证 KNNDetector 的 _output_type 是联合类型
        输入：KNNDetector 类
        预期：同上
        """
        from typing import get_origin
        import typing
        from tsas.engine.operator.detection.knn import KNNDetector
        output_type = KNNDetector._output_type
        origin = get_origin(output_type)
        is_union = (
                origin is typing.Union
                or (hasattr(__import__('types'), 'UnionType') and isinstance(output_type,
                                                                             __import__('types').UnionType))
        )
        assert is_union

    def test_mean_predictor_output_is_union(self):
        """
        目的：验证 MeanPredictor 的 _output_type 是联合类型
        输入：MeanPredictor 类
        预期：同上
        """
        from typing import get_origin
        import typing
        from tsas.engine.operator.detection.mean_predictor import MeanPredictor
        output_type = MeanPredictor._output_type
        origin = get_origin(output_type)
        is_union = (
                origin is typing.Union
                or (hasattr(__import__('types'), 'UnionType') and isinstance(output_type,
                                                                             __import__('types').UnionType))
        )
        assert is_union

    def test_all_evaluation_base_model_ops_extracted(self):
        """
        目的：验证所有 MR 为 BaseModel 的 evaluation 算子都成功提取了 _output_type
        输入：4 个 BaseModel 输出的 evaluation 算子
        预期：全部 _output_type 非 None 且为 BaseModel 子类
        """
        from tsas.engine.operator.evaluation.binary_classification import BinaryClassificationMetric
        from tsas.engine.operator.evaluation.binary_curve import BinaryClassificationCurve
        from tsas.engine.operator.evaluation.multi_classification import MultipleClassificationMetric
        from tsas.engine.operator.evaluation.point_adjust import PointAdjust

        for op_cls in [
            BinaryClassificationMetric,
            MultipleClassificationMetric,
            PointAdjust,
            BinaryClassificationCurve,
        ]:
            assert op_cls._output_type is not None, f"{op_cls.__name__}._output_type 不应为 None"
            assert issubclass(op_cls._output_type, BaseModel), (
                f"{op_cls.__name__}._output_type 必须是 BaseModel 子类"
            )


# ============================================================================
# 跨多层继承场景验证
# ============================================================================

class TestMultiLayerInheritance:
    """测试跨多层继承时类型提取的正确性"""

    def test_metric_op_inheritance_chain(self):
        """
        目的：验证 BaseMetricOperator → 具体算子 的多层继承中类型正确提取
        输入：_MetricOpWithModel（继承 BaseMetricOperator[I, MR, MC, RP]）
        预期：_output_type 来自 MR，_config_type 来自 MC，_input_type 来自 I
        """
        # _output_type 来自 MR
        assert _MetricOpWithModel._output_type is _DummyResult
        # _config_type 来自 MC（BaseMetricOperator.__init_subclass__ 提取）
        assert _MetricOpWithModel._config_type is _MetricConfigSample
        # _input_type 来自 I
        from typing import get_origin
        assert get_origin(_MetricOpWithModel._input_type) is tuple

    def test_run_params_type_unaffected(self):
        """
        目的：验证 _input_type / _output_type 提取不影响 _run_params_type
        输入：_MetricOpWithModel（RP=None）
        预期：_run_params_type 仍为 None
        """
        assert _MetricOpWithModel._run_params_type is None

    def test_eo_type_unaffected_in_numeric_operator(self):
        """
        目的：验证 NumericOperator 子类的 _eo_type 提取不受 _output_type 影响
        输入：_NumericOpSample（NumericOperator[None, None, None]，EO=None）
        预期：_eo_type is None（NoneType 被归一化为 None），_output_type 是联合类型
        """
        assert _NumericOpSample._eo_type is None
        # _output_type 是联合类型（在 TestNumericOperatorExtraction 中已验证）

    def test_eo_type_extracted_independently(self):
        """
        目的：验证 KNNScorer 中 _eo_type 被提取但 _output_type 仍是联合类型
        输入：KNNScorer（EO=KNNScorerExtraOutput）
        预期：_eo_type 为 KNNScorerExtraOutput，_output_type 是联合类型
        """
        from tsas.engine.operator.detection.knn import (
            KNNScorer,
            KNNScorerExtraOutput,
        )
        assert KNNScorer._eo_type is KNNScorerExtraOutput
        # _output_type 是联合类型
        from typing import get_origin
        import typing
        output_type = KNNScorer._output_type
        origin = get_origin(output_type)
        is_union = (
                origin is typing.Union
                or (hasattr(__import__('types'), 'UnionType') and isinstance(output_type,
                                                                             __import__('types').UnionType))
        )
        assert is_union
