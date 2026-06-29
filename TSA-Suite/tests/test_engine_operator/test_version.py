# -*- coding: utf-8 -*-

"""
版本号系统单元测试

对应源文件：
- base.py: BaseOperator.version(), BaseOperator.min_compatible_version(),
  BaseOperator._validate_version_tuple(), BaseOperator._format_version(),
  BaseOperator.__init_subclass__ 版本校验,
  BaseOperator.save() 版本持久化, BaseOperator.load() 版本兼容性校验

测试范围：
1. _validate_version_tuple 静态方法的格式校验
   - 合法版本元组通过校验
   - 非 tuple 类型 → TypeError
   - 空元组 → TypeError
   - 含非 int 元素 → TypeError
2. __init_subclass__ 中的版本校验
   - 非抽象子类必须实现 version()
   - 版本格式非法时 raise TypeError
   - min_compatible_version() > version() 时 raise ValueError
   - 抽象中间类跳过版本校验
4. version() 和 min_compatible_version() 的默认行为
   - 默认 min_compatible_version() 返回 version()
   - 子类可 override min_compatible_version()
5. 所有具体算子的 version() 返回值校验
   - 返回非空 tuple[int, ...]
   - 不变量 min_compatible_version() <= version() 成立
6. save() / load() 版本持久化与兼容性校验
   - save() 创建 version.json 且内容与 cls.version() 一致
   - load() 版本匹配时无警告
   - load() saved > version() 时产生警告
   - load() saved < min_compatible_version() 时产生警告
   - load() 无 version.json 时跳过校验（兼容旧版数据）
   - 兼容区间内和边界值无警告
   - save/load 往返后 config + version 均正确恢复
7. _format_version 静态方法的格式化
   - 标准三元组 (1, 0, 0) → "1.0.0"
   - 单元素 (42,) → "42"
   - 多位数字 (1, 0, 99) → "1.0.99"
   - 长元组 (1, 2, 3, 4, 5) → "1.2.3.4.5"
"""

import json

import pytest
from loguru import logger

from tsas.engine.operator.base import (
    BaseOperator,
    NumericOperator,
)


@pytest.fixture
def loguru_capture():
    """捕获 loguru 日志消息的 fixture

    loguru 使用自己的 sink 机制而非 Python 标准 logging 或 warnings，
    因此 pytest 的 caplog/capfd 无法直接捕获。
    此 fixture 通过添加临时 sink 来捕获日志消息。
    """
    messages = []
    handler_id = logger.add(lambda msg: messages.append(str(msg)), format="{message}")
    yield messages
    logger.remove(handler_id)


# ============================================================================
# 辅助：用于版本校验测试的算子类
# ============================================================================

class _ValidOperator(NumericOperator[None, None, None]):
    """版本号合法的具体算子"""

    @classmethod
    def name(cls) -> str:
        return "valid_op"

    @classmethod
    def version(cls) -> tuple[int, ...]:
        return (1, 0, 0)

    def _run_data(self, x, params, idx=None):
        return x

    def _name_output_columns(self, output_data, meta, params):
        return ["result"]


class _CustomMinCompatOp(NumericOperator[None, None, None]):
    """override min_compatible_version 的具体算子"""

    @classmethod
    def name(cls) -> str:
        return "custom_min_compat"

    @classmethod
    def version(cls) -> tuple[int, ...]:
        return (2, 0, 0)

    @classmethod
    def min_compatible_version(cls) -> tuple[int, ...]:
        return (1, 0, 0)

    def _run_data(self, x, params, idx=None):
        return x

    def _name_output_columns(self, output_data, meta, params):
        return ["result"]


# ============================================================================
# _validate_version_tuple 格式校验测试
# ============================================================================

class TestValidateVersionTuple:
    """测试 BaseOperator._validate_version_tuple 静态校验方法"""

    def test_valid_triplet_passes(self):
        """
        目的：验证标准三元组 (1, 0, 0) 通过校验
        输入：(1, 0, 0), source="test"
        预期：无异常抛出
        """
        BaseOperator._validate_version_tuple((1, 0, 0), "test")

    def test_valid_single_element_passes(self):
        """
        目的：验证单元素元组 (42,) 也能通过校验
        输入：(42,), source="test"
        预期：无异常抛出（不限制元组长度，只要求非空且全部 int）
        """
        BaseOperator._validate_version_tuple((42,), "test")

    def test_valid_long_tuple_passes(self):
        """
        目的：验证长元组 (1, 2, 3, 4, 5) 也能通过校验
        输入：(1, 2, 3, 4, 5), source="test"
        预期：无异常抛出
        """
        BaseOperator._validate_version_tuple((1, 2, 3, 4, 5), "test")

    def test_non_tuple_raises_type_error(self):
        """
        目的：验证非 tuple 类型（如 list）抛出 TypeError
        输入：[1, 0, 0], source="MyOp.version()"
        预期：TypeError，消息包含 "必须返回 tuple[int, ...]" 和来源描述
        """
        with pytest.raises(TypeError, match="必须返回 tuple"):
            BaseOperator._validate_version_tuple([1, 0, 0], "MyOp.version()")

    def test_empty_tuple_raises_type_error(self):
        """
        目的：验证空元组抛出 TypeError
        输入：(), source="MyOp.version()"
        预期：TypeError，消息包含 "不能为空"
        """
        with pytest.raises(TypeError, match="不能为空"):
            BaseOperator._validate_version_tuple((), "MyOp.version()")

    def test_non_int_element_raises_type_error(self):
        """
        目的：验证含非 int 元素的元组抛出 TypeError
        输入：(1, "2", 0), source="MyOp.version()"
        预期：TypeError，消息包含元素索引和实际类型
        """
        with pytest.raises(TypeError, match="必须是 int"):
            BaseOperator._validate_version_tuple((1, "2", 0), "MyOp.version()")

    def test_float_element_raises_type_error(self):
        """
        目的：验证含 float 元素的元组抛出 TypeError（float 不算 int）
        输入：(1, 2.0, 0), source="Test.version()"
        预期：TypeError，消息包含元素索引 1 和 "float"
        """
        with pytest.raises(TypeError, match="float"):
            BaseOperator._validate_version_tuple((1, 2.0, 0), "Test.version()")

    def test_none_raises_type_error(self):
        """
        目的：验证 None 输入抛出 TypeError
        输入：None, source="Test.version()"
        预期：TypeError，消息包含 "必须返回 tuple"
        """
        with pytest.raises(TypeError, match="必须返回 tuple"):
            BaseOperator._validate_version_tuple(None, "Test.version()")

    def test_string_raises_type_error(self):
        """
        目的：验证字符串输入抛出 TypeError
        输入："1.0.0", source="Test.version()"
        预期：TypeError，因为 str 不是 tuple 类型
        """
        with pytest.raises(TypeError, match="必须返回 tuple"):
            BaseOperator._validate_version_tuple("1.0.0", "Test.version()")


# ============================================================================
# __init_subclass__ 版本校验测试
# ============================================================================

class TestInitSubclassVersionValidation:
    """测试 __init_subclass__ 中的版本校验逻辑"""

    def test_non_tuple_version_raises_type_error(self):
        """
        目的：验证 version() 返回非 tuple 时，类定义阶段就抛出 TypeError
        输入：version() 返回 list [1, 0, 0]
        预期：TypeError，消息包含类名和 "必须返回 tuple"
        """
        with pytest.raises(TypeError, match="必须返回 tuple"):
            class _BadListVersion(NumericOperator[None, None, None]):
                @classmethod
                def name(cls) -> str:
                    return "bad_list_version"

                @classmethod
                def version(cls) -> tuple[int, ...]:
                    return [1, 0, 0]  # type: ignore

                def _run_data(self, x, params, idx=None):
                    return x

                def _name_output_columns(self, output_data, meta, params):
                    return ["result"]

    def test_empty_version_raises_type_error(self):
        """
        目的：验证 version() 返回空元组时抛出 TypeError
        输入：version() 返回 ()
        预期：TypeError，消息包含 "不能为空"
        """
        with pytest.raises(TypeError, match="不能为空"):
            class _BadEmptyVersion(NumericOperator[None, None, None]):
                @classmethod
                def name(cls) -> str:
                    return "bad_empty_version"

                @classmethod
                def version(cls) -> tuple[int, ...]:
                    return ()

                def _run_data(self, x, params, idx=None):
                    return x

                def _name_output_columns(self, output_data, meta, params):
                    return ["result"]

    def test_non_int_version_raises_type_error(self):
        """
        目的：验证 version() 含非 int 元素时抛出 TypeError
        输入：version() 返回 (1, "0", 0)
        预期：TypeError，消息包含 "必须是 int"
        """
        with pytest.raises(TypeError, match="必须是 int"):
            class _BadNonIntVersion(NumericOperator[None, None, None]):
                @classmethod
                def name(cls) -> str:
                    return "bad_non_int_version"

                @classmethod
                def version(cls) -> tuple[int, ...]:
                    return (1, "0", 0)  # type: ignore

                def _run_data(self, x, params, idx=None):
                    return x

                def _name_output_columns(self, output_data, meta, params):
                    return ["result"]

    def test_min_compat_greater_than_version_raises_value_error(self):
        """
        目的：验证 min_compatible_version() > version() 时抛出 ValueError
        输入：version() = (1, 0, 0), min_compatible_version() = (2, 0, 0)
        预期：ValueError，消息包含 "违反不变量"
        """
        with pytest.raises(ValueError, match="违反不变量"):
            class _BadInvariant(NumericOperator[None, None, None]):
                @classmethod
                def name(cls) -> str:
                    return "bad_invariant"

                @classmethod
                def version(cls) -> tuple[int, ...]:
                    return (1, 0, 0)

                @classmethod
                def min_compatible_version(cls) -> tuple[int, ...]:
                    return (2, 0, 0)

                def _run_data(self, x, params, idx=None):
                    return x

                def _name_output_columns(self, output_data, meta, params):
                    return ["result"]

    def test_abstract_class_skips_version_check(self):
        """
        目的：验证抽象中间类跳过版本校验
        输入：一个不实现 version() 的抽象中间类
        预期：类定义不报错（inspect.isabstract 返回 True 时跳过）
        """
        from abc import abstractmethod

        class _AbstractIntermediate(NumericOperator[None, None, None]):
            """抽象中间类，不实现 version()，不应触发校验"""

            @classmethod
            def name(cls) -> str:
                return "abstract_intermediate"

            @abstractmethod
            def some_method(self):
                ...

        # 类定义成功，不抛出异常
        assert True

    def test_valid_version_passes(self):
        """
        目的：验证合法版本号的类定义成功
        输入：version() = (1, 0, 0)
        预期：类定义成功，version() 返回 (1, 0, 0)
        """
        assert _ValidOperator.version() == (1, 0, 0)


# ============================================================================
# min_compatible_version() 默认行为测试
# ============================================================================

class TestMinCompatibleVersion:
    """测试 min_compatible_version() 的默认实现和 override"""

    def test_default_returns_version(self):
        """
        目的：验证默认 min_compatible_version() 返回 version()
        输入：_ValidOperator, version() = (1, 0, 0)
        预期：min_compatible_version() == (1, 0, 0)
        """
        assert _ValidOperator.min_compatible_version() == _ValidOperator.version()
        assert _ValidOperator.min_compatible_version() == (1, 0, 0)

    def test_override_min_compatible_version(self):
        """
        目的：验证子类可 override min_compatible_version()
        输入：_CustomMinCompatOp, version() = (2, 0, 0), min_compatible_version() = (1, 0, 0)
        预期：min_compatible_version() == (1, 0, 0) != version()
        """
        assert _CustomMinCompatOp.version() == (2, 0, 0)
        assert _CustomMinCompatOp.min_compatible_version() == (1, 0, 0)
        assert _CustomMinCompatOp.min_compatible_version() <= _CustomMinCompatOp.version()

    def test_min_compat_equals_version_invariant(self):
        """
        目的：验证默认情况下 min_compatible_version() == version() 满足不变量
        输入：_ValidOperator
        预期：min_compatible_version() <= version() 成立
        """
        assert _ValidOperator.min_compatible_version() <= _ValidOperator.version()


# ============================================================================
# 版本号比较规则测试
# ============================================================================

class TestVersionComparison:
    """测试版本元组的比较规则（Python tuple 字典序比较）"""

    def test_patch_comparison(self):
        """
        目的：验证 patch 版本号比较
        输入：(1, 0, 3) < (1, 0, 4)
        预期：True
        """
        assert (1, 0, 3) < (1, 0, 4)

    def test_minor_comparison(self):
        """
        目的：验证 minor 版本号比较
        输入：(1, 0, 9) < (1, 1, 0)
        预期：True
        """
        assert (1, 0, 9) < (1, 1, 0)

    def test_major_comparison(self):
        """
        目的：验证 major 版本号比较
        输入：(1, 9, 9) < (2, 0, 0)
        预期：True
        """
        assert (1, 9, 9) < (2, 0, 0)

    def test_equal_versions(self):
        """
        目的：验证相同版本号比较
        输入：(1, 0, 0) == (1, 0, 0)
        预期：True
        """
        assert (1, 0, 0) == (1, 0, 0)


# ============================================================================
# 具体算子版本号一致性测试
# ============================================================================

class TestConcreteOperatorVersions:
    """测试所有具体算子的版本号符合规范"""

    def test_detection_operators_version(self):
        """
        目的：验证 detection 模块所有算子版本号为合法非空 tuple[int, ...]
        输入：detection 模块全部 19 个具体算子
        预期：version() 返回非空 tuple，min_compatible_version() <= version()
        """
        from tsas.engine.operator.detection.zscore import ZScoreScorer, ZScoreDetector
        from tsas.engine.operator.detection.knn import KNNScorer, KNNDetector
        from tsas.engine.operator.detection.pca import PCAPredictor, PCAScorer, PCADetector
        from tsas.engine.operator.detection.xihe import XiHeGammaScorer
        from tsas.engine.operator.detection.cicada import CICADAPredictor, CICADAScorer
        from tsas.engine.operator.detection.mean_predictor import MeanPredictor
        from tsas.engine.operator.detection.mean_scorer import MeanScorer
        from tsas.engine.operator.detection.residual_scorer import ResidualScorer, ResidualMapScorer
        from tsas.engine.operator.detection.threshold_decider import ThresholdDecider
        from tsas.engine.operator.detection.percentile_decider import PercentileDecider
        from tsas.engine.operator.detection.composite import CompositeScorer, CompositeDetector

        all_ops = [
            ZScoreScorer, ZScoreDetector,
            KNNScorer, KNNDetector,
            PCAPredictor, PCAScorer, PCADetector,
            XiHeGammaScorer,
            CICADAPredictor, CICADAScorer,
            MeanPredictor, MeanScorer,
            ResidualScorer, ResidualMapScorer,
            ThresholdDecider, PercentileDecider,
            CompositeScorer, CompositeDetector,
        ]

        for op_cls in all_ops:
            v = op_cls.version()
            # 版本号为非空 tuple
            assert isinstance(v, tuple), f"{op_cls.__name__}.version() 不是 tuple"
            assert len(v) > 0, f"{op_cls.__name__}.version() 为空元组"
            assert all(isinstance(e, int) for e in v), f"{op_cls.__name__}.version() 含非 int 元素"
            # 不变量
            min_v = op_cls.min_compatible_version()
            assert min_v <= v, (
                f"{op_cls.__name__}: min_compatible_version()={min_v} > version()={v}"
            )

    def test_evaluation_operators_version(self):
        """
        目的：验证 evaluation 模块所有算子版本号为合法非空 tuple[int, ...]
        输入：evaluation 模块全部 5 个具体算子
        预期：version() 返回非空 tuple，min_compatible_version() <= version()
        """
        from tsas.engine.operator.evaluation.binary_classification import BinaryClassificationMetric
        from tsas.engine.operator.evaluation.binary_curve import BinaryClassificationCurve
        from tsas.engine.operator.evaluation.multi_classification import MultipleClassificationMetric
        from tsas.engine.operator.evaluation.point_adjust import PointAdjust
        from tsas.engine.operator.evaluation.self_evaluation import SelfEvaluation

        all_ops = [
            BinaryClassificationMetric,
            BinaryClassificationCurve,
            MultipleClassificationMetric,
            PointAdjust,
            SelfEvaluation,
        ]

        for op_cls in all_ops:
            v = op_cls.version()
            assert isinstance(v, tuple), f"{op_cls.__name__}.version() 不是 tuple"
            assert len(v) > 0, f"{op_cls.__name__}.version() 为空元组"
            assert all(isinstance(e, int) for e in v), f"{op_cls.__name__}.version() 含非 int 元素"
            min_v = op_cls.min_compatible_version()
            assert min_v <= v, (
                f"{op_cls.__name__}: min_compatible_version()={min_v} > version()={v}"
            )

    def test_feature_construction_operators_version(self):
        """
        目的：验证 feature/construction 模块所有算子版本号为合法非空 tuple[int, ...]
        输入：feature/construction 模块全部具体算子
        预期：version() 返回非空 tuple，min_compatible_version() <= version()
        """
        from tsas.engine.operator.feature.construction import signal_feature, simple_feature, smooth_feature

        # 收集所有具体算子类
        all_ops = []
        for module in [signal_feature, simple_feature, smooth_feature]:
            for attr_name in dir(module):
                attr = getattr(module, attr_name)
                if (isinstance(attr, type)
                        and issubclass(attr, BaseOperator)
                        and not getattr(attr, '__abstractmethods__', None)
                        and hasattr(attr, 'version')):
                    all_ops.append(attr)

        assert len(all_ops) >= 30, f"期望至少 30 个 feature 算子，实际找到 {len(all_ops)}"

        for op_cls in all_ops:
            v = op_cls.version()
            assert isinstance(v, tuple), f"{op_cls.__name__}.version() 不是 tuple"
            assert len(v) > 0, f"{op_cls.__name__}.version() 为空元组"
            assert all(isinstance(e, int) for e in v), f"{op_cls.__name__}.version() 含非 int 元素"
            min_v = op_cls.min_compatible_version()
            assert min_v <= v, (
                f"{op_cls.__name__}: min_compatible_version()={min_v} > version()={v}"
            )

    def test_feature_selection_operators_version(self):
        """
        目的：验证 feature/selection 模块所有算子版本号为合法非空 tuple[int, ...]
        输入：feature/selection 模块全部 2 个具体算子
        预期：version() 返回非空 tuple，min_compatible_version() <= version()
        """
        from tsas.engine.operator.feature.selection.simple_selector import (
            ColumnSelector,
            VarianceThresholdSelector,
        )

        all_ops = [ColumnSelector, VarianceThresholdSelector]

        for op_cls in all_ops:
            v = op_cls.version()
            assert isinstance(v, tuple), f"{op_cls.__name__}.version() 不是 tuple"
            assert len(v) > 0, f"{op_cls.__name__}.version() 为空元组"
            assert all(isinstance(e, int) for e in v), f"{op_cls.__name__}.version() 含非 int 元素"
            min_v = op_cls.min_compatible_version()
            assert min_v <= v, (
                f"{op_cls.__name__}: min_compatible_version()={min_v} > version()={v}"
            )

    def test_all_operators_return_1_0_0(self):
        """
        目的：验证所有现有算子的初始版本号为 (1, 0, 0)
        输入：各模块全部具体算子
        预期：version() == (1, 0, 0)
        """
        from tsas.engine.operator.detection.zscore import ZScoreScorer
        from tsas.engine.operator.evaluation.self_evaluation import SelfEvaluation
        from tsas.engine.operator.feature.construction.simple_feature import SquareFeature
        from tsas.engine.operator.feature.selection.simple_selector import ColumnSelector

        # 抽样验证代表性算子
        sample_ops = [ZScoreScorer, SelfEvaluation, SquareFeature, ColumnSelector]
        for op_cls in sample_ops:
            assert op_cls.version() == (1, 0, 0), (
                f"{op_cls.__name__}.version() 应为 (1, 0, 0)，实际为 {op_cls.version()}"
            )


# ============================================================================
# save() / load() 版本持久化与兼容性校验测试
# ============================================================================

class TestSaveLoadVersion:
    """测试 save() 写入 version.json 和 load() 版本兼容性校验"""

    def test_save_creates_version_json(self, tmp_path):
        """
        目的：验证 save() 在目标目录中创建 version.json 文件
        输入：_ValidOperator 保存后
        预期：version.json 文件存在
        """
        op = _ValidOperator()
        save_path = tmp_path / "op_dir"
        op.save(save_path)
        assert (save_path / "version.json").exists()

    def test_save_version_json_content(self, tmp_path):
        """
        目的：验证 save() 写入的 version.json 内容与 cls.version() 一致
        输入：_ValidOperator(version=(1,0,0)) 保存后
        预期：version.json 包含 {"version": [1, 0, 0]}
        """
        op = _ValidOperator()
        save_path = tmp_path / "op_dir"
        op.save(save_path)
        data = json.loads((save_path / "version.json").read_text(encoding='utf-8'))
        assert data["version"] == [1, 0, 0]

    def test_save_version_json_with_custom_version(self, tmp_path):
        """
        目的：验证 save() 对非 (1,0,0) 版本也正确写入
        输入：_CustomMinCompatOp(version=(2,0,0)) 保存后
        预期：version.json 包含 {"version": [2, 0, 0]}
        """
        op = _CustomMinCompatOp()
        save_path = tmp_path / "op_dir"
        op.save(save_path)
        data = json.loads((save_path / "version.json").read_text(encoding='utf-8'))
        assert data["version"] == [2, 0, 0]

    def test_load_matching_version_no_warning(self, tmp_path, loguru_capture):
        """
        目的：验证加载版本与当前版本完全匹配时不产生任何警告
        输入：save() 后直接 load()
        预期：无版本相关警告日志
        """
        op = _ValidOperator()
        save_path = tmp_path / "op_dir"
        op.save(save_path)

        loaded = _ValidOperator.load(save_path)
        assert loaded is not None
        # 无版本兼容性警告
        version_warnings = [m for m in loguru_capture if "版本" in m and "不兼容" in m]
        assert len(version_warnings) == 0

    def test_load_future_version_triggers_warning(self, tmp_path, loguru_capture):
        """
        目的：验证 saved_version > version() 时产生警告
        输入：手动将 version.json 修改为 (9, 0, 0)，然后 load()
        预期：日志中包含“持久化版本”和“可能存在不兼容”
        """
        op = _ValidOperator()
        save_path = tmp_path / "op_dir"
        op.save(save_path)
        # 手动将版本修改为未来版本
        (save_path / "version.json").write_text(
            json.dumps({"version": [9, 0, 0]}), encoding='utf-8'
        )

        loaded = _ValidOperator.load(save_path)
        assert loaded is not None
        # 验证警告已产生
        warnings = [m for m in loguru_capture if "高于当前算子版本" in m]
        assert len(warnings) == 1

    def test_load_incompatible_old_version_triggers_warning(self, tmp_path, loguru_capture):
        """
        目的：验证 saved_version < min_compatible_version() 时产生警告
        输入：_CustomMinCompatOp(min_compat=(1,0,0))，version.json 写为 (0, 5, 0)
        预期：日志中包含“低于当前算子最低兼容版本”和“可能存在不兼容”
        """
        op = _CustomMinCompatOp()
        save_path = tmp_path / "op_dir"
        op.save(save_path)
        # 手动将版本修改为低于 min_compat 的老版本
        (save_path / "version.json").write_text(
            json.dumps({"version": [0, 5, 0]}), encoding='utf-8'
        )

        loaded = _CustomMinCompatOp.load(save_path)
        assert loaded is not None
        warnings = [m for m in loguru_capture if "低于当前算子最低兼容版本" in m]
        assert len(warnings) == 1

    def test_load_without_version_json_no_warning(self, tmp_path, loguru_capture):
        """
        目的：验证 version.json 不存在时跳过版本校验，不产生警告（兼容旧版数据）
        输入：保存后删除 version.json，再 load()
        预期：加载成功，无版本相关警告
        """
        op = _ValidOperator()
        save_path = tmp_path / "op_dir"
        op.save(save_path)
        # 删除 version.json 模拟旧版持久化数据
        (save_path / "version.json").unlink()

        loaded = _ValidOperator.load(save_path)
        assert loaded is not None
        version_warnings = [m for m in loguru_capture if "版本" in m and "不兼容" in m]
        assert len(version_warnings) == 0

    def test_load_within_compat_range_no_warning(self, tmp_path, loguru_capture):
        """
        目的：验证 saved_version 在 [min_compat, version] 区间内时不产生警告
        输入：_CustomMinCompatOp(version=(2,0,0), min_compat=(1,0,0))，version.json 写为 (1, 5, 0)
        预期：加载成功，无版本相关警告
        """
        op = _CustomMinCompatOp()
        save_path = tmp_path / "op_dir"
        op.save(save_path)
        # 版本在兼容区间 [1,0,0] ~ [2,0,0] 内
        (save_path / "version.json").write_text(
            json.dumps({"version": [1, 5, 0]}), encoding='utf-8'
        )

        loaded = _CustomMinCompatOp.load(save_path)
        assert loaded is not None
        version_warnings = [m for m in loguru_capture if "版本" in m and "不兼容" in m]
        assert len(version_warnings) == 0

    def test_load_at_min_compat_boundary_no_warning(self, tmp_path, loguru_capture):
        """
        目的：验证 saved_version == min_compatible_version() 时不产生警告（边界值）
        输入：_CustomMinCompatOp(min_compat=(1,0,0))，version.json 写为 (1, 0, 0)
        预期：加载成功，无版本相关警告
        """
        op = _CustomMinCompatOp()
        save_path = tmp_path / "op_dir"
        op.save(save_path)
        # 版本正好等于 min_compat
        (save_path / "version.json").write_text(
            json.dumps({"version": [1, 0, 0]}), encoding='utf-8'
        )

        loaded = _CustomMinCompatOp.load(save_path)
        assert loaded is not None
        version_warnings = [m for m in loguru_capture if "版本" in m and "不兼容" in m]
        assert len(version_warnings) == 0

    def test_save_load_roundtrip_with_config(self, tmp_path):
        """
        目的：验证 save/load 往返后 version.json、config.json 均正确恢复
        输入：带 Config 的算子保存后加载
        预期：config 正确恢复，version.json 内容一致
        """
        from pydantic import BaseModel, Field

        class _SaveTestConfig(BaseModel):
            threshold: float = Field(default=3.0, gt=0)

        class _SaveTestOp(NumericOperator[None, _SaveTestConfig, None]):
            @classmethod
            def name(cls) -> str:
                return "save_test_op"

            @classmethod
            def version(cls) -> tuple[int, ...]:
                return (1, 2, 3)

            def _run_data(self, x, params, idx=None):
                return x

            def _name_output_columns(self, output_data, meta, params):
                return ["result"]

        op = _SaveTestOp(config=_SaveTestConfig(threshold=5.0))
        save_path = tmp_path / "op_dir"
        op.save(save_path)

        # 验证 version.json
        vdata = json.loads((save_path / "version.json").read_text(encoding='utf-8'))
        assert vdata["version"] == [1, 2, 3]

        # 验证 load 后 config 恢复
        loaded = _SaveTestOp.load(save_path)
        assert loaded.config is not None
        assert loaded.config.threshold == 5.0


# ============================================================================
# _format_version 格式化测试
# ============================================================================

class TestFormatVersion:
    """测试 BaseOperator._format_version 静态方法"""

    def test_standard_triplet(self):
        """
        目的：验证标准三元组格式化为点分字符串
        输入：(1, 0, 0)
        预期：返回 "1.0.0"
        """
        assert BaseOperator._format_version((1, 0, 0)) == "1.0.0"

    def test_single_element(self):
        """
        目的：验证单元素元组格式化
        输入：(42,)
        预期：返回 "42"
        """
        assert BaseOperator._format_version((42,)) == "42"

    def test_large_patch_number(self):
        """
        目的：验证多位数字的 patch 版本号格式化
        输入：(1, 0, 99)
        预期：返回 "1.0.99"
        """
        assert BaseOperator._format_version((1, 0, 99)) == "1.0.99"

    def test_long_tuple(self):
        """
        目的：验证长元组格式化
        输入：(1, 2, 3, 4, 5)
        预期：返回 "1.2.3.4.5"
        """
        assert BaseOperator._format_version((1, 2, 3, 4, 5)) == "1.2.3.4.5"

    def test_all_zeros(self):
        """
        目的：验证全零版本号格式化
        输入：(0, 0, 0)
        预期：返回 "0.0.0"
        """
        assert BaseOperator._format_version((0, 0, 0)) == "0.0.0"

    def test_major_version_only(self):
        """
        目的：验证主版本为 2 的情况
        输入：(2, 0, 0)
        预期：返回 "2.0.0"
        """
        assert BaseOperator._format_version((2, 0, 0)) == "2.0.0"
