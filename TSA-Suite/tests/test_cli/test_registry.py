# -*- coding: utf-8 -*-

"""
通用算子注册中心单元测试

对应源文件：
- cli/registry.py: OperatorRegistry

测试范围：
- discover 自动扫描注册
- get 按名称查找
- list_all 列出所有
- register 手动注册
- 自动触发 discover（懒加载）
- 过滤函数 filter_fn
- 版本覆盖策略（高版本胜出 / 同名同版本不同类 raise / 手动注册被拒 warning）
- 异常场景（未知名称、无 name 方法等）
"""

import pytest
from loguru import logger

from tsas.engine.operator.base import BaseOperator
from tsas.engine.operator.cli.registry import OperatorRegistry


# ============================================================================
# 辅助：用于测试的简单算子类
# ============================================================================

class _DummyOperator(BaseOperator[None, None, None, None]):
    """测试用的简单算子"""

    @classmethod
    def name(cls) -> str:
        return "dummy_op"

    @classmethod
    def version(cls) -> tuple[int, ...]:
        return (1, 0, 0)

    def _run(self, x, *, params):
        return x


class _AnotherOperator(BaseOperator[None, None, None, None]):
    """另一个测试用算子"""

    @classmethod
    def name(cls) -> str:
        return "another_op"

    @classmethod
    def version(cls) -> tuple[int, ...]:
        return (1, 0, 0)

    def _run(self, x, *, params):
        return x


# ============================================================================
# 测试类
# ============================================================================

class TestOperatorRegistryDiscover:
    """测试 OperatorRegistry 的 discover 扫描功能"""

    def test_discover_feature_construction(self):
        """
        目的：验证 discover 能扫描 feature.construction 包
        输入：扫描 tsas.engine.operator.feature.construction
        预期：至少发现 5 个特征算子（square_feature 等）
        """
        from tsas.engine.operator.feature.construction.base import BaseFeatureMixin

        registry = OperatorRegistry(
            base_class=BaseFeatureMixin,
            scan_packages=['tsas.engine.operator.feature.construction'],
        )
        registry.discover()
        operators = registry.list_all()

        assert len(operators) >= 5
        assert 'square_feature' in operators
        assert 'polynomial_feature' in operators
        assert 'rolling_mean_feature' in operators
        assert 'column_median_feature' in operators
        assert 'pca_feature' in operators

    def test_discover_detection(self):
        """
        目的：验证 discover 能扫描 detection 包并通过 filter_fn 过滤
        输入：扫描 tsas.engine.operator.detection，过滤 Scorer 和 Decider
        预期：发现 Scorer、Decider、Detector 类型的算子
        """
        from tsas.engine.operator.detection.base import (
            BaseScorerMixin, BaseDeciderMixin, BaseDetector,
        )

        def _filter(cls):
            return issubclass(cls, (BaseScorerMixin, BaseDeciderMixin, BaseDetector))

        registry = OperatorRegistry(
            base_class=BaseOperator,
            scan_packages=['tsas.engine.operator.detection'],
            filter_fn=_filter,
        )
        registry.discover()
        operators = registry.list_all()

        # 应该包含 Scorer、Detector
        assert 'knn_scorer' in operators
        assert 'knn_detector' in operators
        assert 'residual_scorer' in operators
        assert 'threshold_decider' in operators

    def test_discover_evaluation(self):
        """
        目的：验证 discover 能扫描 evaluation 包
        输入：扫描 tsas.engine.operator.evaluation
        预期：发现 5 个评价指标算子
        """
        from tsas.engine.operator.evaluation.base import BaseMetricOperator

        registry = OperatorRegistry(
            base_class=BaseMetricOperator,
            scan_packages=['tsas.engine.operator.evaluation'],
        )
        registry.discover()
        operators = registry.list_all()

        assert len(operators) >= 5
        assert 'binary_classification' in operators
        assert 'self_evaluation' in operators

    def test_discover_sets_flag(self):
        """
        目的：验证 discover 后 discovered 属性为 True
        输入：创建注册中心并调用 discover
        预期：discovered 从 False 变为 True
        """
        from tsas.engine.operator.feature.construction.base import BaseFeatureMixin

        registry = OperatorRegistry(
            base_class=BaseFeatureMixin,
            scan_packages=['tsas.engine.operator.feature.construction'],
        )
        assert registry.discovered is False
        registry.discover()
        assert registry.discovered is True

    def test_discover_incremental(self):
        """
        目的：验证 discover 重复调用会增量合并
        输入：先 discover 一次，手动注册一个，再 discover
        预期：两次 discover 的结果加上手动注册的都存在
        """
        from tsas.engine.operator.feature.construction.base import BaseFeatureMixin

        registry = OperatorRegistry(
            base_class=BaseFeatureMixin,
            scan_packages=['tsas.engine.operator.feature.construction'],
        )
        registry.discover()
        count1 = len(registry.list_all())

        # 手动注册一个
        registry.register(_DummyOperator, name="test_dummy")

        # 再次 discover
        registry.discover()
        count2 = len(registry.list_all())

        # 手动注册的仍在
        assert 'test_dummy' in registry.list_all()
        assert count2 >= count1


class TestOperatorRegistryGet:
    """测试 OperatorRegistry 的 get 查找功能"""

    def test_get_existing(self):
        """
        目的：验证 get 能找到已注册的算子
        输入：手动注册后 get
        预期：返回对应的类
        """
        registry = OperatorRegistry(
            base_class=BaseOperator,
            scan_packages=[],
        )
        registry.register(_DummyOperator)
        assert registry.get("dummy_op") is _DummyOperator

    def test_get_not_found_raises(self):
        """
        目的：验证 get 未找到时抛出 KeyError
        输入：查找一个不存在的算子名称
        预期：抛出 KeyError，错误信息包含可用算子列表
        """
        registry = OperatorRegistry(
            base_class=BaseOperator,
            scan_packages=[],
        )
        registry._discovered = True  # 跳过 discover

        with pytest.raises(KeyError, match="未找到名为"):
            registry.get("nonexistent_operator")

    def test_get_triggers_discover(self):
        """
        目的：验证 get 在未 discover 时自动触发 discover
        输入：不调用 discover 直接 get
        预期：自动触发 discover 后能找到算子
        """
        from tsas.engine.operator.feature.construction.base import BaseFeatureMixin

        registry = OperatorRegistry(
            base_class=BaseFeatureMixin,
            scan_packages=['tsas.engine.operator.feature.construction'],
        )
        assert registry.discovered is False

        # get 应自动触发 discover
        cls = registry.get('square_feature')
        assert cls is not None
        assert registry.discovered is True


class TestOperatorRegistryListAll:
    """测试 OperatorRegistry 的 list_all 功能"""

    def test_list_all_sorted(self):
        """
        目的：验证 list_all 返回按名称排序的字典
        输入：注册两个算子（名称逆序）
        预期：返回按名称升序排列的字典
        """
        registry = OperatorRegistry(base_class=BaseOperator, scan_packages=[])
        registry._discovered = True
        registry.register(_AnotherOperator)
        registry.register(_DummyOperator)

        result = registry.list_all()
        keys = list(result.keys())
        assert keys == sorted(keys)

    def test_list_all_triggers_discover(self):
        """
        目的：验证 list_all 在未 discover 时自动触发 discover
        输入：不调用 discover 直接 list_all
        预期：自动触发 discover 后返回结果
        """
        from tsas.engine.operator.feature.construction.base import BaseFeatureMixin

        registry = OperatorRegistry(
            base_class=BaseFeatureMixin,
            scan_packages=['tsas.engine.operator.feature.construction'],
        )
        result = registry.list_all()
        assert len(result) >= 5
        assert registry.discovered is True


class TestOperatorRegistryRegister:
    """测试 OperatorRegistry 的 register 手动注册功能"""

    def test_register_with_auto_name(self):
        """
        目的：验证 register 不指定 name 时使用 cls.name()
        输入：注册 _DummyOperator 不指定 name
        预期：以 "dummy_op" 为 key 注册
        """
        registry = OperatorRegistry(base_class=BaseOperator, scan_packages=[])
        registry.register(_DummyOperator)
        assert 'dummy_op' in registry._registry

    def test_register_with_explicit_name(self):
        """
        目的：验证 register 指定 name 时使用显式名称
        输入：注册 _DummyOperator 指定 name="custom_name"
        预期：以 "custom_name" 为 key 注册
        """
        registry = OperatorRegistry(base_class=BaseOperator, scan_packages=[])
        registry.register(_DummyOperator, name="custom_name")
        assert 'custom_name' in registry._registry
        assert registry._registry['custom_name'] is _DummyOperator

    def test_register_no_name_method_raises(self):
        """
        目的：验证类没有 name 方法且未提供 name 参数时抛出 ValueError
        输入：注册一个没有 name 方法的普通类
        预期：抛出 ValueError
        """
        class _NoNameClass:
            pass

        registry = OperatorRegistry(base_class=BaseOperator, scan_packages=[])
        with pytest.raises(ValueError, match="没有 name\\(\\) 方法"):
            registry.register(_NoNameClass)

    def test_register_name_conflict_raises(self):
        """
        目的：验证同名同版本不同类时抛出 ValueError
        输入：先注册 _DummyOperator(v1.0.0)，再注册 _AnotherOperator 同名且同为 v1.0.0
        预期：抛出 ValueError，错误信息包含 '算子名称冲突'
        """
        registry = OperatorRegistry(base_class=BaseOperator, scan_packages=[])
        registry.register(_DummyOperator)

        # 另一个类用相同名称和相同版本注册 → 冲突
        with pytest.raises(ValueError, match="算子名称冲突"):
            registry.register(_AnotherOperator, name="dummy_op")

    def test_register_same_class_same_name_idempotent(self):
        """
        目的：验证同一个类以相同名称重复注册是幂等的，不抛异常
        输入：连续两次注册 _DummyOperator
        预期：不抛异常，注册表中只有一个条目
        """
        registry = OperatorRegistry(base_class=BaseOperator, scan_packages=[])
        registry.register(_DummyOperator)
        # 同一个类的重复注册应该是幂等的
        registry.register(_DummyOperator)
        assert len(registry._registry) == 1
        assert registry._registry['dummy_op'] is _DummyOperator


class TestOperatorRegistryFilterFn:
    """测试 OperatorRegistry 的 filter_fn 过滤功能"""

    def test_filter_fn_excludes(self):
        """
        目的：验证 filter_fn 返回 False 时不注册
        输入：filter_fn 排除所有类
        预期：注册表为空
        """
        from tsas.engine.operator.feature.construction.base import BaseFeatureMixin

        registry = OperatorRegistry(
            base_class=BaseFeatureMixin,
            scan_packages=['tsas.engine.operator.feature.construction'],
            filter_fn=lambda cls: False,
        )
        registry.discover()
        assert len(registry.list_all()) == 0

    def test_filter_fn_includes_selectively(self):
        """
        目的：验证 filter_fn 可以选择性注册
        输入：filter_fn 只注册 name() 以 "square" 开头的算子
        预期：只注册 square_feature
        """
        from tsas.engine.operator.feature.construction.base import BaseFeatureMixin

        registry = OperatorRegistry(
            base_class=BaseFeatureMixin,
            scan_packages=['tsas.engine.operator.feature.construction'],
            filter_fn=lambda cls: cls.name().startswith('square'),
        )
        registry.discover()
        operators = registry.list_all()
        assert len(operators) == 1
        assert 'square_feature' in operators


# ============================================================================
# 辅助：用于版本覆盖测试的算子类（同名不同版本）
# ============================================================================

class _VersionedOpV1(BaseOperator[None, None, None, None]):
    """版本 1.0.0 的测试算子"""

    @classmethod
    def name(cls) -> str:
        return "versioned_op"

    @classmethod
    def version(cls) -> tuple[int, ...]:
        return (1, 0, 0)

    def _run(self, x, *, params):
        return x


class _VersionedOpV2(BaseOperator[None, None, None, None]):
    """版本 2.0.0 的测试算子（同名）"""

    @classmethod
    def name(cls) -> str:
        return "versioned_op"

    @classmethod
    def version(cls) -> tuple[int, ...]:
        return (2, 0, 0)

    def _run(self, x, *, params):
        return x


class _VersionedOpV1Again(BaseOperator[None, None, None, None]):
    """版本 1.0.0 的另一个同名算子类（用于测试同版本冲突）"""

    @classmethod
    def name(cls) -> str:
        return "versioned_op"

    @classmethod
    def version(cls) -> tuple[int, ...]:
        return (1, 0, 0)

    def _run(self, x, *, params):
        return x


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
# 版本覆盖策略测试
# ============================================================================

class TestVersionOverride:
    """测试注册中心的版本覆盖策略：高版本始终胜出"""

    def test_register_higher_version_overrides(self):
        """
        目的：验证手动注册更高版本时覆盖已注册的低版本
        输入：先注册 v1.0.0，再注册 v2.0.0（同名）
        预期：注册表中保留 v2.0.0 的类
        """
        registry = OperatorRegistry(base_class=BaseOperator, scan_packages=[])
        registry.register(_VersionedOpV1)
        assert registry._registry['versioned_op'] is _VersionedOpV1

        registry.register(_VersionedOpV2)
        assert registry._registry['versioned_op'] is _VersionedOpV2

    def test_register_lower_version_keeps_existing(self):
        """
        目的：验证手动注册低版本时保留已注册的高版本
        输入：先注册 v2.0.0，再注册 v1.0.0（同名）
        预期：注册表中仍保留 v2.0.0 的类
        """
        registry = OperatorRegistry(base_class=BaseOperator, scan_packages=[])
        registry.register(_VersionedOpV2)
        registry.register(_VersionedOpV1)
        assert registry._registry['versioned_op'] is _VersionedOpV2

    def test_register_same_version_different_class_raises(self):
        """
        目的：验证同名同版本不同类时抛出 ValueError
        输入：先注册 _VersionedOpV1(1.0.0)，再注册 _VersionedOpV1Again(1.0.0)
        预期：抛出 ValueError，错误信息包含 '算子名称冲突'
        """
        registry = OperatorRegistry(base_class=BaseOperator, scan_packages=[])
        registry.register(_VersionedOpV1)

        with pytest.raises(ValueError, match="算子名称冲突"):
            registry.register(_VersionedOpV1Again)

    def test_register_same_class_idempotent(self):
        """
        目的：验证同一类重复注册是幂等的
        输入：连续两次注册 _VersionedOpV1
        预期：不抛异常，注册表中只有一个条目
        """
        registry = OperatorRegistry(base_class=BaseOperator, scan_packages=[])
        registry.register(_VersionedOpV1)
        registry.register(_VersionedOpV1)
        assert len(registry._registry) == 1
        assert registry._registry['versioned_op'] is _VersionedOpV1

    def test_manual_register_lower_version_logs_warning(self, loguru_capture):
        """
        目的：验证手动注册低版本时记录 warning 日志
        输入：先注册 v2.0.0，再手动注册 v1.0.0（同名）
        预期：loguru 输出包含 "手动注册算子" 和 "无效" 的 warning 日志
        """
        registry = OperatorRegistry(base_class=BaseOperator, scan_packages=[])
        registry.register(_VersionedOpV2)
        registry.register(_VersionedOpV1)

        # 检查日志中包含 warning
        warning_messages = [m for m in loguru_capture if '手动注册算子' in m and '无效' in m]
        assert len(warning_messages) >= 1

    def test_auto_scan_higher_version_overrides_with_debug(self, loguru_capture):
        """
        目的：验证自动扫描中高版本覆盖低版本时记录 debug 日志
        输入：先手动注册 v1.0.0，然后通过 _register_with_version(is_manual=False) 注册 v2.0.0
        预期：v2.0.0 覆盖 v1.0.0，日志中包含 "版本覆盖" 的 debug 记录
        """
        registry = OperatorRegistry(base_class=BaseOperator, scan_packages=[])
        registry.register(_VersionedOpV1)

        # 模拟自动扫描入口注册高版本
        registry._register_with_version(
            name="versioned_op", cls=_VersionedOpV2, is_manual=False
        )

        assert registry._registry['versioned_op'] is _VersionedOpV2
        debug_messages = [m for m in loguru_capture if '版本覆盖' in m]
        assert len(debug_messages) >= 1

    def test_auto_scan_lower_version_keeps_existing_no_extra_log(self, loguru_capture):
        """
        目的：验证自动扫描中低版本被忽略时不记录额外日志
        输入：先注册 v2.0.0，然后通过 _register_with_version(is_manual=False) 注册 v1.0.0
        预期：仍保留 v2.0.0，且没有额外的覆盖或 warning 日志
        """
        registry = OperatorRegistry(base_class=BaseOperator, scan_packages=[])
        registry.register(_VersionedOpV2)

        # 模拟自动扫描入口注册低版本
        registry._register_with_version(
            name="versioned_op", cls=_VersionedOpV1, is_manual=False
        )

        assert registry._registry['versioned_op'] is _VersionedOpV2
        # 自动扫描低版本被忽略时不应该有额外日志
        override_messages = [m for m in loguru_capture if '版本覆盖' in m or '无效' in m]
        assert len(override_messages) == 0

    def test_version_format_in_log_message(self, loguru_capture):
        """
        目的：验证日志中版本号以点分字符串格式显示
        输入：先注册 v1.0.0，再手动注册 v2.0.0（触发覆盖日志）
        预期：日志消息中包含 "1.0.0" 和 "2.0.0"
        """
        registry = OperatorRegistry(base_class=BaseOperator, scan_packages=[])
        registry.register(_VersionedOpV1)

        # 通过自动扫描入口注册高版本（触发 debug 日志）
        registry._register_with_version(
            name="versioned_op", cls=_VersionedOpV2, is_manual=False
        )

        all_log = ''.join(loguru_capture)
        assert '1.0.0' in all_log
        assert '2.0.0' in all_log


# ============================================================================
# discover 边界场景测试（提升覆盖率）
# ============================================================================

class TestDiscoverEdgeCases:
    """测试 discover 方法的边界场景，覆盖普通模块扫描和 ImportError 分支"""

    def test_discover_plain_module_no_path(self):
        """
        目的：验证 scan_packages 指向普通模块（非包）时的处理逻辑
        输入：scan_packages=['tsas.engine.operator.cli.io']（io.py 是普通模块，无 __path__）
        预期：不崩溃，discover 正常完成，discovered=True
        """
        registry = OperatorRegistry(
            base_class=BaseOperator,
            scan_packages=['tsas.engine.operator.cli.io'],
        )
        registry.discover()
        assert registry.discovered is True
        # io.py 中不含 BaseOperator 子类，注册表应为空
        assert len(registry.list_all()) == 0

    def test_discover_import_error_skips_module(self):
        """
        目的：验证 walk_packages 过程中 import 失败的模块被静默跳过
        输入：扫描 detection 包，但 mock importlib.import_module 对特定子模块抛 ImportError
        预期：discover 不崩溃，其他正常模块仍被注册
        """
        import importlib
        from unittest.mock import patch

        registry = OperatorRegistry(
            base_class=BaseOperator,
            scan_packages=['tsas.engine.operator.detection'],
        )

        real_import = importlib.import_module
        blocked = {'tsas.engine.operator.detection.cicada'}

        def mock_import(name, *args, **kwargs):
            if name in blocked:
                raise ImportError("mocked missing dependency")
            return real_import(name, *args, **kwargs)

        with patch('importlib.import_module', side_effect=mock_import):
            registry.discover()

        # cicada 模块被跳过，其他算子仍正常注册
        assert registry.discovered is True
        operators = registry.list_all()
        assert 'cicada_predictor' not in operators
        assert 'cicada_scorer' not in operators
        # 其他算子仍存在
        assert 'knn_scorer' in operators
        assert 'threshold_decider' in operators

    def test_scan_module_class_name_call_raises(self):
        """
        目的：验证 _scan_module 中 name() 调用抛异常时被静默跳过
        输入：构造一个 name() 抛 RuntimeError 的算子类，放入模拟模块中
        预期：discover 不崩溃，该算子不被注册
        """

        class _BrokenNameOp(BaseOperator[None, None, None, None]):
            """name() 抛异常的算子"""

            @classmethod
            def name(cls) -> str:
                raise RuntimeError("name() broken")

            @classmethod
            def version(cls) -> tuple[int, ...]:
                return (1, 0, 0)

            def _run(self, x, *, params):
                return x

        # 构造一个模拟模块，包含可正常注册的算子和 name() 异常的算子
        import types
        fake_module = types.ModuleType("fake_module")
        fake_module._DummyOperator = _DummyOperator
        fake_module._BrokenNameOp = _BrokenNameOp

        registry = OperatorRegistry(base_class=BaseOperator, scan_packages=[])
        registry._scan_module(fake_module)

        # _DummyOperator 正常注册，_BrokenNameOp 被跳过
        assert 'dummy_op' in registry._registry
        assert len(registry._registry) == 1
