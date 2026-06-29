# -*- coding: utf-8 -*-

"""
算子基类单元测试

对应源文件：
- base.py: BaseOperator, LearnableOperatorMixin, NumericOperator,
  BiNumericOperator, SupervisedNumericOperatorMixin, UnsupervisedNumericOperatorMixin

测试范围：
- BaseOperator 参数校验 (_validate_params, _resolve_param)
- BaseOperator 属性 (last_run_params, config, oid)
- BaseOperator 持久化 (save / load)
- LearnableOperatorMixin __init_subclass__ (MRO 检查、FP 提取)
- LearnableOperatorMixin 属性 (is_fitted, can_additional_fit, last_fit_params)
- LearnableOperatorMixin 训练与持久化
- _NumericOperatorMixin IO 校验 (_validate_input, _unwrap_data, _validate_and_wrap_output)
- BiNumericOperator 默认列名
- SupervisedNumericOperatorMixin 模板管线
- UnsupervisedNumericOperatorMixin 模板管线
"""

import tempfile
from pathlib import Path
from typing import TypeVar

import numpy as np
import pandas as pd
import pytest
from loguru import logger
from pydantic import BaseModel, Field, ValidationError

from tsas.engine.operator.base import (
    BaseOperator,
    LearnableOperatorMixin,
    NumericOperator,
    BiNumericOperator,
    SupervisedNumericOperatorMixin,
    UnsupervisedNumericOperatorMixin,
    NumericData,
    DataFrameMeta,
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
# 测试用 Pydantic 模型和具体算子子类
# ============================================================================

class SimpleConfig(BaseModel):
    threshold: float = Field(default=0.5, gt=0)


class SimpleRunParams(BaseModel):
    verbose: bool = False


class MyFitParams(BaseModel):
    epochs: int = 10


class MyExtraOutput(BaseModel):
    info: str = "default"


class SimpleOperator(NumericOperator[None, SimpleConfig, SimpleRunParams]):
    """简单的 NumericOperator 子类，用于测试基类功能"""

    @classmethod
    def name(cls) -> str:
        return "simple_operator"

    @classmethod
    def version(cls) -> tuple[int, ...]:
        return (1, 0, 0)

    def _run_data(self, x, params, idx=None):
        return x

    def _name_output_columns(self, output_data, meta, params):
        return ["result"]


class SimpleOperatorNoParams(NumericOperator[None, None, None]):
    """无参数类型的 NumericOperator 子类"""

    @classmethod
    def name(cls) -> str:
        return "simple_no_params"

    @classmethod
    def version(cls) -> tuple[int, ...]:
        return (1, 0, 0)

    def _run_data(self, x, params, idx=None):
        return x

    def _name_output_columns(self, output_data, meta, params):
        return ["result"]


class SimpleLearnableOp(
    UnsupervisedNumericOperatorMixin[MyFitParams],
    NumericOperator[None, None, None],
):
    """无监督可训练算子，用于测试 LearnableOperatorMixin 功能"""

    @classmethod
    def name(cls) -> str:
        return "simple_learnable"

    @classmethod
    def version(cls) -> tuple[int, ...]:
        return (1, 0, 0)

    def _run_data(self, x, params, idx=None):
        return x

    def _name_output_columns(self, output_data, meta, params):
        return ["result"]

    def _fit_data(self, x, params):
        pass


class SimpleLearnableOpWithConfig(
    UnsupervisedNumericOperatorMixin[MyFitParams],
    NumericOperator[None, SimpleConfig, SimpleRunParams],
):
    """带配置参数的无监督可训练算子"""

    @classmethod
    def name(cls) -> str:
        return "learnable_with_config"

    @classmethod
    def version(cls) -> tuple[int, ...]:
        return (1, 0, 0)

    def _run_data(self, x, params, idx=None):
        return x

    def _name_output_columns(self, output_data, meta, params):
        return ["result"]

    def _fit_data(self, x, params):
        pass


class SupervisedLearnableOp(
    SupervisedNumericOperatorMixin[MyFitParams],
    NumericOperator[None, None, None],
):
    """有监督可训练算子"""

    @classmethod
    def name(cls) -> str:
        return "supervised_learnable"

    @classmethod
    def version(cls) -> tuple[int, ...]:
        return (1, 0, 0)

    def _run_data(self, x, params, idx=None):
        return x

    def _name_output_columns(self, output_data, meta, params):
        return ["result"]

    def _fit_data(self, x, y, params):
        pass

    def _filter_fit_data(self, x, y, *, params=None):
        """默认原样返回"""
        return x, y


class SimpleBiOperator(BiNumericOperator[None, None, None]):
    """简单的 BiNumericOperator 子类"""

    @classmethod
    def name(cls) -> str:
        return "simple_bi"

    @classmethod
    def version(cls) -> tuple[int, ...]:
        return (1, 0, 0)

    def _run_data(self, x_real, x_pred, params, real_idx=None, pred_idx=None):
        return x_pred


class OperatorWithEO(NumericOperator[MyExtraOutput, None, None]):
    """带附加输出的 NumericOperator"""

    @classmethod
    def name(cls) -> str:
        return "op_with_eo"

    @classmethod
    def version(cls) -> tuple[int, ...]:
        return (1, 0, 0)

    def _run_data(self, x, params, idx=None):
        return x, MyExtraOutput(info="computed")

    def _name_output_columns(self, output_data, meta, params):
        return ["result"]


# ============================================================================
# 测试数据 fixtures
# ============================================================================

@pytest.fixture
def sample_ndarray():
    """标准 ndarray 测试数据 (10, 3)"""
    np.random.seed(42)
    return np.random.randn(10, 3)


@pytest.fixture
def sample_dataframe():
    """标准 DataFrame 测试数据"""
    np.random.seed(42)
    data = np.random.randn(10, 3)
    return pd.DataFrame(data, columns=["a", "b", "c"])


# ============================================================================
# BaseOperator._extract_type_from_typevar 测试
# ============================================================================

class TestExtractTypeFromTypevar:
    """测试 _extract_type_from_typevar 静态方法"""

    def test_no_args_or_params_returns_none(self):
        """
        目的：验证当泛型基类没有 __args__ 或 __parameters__ 时返回 None
        输入：构造一个 __orig_bases__ 中含无参数泛型基类的类
        预期：返回 None
        """
        # Line 160: args or origin_params is empty -> continue -> return None
        # Using a non-existent TypeVar ensures no match in any base
        result = BaseOperator._extract_type_from_typevar(
            SimpleOperatorNoParams, BaseOperator, TypeVar("X")
        )
        assert result is None

    def test_extract_config_type(self):
        """
        目的：验证能从泛型参数中正确提取 Config 类型
        输入：SimpleOperator[None, SimpleConfig, SimpleRunParams]
        预期：_config_type 为 SimpleConfig
        """
        assert SimpleOperator._config_type is SimpleConfig

    def test_extract_run_params_type(self):
        """
        目的：验证能从泛型参数中正确提取 RunParams 类型
        输入：SimpleOperator[None, SimpleConfig, SimpleRunParams]
        预期：_run_params_type 为 SimpleRunParams
        """
        assert SimpleOperator._run_params_type is SimpleRunParams

    def test_no_type_returns_none(self):
        """
        目的：验证无参数类型时 ClassVar 为 None
        输入：SimpleOperatorNoParams[None, None, None]
        预期：_config_type 和 _run_params_type 均为 None
        """
        assert SimpleOperatorNoParams._config_type is None
        assert SimpleOperatorNoParams._run_params_type is None


# ============================================================================
# BaseOperator._validate_params 测试
# ============================================================================

class TestValidateParams:
    """测试 _validate_params 静态方法"""

    def test_params_type_none_with_params_warns(self, loguru_capture):
        """
        目的：验证 params_type 为 None 时传入 params 触发 warning 日志
        输入：params_type=None, params=SimpleConfig()
        预期：返回 None，日志包含 warning 消息
        """
        result = BaseOperator._validate_params(
            "config", None, SimpleConfig()
        )
        assert result is None
        assert len(loguru_capture) > 0

    def test_params_type_none_with_kwargs_warns(self, loguru_capture):
        """
        目的：验证 params_type 为 None 时传入 kwargs 触发 warning 日志
        输入：params_type=None, kwargs={"key": "value"}
        预期：返回 None，日志包含 warning 消息
        """
        result = BaseOperator._validate_params(
            "config", None, None, key="value"
        )
        assert result is None
        assert len(loguru_capture) > 0

    def test_typed_params_and_kwargs_warns(self, loguru_capture):
        """
        目的：验证同时提供类型化参数和 kwargs 时发出 warning 日志
        输入：params_type=SimpleConfig, params=SimpleConfig(), kwargs={"extra": 1}
        预期：返回 params 实例，日志包含 warning 消息
        """
        config = SimpleConfig()
        result = BaseOperator._validate_params(
            "config", SimpleConfig, config, extra=1
        )
        assert result is config
        assert len(loguru_capture) > 0

    def test_wrong_params_type_warns(self, loguru_capture):
        """
        目的：验证 params 类型不匹配时发出 warning 日志
        输入：params_type=SimpleConfig, params=SimpleRunParams()
        预期：发出 warning，从 kwargs 构造（空 kwargs 返回默认值）
        """
        result = BaseOperator._validate_params(
            "config", SimpleConfig, SimpleRunParams()
        )
        # Since no kwargs, SimpleConfig() is constructed with defaults
        assert isinstance(result, SimpleConfig)
        assert result.threshold == 0.5
        assert len(loguru_capture) > 0

    def test_valid_params_returned(self):
        """
        目的：验证类型匹配的 params 直接返回
        输入：params_type=SimpleConfig, params=SimpleConfig(threshold=1.0)
        预期：返回同一个 params 实例
        """
        config = SimpleConfig(threshold=1.0)
        result = BaseOperator._validate_params(
            "config", SimpleConfig, config
        )
        assert result is config

    def test_kwargs_constructs_model(self):
        """
        目的：验证用 kwargs 构造 Pydantic 模型
        输入：params_type=SimpleConfig, params=None, kwargs={"threshold": 2.0}
        预期：返回 SimpleConfig(threshold=2.0)
        """
        result = BaseOperator._validate_params(
            "config", SimpleConfig, None, threshold=2.0
        )
        assert isinstance(result, SimpleConfig)
        assert result.threshold == 2.0

    def test_params_type_none_no_args_returns_none(self):
        """
        目的：验证 params_type 为 None 且无 params/kwargs 时返回 None
        输入：params_type=None, params=None
        预期：返回 None
        """
        result = BaseOperator._validate_params("config", None, None)
        assert result is None


# ============================================================================
# BaseOperator 属性测试
# ============================================================================

class TestBaseOperatorProperties:
    """测试 BaseOperator 的属性访问"""

    def test_oid_auto_generated(self):
        """
        目的：验证 oid 自动生成包含算子名
        输入：无 oid 参数
        预期：oid 以 "simple_operator$" 开头
        """
        op = SimpleOperator()
        assert op.oid.startswith("simple_operator$")

    def test_oid_custom(self):
        """
        目的：验证自定义 oid 正确传递
        输入：oid="custom_id"
        预期：oid 为 "simple_operator$custom_id"
        """
        op = SimpleOperator(oid="custom_id")
        assert op.oid == "simple_operator$custom_id"

    def test_config_property(self):
        """
        目的：验证 config 属性返回实例参数
        输入：config=SimpleConfig(threshold=1.0)
        预期：config.threshold == 1.0
        """
        op = SimpleOperator(config=SimpleConfig(threshold=1.0))
        assert op.config is not None
        assert op.config.threshold == 1.0

    def test_config_none(self):
        """
        目的：验证无参数类型算子的 config 为 None
        输入：SimpleOperatorNoParams
        预期：config 为 None
        """
        op = SimpleOperatorNoParams()
        assert op.config is None

    def test_last_run_params_property(self):
        """
        目的：验证 last_run_params 属性初始为 None，run 后更新
        输入：创建算子并 run
        预期：初始为 None，run 后非 None
        """
        op = SimpleOperator()
        assert op.last_run_params is None

        data = np.random.randn(5, 3)
        op.run(data)
        assert op.last_run_params is not None


# ============================================================================
# BaseOperator._resolve_param 测试
# ============================================================================

class TestResolveParam:
    """测试 _resolve_param 三级参数解析"""

    def test_runtime_params_precedence(self):
        """
        目的：验证运行参数优先级最高（BaseModel 无目标 key 时回退到 config）
        输入：runtime params = SimpleRunParams(verbose=True), config 有 threshold=1.0
        预期：SimpleRunParams 没有 threshold 属性 -> getattr 返回 None -> 回退到 config -> 1.0
        """
        op = SimpleOperator(config=SimpleConfig(threshold=1.0))
        params = SimpleRunParams(verbose=True)
        value = op._resolve_param(params, "threshold", default=0.1)
        # SimpleRunParams has no "threshold" -> falls through to config
        assert value == 1.0

    def test_runtime_params_basemodel_with_key(self):
        """
        目的：验证运行参数（BaseModel）包含目标 key 时直接返回
        输入：运行参数为 dict 包含 threshold
        预期：返回运行参数中的值
        """
        op = SimpleOperator(config=SimpleConfig(threshold=1.0))
        value = op._resolve_param({"threshold": 3.0}, "threshold", default=0.1)
        assert value == 3.0

    def test_config_params_precedence(self):
        """
        目的：验证运行参数无目标 key 时回退到实例参数
        输入：runtime params 为 None, config 有 threshold=1.0
        预期：返回 1.0
        """
        op = SimpleOperator(config=SimpleConfig(threshold=1.0))
        value = op._resolve_param(None, "threshold", default=0.1)
        assert value == 1.0

    def test_default_fallback(self):
        """
        目的：验证运行参数和实例参数均无目标 key 时返回默认值
        输入：runtime params=None, config=None
        预期：返回默认值 0.1
        """
        op = SimpleOperatorNoParams()
        value = op._resolve_param(None, "nonexistent", default=0.1)
        assert value == 0.1

    def test_dict_runtime_params(self):
        """
        目的：验证运行参数为 dict 时的解析
        输入：runtime params={"threshold": 2.5}
        预期：返回 2.5
        """
        op = SimpleOperator(config=SimpleConfig(threshold=1.0))
        value = op._resolve_param({"threshold": 2.5}, "threshold")
        assert value == 2.5

    def test_dict_runtime_params_missing_key(self):
        """
        目的：验证 dict 中不包含 key 时回退到 config
        输入：runtime params={"other": 1}, config 有 threshold=1.0
        预期：返回 1.0
        """
        op = SimpleOperator(config=SimpleConfig(threshold=1.0))
        value = op._resolve_param({"other": 1}, "threshold")
        assert value == 1.0

    def test_non_basemodel_non_dict_params(self):
        """
        目的：验证 params 既非 BaseModel 也非 dict 时回退
        输入：params="invalid"
        预期：回退到 config
        """
        op = SimpleOperator(config=SimpleConfig(threshold=1.0))
        value = op._resolve_param("invalid", "threshold", default=0.1)
        assert value == 1.0

    def test_basemodel_runtime_value_is_none_falls_through(self):
        """
        目的：验证 BaseModel 运行参数中 getattr 返回 None 时回退到 config
        输入：SimpleRunParams(verbose=False) 没有 threshold 字段
        预期：回退到 config 中的 threshold
        """
        op = SimpleOperator(config=SimpleConfig(threshold=1.0))
        params = SimpleRunParams(verbose=True)
        value = op._resolve_param(params, "threshold", default=0.1)
        assert value == 1.0


# ============================================================================
# BaseOperator.save / load 测试
# ============================================================================

class TestBaseOperatorSaveLoad:
    """测试 BaseOperator 持久化"""

    def test_save_raises_when_path_is_file(self, tmp_path):
        """
        目的：验证 save() 当 path 是文件时抛出 ValueError
        输入：path 指向一个已存在的文件
        预期：抛出 ValueError，消息含 "已存在的文件"
        """
        file_path = tmp_path / "a_file.txt"
        file_path.write_text("hello")
        op = SimpleOperator()
        with pytest.raises(ValueError, match="已存在的文件"):
            op.save(file_path)

    def test_save_with_last_run_params(self, tmp_path, sample_ndarray):
        """
        目的：验证 save() 当 _last_run_params 非 None 时写入文件
        输入：执行一次 run 后调用 save
        预期：目录中存在 last_run_params.json
        """
        op = SimpleOperator()
        op.run(sample_ndarray, verbose=True)
        save_path = tmp_path / "op_dir"
        op.save(save_path)
        assert (save_path / "last_run_params.json").exists()

    def test_save_creates_config_json(self, tmp_path):
        """
        目的：验证 save() 正确保存 config.json
        输入：SimpleOperator(config=SimpleConfig(threshold=2.0))
        预期：目录中存在 config.json
        """
        op = SimpleOperator(config=SimpleConfig(threshold=2.0))
        save_path = tmp_path / "op_dir"
        op.save(save_path)
        assert (save_path / "config.json").exists()

    def test_load_with_config_and_run_params(self, tmp_path, sample_ndarray):
        """
        目的：验证 load() 恢复 config 和 last_run_params
        输入：保存后加载
        预期：config 和 last_run_params 均正确恢复
        """
        op = SimpleOperator(config=SimpleConfig(threshold=2.0))
        op.run(sample_ndarray, verbose=True)
        save_path = tmp_path / "op_dir"
        op.save(save_path)

        loaded = SimpleOperator.load(save_path)
        assert loaded.config is not None
        assert loaded.config.threshold == 2.0
        assert loaded.last_run_params is not None
        assert loaded.last_run_params.verbose is True

    def test_load_without_config(self, tmp_path):
        """
        目的：验证 load() 在无 config 文件时正常加载
        输入：无参数算子保存后加载
        预期：config 为 None
        """
        op = SimpleOperatorNoParams()
        save_path = tmp_path / "op_dir"
        op.save(save_path)

        loaded = SimpleOperatorNoParams.load(save_path)
        assert loaded.config is None

    def test_load_with_oid(self, tmp_path):
        """
        目的：验证 load() 的 oid 参数正确传递
        输入：oid="loaded_id"
        预期：loaded.oid 含 "loaded_id"
        """
        op = SimpleOperator()
        save_path = tmp_path / "op_dir"
        op.save(save_path)

        loaded = SimpleOperator.load(save_path, oid="loaded_id")
        assert "loaded_id" in loaded.oid


# ============================================================================
# LearnableOperatorMixin.__init_subclass__ 测试
# ============================================================================

class TestLearnableInitSubclass:
    """测试 LearnableOperatorMixin.__init_subclass__"""

    def test_mro_error_when_mixin_after_base(self):
        """
        目的：验证 LearnableOperatorMixin 放在 BaseOperator 后面时报 TypeError
        输入：错误继承顺序的类定义
        预期：抛出 TypeError
        """
        with pytest.raises(TypeError, match="继承顺序错误"):
            class BadOrder(
                NumericOperator[None, None, None],
                LearnableOperatorMixin[None, None, None],
            ):
                @classmethod
                def name(cls) -> str:
                    return "bad"

                @classmethod
                def version(cls) -> tuple[int, ...]:
                    return (1, 0, 0)

                def _run_data(self, x, params):
                    return x

                def _name_output_columns(self, output_data, meta, params):
                    return ["result"]

                def _fit(self, x, y, params):
                    pass

    def test_skip_origin_none(self):
        """
        目的：验证 __orig_bases__ 中 origin 为 None 时跳过
        输入：中间无泛型参数的继承
        预期：不报错，_fit_params_type 为 None
        """

        # Create a class where a base in __orig_bases__ has origin=None
        # This happens with non-generic intermediate classes
        class IntermediateOp(
            UnsupervisedNumericOperatorMixin[None],
            NumericOperator[None, None, None],
        ):
            @classmethod
            def name(cls) -> str:
                return "intermediate"

            @classmethod
            def version(cls) -> tuple[int, ...]:
                return (1, 0, 0)

            def _run_data(self, x, params):
                return x

            def _name_output_columns(self, output_data, meta, params):
                return ["result"]

            def _fit_data(self, x, params):
                pass

        assert IntermediateOp._fit_params_type is None

    def test_skip_no_args_or_params(self):
        """
        目的：验证泛型基类无 __args__ 或 __parameters__ 时跳过
        输入：SimpleLearnableOp 正常继承
        预期：不报错，FP 正确提取
        """
        assert SimpleLearnableOp._fit_params_type is MyFitParams

    def test_fp_type_extraction(self):
        """
        目的：验证 FP 类型从泛型参数中正确提取
        输入：UnsupervisedNumericOperatorMixin[MyFitParams]
        预期：_fit_params_type 为 MyFitParams
        """
        assert SimpleLearnableOp._fit_params_type is MyFitParams

    def test_fp_type_none_when_not_specified(self):
        """
        目的：验证 NumericOperator（非 Learnable）的 _fit_params_type 不存在
        输入：SimpleOperator（纯 NumericOperator，无 LearnableOperatorMixin）
        预期：SimpleOperator 无 _fit_params_type 属性
        """
        # SimpleOperator is a plain NumericOperator without LearnableOperatorMixin
        # So it does not have _fit_params_type
        assert not hasattr(SimpleOperator, '_fit_params_type')


# ============================================================================
# LearnableOperatorMixin 属性测试
# ============================================================================

class TestLearnableProperties:
    """测试 LearnableOperatorMixin 属性"""

    def test_is_fitted_initially_false(self):
        """
        目的：验证初始训练状态为 False
        输入：新创建的 SimpleLearnableOp
        预期：is_fitted 为 False
        """
        op = SimpleLearnableOp()
        assert op.is_fitted is False

    def test_is_fitted_after_fit(self, sample_ndarray):
        """
        目的：验证训练后 is_fitted 为 True
        输入：fit(sample_ndarray)
        预期：is_fitted 为 True
        """
        op = SimpleLearnableOp()
        op.fit(sample_ndarray)
        assert op.is_fitted is True

    def test_can_additional_fit_default_false(self):
        """
        目的：验证默认不支持增训
        输入：SimpleLearnableOp
        预期：can_additional_fit 为 False
        """
        op = SimpleLearnableOp()
        assert op.can_additional_fit is False

    def test_last_fit_params_initially_none(self):
        """
        目的：验证初始 last_fit_params 为 None
        输入：新创建的 SimpleLearnableOp
        预期：last_fit_params 为 None
        """
        op = SimpleLearnableOp()
        assert op.last_fit_params is None

    def test_last_fit_params_after_fit(self, sample_ndarray):
        """
        目的：验证训练后 last_fit_params 有值
        输入：fit(sample_ndarray, epochs=5)
        预期：last_fit_params.epochs == 5
        """
        op = SimpleLearnableOp()
        op.fit(sample_ndarray, epochs=5)
        assert op.last_fit_params is not None
        assert op.last_fit_params.epochs == 5


# ============================================================================
# LearnableOperatorMixin.fit 测试
# ============================================================================

class TestLearnableFit:
    """测试 LearnableOperatorMixin.fit"""

    def test_fit_already_fitted_raises(self, sample_ndarray):
        """
        目的：验证已训练且不支持增训时 fit 抛出 RuntimeError
        输入：连续两次 fit
        预期：第二次 fit 抛出 RuntimeError
        """
        op = SimpleLearnableOp()
        op.fit(sample_ndarray)
        with pytest.raises(RuntimeError, match="训练已完成"):
            op.fit(sample_ndarray)

    def test_fit_returns_self(self, sample_ndarray):
        """
        目的：验证 fit 返回 self
        输入：fit(sample_ndarray)
        预期：返回值是 self
        """
        op = SimpleLearnableOp()
        result = op.fit(sample_ndarray)
        assert result is op

    def test_run_before_fit_raises(self, sample_ndarray):
        """
        目的：验证未训练时 run 抛出 RuntimeError
        输入：未 fit 直接 run
        预期：抛出 RuntimeError
        """
        op = SimpleLearnableOp()
        with pytest.raises(RuntimeError, match="训练尚未完成"):
            op.run(sample_ndarray)

    def test_run_after_fit_succeeds(self, sample_ndarray):
        """
        目的：验证训练后 run 正常执行
        输入：fit 后 run
        预期：正常返回结果
        """
        op = SimpleLearnableOp()
        op.fit(sample_ndarray)
        result = op.run(sample_ndarray)
        assert isinstance(result, np.ndarray)


# ============================================================================
# LearnableOperatorMixin 持久化测试
# ============================================================================

class TestLearnableSaveLoad:
    """测试 LearnableOperatorMixin 持久化"""

    def test_save_fit_state(self, tmp_path, sample_ndarray):
        """
        目的：验证 _save_fit_state 写入 last_fit_params 文件
        输入：训练后保存
        预期：目录中存在 last_fit_params.json
        """
        op = SimpleLearnableOp()
        op.fit(sample_ndarray, epochs=20)
        save_path = tmp_path / "learnable_op"
        op.save(save_path)
        assert (save_path / "last_fit_params.json").exists()

    def test_load_fit_state(self, tmp_path, sample_ndarray):
        """
        目的：验证 _load_fit_state 恢复 last_fit_params
        输入：训练后保存，然后加载
        预期：last_fit_params 正确恢复
        """
        op = SimpleLearnableOp()
        op.fit(sample_ndarray, epochs=20)
        save_path = tmp_path / "learnable_op"
        op.save(save_path)

        loaded = SimpleLearnableOp.load(save_path)
        assert loaded.last_fit_params is not None
        assert loaded.last_fit_params.epochs == 20

    def test_load_mixin_classmethod(self, tmp_path, sample_ndarray):
        """
        目的：验证 LearnableOperatorMixin.load() classmethod 正常工作
        输入：保存后调用 load
        预期：返回正确类型的实例，config 和 fit_params 恢复
        """
        op = SimpleLearnableOpWithConfig(config=SimpleConfig(threshold=3.0))
        op.fit(sample_ndarray, epochs=15)
        op.run(sample_ndarray, verbose=True)
        save_path = tmp_path / "learnable_config_op"
        op.save(save_path)

        loaded = SimpleLearnableOpWithConfig.load(save_path)
        assert loaded.config.threshold == 3.0
        assert loaded.last_fit_params.epochs == 15
        assert loaded.last_run_params.verbose is True

    def test_save_fit_state_no_fit_params(self, tmp_path):
        """
        目的：验证 _save_fit_state 在 last_fit_params 为 None 时不写文件
        输入：未训练的算子
        预期：目录中不存在 last_fit_params.json
        """
        op = SimpleLearnableOp()
        save_path = tmp_path / "learnable_op"
        save_path.mkdir()
        op._save_fit_state(save_path)
        assert not (save_path / "last_fit_params.json").exists()

    def test_load_fit_state_no_file(self, tmp_path):
        """
        目的：验证 _load_fit_state 在文件不存在时正常处理
        输入：空目录
        预期：不报错，last_fit_params 保持不变
        """
        op = SimpleLearnableOp()
        save_path = tmp_path / "empty_dir"
        save_path.mkdir()
        op._load_fit_state(save_path)
        assert op.last_fit_params is None


# ============================================================================
# _NumericOperatorMixin._validate_input 测试
# ============================================================================

class TestValidateInput:
    """测试 _NumericOperatorMixin._validate_input"""

    def test_dataframe_passes(self, sample_dataframe):
        """
        目的：验证 DataFrame 输入通过校验
        输入：pd.DataFrame
        预期：无异常
        """
        op = SimpleOperator()
        op._validate_input(sample_dataframe, None)

    def test_ndarray_passes(self, sample_ndarray):
        """
        目的：验证数值 ndarray 输入通过校验
        输入：np.ndarray (float64)
        预期：无异常
        """
        op = SimpleOperator()
        op._validate_input(sample_ndarray, None)

    def test_invalid_type_raises(self):
        """
        目的：验证非法输入类型抛出 TypeError
        输入：字符串列表
        预期：抛出 TypeError
        """
        op = SimpleOperator()
        with pytest.raises(TypeError, match="输入数据类型必须是"):
            op._validate_input([1, 2, 3], None)

    def test_string_ndarray_raises(self):
        """
        目的：验证非数值 dtype 的 ndarray 抛出 TypeError
        输入：dtype=str 的 ndarray
        预期：抛出 TypeError
        """
        op = SimpleOperator()
        with pytest.raises(TypeError, match="输入数据类型必须是"):
            op._validate_input(np.array(["a", "b", "c"]), None)


# ============================================================================
# _NumericOperatorMixin._unwrap_data 测试
# ============================================================================

class TestUnwrapData:
    """测试 _NumericOperatorMixin._unwrap_data"""

    def test_dataframe_unwrap(self, sample_dataframe):
        """
        目的：验证 DataFrame 输入正确解包
        输入：DataFrame
        预期：meta 非 None，ndarray 数据正确
        """
        op = SimpleOperator()
        meta, arr = op._unwrap_data(sample_dataframe, None)
        assert meta is not None
        assert isinstance(meta, DataFrameMeta)
        assert isinstance(arr, np.ndarray)
        np.testing.assert_array_equal(arr, sample_dataframe.to_numpy())

    def test_ndarray_unwrap(self, sample_ndarray):
        """
        目的：验证 ndarray 输入正确解包
        输入：ndarray
        预期：meta 为 None，ndarray 原样返回
        """
        op = SimpleOperator()
        meta, arr = op._unwrap_data(sample_ndarray, None)
        assert meta is None
        assert arr is sample_ndarray

    def test_invalid_type_raises(self):
        """
        目的：验证非法输入类型抛出 TypeError
        输入：字符串
        预期：抛出 TypeError
        """
        op = SimpleOperator()
        with pytest.raises(TypeError, match="数据类型必须是"):
            op._unwrap_data("invalid", None)


# ============================================================================
# _NumericOperatorMixin._validate_and_wrap_output 测试
# ============================================================================

class TestValidateAndWrapOutput:
    """测试 _NumericOperatorMixin._validate_and_wrap_output"""

    def test_eo_none_non_ndarray_raises(self):
        """
        目的：验证 EO 为 None 时返回非 ndarray 抛出 TypeError
        输入：output_data="string"
        预期：抛出 TypeError
        """
        op = SimpleOperator()
        with pytest.raises(TypeError, match="必须返回 np.ndarray"):
            op._validate_and_wrap_output("not_an_array", None, None)

    def test_eo_non_none_non_tuple_raises(self):
        """
        目的：验证 EO 非 None 时返回非 tuple 抛出 TypeError
        输入：output_data=ndarray（期望 tuple）
        预期：抛出 TypeError
        """
        op = OperatorWithEO()
        with pytest.raises(TypeError, match="必须返回 tuple"):
            op._validate_and_wrap_output(np.array([1.0]), None, None)

    def test_eo_non_none_wrong_eo_type_raises(self):
        """
        目的：验证 EO 非 None 时 EO 实例类型不匹配抛出 TypeError
        输入：tuple(ndarray, SimpleConfig) 而非 tuple(ndarray, MyExtraOutput)
        预期：抛出 TypeError
        """
        op = OperatorWithEO()
        with pytest.raises(TypeError, match="附加输出类型必须是"):
            op._validate_and_wrap_output(
                (np.array([1.0]), SimpleConfig()), None, None
            )

    def test_eo_none_valid_ndarray(self):
        """
        目的：验证 EO 为 None 时 ndarray 正常打包
        输入：ndarray + meta=None
        预期：返回 ndarray
        """
        op = SimpleOperator()
        arr = np.array([[1.0, 2.0]])
        result = op._validate_and_wrap_output(arr, None, None)
        assert isinstance(result, np.ndarray)
        np.testing.assert_array_equal(result, arr)

    def test_eo_non_none_valid_tuple(self):
        """
        目的：验证 EO 非 None 时 tuple(ndarray, EO) 正常打包
        输入：(ndarray, MyExtraOutput)
        预期：返回 (NumericData, MyExtraOutput)
        """
        op = OperatorWithEO()
        arr = np.array([[1.0, 2.0]])
        eo = MyExtraOutput(info="test")
        result = op._validate_and_wrap_output((arr, eo), None, None)
        assert isinstance(result, tuple)
        assert isinstance(result[0], np.ndarray)
        assert result[1] is eo


# ============================================================================
# BiNumericOperator 测试
# ============================================================================

class TestBiNumericOperator:
    """测试 BiNumericOperator"""

    def test_default_name_output_columns(self, sample_dataframe):
        """
        目的：验证 BiNumericOperator 默认列名沿用 x_pred 列名
        输入：DataFrame 元信息
        预期：返回 x_pred 的列名
        """
        op = SimpleBiOperator()
        meta = DataFrameMeta.from_dataframe(sample_dataframe)
        columns = op._name_output_columns(np.array([[1.0]]), meta, None)
        assert columns == ["a", "b", "c"]

    def test_run_with_ndarray(self, sample_ndarray):
        """
        目的：验证 BiNumericOperator 正常运行
        输入：tuple(ndarray, ndarray)
        预期：返回 ndarray
        """
        op = SimpleBiOperator()
        result = op.run((sample_ndarray, sample_ndarray))
        assert isinstance(result, np.ndarray)

    def test_run_with_dataframe(self, sample_dataframe):
        """
        目的：验证 BiNumericOperator DataFrame 输入输出
        输入：tuple(DataFrame, DataFrame)
        预期：返回 DataFrame
        """
        op = SimpleBiOperator()
        result = op.run((sample_dataframe, sample_dataframe))
        assert isinstance(result, pd.DataFrame)


# ============================================================================
# SupervisedNumericOperatorMixin 测试
# ============================================================================

class TestSupervisedNumericOperatorMixin:
    """测试 SupervisedNumericOperatorMixin 模板方法管线"""

    def test_fit_template_method(self, sample_ndarray):
        """
        目的：验证有监督训练模板方法正确执行
        输入：x=ndarray, y=ndarray
        预期：训练后 is_fitted 为 True
        """
        op = SupervisedLearnableOp()
        y = np.random.randn(len(sample_ndarray), 1)
        op.fit(sample_ndarray, y)
        assert op.is_fitted is True

    def test_fit_with_dataframe(self, sample_dataframe):
        """
        目的：验证 DataFrame 输入训练正常执行
        输入：x=DataFrame, y=DataFrame
        预期：训练后 is_fitted 为 True
        """
        op = SupervisedLearnableOp()
        y = pd.DataFrame(np.random.randn(len(sample_dataframe), 1), columns=["label"])
        op.fit(sample_dataframe, y)
        assert op.is_fitted is True

    def test_validate_fit_input_invalid_x_raises(self):
        """
        目的：验证训练输入 x 类型不合法时抛出 TypeError
        输入：x="invalid", y=ndarray, params=None
        预期：抛出 TypeError
        """
        op = SupervisedLearnableOp()
        with pytest.raises(TypeError, match="输入数据类型必须是"):
            op._validate_fit_input("invalid", np.array([1.0]), params=None)

    def test_validate_fit_input_invalid_y_raises(self):
        """
        目的：验证训练输入 y 类型不合法时抛出 TypeError
        输入：x=ndarray, y="invalid", params=None
        预期：抛出 TypeError
        """
        op = SupervisedLearnableOp()
        with pytest.raises(TypeError, match="输入数据类型必须是"):
            op._validate_fit_input(np.array([1.0]), "invalid", params=None)

    def test_unwrap_fit_data_dataframe(self, sample_dataframe):
        """
        目的：验证 DataFrame 训练数据正确解包
        输入：x=DataFrame, y=DataFrame
        预期：x_meta 和 y_meta 非 None，数据为 ndarray
        """
        op = SupervisedLearnableOp()
        y = pd.DataFrame(np.random.randn(len(sample_dataframe), 1), columns=["label"])
        x_meta, x_data, y_meta, y_data = op._unwrap_fit_data(sample_dataframe, y, None)
        assert x_meta is not None
        assert y_meta is not None
        assert isinstance(x_data, np.ndarray)
        assert isinstance(y_data, np.ndarray)

    def test_unwrap_fit_data_ndarray(self, sample_ndarray):
        """
        目的：验证 ndarray 训练数据正确解包
        输入：x=ndarray, y=ndarray
        预期：x_meta 和 y_meta 均为 None
        """
        op = SupervisedLearnableOp()
        y = np.random.randn(len(sample_ndarray), 1)
        x_meta, x_data, y_meta, y_data = op._unwrap_fit_data(sample_ndarray, y, None)
        assert x_meta is None
        assert y_meta is None

    def test_unwrap_fit_data_invalid_x_raises(self):
        """
        目的：验证 _unwrap_fit_data 中 x 类型不合法时抛出 TypeError
        输入：x="invalid"
        预期：抛出 TypeError
        """
        op = SupervisedLearnableOp()
        with pytest.raises(TypeError, match="数据类型必须是"):
            op._unwrap_fit_data("invalid", np.array([1.0]), None)

    def test_unwrap_fit_data_invalid_y_raises(self):
        """
        目的：验证 _unwrap_fit_data 中 y 类型不合法时抛出 TypeError
        输入：y="invalid"
        预期：抛出 TypeError
        """
        op = SupervisedLearnableOp()
        with pytest.raises(TypeError, match="数据类型必须是"):
            op._unwrap_fit_data(np.array([1.0]), "invalid", None)


# ============================================================================
# UnsupervisedNumericOperatorMixin 测试
# ============================================================================

class TestUnsupervisedNumericOperatorMixin:
    """测试 UnsupervisedNumericOperatorMixin 模板方法管线"""

    def test_validate_fit_input_invalid_raises(self):
        """
        目的：验证训练输入类型不合法时抛出 TypeError
        输入：x="invalid"
        预期：抛出 TypeError
        """
        op = SimpleLearnableOp()
        with pytest.raises(TypeError, match="输入数据类型必须是"):
            op._validate_fit_input("invalid", None)

    def test_validate_fit_input_ndarray_passes(self, sample_ndarray):
        """
        目的：验证 ndarray 训练输入通过校验
        输入：ndarray
        预期：无异常
        """
        op = SimpleLearnableOp()
        op._validate_fit_input(sample_ndarray, None)

    def test_validate_fit_input_dataframe_passes(self, sample_dataframe):
        """
        目的：验证 DataFrame 训练输入通过校验
        输入：DataFrame
        预期：无异常
        """
        op = SimpleLearnableOp()
        op._validate_fit_input(sample_dataframe, None)

    def test_unwrap_fit_data_invalid_raises(self):
        """
        目的：验证 _unwrap_fit_data 输入类型不合法时抛出 TypeError
        输入：x="invalid"
        预期：抛出 TypeError
        """
        op = SimpleLearnableOp()
        with pytest.raises(TypeError, match="数据类型必须是"):
            op._unwrap_fit_data("invalid", None)

    def test_unwrap_fit_data_dataframe(self, sample_dataframe):
        """
        目的：验证 DataFrame 训练数据正确解包
        输入：DataFrame
        预期：meta 非 None，数据为 ndarray
        """
        op = SimpleLearnableOp()
        meta, data = op._unwrap_fit_data(sample_dataframe, None)
        assert meta is not None
        assert isinstance(data, np.ndarray)

    def test_unwrap_fit_data_ndarray(self, sample_ndarray):
        """
        目的：验证 ndarray 训练数据正确解包
        输入：ndarray
        预期：meta 为 None
        """
        op = SimpleLearnableOp()
        meta, data = op._unwrap_fit_data(sample_ndarray, None)
        assert meta is None
        assert data is sample_ndarray

    def test_fit_sets_fitted(self, sample_ndarray):
        """
        目的：验证无监督训练后 is_fitted 为 True
        输入：fit(sample_ndarray)
        预期：is_fitted 为 True
        """
        op = SimpleLearnableOp()
        op.fit(sample_ndarray)
        assert op.is_fitted is True

    def test_fit_with_params(self, sample_ndarray):
        """
        目的：验证带训练参数的 fit 正常执行
        输入：fit(x, params=MyFitParams(epochs=5))
        预期：last_fit_params.epochs == 5
        """
        op = SimpleLearnableOp()
        op.fit(sample_ndarray, params=MyFitParams(epochs=5))
        assert op.last_fit_params.epochs == 5

    def test_fit_with_kwargs(self, sample_ndarray):
        """
        目的：验证用 kwargs 传递训练参数
        输入：fit(x, epochs=8)
        预期：last_fit_params.epochs == 8
        """
        op = SimpleLearnableOp()
        op.fit(sample_ndarray, epochs=8)
        assert op.last_fit_params.epochs == 8
