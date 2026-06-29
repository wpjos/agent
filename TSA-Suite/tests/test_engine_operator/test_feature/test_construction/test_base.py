# -*- coding: utf-8 -*-

"""
特征构造基类测试

测试覆盖:
    1. 枚举类型: Alignment / Padding 值正确性
    2. Config 校验: input_columns min_length、window_size gt=0
    3. BaseFeatureMixin 方法: state 属性、_filter_data、_make_output_column_name、
       _get_compute_params / _get_train_params 返回空字典、train 返回 None
    4. BaseFeature: run 管线（DataFrame / ndarray 输入）、输入校验
    5. LearnableFeature: fit / run 管线、state 属性、is_fitted
    6. IndependentFeatureMixin: _name_output_columns 分组命名
    7. JointFeatureMixin: 行为透传
    8. MapFeatureMixin: _run_data 调用 compute
    9. WindowFeatureMixin: 滑动窗口、填充模式、对齐、_adjust_index
    10. 8 个编排基类端到端
    11. Learnable 特征的 fit/run/state
    12. save/load
    13. 边界情况: 1D 输入、多列输入、DataFrame 输入
"""

from pathlib import Path

import numpy as np
import pandas as pd
import pytest
from pydantic import BaseModel, ConfigDict, Field, ValidationError

from tsas.engine.operator.feature.construction.base import (
    Alignment,
    Padding,
    BaseFeatureConfig,
    WindowFeatureConfig,
    BaseFeatureMixin,
    BaseFeature,
    LearnableFeature,
    IndependentFeatureMixin,
    JointFeatureMixin,
    MapFeatureMixin,
    WindowFeatureMixin,
    IndependentMapFeature,
    IndependentWindowFeature,
    JointMapFeature,
    JointWindowFeature,
    LearnableIndependentMapFeature,
    LearnableIndependentWindowFeature,
    LearnableJointMapFeature,
    LearnableJointWindowFeature,
)


# ============================================================================
# 测试辅助：具体子类定义
# ============================================================================

class _SimpleConfig(BaseFeatureConfig):
    """测试用简单 Config"""
    pass


class _SimpleWindowConfig(WindowFeatureConfig):
    """测试用窗口 Config"""
    pass


class _ConcreteMapFeature(IndependentMapFeature[_SimpleConfig]):
    """测试用 IndependentMapFeature 具体实现：逐元素乘 2"""

    @classmethod
    def name(cls) -> str:
        return "concrete_map_feature"

    @classmethod
    def version(cls) -> tuple[int, ...]:
        return (1, 0, 0)

    @staticmethod
    def compute(x: np.ndarray, *, state=None, **params) -> np.ndarray:
        return x * 2

    def _name_output_column(self, input_col: str, output_val) -> str:
        return self._make_output_column_name(input_col, "double")


class _ConcreteWindowFeature(IndependentWindowFeature[_SimpleWindowConfig]):
    """测试用 IndependentWindowFeature 具体实现：窗口求和"""

    @classmethod
    def name(cls) -> str:
        return "concrete_window_feature"

    @classmethod
    def version(cls) -> tuple[int, ...]:
        return (1, 0, 0)

    @staticmethod
    def compute(x: np.ndarray, *, state=None, **params) -> np.ndarray:
        return np.sum(x, axis=0)

    def _name_output_column(self, input_col: str, output_val) -> str:
        return self._make_output_column_name(input_col, "win_sum")


class _ConcreteJointMapFeature(JointMapFeature[_SimpleConfig]):
    """测试用 JointMapFeature 具体实现：行求和"""

    @classmethod
    def name(cls) -> str:
        return "concrete_joint_map_feature"

    @classmethod
    def version(cls) -> tuple[int, ...]:
        return (1, 0, 0)

    @staticmethod
    def compute(x: np.ndarray, *, state=None, **params) -> np.ndarray:
        return np.sum(x, axis=1)

    def _name_output_columns(self, output_data: np.ndarray, meta, params) -> list[str]:
        return ["joint_sum"]


class _ConcreteJointWindowFeature(JointWindowFeature[_SimpleWindowConfig]):
    """测试用 JointWindowFeature 具体实现：多列窗口求和"""

    @classmethod
    def name(cls) -> str:
        return "concrete_joint_window_feature"

    @classmethod
    def version(cls) -> tuple[int, ...]:
        return (1, 0, 0)

    @staticmethod
    def compute(x: np.ndarray, *, state=None, **params) -> np.ndarray:
        return np.array([np.sum(x)])

    def _name_output_columns(self, output_data: np.ndarray, meta, params) -> list[str]:
        return ["joint_win_sum"]


class _LearnableState(BaseModel):
    """测试用 Learnable 状态"""
    model_config = ConfigDict(arbitrary_types_allowed=True)
    offset: np.ndarray


class _ConcreteLearnableMapFeature(LearnableIndependentMapFeature[_SimpleConfig, _LearnableState]):
    """测试用 LearnableIndependentMapFeature：用训练均值做偏移"""

    @classmethod
    def name(cls) -> str:
        return "concrete_learnable_map_feature"

    @classmethod
    def version(cls) -> tuple[int, ...]:
        return (1, 0, 0)

    @staticmethod
    def train(x: np.ndarray, **params) -> _LearnableState:
        offset = np.atleast_1d(x.mean(axis=0))
        return _LearnableState(offset=offset)

    @staticmethod
    def compute(x: np.ndarray, *, state=None, **params) -> np.ndarray:
        if state is None:
            raise ValueError("需要 state")
        return x - state.offset

    def _name_output_column(self, input_col: str, output_val) -> str:
        return self._make_output_column_name(input_col, "centered")

    def save(self, path):
        super().save(path)
        path = Path(path)
        if self._state is not None:
            np.save(path / "offset.npy", self._state.offset)

    @classmethod
    def load(cls, path, *, oid=None):
        from pathlib import Path as P
        instance = super().load(path, oid=oid)
        p = P(path)
        offset_file = p / "offset.npy"
        if offset_file.exists():
            instance._state = _LearnableState(offset=np.load(offset_file))
            instance._fitted = True
        return instance


class _ConcreteLearnableWindowFeature(LearnableIndependentWindowFeature[_SimpleWindowConfig, _LearnableState]):
    """测试用 LearnableIndependentWindowFeature"""

    @classmethod
    def name(cls) -> str:
        return "concrete_learnable_window_feature"

    @classmethod
    def version(cls) -> tuple[int, ...]:
        return (1, 0, 0)

    @staticmethod
    def train(x: np.ndarray, **params) -> _LearnableState:
        offset = np.atleast_1d(x.mean(axis=0))
        return _LearnableState(offset=offset)

    @staticmethod
    def compute(x: np.ndarray, *, state=None, **params) -> np.ndarray:
        if state is None:
            raise ValueError("需要 state")
        return np.sum(x - state.offset, axis=0)

    def _name_output_column(self, input_col: str, output_val) -> str:
        return self._make_output_column_name(input_col, "lwin")


class _ConcreteLearnableJointMapFeature(LearnableJointMapFeature[_SimpleConfig, _LearnableState]):
    """测试用 LearnableJointMapFeature"""

    @classmethod
    def name(cls) -> str:
        return "concrete_learnable_joint_map_feature"

    @classmethod
    def version(cls) -> tuple[int, ...]:
        return (1, 0, 0)

    @staticmethod
    def train(x: np.ndarray, **params) -> _LearnableState:
        offset = np.atleast_1d(x.mean(axis=0))
        return _LearnableState(offset=offset)

    @staticmethod
    def compute(x: np.ndarray, *, state=None, **params) -> np.ndarray:
        if state is None:
            raise ValueError("需要 state")
        return np.sum(x - state.offset, axis=1)

    def _name_output_columns(self, output_data: np.ndarray, meta, params) -> list[str]:
        return ["ljoint_centered_sum"]


class _ConcreteLearnableJointWindowFeature(LearnableJointWindowFeature[_SimpleWindowConfig, _LearnableState]):
    """测试用 LearnableJointWindowFeature"""

    @classmethod
    def name(cls) -> str:
        return "concrete_learnable_joint_window_feature"

    @classmethod
    def version(cls) -> tuple[int, ...]:
        return (1, 0, 0)

    @staticmethod
    def train(x: np.ndarray, **params) -> _LearnableState:
        offset = np.atleast_1d(x.mean(axis=0))
        return _LearnableState(offset=offset)

    @staticmethod
    def compute(x: np.ndarray, *, state=None, **params) -> np.ndarray:
        if state is None:
            raise ValueError("需要 state")
        return np.array([np.sum(x - state.offset)])

    def _name_output_columns(self, output_data: np.ndarray, meta, params) -> list[str]:
        return ["ljoint_win_sum"]


# ============================================================================
# 1. 枚举类型测试
# ============================================================================

class TestAlignment:
    """Alignment 枚举值测试"""

    def test_values(self):
        assert Alignment.LEFT == "left"
        assert Alignment.RIGHT == "right"

    def test_members(self):
        assert set(Alignment.__members__.keys()) == {"LEFT", "RIGHT"}

    def test_is_string_enum(self):
        assert isinstance(Alignment.LEFT, str)
        assert isinstance(Alignment.RIGHT, str)


class TestPadding:
    """Padding 枚举值测试"""

    def test_values(self):
        assert Padding.EDGE == "edge"
        assert Padding.NAN == "nan"
        assert Padding.REFLECT == "reflect"
        assert Padding.RING == "ring"

    def test_members(self):
        assert set(Padding.__members__.keys()) == {"EDGE", "NAN", "REFLECT", "RING"}

    def test_is_string_enum(self):
        for member in Padding:
            assert isinstance(member, str)


# ============================================================================
# 2. Config 校验测试
# ============================================================================

class TestBaseFeatureConfig:
    """BaseFeatureConfig 校验测试"""

    def test_valid_config(self):
        cfg = BaseFeatureConfig(input_columns=["a", "b"])
        assert cfg.input_columns == ["a", "b"]

    def test_single_column(self):
        cfg = BaseFeatureConfig(input_columns=["x"])
        assert cfg.input_columns == ["x"]

    def test_empty_columns_raises(self):
        with pytest.raises(ValidationError):
            BaseFeatureConfig(input_columns=[])

    def test_frozen(self):
        cfg = BaseFeatureConfig(input_columns=["a"])
        with pytest.raises(ValidationError):
            cfg.input_columns = ["b"]


class TestWindowFeatureConfig:
    """WindowFeatureConfig 校验测试"""

    def test_valid_config(self):
        cfg = WindowFeatureConfig(
            input_columns=["a"],
            window_size=3,
        )
        assert cfg.window_size == 3
        assert cfg.padding is None
        assert cfg.alignment == Alignment.RIGHT

    def test_window_size_zero_raises(self):
        with pytest.raises(ValidationError):
            WindowFeatureConfig(input_columns=["a"], window_size=0)

    def test_window_size_negative_raises(self):
        with pytest.raises(ValidationError):
            WindowFeatureConfig(input_columns=["a"], window_size=-1)

    def test_padding_edge(self):
        cfg = WindowFeatureConfig(
            input_columns=["a"],
            window_size=3,
            padding=Padding.EDGE,
        )
        assert cfg.padding == Padding.EDGE

    def test_padding_numeric(self):
        cfg = WindowFeatureConfig(
            input_columns=["a"],
            window_size=3,
            padding=0.0,
        )
        assert cfg.padding == 0.0

    def test_alignment_left(self):
        cfg = WindowFeatureConfig(
            input_columns=["a"],
            window_size=3,
            alignment=Alignment.LEFT,
        )
        assert cfg.alignment == Alignment.LEFT

    def test_inherits_base(self):
        assert issubclass(WindowFeatureConfig, BaseFeatureConfig)


# ============================================================================
# 3. BaseFeatureMixin 方法测试
# ============================================================================

class TestBaseFeatureMixinMethods:
    """BaseFeatureMixin 各方法的独立测试"""

    def _make_instance(self, input_columns=None, window_size=None):
        """创建一个挂载了 config 的 _ConcreteMapFeature 实例"""
        if window_size is not None:
            config = _SimpleWindowConfig(input_columns=input_columns or ["a"], window_size=window_size)
            return _ConcreteWindowFeature(config=config)
        config = _SimpleConfig(input_columns=input_columns or ["a"])
        return _ConcreteMapFeature(config=config)

    def test_state_returns_none(self):
        feat = self._make_instance()
        assert feat.state is None

    def test_get_compute_params_default_empty(self):
        feat = self._make_instance()
        assert feat._get_compute_params() == {}

    def test_get_train_params_default_empty(self):
        feat = self._make_instance()
        assert feat._get_train_params() == {}

    def test_train_default_returns_none(self):
        x = np.array([1.0, 2.0, 3.0])
        assert BaseFeatureMixin.train(x) is None

    def test_make_output_column_name_without_value(self):
        feat = self._make_instance()
        result = feat._make_output_column_name("col_a", "square")
        assert result == "col_a_square"

    def test_make_output_column_name_with_value(self):
        feat = self._make_instance()
        result = feat._make_output_column_name("col_a", "poly", "2")
        assert result == "col_a_poly_2"

    def test_filter_data_dataframe(self):
        feat = self._make_instance(input_columns=["b", "a"])
        df = pd.DataFrame({"a": [1, 2], "b": [3, 4], "c": [5, 6]})
        filtered = feat._filter_data(df, None)
        assert list(filtered.columns) == ["b", "a"]
        assert filtered.shape == (2, 2)

    def test_filter_data_ndarray_1d(self):
        feat = self._make_instance(input_columns=["a"])
        arr = np.array([1.0, 2.0, 3.0])
        filtered = feat._filter_data(arr, None)
        np.testing.assert_array_equal(filtered, arr)

    def test_filter_data_ndarray_2d(self):
        feat = self._make_instance(input_columns=["a"])
        arr = np.array([[1.0, 10.0], [2.0, 20.0], [3.0, 30.0]])
        filtered = feat._filter_data(arr, None)
        assert filtered.shape == (3, 1)
        np.testing.assert_array_equal(filtered[:, 0], [1.0, 2.0, 3.0])

    def test_filter_data_unsupported_type_raises(self):
        feat = self._make_instance()
        with pytest.raises(TypeError, match="数据类型必须是"):
            feat._filter_data([1, 2, 3], None)

    def test_validate_dataframe_input_missing_columns(self):
        feat = self._make_instance(input_columns=["a", "b", "missing"])
        df = pd.DataFrame({"a": [1], "b": [2]})
        with pytest.raises(ValueError, match="缺少以下列"):
            feat._validate_dataframe_input(df, None)

    def test_validate_dataframe_input_ok(self):
        feat = self._make_instance(input_columns=["a", "b"])
        df = pd.DataFrame({"a": [1], "b": [2], "c": [3]})
        feat._validate_dataframe_input(df, None)  # 不抛异常

    def test_validate_ndarray_input_1d_ok(self):
        feat = self._make_instance(input_columns=["a"])
        arr = np.array([1.0, 2.0])
        feat._validate_ndarray_input(arr, None)  # 不抛异常

    def test_validate_ndarray_input_1d_insufficient(self):
        feat = self._make_instance(input_columns=["a", "b"])
        arr = np.array([1.0, 2.0])
        with pytest.raises(ValueError, match="列数要求至少为 2"):
            feat._validate_ndarray_input(arr, None)

    def test_validate_ndarray_input_2d_ok(self):
        feat = self._make_instance(input_columns=["a", "b"])
        arr = np.array([[1.0, 2.0], [3.0, 4.0]])
        feat._validate_ndarray_input(arr, None)

    def test_validate_ndarray_input_2d_insufficient(self):
        feat = self._make_instance(input_columns=["a", "b", "c"])
        arr = np.array([[1.0, 2.0]])
        with pytest.raises(ValueError, match="列数要求至少为 3"):
            feat._validate_ndarray_input(arr, None)


# ============================================================================
# 4. BaseFeature 测试 (不可训练)
# ============================================================================

class TestBaseFeatureRun:
    """BaseFeature run 管线测试"""

    def test_run_with_dataframe(self):
        config = _SimpleConfig(input_columns=["a"])
        feat = _ConcreteMapFeature(config=config)
        df = pd.DataFrame({"a": [1.0, 2.0, 3.0], "b": [4.0, 5.0, 6.0]})
        result = feat.run(df)
        assert isinstance(result, pd.DataFrame)
        assert "a_double" in result.columns
        np.testing.assert_array_equal(result["a_double"].values, [2.0, 4.0, 6.0])

    def test_run_with_ndarray(self):
        config = _SimpleConfig(input_columns=["a"])
        feat = _ConcreteMapFeature(config=config)
        arr = np.array([1.0, 2.0, 3.0])
        result = feat.run(arr)
        assert isinstance(result, np.ndarray)
        np.testing.assert_array_equal(result, [2.0, 4.0, 6.0])

    def test_run_preserves_index(self):
        config = _SimpleConfig(input_columns=["a"])
        feat = _ConcreteMapFeature(config=config)
        idx = pd.Index([10, 20, 30])
        df = pd.DataFrame({"a": [1.0, 2.0, 3.0]}, index=idx)
        result = feat.run(df)
        assert list(result.index) == [10, 20, 30]

    def test_run_multi_column_dataframe(self):
        config = _SimpleConfig(input_columns=["a", "b"])
        feat = _ConcreteMapFeature(config=config)
        df = pd.DataFrame({"a": [1.0, 2.0], "b": [3.0, 4.0]})
        result = feat.run(df)
        assert isinstance(result, pd.DataFrame)
        assert "a_double" in result.columns
        assert "b_double" in result.columns
        np.testing.assert_array_equal(result["a_double"].values, [2.0, 4.0])
        np.testing.assert_array_equal(result["b_double"].values, [6.0, 8.0])

    def test_run_multi_column_ndarray(self):
        config = _SimpleConfig(input_columns=["a", "b"])
        feat = _ConcreteMapFeature(config=config)
        arr = np.array([[1.0, 3.0], [2.0, 4.0]])
        result = feat.run(arr)
        assert isinstance(result, np.ndarray)
        np.testing.assert_array_equal(result[:, 0], [2.0, 4.0])
        np.testing.assert_array_equal(result[:, 1], [6.0, 8.0])

    def test_run_invalid_type_raises(self):
        config = _SimpleConfig(input_columns=["a"])
        feat = _ConcreteMapFeature(config=config)
        with pytest.raises(TypeError, match="输入数据类型必须是"):
            feat.run("not_valid")


# ============================================================================
# 5. LearnableFeature 测试
# ============================================================================

class TestLearnableFeature:
    """LearnableFeature fit/run/state 测试"""

    def test_initial_state_is_none(self):
        config = _SimpleConfig(input_columns=["a"])
        feat = _ConcreteLearnableMapFeature(config=config)
        assert feat.state is None
        assert feat.is_fitted is False

    def test_fit_sets_state(self):
        config = _SimpleConfig(input_columns=["a"])
        feat = _ConcreteLearnableMapFeature(config=config)
        df = pd.DataFrame({"a": [1.0, 2.0, 3.0]})
        feat.fit(df)
        assert feat.is_fitted is True
        assert feat.state is not None
        np.testing.assert_almost_equal(feat.state.offset, np.array([2.0]))

    def test_run_after_fit(self):
        config = _SimpleConfig(input_columns=["a"])
        feat = _ConcreteLearnableMapFeature(config=config)
        train_df = pd.DataFrame({"a": [1.0, 2.0, 3.0]})
        feat.fit(train_df)
        test_df = pd.DataFrame({"a": [4.0, 5.0, 6.0]})
        result = feat.run(test_df)
        assert isinstance(result, pd.DataFrame)
        np.testing.assert_array_almost_equal(result["a_centered"].values, [2.0, 3.0, 4.0])

    def test_run_before_fit_raises(self):
        config = _SimpleConfig(input_columns=["a"])
        feat = _ConcreteLearnableMapFeature(config=config)
        df = pd.DataFrame({"a": [1.0]})
        with pytest.raises(RuntimeError, match="训练尚未完成"):
            feat.run(df)

    def test_fit_returns_self(self):
        config = _SimpleConfig(input_columns=["a"])
        feat = _ConcreteLearnableMapFeature(config=config)
        df = pd.DataFrame({"a": [1.0, 2.0]})
        result = feat.fit(df)
        assert result is feat

    def test_fit_with_ndarray(self):
        config = _SimpleConfig(input_columns=["a"])
        feat = _ConcreteLearnableMapFeature(config=config)
        arr = np.array([1.0, 2.0, 3.0])
        feat.fit(arr)
        assert feat.is_fitted is True
        assert feat.state is not None

    def test_run_with_ndarray_after_fit(self):
        config = _SimpleConfig(input_columns=["a"])
        feat = _ConcreteLearnableMapFeature(config=config)
        feat.fit(np.array([1.0, 2.0, 3.0]))
        result = feat.run(np.array([4.0, 5.0]))
        assert isinstance(result, np.ndarray)
        np.testing.assert_array_almost_equal(result, [2.0, 3.0])


# ============================================================================
# 6. IndependentFeatureMixin 测试
# ============================================================================

class TestIndependentFeatureMixin:
    """IndependentFeatureMixin._name_output_columns 分组命名测试"""

    def test_1_to_1_column_naming(self):
        config = _SimpleConfig(input_columns=["x"])
        feat = _ConcreteMapFeature(config=config)
        df = pd.DataFrame({"x": [1.0, 2.0]})
        result = feat.run(df)
        assert list(result.columns) == ["x_double"]

    def test_multi_column_naming(self):
        config = _SimpleConfig(input_columns=["a", "b"])
        feat = _ConcreteMapFeature(config=config)
        df = pd.DataFrame({"a": [1.0], "b": [2.0]})
        result = feat.run(df)
        assert "a_double" in result.columns
        assert "b_double" in result.columns


# ============================================================================
# 7. JointFeatureMixin 测试
# ============================================================================

class TestJointFeatureMixin:
    """JointFeatureMixin 行为测试"""

    def test_joint_map_runs_on_all_columns(self):
        config = _SimpleConfig(input_columns=["a", "b"])
        feat = _ConcreteJointMapFeature(config=config)
        df = pd.DataFrame({"a": [1.0, 2.0], "b": [3.0, 4.0]})
        result = feat.run(df)
        assert isinstance(result, pd.DataFrame)
        assert "joint_sum" in result.columns
        np.testing.assert_array_equal(result["joint_sum"].values, [4.0, 6.0])


# ============================================================================
# 8. MapFeatureMixin 测试
# ============================================================================

class TestMapFeatureMixin:
    """MapFeatureMixin._run_data 测试"""

    def test_run_data_calls_compute_with_full_data(self):
        config = _SimpleConfig(input_columns=["a"])
        feat = _ConcreteMapFeature(config=config)
        x = np.array([[1.0], [2.0], [3.0]])
        result = feat._run_data(x, None)
        np.testing.assert_array_equal(result, [[2.0], [4.0], [6.0]])

    def test_run_data_1d_input(self):
        config = _SimpleConfig(input_columns=["a"])
        feat = _ConcreteMapFeature(config=config)
        x = np.array([1.0, 2.0, 3.0])
        result = feat._run_data(x, None)
        np.testing.assert_array_equal(result, [2.0, 4.0, 6.0])


# ============================================================================
# 9. WindowFeatureMixin 测试
# ============================================================================

class TestWindowFeatureMixin:
    """WindowFeatureMixin 滑动窗口、填充、对齐测试"""

    def test_sliding_window_no_padding_right_align(self):
        """无填充右对齐: 输出行数 = 输入行数 - window_size + 1"""
        config = _SimpleWindowConfig(input_columns=["a"], window_size=3, alignment=Alignment.RIGHT)
        feat = _ConcreteWindowFeature(config=config)
        df = pd.DataFrame({"a": [1.0, 2.0, 3.0, 4.0, 5.0]})
        result = feat.run(df)
        assert isinstance(result, pd.DataFrame)
        assert len(result) == 3  # 5 - 3 + 1
        # 窗口求和: [1+2+3, 2+3+4, 3+4+5]
        np.testing.assert_array_equal(result["a_win_sum"].values, [6.0, 9.0, 12.0])

    def test_sliding_window_no_padding_left_align(self):
        """无填充左对齐: 输出行数 = 输入行数 - window_size + 1"""
        config = _SimpleWindowConfig(input_columns=["a"], window_size=3, alignment=Alignment.LEFT)
        feat = _ConcreteWindowFeature(config=config)
        df = pd.DataFrame({"a": [1.0, 2.0, 3.0, 4.0, 5.0]})
        result = feat.run(df)
        assert len(result) == 3
        # 窗口求和: [1+2+3, 2+3+4, 3+4+5] 相同数值，但索引对齐不同
        np.testing.assert_array_equal(result["a_win_sum"].values, [6.0, 9.0, 12.0])

    def test_no_padding_right_align_index(self):
        """无填充右对齐索引: 取后 num_windows 个"""
        config = _SimpleWindowConfig(input_columns=["a"], window_size=3, alignment=Alignment.RIGHT)
        feat = _ConcreteWindowFeature(config=config)
        idx = pd.Index([0, 1, 2, 3, 4])
        df = pd.DataFrame({"a": [1.0, 2.0, 3.0, 4.0, 5.0]}, index=idx)
        result = feat.run(df)
        assert list(result.index) == [2, 3, 4]

    def test_no_padding_left_align_index(self):
        """无填充左对齐索引: 取前 num_windows 个"""
        config = _SimpleWindowConfig(input_columns=["a"], window_size=3, alignment=Alignment.LEFT)
        feat = _ConcreteWindowFeature(config=config)
        idx = pd.Index([0, 1, 2, 3, 4])
        df = pd.DataFrame({"a": [1.0, 2.0, 3.0, 4.0, 5.0]}, index=idx)
        result = feat.run(df)
        assert list(result.index) == [0, 1, 2]

    def test_edge_padding_right_align(self):
        """EDGE 填充右对齐: 首行重复填充到头部"""
        config = _SimpleWindowConfig(
            input_columns=["a"],
            window_size=3,
            padding=Padding.EDGE,
            alignment=Alignment.RIGHT,
        )
        feat = _ConcreteWindowFeature(config=config)
        df = pd.DataFrame({"a": [1.0, 2.0, 3.0, 4.0]})
        result = feat.run(df)
        assert len(result) == 4  # 填充后输出行数 = 输入行数
        # 填充后输入: [1,1, 1, 2, 3, 4]
        # 窗口求和: [1+1+1, 1+1+2, 1+2+3, 2+3+4] = [3, 4, 6, 9]
        np.testing.assert_array_equal(result["a_win_sum"].values, [3.0, 4.0, 6.0, 9.0])

    def test_nan_padding_right_align(self):
        """NAN 填充: 边界处结果为 NaN"""
        config = _SimpleWindowConfig(
            input_columns=["a"],
            window_size=3,
            padding=Padding.NAN,
            alignment=Alignment.RIGHT,
        )
        feat = _ConcreteWindowFeature(config=config)
        df = pd.DataFrame({"a": [1.0, 2.0, 3.0]})
        result = feat.run(df)
        assert len(result) == 3
        # 填充后输入: [NaN, NaN, 1, 2, 3]
        # 窗口求和: [NaN, NaN, 6.0]
        assert np.isnan(result["a_win_sum"].values[0])
        assert np.isnan(result["a_win_sum"].values[1])
        np.testing.assert_almost_equal(result["a_win_sum"].values[2], 6.0)

    def test_reflect_padding_right_align(self):
        """REFLECT 填充右对齐"""
        config = _SimpleWindowConfig(
            input_columns=["a"],
            window_size=3,
            padding=Padding.REFLECT,
            alignment=Alignment.RIGHT,
        )
        feat = _ConcreteWindowFeature(config=config)
        df = pd.DataFrame({"a": [1.0, 2.0, 3.0, 4.0]})
        result = feat.run(df)
        assert len(result) == 4
        # 填充后输入: data[1:3][::-1] + data = [3,2, 1,2,3,4]
        # 窗口求和: [3+2+1, 2+1+2, 1+2+3, 2+3+4] = [6, 5, 6, 9]
        np.testing.assert_array_equal(result["a_win_sum"].values, [6.0, 5.0, 6.0, 9.0])

    def test_ring_padding_right_align(self):
        """RING 填充右对齐: 首尾相接"""
        config = _SimpleWindowConfig(
            input_columns=["a"],
            window_size=3,
            padding=Padding.RING,
            alignment=Alignment.RIGHT,
        )
        feat = _ConcreteWindowFeature(config=config)
        df = pd.DataFrame({"a": [1.0, 2.0, 3.0, 4.0]})
        result = feat.run(df)
        assert len(result) == 4
        # 填充后输入: data[-2:] + data = [3,4, 1,2,3,4]
        # 窗口求和: [3+4+1, 4+1+2, 1+2+3, 2+3+4] = [8, 7, 6, 9]
        np.testing.assert_array_equal(result["a_win_sum"].values, [8.0, 7.0, 6.0, 9.0])

    def test_edge_padding_left_align(self):
        """EDGE 填充左对齐: 末行重复填充到尾部"""
        config = _SimpleWindowConfig(
            input_columns=["a"],
            window_size=3,
            padding=Padding.EDGE,
            alignment=Alignment.LEFT,
        )
        feat = _ConcreteWindowFeature(config=config)
        df = pd.DataFrame({"a": [1.0, 2.0, 3.0, 4.0]})
        result = feat.run(df)
        assert len(result) == 4
        # 填充后输入: [1,2,3,4, 4,4]
        # 窗口求和: [1+2+3, 2+3+4, 3+4+4, 4+4+4] = [6, 9, 11, 12]
        np.testing.assert_array_equal(result["a_win_sum"].values, [6.0, 9.0, 11.0, 12.0])

    def test_numeric_padding(self):
        """数值填充: 用 0.0 填充"""
        config = _SimpleWindowConfig(
            input_columns=["a"],
            window_size=3,
            padding=0.0,
            alignment=Alignment.RIGHT,
        )
        feat = _ConcreteWindowFeature(config=config)
        df = pd.DataFrame({"a": [1.0, 2.0, 3.0]})
        result = feat.run(df)
        assert len(result) == 3
        # 填充后输入: [0, 0, 1, 2, 3]
        # 窗口求和: [0+0+1, 0+1+2, 1+2+3] = [1, 3, 6]
        np.testing.assert_array_equal(result["a_win_sum"].values, [1.0, 3.0, 6.0])

    def test_reflect_insufficient_data_raises(self):
        """REFLECT 填充数据不足时抛异常"""
        config = _SimpleWindowConfig(
            input_columns=["a"],
            window_size=4,
            padding=Padding.REFLECT,
            alignment=Alignment.RIGHT,
        )
        feat = _ConcreteWindowFeature(config=config)
        df = pd.DataFrame({"a": [1.0, 2.0, 3.0]})  # len=3, pad_width=3 >= len=3
        with pytest.raises(ValueError, match="镜像填充"):
            feat.run(df)

    def test_ring_insufficient_data_raises(self):
        """RING 填充数据不足时抛异常"""
        config = _SimpleWindowConfig(
            input_columns=["a"],
            window_size=4,
            padding=Padding.RING,
            alignment=Alignment.RIGHT,
        )
        feat = _ConcreteWindowFeature(config=config)
        df = pd.DataFrame({"a": [1.0, 2.0, 3.0]})
        with pytest.raises(ValueError, match="循环填充"):
            feat.run(df)

    def test_window_run_with_ndarray(self):
        """窗口特征 ndarray 输入"""
        config = _SimpleWindowConfig(input_columns=["a"], window_size=2, alignment=Alignment.RIGHT)
        feat = _ConcreteWindowFeature(config=config)
        arr = np.array([1.0, 2.0, 3.0, 4.0])
        result = feat.run(arr)
        assert isinstance(result, np.ndarray)
        np.testing.assert_array_equal(result, [3.0, 5.0, 7.0])


# ============================================================================
# 10. 八个编排基类端到端测试
# ============================================================================

class TestIndependentMapFeatureEndToEnd:
    """IndependentMapFeature 端到端"""

    def test_single_column(self):
        config = _SimpleConfig(input_columns=["val"])
        feat = _ConcreteMapFeature(config=config)
        df = pd.DataFrame({"val": [10.0, 20.0, 30.0]})
        result = feat.run(df)
        assert list(result.columns) == ["val_double"]
        np.testing.assert_array_equal(result["val_double"].values, [20.0, 40.0, 60.0])

    def test_multiple_columns(self):
        config = _SimpleConfig(input_columns=["x", "y"])
        feat = _ConcreteMapFeature(config=config)
        df = pd.DataFrame({"x": [1.0], "y": [2.0]})
        result = feat.run(df)
        assert set(result.columns) == {"x_double", "y_double"}

    def test_name(self):
        assert _ConcreteMapFeature.name() == "concrete_map_feature"

    def test_oid_property(self):
        config = _SimpleConfig(input_columns=["a"])
        feat = _ConcreteMapFeature(config=config)
        assert feat.oid.startswith("concrete_map_feature$")


class TestIndependentWindowFeatureEndToEnd:
    """IndependentWindowFeature 端到端"""

    def test_basic_window(self):
        config = _SimpleWindowConfig(input_columns=["a"], window_size=2)
        feat = _ConcreteWindowFeature(config=config)
        df = pd.DataFrame({"a": [1.0, 2.0, 3.0, 4.0]})
        result = feat.run(df)
        assert len(result) == 3
        np.testing.assert_array_equal(result["a_win_sum"].values, [3.0, 5.0, 7.0])

    def test_name(self):
        assert _ConcreteWindowFeature.name() == "concrete_window_feature"


class TestJointMapFeatureEndToEnd:
    """JointMapFeature 端到端"""

    def test_basic_joint_map(self):
        config = _SimpleConfig(input_columns=["a", "b"])
        feat = _ConcreteJointMapFeature(config=config)
        df = pd.DataFrame({"a": [1.0, 2.0], "b": [3.0, 4.0]})
        result = feat.run(df)
        assert "joint_sum" in result.columns
        np.testing.assert_array_equal(result["joint_sum"].values, [4.0, 6.0])

    def test_name(self):
        assert _ConcreteJointMapFeature.name() == "concrete_joint_map_feature"


class TestJointWindowFeatureEndToEnd:
    """JointWindowFeature 端到端"""

    def test_basic_joint_window(self):
        config = _SimpleWindowConfig(input_columns=["a", "b"], window_size=2)
        feat = _ConcreteJointWindowFeature(config=config)
        df = pd.DataFrame({"a": [1.0, 2.0, 3.0], "b": [4.0, 5.0, 6.0]})
        result = feat.run(df)
        assert len(result) == 2
        # 窗口0: sum of [[1,4],[2,5]] = 12
        # 窗口1: sum of [[2,5],[3,6]] = 16
        np.testing.assert_array_equal(result["joint_win_sum"].values, [12.0, 16.0])

    def test_name(self):
        assert _ConcreteJointWindowFeature.name() == "concrete_joint_window_feature"


class TestLearnableIndependentMapFeatureEndToEnd:
    """LearnableIndependentMapFeature 端到端"""

    def test_fit_and_run(self):
        config = _SimpleConfig(input_columns=["a"])
        feat = _ConcreteLearnableMapFeature(config=config)
        train_df = pd.DataFrame({"a": [10.0, 20.0, 30.0]})
        feat.fit(train_df)
        assert feat.is_fitted
        test_df = pd.DataFrame({"a": [10.0, 20.0, 30.0]})
        result = feat.run(test_df)
        np.testing.assert_array_almost_equal(result["a_centered"].values, [-10.0, 0.0, 10.0])

    def test_name(self):
        assert _ConcreteLearnableMapFeature.name() == "concrete_learnable_map_feature"


class TestLearnableIndependentWindowFeatureEndToEnd:
    """LearnableIndependentWindowFeature 端到端"""

    def test_fit_and_run(self):
        config = _SimpleWindowConfig(input_columns=["a"], window_size=2, padding=Padding.EDGE)
        feat = _ConcreteLearnableWindowFeature(config=config)
        train_df = pd.DataFrame({"a": [1.0, 2.0, 3.0]})
        feat.fit(train_df)
        assert feat.is_fitted
        test_df = pd.DataFrame({"a": [1.0, 2.0, 3.0]})
        result = feat.run(test_df)
        assert isinstance(result, pd.DataFrame)

    def test_name(self):
        assert _ConcreteLearnableWindowFeature.name() == "concrete_learnable_window_feature"


class TestLearnableJointMapFeatureEndToEnd:
    """LearnableJointMapFeature 端到端"""

    def test_fit_and_run(self):
        config = _SimpleConfig(input_columns=["a", "b"])
        feat = _ConcreteLearnableJointMapFeature(config=config)
        train_df = pd.DataFrame({"a": [1.0, 2.0], "b": [3.0, 4.0]})
        feat.fit(train_df)
        test_df = pd.DataFrame({"a": [1.0, 2.0], "b": [3.0, 4.0]})
        result = feat.run(test_df)
        assert "ljoint_centered_sum" in result.columns

    def test_name(self):
        assert _ConcreteLearnableJointMapFeature.name() == "concrete_learnable_joint_map_feature"


class TestLearnableJointWindowFeatureEndToEnd:
    """LearnableJointWindowFeature 端到端"""

    def test_fit_and_run(self):
        config = _SimpleWindowConfig(input_columns=["a", "b"], window_size=2, padding=Padding.EDGE)
        feat = _ConcreteLearnableJointWindowFeature(config=config)
        train_df = pd.DataFrame({"a": [1.0, 2.0, 3.0], "b": [4.0, 5.0, 6.0]})
        feat.fit(train_df)
        test_df = pd.DataFrame({"a": [1.0, 2.0, 3.0], "b": [4.0, 5.0, 6.0]})
        result = feat.run(test_df)
        assert "ljoint_win_sum" in result.columns

    def test_name(self):
        assert _ConcreteLearnableJointWindowFeature.name() == "concrete_learnable_joint_window_feature"


# ============================================================================
# 12. save/load 测试
# ============================================================================

class TestLearnableFeatureSaveLoad:
    """LearnableFeature save/load 回环测试"""

    def test_save_load_roundtrip(self, tmp_path):
        config = _SimpleConfig(input_columns=["a"])
        feat = _ConcreteLearnableMapFeature(config=config)
        df = pd.DataFrame({"a": [1.0, 2.0, 3.0]})
        feat.fit(df)

        save_dir = tmp_path / "feature"
        feat.save(save_dir)

        loaded = _ConcreteLearnableMapFeature.load(save_dir)
        assert loaded.is_fitted is True
        assert loaded.state is not None
        np.testing.assert_array_almost_equal(loaded.state.offset, feat.state.offset)

    def test_save_load_run_consistency(self, tmp_path):
        config = _SimpleConfig(input_columns=["a"])
        feat = _ConcreteLearnableMapFeature(config=config)
        df = pd.DataFrame({"a": [1.0, 2.0, 3.0]})
        feat.fit(df)

        save_dir = tmp_path / "feature"
        feat.save(save_dir)

        loaded = _ConcreteLearnableMapFeature.load(save_dir)
        test_df = pd.DataFrame({"a": [10.0, 20.0]})
        result = loaded.run(test_df)
        np.testing.assert_array_almost_equal(result["a_centered"].values, [8.0, 18.0])


class TestNonLearnableFeatureSaveLoad:
    """非 Learnable BaseFeature save/load 测试"""

    def test_save_load_roundtrip(self, tmp_path):
        config = _SimpleConfig(input_columns=["a"])
        feat = _ConcreteMapFeature(config=config)
        save_dir = tmp_path / "feature"
        feat.save(save_dir)

        loaded = _ConcreteMapFeature.load(save_dir)
        assert loaded.config.input_columns == ["a"]
        df = pd.DataFrame({"a": [1.0, 2.0]})
        result = loaded.run(df)
        np.testing.assert_array_equal(result["a_double"].values, [2.0, 4.0])


# ============================================================================
# 13. 边界情况
# ============================================================================

class TestEdgeCases:
    """边界情况测试"""

    def test_1d_ndarray_input(self):
        """1D ndarray 输入"""
        config = _SimpleConfig(input_columns=["a"])
        feat = _ConcreteMapFeature(config=config)
        result = feat.run(np.array([1.0, 2.0]))
        assert isinstance(result, np.ndarray)
        np.testing.assert_array_equal(result, [2.0, 4.0])

    def test_single_row_dataframe(self):
        """单行 DataFrame"""
        config = _SimpleConfig(input_columns=["a"])
        feat = _ConcreteMapFeature(config=config)
        df = pd.DataFrame({"a": [5.0]})
        result = feat.run(df)
        assert len(result) == 1
        np.testing.assert_array_equal(result["a_double"].values, [10.0])

    def test_dataframe_with_extra_columns(self):
        """DataFrame 含额外列，只取 input_columns"""
        config = _SimpleConfig(input_columns=["a"])
        feat = _ConcreteMapFeature(config=config)
        df = pd.DataFrame({"a": [1.0], "b": [2.0], "c": [3.0]})
        result = feat.run(df)
        assert list(result.columns) == ["a_double"]

    def test_window_equal_to_data_length_no_padding(self):
        """window_size == 数据长度, 无填充"""
        config = _SimpleWindowConfig(input_columns=["a"], window_size=3)
        feat = _ConcreteWindowFeature(config=config)
        df = pd.DataFrame({"a": [1.0, 2.0, 3.0]})
        result = feat.run(df)
        assert len(result) == 1
        np.testing.assert_array_equal(result["a_win_sum"].values, [6.0])

    def test_oid_format(self):
        config = _SimpleConfig(input_columns=["a"])
        feat = _ConcreteMapFeature(config=config, oid="test123")
        assert feat.oid == "concrete_map_feature$test123"

    def test_oid_auto_generated(self):
        config = _SimpleConfig(input_columns=["a"])
        feat = _ConcreteMapFeature(config=config)
        assert feat.oid.startswith("concrete_map_feature$")
        parts = feat.oid.split("$")
        assert len(parts) == 2
        assert len(parts[1]) == 8
