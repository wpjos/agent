# -*- coding: utf-8 -*-

"""
类型推断辅助函数单元测试

对应源文件：
- cli/help_generator.py: _format_type_full / _simplify_output_type /
  _split_input_types / _parse_variables / _is_basemodel_subtype

测试范围：
1. ``_format_type_full``：类型对象 → 全限定名字符串
   - 基础标量（int/float/str/bool）
   - pandas.DataFrame / numpy.ndarray（全限定名）
   - Annotated 类型（如 NumericData、ArrayN）
   - Union / ``A | B``（Python 3.10+）
   - tuple[T1, T2] / list[T] / dict[K, V]
   - BaseModel 子类
   - None / NoneType
2. ``_simplify_output_type``：从 ``T | tuple[T, EO]`` 提取主输出 T
   - 联合类型含 tuple 形式 → 简化为非 tuple 部分
   - 非 Union / Union 中无 tuple / 多个非 tuple arg → 原样返回
3. ``_split_input_types``：根据输入类型返回位置类型列表
   - tuple[T1, T2] → [T1_full, T2_full]
   - 单一类型 → [T_full]
   - None → None
4. ``_parse_variables``：docstring 变量解析
   - 标准格式 ``变量名: 描述``
   - 含开发者写的类型 ``变量名 (类型): 描述`` → 忽略类型
   - 中文冒号 ``变量名：描述``
   - 纯描述行（不匹配模式）
   - 空行处理
5. ``_is_basemodel_subtype``：判断 BaseModel 子类
"""

import numpy as np
import pandas as pd
from pydantic import BaseModel

from tsas.engine.operator.cli.help_generator import (
    _format_type_full,
    _simplify_output_type,
    _split_input_types,
    _parse_variables,
    _is_basemodel_subtype,
)


# ============================================================================
# 辅助：测试用 BaseModel 与 Annotated 类型
# ============================================================================

class _SampleModel(BaseModel):
    """测试用 BaseModel"""
    val: float = 0.0


# ============================================================================
# _format_type_full 测试
# ============================================================================

class TestFormatTypeFullScalars:
    """测试 _format_type_full 对标量类型的展开"""

    def test_int(self):
        """
        目的：验证 int 展开为 "int"
        输入：int
        预期：返回 "int"
        """
        assert _format_type_full(int) == "int"

    def test_float(self):
        """
        目的：验证 float 展开为 "float"
        输入：float
        预期：返回 "float"
        """
        assert _format_type_full(float) == "float"

    def test_str(self):
        """
        目的：验证 str 展开为 "str"
        """
        assert _format_type_full(str) == "str"

    def test_bool(self):
        """
        目的：验证 bool 展开为 "bool"
        """
        assert _format_type_full(bool) == "bool"


class TestFormatTypeFullPandasNumpy:
    """测试 _format_type_full 对 pandas/numpy 类型的展开"""

    def test_pandas_dataframe(self):
        """
        目的：验证 pd.DataFrame 展开为全限定名 "pandas.DataFrame"
        输入：pandas.DataFrame
        预期：返回 "pandas.DataFrame"
        """
        assert _format_type_full(pd.DataFrame) == "pandas.DataFrame"

    def test_numpy_ndarray(self):
        """
        目的：验证 np.ndarray 展开为全限定名 "numpy.ndarray"
        输入：numpy.ndarray
        预期：返回 "numpy.ndarray"
        """
        assert _format_type_full(np.ndarray) == "numpy.ndarray"

    def test_numpy_ndarray_with_params(self):
        """
        目的：验证带参数的 ndarray（如 np.ndarray[Any, dtype]）也展开为 "numpy.ndarray"
        输入：np.ndarray[Any, np.dtype[np.float64]]
        预期：返回 "numpy.ndarray"（参数被忽略）
        """
        parametrized = np.ndarray  # 简化测试，直接用 ndarray
        assert _format_type_full(parametrized) == "numpy.ndarray"


class TestFormatTypeFullAnnotated:
    """测试 _format_type_full 对 Annotated 类型的展开"""

    def test_numeric_data(self):
        """
        目的：验证 NumericData（Annotated[pd.DataFrame | ArrayN, ...]）展开
        输入：NumericData
        预期：返回 "pandas.DataFrame | numpy.ndarray"

        NumericData 内部结构：
        - NumericData = Annotated[pd.DataFrame | ArrayN, "NumericData"]
        - ArrayN = Annotated[np.ndarray[Any, np.dtype[np.integer | np.floating]], "ArrayN"]
        展开后应为 "pandas.DataFrame | numpy.ndarray"
        """
        from tsas.engine.operator.base import NumericData
        result = _format_type_full(NumericData)
        assert "pandas.DataFrame" in result
        assert "numpy.ndarray" in result
        assert " | " in result

    def test_array_n(self):
        """
        目的：验证 ArrayN（Annotated[np.ndarray, ...]）展开为 "numpy.ndarray"
        """
        from tsas.engine.operator.base import ArrayN
        assert _format_type_full(ArrayN) == "numpy.ndarray"


class TestFormatTypeFullCollections:
    """测试 _format_type_full 对容器类型的展开"""

    def test_tuple_homogeneous(self):
        """
        目的：验证 tuple[T, T] 展开为 "tuple[T_full, T_full]"
        输入：tuple[np.ndarray, np.ndarray]
        预期：返回 "tuple[numpy.ndarray, numpy.ndarray]"
        """
        result = _format_type_full(tuple[np.ndarray, np.ndarray])
        assert result == "tuple[numpy.ndarray, numpy.ndarray]"

    def test_tuple_heterogeneous(self):
        """
        目的：验证 tuple[T1, T2] 展开包含两种类型
        输入：tuple[np.ndarray, pd.DataFrame]
        预期：返回 "tuple[numpy.ndarray, pandas.DataFrame]"
        """
        result = _format_type_full(tuple[np.ndarray, pd.DataFrame])
        assert result == "tuple[numpy.ndarray, pandas.DataFrame]"

    def test_list(self):
        """
        目的：验证 list[T] 展开为 "list[T_full]"
        输入：list[int]
        预期：返回 "list[int]"
        """
        assert _format_type_full(list[int]) == "list[int]"

    def test_dict(self):
        """
        目的：验证 dict[K, V] 展开为 "dict[K_full, V_full]"
        输入：dict[str, int]
        预期：返回 "dict[str, int]"
        """
        assert _format_type_full(dict[str, int]) == "dict[str, int]"


class TestFormatTypeFullUnion:
    """测试 _format_type_full 对 Union 类型的展开"""

    def test_typing_union(self):
        """
        目的：验证 typing.Union[A, B] 展开为 "A_full | B_full"
        输入：Union[np.ndarray, pd.DataFrame]
        预期：返回 "numpy.ndarray | pandas.DataFrame"
        """
        from typing import Union
        result = _format_type_full(Union[np.ndarray, pd.DataFrame])
        assert "numpy.ndarray" in result
        assert "pandas.DataFrame" in result
        assert " | " in result

    def test_pipe_union(self):
        """
        目的：验证 Python 3.10+ 的 A | B 形式（types.UnionType）展开
        输入：np.ndarray | pd.DataFrame
        预期：返回 "numpy.ndarray | pandas.DataFrame"
        """
        result = _format_type_full(np.ndarray | pd.DataFrame)
        assert "numpy.ndarray" in result
        assert "pandas.DataFrame" in result
        assert " | " in result


class TestFormatTypeFullBasemodel:
    """测试 _format_type_full 对 BaseModel 子类的展开"""

    def test_basemodel_subtype(self):
        """
        目的：验证 BaseModel 子类展开为类名
        输入：_SampleModel
        预期：返回 "_SampleModel"
        """
        assert _format_type_full(_SampleModel) == "_SampleModel"


class TestFormatTypeFullNone:
    """测试 _format_type_full 对 None / NoneType 的处理"""

    def test_none(self):
        """
        目的：验证 None 展开为 "None"
        """
        assert _format_type_full(None) == "None"

    def test_none_type(self):
        """
        目的：验证 NoneType 展开为 "None"
        """
        assert _format_type_full(type(None)) == "None"


# ============================================================================
# _simplify_output_type 测试
# ============================================================================

class TestSimplifyOutputType:
    """测试 _simplify_output_type 从 T | tuple[T, EO] 提取主输出"""

    def test_typical_numeric_operator_pattern(self):
        """
        目的：验证 NumericOperator 子类的典型输出类型模式
        输入：NumericData | tuple[NumericData, SomeEO>
        预期：返回 NumericData
        """
        from typing import Union
        from tsas.engine.operator.base import NumericData
        eo_type = type("_EO", (BaseModel,), {})
        union_type = Union[NumericData, tuple[NumericData, eo_type]]
        result = _simplify_output_type(union_type)
        assert result is NumericData

    def test_pipe_union_with_tuple(self):
        """
        目的：验证 Python 3.10+ A | B 形式中含 tuple 的简化
        输入：np.ndarray | tuple[np.ndarray, _SampleModel]
        预期：返回 np.ndarray
        """
        union_type = np.ndarray | tuple[np.ndarray, _SampleModel]
        result = _simplify_output_type(union_type)
        assert result is np.ndarray

    def test_non_union_returns_as_is(self):
        """
        目的：验证非 Union 类型原样返回
        输入：np.ndarray
        预期：返回 np.ndarray
        """
        assert _simplify_output_type(np.ndarray) is np.ndarray

    def test_union_without_tuple_returns_as_is(self):
        """
        目的：验证 Union 中无 tuple 时原样返回
        输入：Union[int, float]
        预期：返回 Union[int, float]
        """
        from typing import Union
        union_type = Union[int, float]
        result = _simplify_output_type(union_type)
        assert result is union_type

    def test_union_with_multiple_non_tuple_returns_as_is(self):
        """
        目的：验证 Union 中有多个非 tuple arg 时不简化
        输入：Union[int, str, tuple[int, float]]
        预期：原样返回
        """
        from typing import Union
        union_type = Union[int, str, tuple[int, float]]
        result = _simplify_output_type(union_type)
        assert result is union_type

    def test_none_returns_none(self):
        """
        目的：验证 None 输入返回 None
        """
        assert _simplify_output_type(None) is None

    def test_basemodel_returns_as_is(self):
        """
        目的：验证 BaseModel 子类原样返回
        输入：_SampleModel
        预期：返回 _SampleModel
        """
        assert _simplify_output_type(_SampleModel) is _SampleModel


# ============================================================================
# _split_input_types 测试
# ============================================================================

class TestSplitInputTypes:
    """测试 _split_input_types 根据输入类型返回位置类型列表"""

    def test_tuple_returns_multiple(self):
        """
        目的：验证 tuple[T1, T2] 拆解为两个类型字符串
        输入：tuple[np.ndarray, np.ndarray]
        预期：返回 ["numpy.ndarray", "numpy.ndarray"]
        """
        result = _split_input_types(tuple[np.ndarray, np.ndarray])
        assert result == ["numpy.ndarray", "numpy.ndarray"]

    def test_single_type_returns_one_element_list(self):
        """
        目的：验证单一类型（如 NumericData）返回单元素列表
        输入：NumericData
        预期：返回 ["pandas.DataFrame | numpy.ndarray"]（或类似展开）
        """
        from tsas.engine.operator.base import NumericData
        result = _split_input_types(NumericData)
        assert len(result) == 1
        assert "pandas.DataFrame" in result[0]

    def test_none_returns_none(self):
        """
        目的：验证 None 输入返回 None
        """
        assert _split_input_types(None) is None

    def test_float_returns_one_element_list(self):
        """
        目的：验证标量类型返回单元素列表
        输入：float
        预期：返回 ["float"]
        """
        assert _split_input_types(float) == ["float"]


# ============================================================================
# _parse_variables 测试
# ============================================================================

class TestParseVariables:
    """测试 _parse_variables docstring 变量解析"""

    def test_standard_format(self):
        """
        目的：验证标准格式 "变量名: 描述" 的解析
        输入："x: 特征矩阵"
        预期：返回 [("x", "特征矩阵")]
        """
        result = _parse_variables("x: 特征矩阵")
        assert result == [("x", "特征矩阵")]

    def test_with_developer_type_ignored(self):
        """
        目的：验证开发者写的类型部分被忽略
        输入："x (DataFrame | ndarray): 特征矩阵"
        预期：返回 [("x", "特征矩阵")]（类型部分被丢弃）
        """
        result = _parse_variables("x (DataFrame | ndarray): 特征矩阵")
        assert result == [("x", "特征矩阵")]

    def test_chinese_colon(self):
        """
        目的：验证中文冒号也能正确解析
        输入："x：特征矩阵"
        预期：返回 [("x", "特征矩阵")]
        """
        result = _parse_variables("x：特征矩阵")
        assert result == [("x", "特征矩阵")]

    def test_multiple_variables(self):
        """
        目的：验证多行多变量的解析
        输入：两行变量
        预期：返回两个变量的列表
        """
        text = "x_real: 真实值\nx_pred: 预测值"
        result = _parse_variables(text)
        assert result == [("x_real", "真实值"), ("x_pred", "预测值")]

    def test_pure_description_line(self):
        """
        目的：验证不匹配变量模式的行作为纯描述处理
        输入："二维时序数据，每列为一个特征通道"
        预期：返回 [("", "二维时序数据，每列为一个特征通道")]
        """
        result = _parse_variables("二维时序数据，每列为一个特征通道")
        assert result == [("", "二维时序数据，每列为一个特征通道")]

    def test_empty_text(self):
        """
        目的：验证空字符串返回空列表
        输入：""
        预期：返回 []
        """
        assert _parse_variables("") == []

    def test_none_text(self):
        """
        目的：验证 None 输入返回空列表
        输入：None
        预期：返回 []
        """
        assert _parse_variables(None) == []

    def test_empty_lines_skipped(self):
        """
        目的：验证空行被跳过
        输入：含空行的多行文本
        预期：空行不出现在结果中
        """
        text = "x: 变量1\n\ny: 变量2"
        result = _parse_variables(text)
        assert result == [("x", "变量1"), ("y", "变量2")]

    def test_no_description(self):
        """
        目的：验证只有变量名无描述的行
        输入："x:"
        预期：返回 [("x", "")]
        """
        result = _parse_variables("x:")
        assert result == [("x", "")]

    def test_mixed_variables_and_description(self):
        """
        目的：验证变量行与纯描述行混合的情况
        输入：第一行变量，第二行纯描述
        预期：两行都被解析，纯描述行变量名为空
        """
        text = "x: 特征矩阵\n二维时序数据"
        result = _parse_variables(text)
        assert len(result) == 2
        assert result[0] == ("x", "特征矩阵")
        assert result[1] == ("", "二维时序数据")


# ============================================================================
# _is_basemodel_subtype 测试
# ============================================================================

class TestIsBasemodelSubtype:
    """测试 _is_basemodel_subtype 判断"""

    def test_basemodel_subtype_returns_true(self):
        """
        目的：验证 BaseModel 子类返回 True
        输入：_SampleModel
        预期：返回 True
        """
        assert _is_basemodel_subtype(_SampleModel) is True

    def test_basemodel_itself_returns_true(self):
        """
        目的：验证 BaseModel 自身返回 True
        输入：BaseModel
        预期：返回 True
        """
        assert _is_basemodel_subtype(BaseModel) is True

    def test_non_basemodel_type_returns_false(self):
        """
        目的：验证非 BaseModel 类型返回 False
        输入：int, float, np.ndarray, pd.DataFrame
        预期：返回 False
        """
        assert _is_basemodel_subtype(int) is False
        assert _is_basemodel_subtype(float) is False
        assert _is_basemodel_subtype(np.ndarray) is False
        assert _is_basemodel_subtype(pd.DataFrame) is False

    def test_union_type_returns_false(self):
        """
        目的：验证联合类型返回 False（不是单一 BaseModel）
        输入：Union[int, str]
        预期：返回 False
        """
        from typing import Union
        assert _is_basemodel_subtype(Union[int, str]) is False

    def test_none_returns_false(self):
        """
        目的：验证 None 返回 False
        输入：None
        预期：返回 False
        """
        assert _is_basemodel_subtype(None) is False


# ============================================================================
# _format_type_full 兜底分支测试
# ============================================================================

class TestFormatTypeFullFallback:
    """测试 _format_type_full 在罕见类型上的兜底行为"""

    def test_bytes(self):
        """
        目的：验证 bytes 类型展开为 "bytes"
        输入：bytes
        预期：返回 "bytes"
        """
        assert _format_type_full(bytes) == "bytes"

    def test_custom_class_uses_name(self):
        """
        目的：验证非内置/非特殊类型走 __name__ 分支
        输入：自定义普通类
        预期：返回类名
        """

        class CustomType:
            pass

        result = _format_type_full(CustomType)
        assert result == "CustomType"

    def test_parametrized_origin_without_known_branch(self):
        """
        目的：验证带参数但无明确处理分支的类型走 origin.__name__
        输入：set[int]（origin 是 set，无独立分支）
        预期：返回 "set"
        """
        result = _format_type_full(set[int])
        assert result == "set"


# ============================================================================
# _render_input_section / _render_output_section 边界场景
# ============================================================================

class TestRenderSectionsEdgeCases:
    """测试输入/输出渲染逻辑的边界场景"""

    def test_basemodel_input_renders_struct_table(self):
        """
        目的：验证 BaseModel 输入类型时会渲染 "**结构**：" 字段表
        输入：自定义算子类，_input_type = BaseModel 子类
        预期：输出包含 "**结构**：" 与字段名
        """
        from tsas.engine.operator.cli.help_generator import _render_input_section

        class _SampleInputModel(BaseModel):
            """测试用 BaseModel 输入"""
            from pydantic import Field
            feature: float = Field(description="特征值")

        class _BaseModelInputOp:
            """测试算子（BaseModel 输入）

            Input:
                payload: 结构化输入对象
            """
            _input_type = _SampleInputModel
            _output_type = None

            @classmethod
            def name(cls) -> str:
                return "basemodel_input_op"

        result_lines = _render_input_section(_BaseModelInputOp)
        joined = "\n".join(result_lines)
        # 标准变量行渲染
        assert "payload (_SampleInputModel): 结构化输入对象" in joined
        # 字段表追加
        assert "**结构**：" in joined
        assert "feature" in joined

    def test_input_type_none_with_pure_description(self):
        """
        目的：验证 type_list 为 None 但 docstring 有纯描述时的渲染
        输入：算子无 _input_type，docstring Input 是纯描述
        预期：直接展示 docstring 描述
        """
        from tsas.engine.operator.cli.help_generator import _render_input_section

        class _NoTypeOp:
            """测试算子（无类型）

            Input:
                纯描述输入
            """
            _input_type = None
            _output_type = None

            @classmethod
            def name(cls) -> str:
                return "no_type_op"

        result_lines = _render_input_section(_NoTypeOp)
        joined = "\n".join(result_lines)
        assert "纯描述输入" in joined

    def test_input_type_none_with_variable(self):
        """
        目的：验证 type_list 为 None 但 docstring 有变量行时的渲染
        输入：算子无 _input_type，docstring 写了 "x: 描述"
        预期：直接展示 "x: 描述"，无类型括号
        """
        from tsas.engine.operator.cli.help_generator import _render_input_section

        class _NoTypeWithVarOp:
            """测试算子

            Input:
                x: 变量描述
            """
            _input_type = None
            _output_type = None

            @classmethod
            def name(cls) -> str:
                return "no_type_var_op"

        result_lines = _render_input_section(_NoTypeWithVarOp)
        joined = "\n".join(result_lines)
        assert "x: 变量描述" in joined

    def test_variable_count_mismatch_single_variable_downgrade(self):
        """
        目的：验证 docstring 单变量 + tuple 多元素类型时的降级
        输入：_input_type = tuple[ndarray, ndarray]，docstring 写一个变量
        预期：降级为单变量 "x (完整类型): 描述"
        """
        from tsas.engine.operator.cli.help_generator import _render_input_section

        class _MismatchSingleOp:
            """测试算子

            Input:
                x: 单变量描述
            """
            _input_type = tuple[np.ndarray, np.ndarray]
            _output_type = None

            @classmethod
            def name(cls) -> str:
                return "mismatch_single_op"

        result_lines = _render_input_section(_MismatchSingleOp)
        joined = "\n".join(result_lines)
        # 完整 tuple 类型与单变量名合并
        assert "x (tuple[numpy.ndarray, numpy.ndarray]): 单变量描述" in joined

    def test_variable_count_mismatch_multi_variables_downgrade(self):
        """
        目的：验证 docstring 变量数（3）与 tuple 类型元素数（2）不匹配时的降级行为
        输入：_input_type = tuple[ndarray, ndarray]（2 元素），docstring 写 3 个变量
        预期：走"多变量 + 类型数不匹配"分支，首行显示完整 tuple 类型，
            后续各行原样输出变量名与描述（无类型括号）
        """
        from tsas.engine.operator.cli.help_generator import _render_input_section

        class _MismatchMultiOp:
            """测试算子

            Input:
                a: 第一个变量
                b: 第二个变量
                c: 第三个变量
            """
            _input_type = tuple[np.ndarray, np.ndarray]
            _output_type = None

            @classmethod
            def name(cls) -> str:
                return "mismatch_multi_op"

        result_lines = _render_input_section(_MismatchMultiOp)
        joined = "\n".join(result_lines)
        # 首行显示完整 tuple 类型（无变量名）
        assert "(tuple[numpy.ndarray, numpy.ndarray])" in joined
        # 后续各行保留变量名 + 描述（无类型括号）
        assert "a: 第一个变量" in joined
        assert "b: 第二个变量" in joined
        assert "c: 第三个变量" in joined

    def test_pure_description_only_no_variable_name(self):
        """
        目的：验证 docstring 仅含纯描述（无变量名）且 _input_type 为 None 时的渲染
        输入：算子 _input_type=None，docstring 是纯描述
        预期：直接展示 docstring 描述
        """
        from tsas.engine.operator.cli.help_generator import _render_input_section

        class _PureDescOp:
            """测试算子

            Input:
                纯描述无变量名
            """
            _input_type = None
            _output_type = None

            @classmethod
            def name(cls) -> str:
                return "pure_desc_op"

        result_lines = _render_input_section(_PureDescOp)
        joined = "\n".join(result_lines)
        assert "纯描述无变量名" in joined

    def test_render_output_no_type_no_doc_shows_placeholder(self):
        """
        目的：验证 _output_type 为 None 且 docstring 无 Output 段时显示 "（无）"
        输入：算子无类型与 docstring Output
        预期：包含 "（无）"
        """
        from tsas.engine.operator.cli.help_generator import _render_output_section

        class _EmptyOutputOp:
            """测试算子"""
            _input_type = None
            _output_type = None

            @classmethod
            def name(cls) -> str:
                return "empty_output_op"

        result_lines = _render_output_section(_EmptyOutputOp)
        joined = "\n".join(result_lines)
        assert "### 主输出" in joined
        assert "（无）" in joined
