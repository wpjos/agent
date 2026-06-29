# -*- coding: utf-8 -*-

"""
帮助文档自动生成模块单元测试

对应源文件：
- cli/help_generator.py: generate_list, generate_detail, generate_config_params_table
  及所有内部辅助函数

测试范围：
- CJK-aware 显示宽度和表格对齐
- docstring section 提取（Input/Output）
- 算子分组判断（管线组件/端到端检测器/组合管线）
- 角色提取（Predictor/Scorer(Single)/Scorer(Multi)/Decider/Detector）
- 可训练/监督类型/分批推理判断
- 类型标签提取（扩展版）
- 列表模式（分组+CJK对齐表格）
- 详情模式（基础分类/输入/主输出/附加输出/训练参数/运行参数）
- 参数表格生成（各种类型+CJK对齐）
- 真实算子集成测试
"""

from enum import Enum
from typing import Literal

from pydantic import BaseModel, Field

from tsas.engine.operator.cli.help_generator import (
    generate_list,
    generate_detail,
    generate_config_params_table,
    _extract_summary,
    _extract_description,
    _extract_docstring_section,
    _extract_type_tags,
    _extract_role,
    _is_learnable,
    _supervision_type,
    _supports_batch_run,
    _classify_operator,
    _format_type,
    _extract_constraints,
    _display_width,
    _pad_cell,
    _build_aligned_table,
    _extract_model_fields_description,
)


# ============================================================================
# 辅助类 — 测试用 Config / ExtraOutput
# ============================================================================

class _TestEnum(Enum):
    """测试用枚举"""
    A = "alpha"
    B = "beta"


class _TestConfig(BaseModel):
    """测试用配置类"""
    n: int = Field(default=5, ge=1, le=20, description="数量参数")
    ratio: float = Field(default=0.5, gt=0, lt=1.0, description="比例参数")
    method: _TestEnum = Field(default=_TestEnum.A, description="方法选择")
    mode: Literal["fast", "slow"] = Field(default="fast", description="模式")
    name: str = Field(default="test", description="名称")
    required_param: int = Field(..., description="必填参数")


class _TestExtraOutput(BaseModel):
    """测试用附加输出"""
    score: float = Field(description="异常分数")
    detail: str = Field(default="", description="详细信息")


class _TestExtraOutputNoDesc(BaseModel):
    """测试用无描述附加输出"""
    value: int


# ============================================================================
# 辅助类 — 测试用算子类
# ============================================================================

class _DocOperator:
    """
    测试算子

    这是一个用于测试的算子类。

    Input:
        DataFrame — 二维时序数据

    Output:
        DataFrame — 重构后的数据

    Args:
        config: 配置参数
    """
    _config_type = _TestConfig
    _run_params_type = None
    _eo_type = None
    _fit_params_type = None

    @classmethod
    def name(cls) -> str:
        return "test_operator"


class _NoDocOperator:
    """   """
    @classmethod
    def name(cls) -> str:
        return "no_doc_op"


class _NoneDocOperator:
    @classmethod
    def name(cls) -> str:
        return "none_doc_op"


class _MinimalOperator:
    """最简算子"""
    _config_type = None
    _run_params_type = None
    _eo_type = None
    _fit_params_type = None
    _output_type = None

    @classmethod
    def name(cls) -> str:
        return "minimal_op"


# ============================================================================
# 测试 _display_width — CJK-aware 显示宽度
# ============================================================================

class TestDisplayWidth:
    """测试 _display_width 辅助函数"""

    def test_ascii_string(self):
        """
        目的：验证纯 ASCII 字符串的显示宽度等于字符数
        输入："hello"
        预期：返回 5
        """
        assert _display_width("hello") == 5

    def test_chinese_string(self):
        """
        目的：验证中文字符串每个字符占 2 列宽
        输入："名称"
        预期：返回 4（2 个中文字符 × 2）
        """
        assert _display_width("名称") == 4

    def test_mixed_string(self):
        """
        目的：验证中英文混合字符串的显示宽度
        输入："名称name"
        预期：返回 8（2×2 + 4×1）
        """
        assert _display_width("名称name") == 8

    def test_empty_string(self):
        """
        目的：验证空字符串宽度为 0
        预期：返回 0
        """
        assert _display_width("") == 0

    def test_backtick_wrapped_name(self):
        """
        目的：验证反引号包裹的名称宽度正确
        输入："`knn_scorer`"
        预期：返回 12
        """
        assert _display_width("`knn_scorer`") == 12


# ============================================================================
# 测试 _pad_cell — 单元格填充
# ============================================================================

class TestPadCell:
    """测试 _pad_cell 辅助函数"""

    def test_ascii_padding(self):
        """
        目的：验证 ASCII 字符串正确填充到目标宽度
        输入："hello", target=10
        预期：返回 "hello     "（5 个空格）
        """
        result = _pad_cell("hello", 10)
        assert len(result) == 10
        assert result == "hello     "

    def test_cjk_padding(self):
        """
        目的：验证中文字符串按显示宽度填充
        输入："名称", target=10
        预期：返回 "名称      "（显示宽度 4 + 6 个空格 = 10）
        """
        result = _pad_cell("名称", 10)
        assert _display_width(result) == 10

    def test_no_padding_needed(self):
        """
        目的：验证已达到目标宽度时不添加填充
        输入："hello", target=3
        预期：原样返回 "hello"
        """
        result = _pad_cell("hello", 3)
        assert result == "hello"

    def test_exact_width(self):
        """
        目的：验证恰好达到目标宽度时不添加填充
        输入："hello", target=5
        预期：原样返回 "hello"
        """
        result = _pad_cell("hello", 5)
        assert result == "hello"


# ============================================================================
# 测试 _build_aligned_table — CJK 对齐表格
# ============================================================================

class TestBuildAlignedTable:
    """测试 _build_aligned_table 辅助函数"""

    def test_basic_alignment(self):
        """
        目的：验证基本表格对齐
        输入：2 列 2 行
        预期：每列对齐到最大宽度
        """
        headers = ["Name", "Desc"]
        rows = [["a", "short"], ["longer_name", "x"]]
        result = _build_aligned_table(headers, rows)
        lines = result.split('\n')
        # 表头、分隔、2 行数据 = 4 行
        assert len(lines) == 4
        # 分隔行全为 -
        assert '---' in lines[1]

    def test_cjk_alignment(self):
        """
        目的：验证中文表头的对齐
        输入：中文表头 + ASCII 数据
        预期：显示宽度正确对齐
        """
        headers = ["名称", "类型"]
        rows = [["abc", "Predictor"]]
        result = _build_aligned_table(headers, rows)
        lines = result.split('\n')
        # 表头行的显示宽度应等于分隔行的显示宽度
        assert _display_width(lines[0]) == _display_width(lines[1])

    def test_empty_headers(self):
        """
        目的：验证空表头返回空字符串
        输入：空表头列表
        预期：返回 ""
        """
        assert _build_aligned_table([], []) == ""

    def test_no_data_rows(self):
        """
        目的：验证无数据行时仅输出表头和分隔
        输入：有表头无数据
        预期：输出 2 行（表头+分隔）
        """
        result = _build_aligned_table(["A", "B"], [])
        lines = result.split('\n')
        assert len(lines) == 2


# ============================================================================
# 测试 _extract_docstring_section — Input/Output 提取
# ============================================================================

class TestExtractDocstringSection:
    """测试 _extract_docstring_section 辅助函数"""

    def test_extract_input_section(self):
        """
        目的：验证从 docstring 提取 Input section
        输入：包含 Input: section 的 docstring
        预期：返回 Input section 的文本内容
        """
        doc = """测试算子

        Input:
            DataFrame — 二维时序数据

        Output:
            DataFrame — 重构数据
        """
        result = _extract_docstring_section(doc, "Input")
        assert "DataFrame" in result
        assert "二维时序数据" in result

    def test_extract_output_section(self):
        """
        目的：验证从 docstring 提取 Output section
        输入：包含 Output: section 的 docstring
        预期：返回 Output section 的文本内容
        """
        doc = """测试算子

        Input:
            输入描述

        Output:
            DataFrame — 重构后的数据
        """
        result = _extract_docstring_section(doc, "Output")
        assert "DataFrame" in result
        assert "重构后的数据" in result

    def test_missing_section(self):
        """
        目的：验证缺少目标 section 时返回空字符串
        输入：不含 Input: section 的 docstring
        预期：返回 ""
        """
        doc = """测试算子

        Args:
            x: 参数
        """
        result = _extract_docstring_section(doc, "Input")
        assert result == ""

    def test_none_docstring(self):
        """
        目的：验证 None docstring 返回空字符串
        输入：None
        预期：返回 ""
        """
        assert _extract_docstring_section(None, "Input") == ""

    def test_empty_docstring(self):
        """
        目的：验证空字符串 docstring 返回空
        输入：""
        预期：返回 ""
        """
        assert _extract_docstring_section("", "Input") == ""


# ============================================================================
# 测试 _extract_role — 角色提取
# ============================================================================

class TestExtractRole:
    """测试 _extract_role 辅助函数"""

    def test_detector_by_name(self):
        """
        目的：验证类名以 Detector 结尾时返回 "Detector"
        输入：KNNDetector（实际不继承 BaseDetector）
        预期：返回 "Detector"
        """
        from tsas.engine.operator.detection.knn import KNNDetector
        assert _extract_role(KNNDetector) == "Detector"

    def test_scorer_single(self):
        """
        目的：验证 SingleScorer 返回 "Scorer(Single)"
        输入：KNNScorer
        预期：返回 "Scorer(Single)"
        """
        from tsas.engine.operator.detection.knn import KNNScorer
        assert _extract_role(KNNScorer) == "Scorer(Single)"

    def test_predictor(self):
        """
        目的：验证 Predictor 角色提取
        输入：MeanPredictor
        预期：返回 "Predictor"
        """
        from tsas.engine.operator.detection.mean_predictor import MeanPredictor
        assert _extract_role(MeanPredictor) == "Predictor"

    def test_decider(self):
        """
        目的：验证 Decider 角色提取
        输入：ThresholdDecider
        预期：返回 "Decider"
        """
        from tsas.engine.operator.detection.threshold_decider import ThresholdDecider
        assert _extract_role(ThresholdDecider) == "Decider"

    def test_unknown_for_plain_class(self):
        """
        目的：验证普通类返回 "未知"
        输入：普通类
        预期：返回 "未知"
        """
        class _Plain:
            pass
        assert _extract_role(_Plain) == "未知"


# ============================================================================
# 测试 _is_learnable / _supervision_type / _supports_batch_run
# ============================================================================

class TestCapabilityChecks:
    """测试能力检查辅助函数"""

    def test_learnable_true(self):
        """
        目的：验证可训练算子返回 True
        输入：KNNDetector（可训练）
        预期：返回 True
        """
        from tsas.engine.operator.detection.knn import KNNDetector
        assert _is_learnable(KNNDetector) is True

    def test_learnable_false(self):
        """
        目的：验证不可训练算子返回 False
        输入：普通类
        预期：返回 False
        """
        class _Plain:
            pass
        assert _is_learnable(_Plain) is False

    def test_supervision_unsupervised(self):
        """
        目的：验证无监督算子返回 "无监督"
        输入：KNNDetector（无监督）
        预期：返回 "无监督"
        """
        from tsas.engine.operator.detection.knn import KNNDetector
        assert _supervision_type(KNNDetector) == "无监督"

    def test_supervision_default(self):
        """
        目的：验证不继承任何监督 Mixin 的类返回 "无监督"
        输入：普通类
        预期：返回 "无监督"
        """
        class _Plain:
            pass
        assert _supervision_type(_Plain) == "无监督"

    def test_batch_run_false(self):
        """
        目的：验证不支持分批推理的算子返回 False
        输入：普通类
        预期：返回 False
        """
        class _Plain:
            pass
        assert _supports_batch_run(_Plain) is False


# ============================================================================
# 测试 _classify_operator — 分组判断
# ============================================================================

class TestClassifyOperator:
    """测试 _classify_operator 辅助函数"""

    def test_pipeline_component_scorer(self):
        """
        目的：验证 Scorer 归入管线组件
        输入：KNNScorer
        预期：返回 "管线组件算子"
        """
        from tsas.engine.operator.detection.knn import KNNScorer
        assert _classify_operator(KNNScorer) == "管线组件算子"

    def test_pipeline_component_predictor(self):
        """
        目的：验证 Predictor 归入管线组件
        输入：MeanPredictor
        预期：返回 "管线组件算子"
        """
        from tsas.engine.operator.detection.mean_predictor import MeanPredictor
        assert _classify_operator(MeanPredictor) == "管线组件算子"

    def test_end_to_end_detector(self):
        """
        目的：验证 Detector 归入端到端检测器
        输入：KNNDetector
        预期：返回 "端到端检测器算子"
        """
        from tsas.engine.operator.detection.knn import KNNDetector
        assert _classify_operator(KNNDetector) == "端到端检测器算子"

    def test_composite_detector(self):
        """
        目的：验证 CompositeDetector 归入组合管线
        输入：CompositeDetector
        预期：返回 "组合管线算子"
        """
        from tsas.engine.operator.detection.composite import CompositeDetector
        assert _classify_operator(CompositeDetector) == "组合管线算子"

    def test_composite_scorer(self):
        """
        目的：验证 CompositeScorer 归入组合管线
        输入：CompositeScorer
        预期：返回 "组合管线算子"
        """
        from tsas.engine.operator.detection.composite import CompositeScorer
        assert _classify_operator(CompositeScorer) == "组合管线算子"

    def test_decider_is_pipeline(self):
        """
        目的：验证 Decider 归入管线组件
        输入：ThresholdDecider
        预期：返回 "管线组件算子"
        """
        from tsas.engine.operator.detection.threshold_decider import ThresholdDecider
        assert _classify_operator(ThresholdDecider) == "管线组件算子"


# ============================================================================
# 测试 _extract_type_tags — 扩展版类型标签
# ============================================================================

class TestExtractTypeTags:
    """测试 _extract_type_tags 类型标签提取"""

    def test_detector_tag(self):
        """
        目的：验证 Detector 标签提取（通过类名）
        输入：KNNDetector
        预期：返回包含 "Detector"
        """
        from tsas.engine.operator.detection.knn import KNNDetector
        tags = _extract_type_tags(KNNDetector)
        assert "Detector" in tags

    def test_scorer_single_tag(self):
        """
        目的：验证 SingleScorer 标签
        输入：KNNScorer
        预期：返回包含 "Scorer(Single)"
        """
        from tsas.engine.operator.detection.knn import KNNScorer
        tags = _extract_type_tags(KNNScorer)
        assert "Scorer(Single)" in tags

    def test_predictor_tag(self):
        """
        目的：验证 Predictor 标签
        输入：MeanPredictor
        预期：返回包含 "Predictor"
        """
        from tsas.engine.operator.detection.mean_predictor import MeanPredictor
        tags = _extract_type_tags(MeanPredictor)
        assert "Predictor" in tags

    def test_decider_tag(self):
        """
        目的：验证 Decider 标签
        输入：ThresholdDecider
        预期：返回包含 "Decider"
        """
        from tsas.engine.operator.detection.threshold_decider import ThresholdDecider
        tags = _extract_type_tags(ThresholdDecider)
        assert "Decider" in tags

    def test_learnable_tag(self):
        """
        目的：验证可训练标签
        输入：KNNDetector（可训练）
        预期：返回包含 "可训练"
        """
        from tsas.engine.operator.detection.knn import KNNDetector
        tags = _extract_type_tags(KNNDetector)
        assert "可训练" in tags

    def test_unsupervised_tag(self):
        """
        目的：验证无监督标签
        输入：KNNDetector（无监督）
        预期：返回包含 "无监督"
        """
        from tsas.engine.operator.detection.knn import KNNDetector
        tags = _extract_type_tags(KNNDetector)
        assert "无监督" in tags

    def test_plain_class_no_tags(self):
        """
        目的：验证普通类无标签
        输入：普通类
        预期：返回空列表
        """
        class _Plain:
            pass
        tags = _extract_type_tags(_Plain)
        assert len(tags) == 0

    def test_evaluation_operator(self):
        """
        目的：验证评价算子标签
        输入：BinaryClassificationMetric
        预期：返回包含 "评价指标"
        """
        from tsas.engine.operator.evaluation.binary_classification import BinaryClassificationMetric
        tags = _extract_type_tags(BinaryClassificationMetric)
        assert "评价指标" in tags

    def test_feature_operator(self):
        """
        目的：验证特征算子标签
        输入：PCAFeature
        预期：返回包含 "特征算子" 和 "可训练"
        """
        from tsas.engine.operator.feature.construction.simple_feature import PCAFeature
        tags = _extract_type_tags(PCAFeature)
        assert "特征算子" in tags
        assert "可训练" in tags


# ============================================================================
# 测试 _extract_description — 含 Input/Output 标记
# ============================================================================

class TestExtractDescription:
    """测试 _extract_description 辅助函数"""

    def test_stops_at_input_section(self):
        """
        目的：验证描述提取在 Input: section 前停止
        输入：包含 Input: section 的 docstring
        预期：不包含 Input section 内容
        """
        result = _extract_description(_DocOperator)
        assert "测试算子" in result
        assert "DataFrame" not in result

    def test_empty_docstring(self):
        """
        目的：验证空 docstring 返回空字符串
        输入：空白 docstring
        预期：返回 ""
        """
        result = _extract_description(_NoDocOperator)
        assert result == ""

    def test_none_docstring(self):
        """
        目的：验证无 docstring 返回空字符串
        输入：无 docstring 的类
        预期：返回 ""
        """
        result = _extract_description(_NoneDocOperator)
        assert result == ""


# ============================================================================
# 测试 _extract_summary
# ============================================================================

class TestExtractSummary:
    """测试 _extract_summary 辅助函数"""

    def test_with_docstring(self):
        """
        目的：验证从 docstring 提取第一行简介
        预期：返回 "测试算子"
        """
        assert _extract_summary(_DocOperator) == "测试算子"

    def test_empty_docstring(self):
        """
        目的：验证空 docstring 返回 "(无描述)"
        """
        assert _extract_summary(_NoDocOperator) == "(无描述)"

    def test_none_docstring(self):
        """
        目的：验证无 docstring 返回 "(无描述)"
        """
        assert _extract_summary(_NoneDocOperator) == "(无描述)"


# ============================================================================
# 测试 _format_type / _extract_constraints
# ============================================================================

class TestFormatType:
    """测试 _format_type 辅助函数"""

    def test_int(self):
        """预期："int\""""
        assert _format_type(int) == "int"

    def test_float(self):
        """预期："float\""""
        assert _format_type(float) == "float"

    def test_str(self):
        """预期："str\""""
        assert _format_type(str) == "str"

    def test_bool(self):
        """预期："bool\""""
        assert _format_type(bool) == "bool"

    def test_enum(self):
        """预期：包含 "enum" 和 "alpha\""""
        result = _format_type(_TestEnum)
        assert "enum" in result and "alpha" in result

    def test_literal(self):
        """预期：包含 "literal" 和 "fast\""""
        result = _format_type(Literal["fast", "slow"])
        assert "literal" in result and "fast" in result

    def test_none(self):
        """预期："Any\""""
        assert _format_type(None) == "Any"

    def test_list_type(self):
        """预期：包含 "list" 和 "str\""""
        result = _format_type(list[str])
        assert "list" in result and "str" in result

    def test_class_with_name(self):
        """预期：返回类名"""
        class MyCls:
            pass
        assert "MyCls" in _format_type(MyCls)


class TestExtractConstraints:
    """测试 _extract_constraints 辅助函数"""

    def test_ge_le(self):
        """预期："[1, 20]\""""
        fi = _TestConfig.model_fields['n']
        assert "[1, 20]" in _extract_constraints(fi)

    def test_gt_lt(self):
        """预期："(0, 1.0)\""""
        fi = _TestConfig.model_fields['ratio']
        assert "(0, 1.0)" in _extract_constraints(fi)

    def test_no_constraints(self):
        """预期："-"\""""
        fi = _TestConfig.model_fields['name']
        assert _extract_constraints(fi) == "-"

    def test_enum_candidates(self):
        """预期：包含 "alpha" 和 "beta\""""
        fi = _TestConfig.model_fields['method']
        result = _extract_constraints(fi)
        assert "alpha" in result and "beta" in result


# ============================================================================
# 测试 generate_config_params_table — 含 CJK 对齐
# ============================================================================

class TestGenerateConfigParamsTable:
    """测试 generate_config_params_table 参数表格生成"""

    def test_has_header(self):
        """预期：包含完整 5 列表头"""
        result = generate_config_params_table(_TestConfig)
        assert "| 参数名" in result

    def test_int_field(self):
        """预期：包含 n 的类型 int、默认值 5、值域 [1, 20]"""
        result = generate_config_params_table(_TestConfig)
        assert "n" in result and "5" in result and "[1, 20]" in result

    def test_float_field_exclusive(self):
        """预期：开区间 (0, 1.0)"""
        result = generate_config_params_table(_TestConfig)
        assert "(0, 1.0)" in result

    def test_enum_field(self):
        """预期：包含候选值 alpha, beta"""
        result = generate_config_params_table(_TestConfig)
        assert "alpha" in result and "beta" in result

    def test_literal_field(self):
        """预期：包含候选值 fast, slow"""
        result = generate_config_params_table(_TestConfig)
        assert "fast" in result and "slow" in result

    def test_required_field(self):
        """预期：显示 **必填**"""
        result = generate_config_params_table(_TestConfig)
        assert "**必填**" in result

    def test_description(self):
        """预期：包含描述文本"""
        result = generate_config_params_table(_TestConfig)
        assert "数量参数" in result and "比例参数" in result

    def test_aligned_columns(self):
        """
        目的：验证表格列对齐
        预期：所有行的显示宽度相等
        """
        result = generate_config_params_table(_TestConfig)
        lines = result.strip().split('\n')
        widths = [_display_width(line) for line in lines]
        # 所有行宽度应一致
        assert len(set(widths)) == 1


# ============================================================================
# 测试 _extract_model_fields_description — 附加输出表格
# ============================================================================

class TestExtractModelFieldsDescription:
    """测试 _extract_model_fields_description"""

    def test_with_descriptions(self):
        """
        目的：验证 Field(description=...) 被提取到说明列
        输入：_TestExtraOutput（有 description）
        预期：输出包含 "异常分数" 和 "详细信息"
        """
        result = _extract_model_fields_description(_TestExtraOutput)
        assert "异常分数" in result
        assert "详细信息" in result

    def test_without_descriptions(self):
        """
        目的：验证无 description 时说明列为空
        输入：_TestExtraOutputNoDesc
        预期：输出不包含额外说明文本
        """
        result = _extract_model_fields_description(_TestExtraOutputNoDesc)
        assert "value" in result


# ============================================================================
# 测试 generate_list — 列表模式
# ============================================================================

class TestGenerateList:
    """测试 generate_list 列表模式"""

    def test_grouped_output(self):
        """
        目的：验证列表模式按分组输出
        输入：含 Scorer 和 Detector 的算子映射
        预期：包含 "管线组件算子" 和 "端到端检测器算子" 标题
        """
        from tsas.engine.operator.detection.knn import KNNScorer, KNNDetector
        operators = {"knn_scorer": KNNScorer, "knn_detector": KNNDetector}
        result = generate_list(operators)
        assert "管线组件算子" in result
        assert "端到端检测器算子" in result

    def test_type_and_learnable_columns(self):
        """
        目的：验证列表包含类型和可训练列
        输入：含 Scorer 的算子映射
        预期：输出包含 "Scorer(Single)" 和 "是" 或 "否"
        """
        from tsas.engine.operator.detection.knn import KNNScorer
        result = generate_list({"knn_scorer": KNNScorer})
        assert "Scorer(Single)" in result

    def test_no_backticks_in_name(self):
        """
        目的：验证名称不含反引号
        输入：含算子的映射
        预期：名称列无 ` 符号
        """
        from tsas.engine.operator.detection.knn import KNNScorer
        result = generate_list({"knn_scorer": KNNScorer})
        # 检查表格数据行中的名称不含反引号
        for line in result.split('\n'):
            if 'knn_scorer' in line and '|' in line:
                assert '`knn_scorer`' not in line

    def test_count(self):
        """
        目的：验证算子计数正确
        输入：3 个算子
        预期：显示 "共 3 个算子"
        """
        from tsas.engine.operator.detection.knn import KNNScorer, KNNDetector
        from tsas.engine.operator.detection.threshold_decider import ThresholdDecider
        operators = {
            "knn_scorer": KNNScorer,
            "knn_detector": KNNDetector,
            "threshold_decider": ThresholdDecider,
        }
        result = generate_list(operators)
        assert "共 3 个算子" in result

    def test_empty_operators(self):
        """
        目的：验证空算子列表
        输入：空字典
        预期：显示 "共 0 个算子"
        """
        result = generate_list({})
        assert "共 0 个算子" in result


# ============================================================================
# 测试 generate_detail — 详情模式
# ============================================================================

class TestGenerateDetail:
    """测试 generate_detail 详情模式"""

    def test_basic_sections(self):
        """
        目的：验证详情模式包含基础分类区块
        输入：_DocOperator
        预期：包含 "### 基础分类" 和 "**类型**"
        """
        result = generate_detail(_DocOperator)
        assert "## test_operator" in result
        assert "### 基础分类" in result
        assert "**类型**" in result

    def test_input_output_sections(self):
        """
        目的：验证 Input/Output section 被提取到详情
        输入：_DocOperator（有 Input/Output docstring sections）
        预期：包含 "### 输入" 和 "### 主输出" 区块
        """
        result = generate_detail(_DocOperator)
        assert "### 输入" in result
        assert "### 主输出" in result
        assert "二维时序数据" in result
        assert "重构后的数据" in result

    def test_input_output_always_shown(self):
        """
        目的：验证输入/主输出区块始终显示
        输入：_MinimalOperator（无 Input/Output docstring 标记）
        预期：仍包含 "### 输入" 和 "### 主输出"，内容为 "（无）"
        """
        result = generate_detail(_MinimalOperator)
        assert "### 输入" in result
        assert "### 主输出" in result
        assert "（无）" in result

    def test_trainable_with_supervision(self):
        """
        目的：验证可训练算子显示监督类型
        输入：KNNDetector（可训练，无监督）
        预期：包含 "是 - 无监督"
        """
        from tsas.engine.operator.detection.knn import KNNDetector
        result = generate_detail(KNNDetector)
        assert "是 - 无监督" in result

    def test_not_learnable(self):
        """
        目的：验证不可训练算子显示 "否"
        输入：ThresholdDecider
        预期：包含 "可训练**: 否"
        """
        from tsas.engine.operator.detection.threshold_decider import ThresholdDecider
        result = generate_detail(ThresholdDecider)
        assert "**可训练**: 否" in result

    def test_batch_run_section(self):
        """
        目的：验证支持分批推理行存在
        输入：任意算子
        预期：包含 "**支持分批推理**"
        """
        result = generate_detail(_DocOperator)
        assert "**支持分批推理**" in result

    def test_always_show_fit_params(self):
        """
        目的：验证训练参数区块始终显示
        输入：无 FitParams 的算子
        预期：包含 "### 训练参数" 和 "（无）"
        """
        result = generate_detail(_DocOperator)
        assert "### 训练参数" in result
        assert "（无）" in result

    def test_always_show_run_params(self):
        """
        目的：验证运行参数区块始终显示
        输入：无 RunParams 的算子
        预期：包含 "### 运行参数" 和 "（无）"
        """
        result = generate_detail(_DocOperator)
        assert "### 运行参数" in result
        assert "（无）" in result

    def test_extra_output_after_main_output(self):
        """
        目的：验证附加输出位于主输出之后
        输入：KNNScorer（有 ExtraOutput）
        预期：附加输出位置在主输出之后
        """
        from tsas.engine.operator.detection.knn import KNNScorer
        result = generate_detail(KNNScorer)
        assert "### 附加输出" in result
        # 附加输出在实例参数之前
        eo_pos = result.index("### 附加输出")
        config_pos = result.index("### 实例参数")
        assert eo_pos < config_pos

    def test_no_config(self):
        """
        目的：验证无 Config 的算子实例参数段显示"（无）"
        输入：_MinimalOperator
        预期：包含 "### 实例参数" 和 "（无）"
        """
        result = generate_detail(_MinimalOperator)
        assert "### 实例参数" in result
        assert "（无）" in result

    def test_section_order(self):
        """
        目的：验证详情区块顺序
        输入：KNNScorer（有 ExtraOutput）
        预期：版本号 → 基础分类 → 实例参数 → 训练参数 → 运行参数
        """
        from tsas.engine.operator.detection.knn import KNNScorer
        result = generate_detail(KNNScorer)
        positions = {
            "版本": result.index("**版本**"),
            "基础分类": result.index("### 基础分类"),
            "实例参数": result.index("### 实例参数"),
            "训练参数": result.index("### 训练参数"),
            "运行参数": result.index("### 运行参数"),
        }
        assert positions["版本"] < positions["基础分类"]
        assert positions["基础分类"] < positions["实例参数"]
        assert positions["实例参数"] < positions["训练参数"]
        assert positions["训练参数"] < positions["运行参数"]

    def test_version_display_for_real_operator(self):
        """
        目的：验证真实算子的版本号在详情中正确显示
        输入：KNNScorer（version=(1,0,0)）
        预期：包含 "**版本**：1.0.0"
        """
        from tsas.engine.operator.detection.knn import KNNScorer
        result = generate_detail(KNNScorer)
        assert "**版本**：1.0.0" in result

    def test_version_between_description_and_classification(self):
        """
        目的：验证版本号位于描述和基础分类之间
        输入：KNNScorer
        预期：版本号位置在描述之后、基础分类之前
        """
        from tsas.engine.operator.detection.knn import KNNScorer
        result = generate_detail(KNNScorer)
        version_pos = result.index("**版本**")
        classification_pos = result.index("### 基础分类")
        assert version_pos < classification_pos

    def test_version_shown_for_operator_without_version_method(self):
        """
        目的：验证无 version() 方法的模拟算子不会崩溃
        输入：_DocOperator（无 version 方法）
        预期：不包含 "**版本**" 行（因为 hasattr 检查失败）
        """
        result = generate_detail(_DocOperator)
        assert "**版本**" not in result


# ============================================================================
# 测试 generate_detail 使用真实算子类
# ============================================================================

class TestGenerateDetailWithRealOperators:
    """使用真实算子验证 generate_detail 集成输出"""

    def test_detection_scorer(self):
        """预期：包含 "Scorer(Single)" 和 "可训练" """
        from tsas.engine.operator.detection.knn import KNNScorer
        result = generate_detail(KNNScorer)
        assert "## knn_scorer" in result
        assert "Scorer(Single)" in result
        assert "可训练" in result

    def test_detection_detector(self):
        """预期：包含 "Detector" 角色"""
        from tsas.engine.operator.detection.knn import KNNDetector
        result = generate_detail(KNNDetector)
        assert "## knn_detector" in result
        assert "Detector" in result

    def test_evaluation_operator(self):
        """预期：包含 "评价指标" 角色"""
        from tsas.engine.operator.evaluation.binary_classification import BinaryClassificationMetric
        result = generate_detail(BinaryClassificationMetric)
        assert "评价指标" in result

    def test_feature_operator(self):
        """预期：包含 "特征算子" 和 "可训练" """
        from tsas.engine.operator.feature.construction.simple_feature import PCAFeature
        result = generate_detail(PCAFeature)
        assert "特征算子" in result
        assert "可训练" in result

    def test_operator_with_extra_output(self):
        """预期：包含 "### 附加输出" """
        from tsas.engine.operator.detection.knn import KNNScorer
        result = generate_detail(KNNScorer)
        assert "### 附加输出" in result


# ============================================================================
# 补充测试 — 提升覆盖率到 90%+
# ============================================================================

class _TestFitParams(BaseModel):
    """测试用训练参数"""
    epochs: int = Field(default=10, ge=1, description="训练轮数")


class _TestRunParams(BaseModel):
    """测试用运行参数"""
    batch_size: int = Field(default=32, ge=1, description="推理批大小")


class _OperatorWithFitAndRunParams:
    """
    有训练和运行参数的算子

    用于测试 generate_detail 中 _fit_params_type 和 _run_params_type
    不为 None 时的输出分支。

    Args:
        config: 配置参数
    """
    _config_type = _TestConfig
    _fit_params_type = _TestFitParams
    _run_params_type = _TestRunParams
    _eo_type = None

    @classmethod
    def name(cls) -> str:
        return "fit_run_op"


class _BlankDocOperator:
    """   


    """
    _config_type = None
    _fit_params_type = None
    _run_params_type = None
    _eo_type = None

    @classmethod
    def name(cls) -> str:
        return "blank_doc_op"


class TestFitAndRunParams:
    """测试 generate_detail 中训练参数和运行参数的实际输出"""

    def test_fit_params_section_with_type(self):
        """
        目的：验证 _fit_params_type 非 None 时输出训练参数表格
        输入：_OperatorWithFitAndRunParams
        预期：包含 "### 训练参数 (_TestFitParams)" 和 "epochs" 字段
        """
        result = generate_detail(_OperatorWithFitAndRunParams)
        assert "### 训练参数 (_TestFitParams)" in result
        assert "epochs" in result
        assert "训练轮数" in result

    def test_run_params_section_with_type(self):
        """
        目的：验证 _run_params_type 非 None 时输出运行参数表格
        输入：_OperatorWithFitAndRunParams
        预期：包含 "### 运行参数 (_TestRunParams)" 和 "batch_size" 字段
        """
        result = generate_detail(_OperatorWithFitAndRunParams)
        assert "### 运行参数 (_TestRunParams)" in result
        assert "batch_size" in result
        assert "推理批大小" in result


class TestMultiScorerRole:
    """测试 MultiScorer 角色提取和类型标签"""

    def test_extract_role_multi_scorer(self):
        """
        目的：验证 _extract_role 对 MultiScorer 返回 Scorer(Multi)
        输入：XiHeGammaScorer（继承 MultiScorerMixin）
        预期：返回 "Scorer(Multi)"
        """
        from tsas.engine.operator.detection.xihe import XiHeGammaScorer
        assert _extract_role(XiHeGammaScorer) == "Scorer(Multi)"

    def test_extract_type_tags_multi_scorer(self):
        """
        目的：验证 _extract_type_tags 对 MultiScorer 包含 Scorer(Multi) 标签
        输入：XiHeGammaScorer
        预期：tags 列表包含 "Scorer(Multi)"
        """
        from tsas.engine.operator.detection.xihe import XiHeGammaScorer
        tags = _extract_type_tags(XiHeGammaScorer)
        assert "Scorer(Multi)" in tags

    def test_extract_role_residual_map_scorer(self):
        """
        目的：验证另一个 MultiScorer 的角色提取
        输入：ResidualMapScorer
        预期：返回 "Scorer(Multi)"
        """
        from tsas.engine.operator.detection.residual_scorer import ResidualMapScorer
        assert _extract_role(ResidualMapScorer) == "Scorer(Multi)"


class TestExtractSummaryEdgeCases:
    """测试 _extract_summary 的边界条件"""

    def test_all_blank_docstring(self):
        """
        目的：验证仅含空白的 docstring 返回 "(无描述)"
        输入：_BlankDocOperator（docstring 仅含空白行）
        预期：返回 "(无描述)"
        """
        result = _extract_summary(_BlankDocOperator)
        assert result == "(无描述)"


class TestExtractDescriptionEdgeCases:
    """测试 _extract_description 的边界条件"""

    def test_section_marker_break(self):
        """
        目的：验证遇到 section 标记时停止提取
        输入：docstring 中第一段后紧跟 Args: 标记
        预期：只返回第一段文本，不含 Args 内容
        """
        class _SectionDocClass:
            """这是第一段描述文本

            Args:
                x: 参数
            """
        result = _extract_description(_SectionDocClass)
        assert "这是第一段描述文本" in result
        assert "Args" not in result
        assert "参数" not in result

    def test_input_section_marker_break(self):
        """
        目的：验证遇到 Input: section 标记时停止提取
        输入：docstring 中第一段后紧跟 Input: 标记
        预期：只返回第一段文本
        """
        class _InputDocClass:
            """功能描述

            Input:
                DataFrame — 数据
            """
        result = _extract_description(_InputDocClass)
        assert "功能描述" in result
        assert "DataFrame" not in result


class TestExtractDocstringSectionEdgeCases:
    """测试 _extract_docstring_section 的边界条件"""

    def test_content_after_colon(self):
        """
        目的：验证 section 标记同行冒号后的内容被提取
        输入："Input: 同行内容" 形式的 docstring
        预期：返回 "同行内容"
        """
        doc = """
        描述段落

        Input: 同行内容
        """
        result = _extract_docstring_section(doc, "Input")
        assert "同行内容" in result

    def test_section_break_on_new_section(self):
        """
        目的：验证遇到新的 section 标记时停止提取
        输入：Input section 后面紧跟 Args section
        预期：只返回 Input 内容，不含 Args 内容
        """
        doc = """
        Input:
            输入数据描述

        Args:
            x: 参数
        """
        result = _extract_docstring_section(doc, "Input")
        assert "输入数据描述" in result
        assert "参数" not in result

    def test_known_section_marker_break(self):
        """
        目的：验证遇到常见 section 标记（如 Returns:）时停止
        输入：Output section 后紧跟 Returns section
        预期：只返回 Output 内容
        """
        doc = """
        Output:
            输出数据描述
        Returns:
            返回值描述
        """
        result = _extract_docstring_section(doc, "Output")
        assert "输出数据描述" in result
        assert "返回值描述" not in result

    def test_empty_line_terminates_section(self):
        """
        目的：验证空行终止 section 提取
        输入：section 内容后有空行再有其他文本
        预期：只返回空行之前的内容
        """
        doc = """
        Input:
            第一行内容

            这不应该被包含
        """
        result = _extract_docstring_section(doc, "Input")
        assert "第一行内容" in result
        assert "这不应该被包含" not in result


class TestFormatTypeEdgeCases:
    """测试 _format_type 的边界条件"""

    def test_origin_without_args(self):
        """
        目的：验证有 origin 但无 args 的类型
        输入：list（原始类型，非 list[X]）
        预期：返回合理的字符串表示
        """
        # list 本身（非 list[int]）在 get_origin 中返回 None，走 __name__ 分支
        result = _format_type(list)
        assert "list" in result

    def test_no_name_no_origin(self):
        """
        目的：验证无 __name__ 且无 origin 的类型回退到 str()
        输入：无 __name__ 的自定义对象
        预期：返回 str(annotation) 的结果
        """
        # 使用一个没有 __name__ 的 typing 构造
        from typing import Union
        # Union[int, str] 有 origin 和 args，走 origin 分支
        result = _format_type(Union[int, str])
        assert "int" in result or "str" in result


class TestImportErrorBranches:
    """通过 mock 模拟 ImportError 测试异常分支"""

    def test_is_learnable_import_error(self):
        """
        目的：验证 _is_learnable 在 ImportError 时返回 False
        输入：mock 导致 ImportError
        预期：返回 False
        """
        from unittest.mock import patch
        import tsas.engine.operator.cli.help_generator as hg

        # 临时替换 import 来模拟 ImportError
        original = hg.__builtins__ if isinstance(hg.__builtins__, dict) else None

        import builtins
        real_import = builtins.__import__

        def mock_import(name, *args, **kwargs):
            if name == 'tsas.engine.operator.base':
                raise ImportError("mocked")
            return real_import(name, *args, **kwargs)

        with patch('builtins.__import__', side_effect=mock_import):
            # 需要清除模块缓存以触发重新导入
            import sys
            saved = sys.modules.pop('tsas.engine.operator.base', None)
            try:
                result = _is_learnable(int)  # int 不会匹配任何 Mixin
                assert result is False
            finally:
                if saved is not None:
                    sys.modules['tsas.engine.operator.base'] = saved

    def test_supervision_type_import_error(self):
        """
        目的：验证 _supervision_type 在 ImportError 时返回 "无监督"
        输入：mock 导致 ImportError
        预期：返回 "无监督"
        """
        import builtins
        from unittest.mock import patch
        real_import = builtins.__import__

        def mock_import(name, *args, **kwargs):
            if name == 'tsas.engine.operator.base':
                raise ImportError("mocked")
            return real_import(name, *args, **kwargs)

        import sys
        saved = sys.modules.pop('tsas.engine.operator.base', None)
        try:
            with patch('builtins.__import__', side_effect=mock_import):
                result = _supervision_type(int)
                assert result == "无监督"
        finally:
            if saved is not None:
                sys.modules['tsas.engine.operator.base'] = saved

    def test_supports_batch_run_import_error(self):
        """
        目的：验证 _supports_batch_run 在 ImportError 时返回 False
        输入：mock 导致 ImportError
        预期：返回 False
        """
        import builtins
        from unittest.mock import patch
        real_import = builtins.__import__

        def mock_import(name, *args, **kwargs):
            if name == 'tsas.engine.operator.base':
                raise ImportError("mocked")
            return real_import(name, *args, **kwargs)

        import sys
        saved = sys.modules.pop('tsas.engine.operator.base', None)
        try:
            with patch('builtins.__import__', side_effect=mock_import):
                result = _supports_batch_run(int)
                assert result is False
        finally:
            if saved is not None:
                sys.modules['tsas.engine.operator.base'] = saved

    def test_extract_role_import_errors(self):
        """
        目的：验证 _extract_role 在各模块 ImportError 时返回 "未知"
        输入：mock 导致所有 detection/feature/evaluation 模块 ImportError
        预期：返回 "未知"
        """
        import builtins
        from unittest.mock import patch
        real_import = builtins.__import__

        blocked = {
            'tsas.engine.operator.detection.base',
            'tsas.engine.operator.feature.construction.base',
            'tsas.engine.operator.evaluation.base',
        }

        def mock_import(name, *args, **kwargs):
            if name in blocked:
                raise ImportError("mocked")
            return real_import(name, *args, **kwargs)

        import sys
        saved = {m: sys.modules.pop(m, None) for m in blocked}
        try:
            with patch('builtins.__import__', side_effect=mock_import):
                result = _extract_role(int)
                assert result == "未知"
        finally:
            for m, mod in saved.items():
                if mod is not None:
                    sys.modules[m] = mod

    def test_extract_type_tags_import_errors(self):
        """
        目的：验证 _extract_type_tags 在各模块 ImportError 时不崩溃
        输入：mock 导致所有相关模块 ImportError
        预期：返回空列表（非 Detector 类）
        """
        import builtins
        from unittest.mock import patch
        real_import = builtins.__import__

        blocked = {
            'tsas.engine.operator.detection.base',
            'tsas.engine.operator.base',
            'tsas.engine.operator.feature.construction.base',
            'tsas.engine.operator.evaluation.base',
        }

        def mock_import(name, *args, **kwargs):
            if name in blocked:
                raise ImportError("mocked")
            return real_import(name, *args, **kwargs)

        import sys
        saved = {m: sys.modules.pop(m, None) for m in blocked}
        try:
            with patch('builtins.__import__', side_effect=mock_import):
                result = _extract_type_tags(int)
                assert isinstance(result, list)
        finally:
            for m, mod in saved.items():
                if mod is not None:
                    sys.modules[m] = mod


# ============================================================================
# 测试 generate_detail 中 _output_type 的渲染（主输出结构字段表）
# ============================================================================

class _TestOutputModel(BaseModel):
    """测试用 BaseModel 主输出结构"""
    metric_value: float = Field(description="指标值")
    flag: bool = Field(default=False, description="标志位")


class _OutputTypeOnlyOperator:
    """仅有 _output_type、无 docstring Output 段的算子（BaseModel 输出）

    用于验证：当 docstring Output 段为空、_output_type 是 BaseModel 时，
    generate_detail 渲染 "### 主输出 (类型名)" + "**结构**：" 字段表。
    """
    _config_type = None
    _run_params_type = None
    _eo_type = None
    _fit_params_type = None
    _input_type = None
    _output_type = _TestOutputModel

    @classmethod
    def name(cls) -> str:
        return "output_type_only_op"


class _OutputTypeWithDocstringOperator:
    """
    测试算子（BaseModel 输出 + 含 Output docstring）

    用于验证合并显示策略：标题带类型、docstring 描述在前、字段表在后。

    Output:
        语义层面的输出说明，描述主输出的用途
    """
    _config_type = None
    _run_params_type = None
    _eo_type = None
    _fit_params_type = None
    _input_type = None
    _output_type = _TestOutputModel

    @classmethod
    def name(cls) -> str:
        return "output_type_with_doc_op"


class _ScalarOutputOperator:
    """标量输出算子（_output_type = float）

    用于验证：标题带标量类型，无字段表。
    """
    _config_type = None
    _run_params_type = None
    _eo_type = None
    _fit_params_type = None
    _input_type = None
    _output_type = float

    @classmethod
    def name(cls) -> str:
        return "scalar_output_op"


class _UnionOutputOperator:
    """联合类型输出算子（_output_type = NumericData | tuple[NumericData, EO]）

    用于验证：标题带简化后的主输出类型（通过 _simplify_output_type 提取）。
    """
    _config_type = None
    _run_params_type = None
    _eo_type = None
    _fit_params_type = None
    _input_type = None
    _output_type = None  # 将在测试中设置

    @classmethod
    def name(cls) -> str:
        return "union_output_op"


class _VariableInputOperator:
    """
    测试算子（含 Input 段，按 Args 风格写变量名）

    用于验证 Input 段渲染：变量名 + 类型 + 描述。

    Input:
        x: 特征矩阵，形状 (n_samples, n_features)
    """
    _config_type = None
    _run_params_type = None
    _eo_type = None
    _fit_params_type = None
    _input_type = None  # 将在测试中设置
    _output_type = None

    @classmethod
    def name(cls) -> str:
        return "variable_input_op"


class TestOutputTypeRendering:
    """测试 generate_detail 中 Input/Output 段的新渲染逻辑"""

    def test_basemodel_output_renders_title_with_type(self):
        """
        目的：验证 BaseModel 输出时标题带类型名
        输入：_OutputTypeOnlyOperator（_output_type=_TestOutputModel）
        预期：标题为 "### 主输出 (_TestOutputModel)"
        """
        result = generate_detail(_OutputTypeOnlyOperator)
        assert "### 主输出 (_TestOutputModel)" in result

    def test_basemodel_output_renders_structure_table(self):
        """
        目的：验证 BaseModel 输出渲染 "**结构**：" 字段表
        输入：_OutputTypeOnlyOperator
        预期：包含 "**结构**：" 和字段 "metric_value" 和说明 "指标值"
        """
        result = generate_detail(_OutputTypeOnlyOperator)
        assert "**结构**：" in result
        assert "metric_value" in result
        assert "指标值" in result

    def test_basemodel_output_with_docstring_merge(self):
        """
        目的：验证 BaseModel 输出 + docstring 描述的合并显示
        输入：_OutputTypeWithDocstringOperator
        预期：标题带类型、描述在前、字段表在后
        """
        result = generate_detail(_OutputTypeWithDocstringOperator)
        # 标题带类型
        assert "### 主输出 (_TestOutputModel)" in result
        # docstring 描述
        assert "语义层面的输出说明" in result
        # 字段表
        assert "**结构**：" in result
        assert "metric_value" in result

    def test_basemodel_output_docstring_before_structure(self):
        """
        目的：验证 docstring 描述位置在字段表之前
        输入：_OutputTypeWithDocstringOperator
        预期：描述位置 < "**结构**" 位置
        """
        result = generate_detail(_OutputTypeWithDocstringOperator)
        docstring_pos = result.index("语义层面的输出说明")
        struct_pos = result.index("**结构**")
        assert docstring_pos < struct_pos

    def test_scalar_output_title_with_type(self):
        """
        目的：验证标量输出（float）标题带类型
        输入：_ScalarOutputOperator（_output_type=float）
        预期：标题为 "### 主输出 (float)"，无字段表
        """
        result = generate_detail(_ScalarOutputOperator)
        assert "### 主输出 (float)" in result
        # 标量输出不应有字段表
        assert "**结构**" not in result

    def test_none_output_no_type_in_title(self):
        """
        目的：验证 _output_type 为 None 时标题不带类型
        输入：_MinimalOperator（_output_type=None）
        预期：标题为 "### 主输出"（无括号类型）
        """
        result = generate_detail(_MinimalOperator)
        assert "### 主输出\n" in result  # 标题后直接换行，无 (类型)

    def test_none_output_renders_placeholder(self):
        """
        目的：验证 _output_type 为 None 且无 docstring 时显示"（无）"
        输入：_MinimalOperator
        预期：包含 "### 主输出" 和 "（无）"
        """
        result = generate_detail(_MinimalOperator)
        assert "### 主输出" in result
        assert "（无）" in result

    def test_knn_scorer_main_output_title_simplified(self):
        """
        目的：验证 KNNScorer 主输出标题是简化后的 NumericData
        输入：KNNScorer（_output_type=NumericData | tuple[...]）
        预期：标题为 "### 主输出 (pandas.DataFrame | numpy.ndarray)"
        """
        from tsas.engine.operator.detection.knn import KNNScorer
        result = generate_detail(KNNScorer)
        assert "### 主输出 (pandas.DataFrame | numpy.ndarray)" in result

    def test_knn_scorer_main_output_with_docstring(self):
        """
        目的：验证 KNNScorer 主输出段含 docstring 描述（异常分数）
        输入：KNNScorer
        预期：包含 "异常分数" 描述
        """
        from tsas.engine.operator.detection.knn import KNNScorer
        result = generate_detail(KNNScorer)
        assert "异常分数" in result

    def test_knn_scorer_no_structure_table(self):
        """
        目的：验证 KNNScorer 主输出是联合类型（简化后为 NumericData），不渲染字段表
        输入：KNNScorer
        预期：不包含 "**结构**"
        """
        from tsas.engine.operator.detection.knn import KNNScorer
        result = generate_detail(KNNScorer)
        assert "**结构**" not in result

    def test_knn_scorer_eo_section_preserved(self):
        """
        目的：验证 KNNScorer 的附加输出（EO）段保留
        输入：KNNScorer（_eo_type=KNNScorerExtraOutput）
        预期：包含 "### 附加输出 (KNNScorerExtraOutput)"
        """
        from tsas.engine.operator.detection.knn import KNNScorer
        result = generate_detail(KNNScorer)
        assert "### 附加输出 (KNNScorerExtraOutput)" in result

    def test_binary_classification_metric_full_rendering(self):
        """
        目的：验证 BinaryClassificationMetric 完整渲染（多层追踪 + 标题带类型 + 字段表）
        输入：BinaryClassificationMetric
        预期：标题为 "### 主输出 (BinaryClassificationResult)"，含字段表
        """
        from tsas.engine.operator.evaluation.binary_classification import (
            BinaryClassificationMetric,
            BinaryClassificationResult,
        )
        assert BinaryClassificationMetric._output_type is BinaryClassificationResult
        result = generate_detail(BinaryClassificationMetric)
        assert "### 主输出 (BinaryClassificationResult)" in result
        assert "**结构**：" in result
        assert "f1" in result  # BinaryClassificationResult 含 f1 字段

    def test_self_evaluation_scalar_output(self):
        """
        目的：验证 SelfEvaluation 标量输出（float）渲染
        输入：SelfEvaluation（MR=float）
        预期：标题为 "### 主输出 (float)"，无字段表
        """
        from tsas.engine.operator.evaluation.self_evaluation import SelfEvaluation
        assert SelfEvaluation._output_type is float
        result = generate_detail(SelfEvaluation)
        assert "### 主输出 (float)" in result
        assert "**结构**" not in result

    def test_knn_scorer_input_with_variable_and_type(self):
        """
        目的：验证 KNNScorer Input 段渲染为 "变量名 (类型): 描述"
        输入：KNNScorer（docstring 含 "x: 特征矩阵..."）
        预期：Input 段含 "x (pandas.DataFrame | numpy.ndarray): 特征矩阵"
        """
        from tsas.engine.operator.detection.knn import KNNScorer
        result = generate_detail(KNNScorer)
        assert "### 输入" in result
        assert "x (pandas.DataFrame | numpy.ndarray): 特征矩阵" in result

    def test_knn_detector_input_and_output(self):
        """
        目的：验证 KNNDetector Input/Output 段渲染
        输入：KNNDetector
        预期：Input 含 "x (...): 特征矩阵"，主输出含 "异常标签"
        """
        from tsas.engine.operator.detection.knn import KNNDetector
        result = generate_detail(KNNDetector)
        assert "x (pandas.DataFrame | numpy.ndarray): 特征矩阵" in result
        assert "异常标签" in result

    def test_cicada_predictor_input_no_variable_renders_type_only(self):
        """
        目的：验证 cicada_predictor（无变量名 docstring）的 Input 段渲染为 "(类型): 描述"
        输入：CICADAPredictor（docstring Input 是纯描述）
        预期：Input 段含 "(pandas.DataFrame | numpy.ndarray): 二维时序数据"
        """
        try:
            from tsas.engine.operator.detection.cicada import CICADAPredictor
        except ImportError:
            self.skipTest("cicada package not available")
        result = generate_detail(CICADAPredictor)
        assert "(pandas.DataFrame | numpy.ndarray)" in result

    def test_self_evaluation_input_renders_type_only(self):
        """
        目的：验证 SelfEvaluation（无 docstring Input）的 Input 段渲染为 "(类型)"
        输入：SelfEvaluation
        预期：Input 段含 "(numpy.ndarray)"
        """
        from tsas.engine.operator.evaluation.self_evaluation import SelfEvaluation
        result = generate_detail(SelfEvaluation)
        assert "(numpy.ndarray)" in result

    def test_binary_classification_metric_input_renders_tuple_type(self):
        """
        目的：验证 BinaryClassificationMetric（tuple 输入，docstring 写两个变量）渲染
        输入：BinaryClassificationMetric（I=tuple[ndarray, ndarray]，docstring 写 y_truth + y_predict）
        预期：Input 段按位置拆解 tuple 元素，每行变量独立带单一元素类型，
            形如 ``y_truth (numpy.ndarray): ...`` 与 ``y_predict (numpy.ndarray): ...``
        """
        from tsas.engine.operator.evaluation.binary_classification import BinaryClassificationMetric
        result = generate_detail(BinaryClassificationMetric)
        # 双变量按位置拆解 tuple[ndarray, ndarray]，每个变量对应 ndarray 类型
        assert "y_truth (numpy.ndarray):" in result
        assert "y_predict (numpy.ndarray):" in result
