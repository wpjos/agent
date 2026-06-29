# -*- coding: utf-8 -*-

"""
算子帮助文档自动生成模块

从算子类的元信息（docstring、Config 类的 Pydantic Field 定义、Enum/Literal 类型注解等）
中自动提取并生成结构化 Markdown 格式的帮助文档。

核心能力:
    - 列表模式: 按分组输出算子的名称、类型、可训练性和简介（CJK 对齐表格）
    - 详情模式: 输出单个算子的版本号、基础分类、输入输出说明、参数表格等完整信息
    - 参数信息直接从 ``model_fields`` 中提取，与 HPO 模块共享同一数据源

提取来源:
    - ``cls.name()`` → 算子名称
    - ``cls.__doc__`` → 功能描述（含 Input/Output 结构化 section）
    - ``cls._config_type.model_fields`` → 实例参数表格
    - ``cls._fit_params_type`` → 训练参数表格
    - ``cls._run_params_type`` → 运行参数表格
    - ``cls._eo_type`` → 附加输出表格
    - ``cls.version()`` → 版本号（点分字符串格式）
    - ``issubclass`` 检查 → 角色/可训练/监督类型/分批推理等元信息
    - ``cls.__module__`` → 组合管线算子分组判断

输出格式面向 Agent Skill 友好，采用结构化 Markdown，支持 CJK-aware 列对齐。

使用示例::

    from tsas.engine.operator.cli.help_generator import generate_list, generate_detail

    # 列表模式
    print(generate_list({"knn_scorer": KNNScorer, "zscore_detector": ZScoreDetector}))

    # 详情模式
    print(generate_detail(KNNScorer))
"""

import enum
import inspect
import re
import types
import typing
from typing import Literal, get_args, get_origin

import numpy as np
import pandas as pd
import unicodedata
from annotated_types import Ge, Gt, Le, Lt
from loguru import logger
from pydantic import BaseModel
from pydantic.fields import PydanticUndefined

__all__ = [
    'generate_list',
    'generate_detail',
    'generate_config_params_table',
]

# ============================================================================
# 列表模式分组常量
# ============================================================================

# 分组显示顺序和标题
_GROUP_PIPELINE = "管线组件算子"
_GROUP_END_TO_END = "端到端检测器算子"
_GROUP_COMPOSITE = "组合管线算子"

# 分组排序权重（越小越靠前）
_GROUP_ORDER = {
    _GROUP_PIPELINE: 0,
    _GROUP_END_TO_END: 1,
    _GROUP_COMPOSITE: 2,
}


# ============================================================================
# 公共函数
# ============================================================================

def generate_list(operators: dict[str, type]) -> str:
    """
    生成所有算子的分组列表概览（列表模式）

    按三类分组（管线组件算子、端到端检测器算子、组合管线算子）展示，
    每组一个 CJK 对齐的 Markdown 表格，包含名称、类型、可训练和简介四列。

    Args:
        operators (dict[str, type]): {算子名称: 算子类} 映射，通常来自
            ``OperatorRegistry.list_all()``

    Returns:
        str: Markdown 格式的分组算子列表
    """
    lines = []
    lines.append("## 可用算子列表")
    lines.append("")

    # ---- 按分组归类算子 ----
    groups: dict[str, list[tuple[str, type]]] = {}
    for op_name, cls in sorted(operators.items()):
        group = _classify_operator(cls)
        groups.setdefault(group, []).append((op_name, cls))

    # ---- 按预定义顺序输出每个分组 ----
    for group_name in sorted(groups.keys(), key=lambda g: _GROUP_ORDER.get(g, 99)):
        lines.append(f"### {group_name}")
        lines.append("")

        # 构建表格数据
        headers = ["名称", "类型", "可训练", "简介"]
        rows = []
        for op_name, cls in groups[group_name]:
            role = _extract_role(cls)
            learnable = "是" if _is_learnable(cls) else "否"
            summary = _extract_summary(cls)
            rows.append([op_name, role, learnable, summary])

        # 生成对齐表格
        table = _build_aligned_table(headers, rows)
        lines.append(table)
        lines.append("")

    # ---- 统计汇总 ----
    lines.append(f"共 {len(operators)} 个算子。使用 `help <算子名称>` 查看详细信息。")
    return '\n'.join(lines)


def generate_detail(cls: type) -> str:
    """
    生成单个算子的详细帮助文档（详情模式）

    按版本号 → 基础分类 → 输入 → 主输出 → 附加输出 → 实例参数 → 训练参数 → 运行参数
    的顺序输出完整的结构化帮助文档。所有表格均采用 CJK-aware 列对齐。

    Args:
        cls (type): 算子类

    Returns:
        str: Markdown 格式的详细帮助文档
    """
    lines = []

    # ---- 标题和名称 ----
    op_name = cls.name() if hasattr(cls, 'name') and callable(cls.name) else cls.__name__
    lines.append(f"## {op_name}")
    lines.append("")

    # ---- 功能描述 ----
    description = _extract_description(cls)
    if description:
        lines.append(description)
        lines.append("")

    # ---- 版本号 ----
    if hasattr(cls, 'version') and callable(cls.version):
        from tsas.engine.operator.base import BaseOperator
        version_str = BaseOperator._format_version(cls.version())
        lines.append(f"**版本**：{version_str}")
        lines.append("")

    # ---- 基础分类 ----
    lines.append("### 基础分类")
    lines.append("")
    role = _extract_role(cls)
    lines.append(f"**类型**: {role}")
    # 可训练
    if _is_learnable(cls):
        supervision = _supervision_type(cls)
        lines.append(f"**可训练**: 是 - {supervision}")
    else:
        lines.append("**可训练**: 否")
    # 支持分批推理
    batch_run = "是" if _supports_batch_run(cls) else "否"
    lines.append(f"**支持分批推理**: {batch_run}")
    lines.append("")

    # ---- 输入段（变量名 + 类型 + 描述，元组拆解，降级处理）----
    # 渲染策略：
    #   1. 从 _input_type 提取类型（可能为单一类型或 tuple[...]）
    #   2. 用 _split_input_types 拆解为类型字符串列表
    #   3. 从 docstring Input 段解析变量名与描述
    #   4. 按变量数与类型数匹配渲染：
    #      - 匹配：每行 "变量名 (类型): 描述"
    #      - 不匹配：降级为单变量 "x (完整类型): 描述"
    #      - 无变量：显示 "(类型): 描述" 或 "(类型)" 或纯描述或"（无）"
    #   5. BaseModel 输入额外渲染字段表
    lines.extend(_render_input_section(cls))

    # ---- 主输出段（标题带类型 + docstring 描述 + BaseModel 字段表）----
    # 渲染策略：
    #   1. 用 _simplify_output_type 从 _output_type 提取主输出（处理 T | tuple[T, EO] 形式）
    #   2. 标题："### 主输出 (类型字符串)" 或 "### 主输出"（无类型时）
    #   3. 内容：
    #      - docstring Output 段非空：先展示描述
    #      - 主输出是 BaseModel：追加 "**结构**：" + 字段表
    #      - 都无内容：显示 "（无）"
    lines.extend(_render_output_section(cls))

    # ---- 附加输出 (ExtraOutput) ----
    eo_type = getattr(cls, '_eo_type', None)
    if eo_type is not None and eo_type is not type(None):
        lines.append(f"### 附加输出 ({eo_type.__name__})")
        lines.append("")
        eo_desc = _extract_model_fields_description(eo_type)
        if eo_desc:
            lines.append(eo_desc)
            lines.append("")

    # ---- 实例参数 (Config) — 始终显示 ----
    config_type = getattr(cls, '_config_type', None)
    if config_type is not None and config_type is not type(None):
        lines.append(f"### 实例参数 ({config_type.__name__})")
        lines.append("")
        table = generate_config_params_table(config_type)
        lines.append(table)
    else:
        lines.append("### 实例参数")
        lines.append("")
        lines.append("（无）")
    lines.append("")

    # ---- 训练参数 (FitParams) — 始终显示 ----
    fp_type = getattr(cls, '_fit_params_type', None)
    if fp_type is not None and fp_type is not type(None):
        lines.append(f"### 训练参数 ({fp_type.__name__})")
        lines.append("")
        table = generate_config_params_table(fp_type)
        lines.append(table)
    else:
        lines.append("### 训练参数")
        lines.append("")
        lines.append("（无）")
    lines.append("")

    # ---- 运行参数 (RunParams) — 始终显示 ----
    rp_type = getattr(cls, '_run_params_type', None)
    if rp_type is not None and rp_type is not type(None):
        lines.append(f"### 运行参数 ({rp_type.__name__})")
        lines.append("")
        table = generate_config_params_table(rp_type)
        lines.append(table)
    else:
        lines.append("### 运行参数")
        lines.append("")
        lines.append("（无）")
    lines.append("")

    return '\n'.join(lines)


def generate_config_params_table(config_cls: type[BaseModel]) -> str:
    """
    从 Pydantic Config 类生成 CJK 对齐的参数表格

    遍历 ``model_fields``，从 field_info 中提取类型、默认值、
    值域/候选值、说明等信息，格式化为列对齐的 Markdown 表格。

    Args:
        config_cls (type[BaseModel]): Pydantic Config/Params 类

    Returns:
        str: Markdown 格式的参数表格（CJK-aware 列对齐）
    """
    headers = ["参数名", "类型", "默认值", "值域/候选", "说明"]
    rows = []

    for field_name, field_info in config_cls.model_fields.items():
        # ---- 类型 ----
        type_str = _format_type(field_info.annotation)

        # ---- 默认值 ----
        default = field_info.default
        if default is PydanticUndefined or default is ...:
            default_str = "**必填**"
        elif default is None:
            default_str = "None"
        elif isinstance(default, enum.Enum):
            default_str = f"{default.value}"
        else:
            default_str = f"{default}"

        # ---- 值域/候选值 ----
        constraints = _extract_constraints(field_info)

        # ---- 说明 ----
        desc = field_info.description or ""

        rows.append([field_name, type_str, default_str, constraints, desc])

    return _build_aligned_table(headers, rows)


# ============================================================================
# 内部辅助函数 — 元信息提取
# ============================================================================

def _extract_role(cls: type) -> str:
    """
    提取算子管线角色，含 Scorer 细分

    通过类名后缀和 issubclass 检查 MRO 中的角色 Mixin 来确定算子角色。
    Scorer 进一步细分为 Scorer(Single) 和 Scorer(Multi)。

    注意: 实际的 Detector 类（KNNDetector 等）不继承 ``BaseDetector``，
    而是直接继承 ``BaseDeciderMixin``，因此通过类名后缀 ``"Detector"`` 判断。

    Args:
        cls (type): 算子类

    Returns:
        str: 角色字符串，如 ``"Predictor"``、``"Scorer(Single)"``、
            ``"Scorer(Multi)"``、``"Decider"``、``"Detector"``
    """
    # ---- Detector: 通过类名后缀判断（实际 Detector 类不继承 BaseDetector） ----
    cls_name = cls.__name__ if hasattr(cls, '__name__') else ''
    if cls_name.endswith('Detector'):
        return "Detector"

    try:
        from tsas.engine.operator.detection.base import (
            BasePredictorMixin, BaseScorerMixin, BaseDeciderMixin,
            SingleScorerMixin, MultiScorerMixin,
        )
        if issubclass(cls, SingleScorerMixin):
            return "Scorer(Single)"
        elif issubclass(cls, MultiScorerMixin):
            return "Scorer(Multi)"
        elif issubclass(cls, BaseScorerMixin):
            return "Scorer"
        elif issubclass(cls, BaseDeciderMixin):
            return "Decider"
        elif issubclass(cls, BasePredictorMixin):
            return "Predictor"
    except ImportError:
        pass

    try:
        from tsas.engine.operator.feature.construction.base import BaseFeatureMixin
        if issubclass(cls, BaseFeatureMixin):
            return "特征算子"
    except ImportError:
        pass

    try:
        from tsas.engine.operator.feature.selection.base import BaseFeatureSelectorMixin
        if issubclass(cls, BaseFeatureSelectorMixin):
            return "特征选择器"
    except ImportError:
        pass

    try:
        from tsas.engine.operator.evaluation.base import BaseMetricOperator
        if issubclass(cls, BaseMetricOperator):
            return "评价指标"
    except ImportError:
        pass

    return "未知"


def _is_learnable(cls: type) -> bool:
    """
    判断算子是否可训练

    Args:
        cls (type): 算子类

    Returns:
        bool: 是否继承 ``LearnableOperatorMixin``
    """
    try:
        from tsas.engine.operator.base import LearnableOperatorMixin
        return issubclass(cls, LearnableOperatorMixin)
    except ImportError:
        return False


def _supervision_type(cls: type) -> str:
    """
    提取训练监督类型

    Args:
        cls (type): 算子类

    Returns:
        str: ``"有监督"`` 或 ``"无监督"``
    """
    try:
        from tsas.engine.operator.base import SupervisedNumericOperatorMixin
        if issubclass(cls, SupervisedNumericOperatorMixin):
            return "有监督"
    except ImportError:
        pass
    return "无监督"


def _supports_batch_run(cls: type) -> bool:
    """
    判断算子是否支持分批推理

    Args:
        cls (type): 算子类

    Returns:
        bool: 是否继承 ``BatchRunNumericOperatorMixin``
    """
    try:
        from tsas.engine.operator.base import BatchRunNumericOperatorMixin
        return issubclass(cls, BatchRunNumericOperatorMixin)
    except ImportError:
        return False


def _classify_operator(cls: type) -> str:
    """
    判断算子所属分组（用于列表模式分类展示）

    分组规则：
    - 组合管线算子: 模块路径包含 ``"composite"`` 的类
    - 端到端检测器算子: 类名以 ``"Detector"`` 结尾的非组合类
    - 管线组件算子: 其他所有算子

    Args:
        cls (type): 算子类

    Returns:
        str: 分组名称（``"管线组件算子"`` / ``"端到端检测器算子"`` / ``"组合管线算子"``）
    """
    module = getattr(cls, '__module__', '')
    cls_name = cls.__name__ if hasattr(cls, '__name__') else ''

    # 组合管线算子: 仅 composite.py 包下的类
    if 'composite' in module:
        return _GROUP_COMPOSITE

    # 端到端检测器: 类名以 "Detector" 结尾且非组合
    if cls_name.endswith('Detector'):
        return _GROUP_END_TO_END

    # 其他均为管线组件
    return _GROUP_PIPELINE


def _extract_type_tags(cls: type) -> list[str]:
    """
    从类的 MRO 中提取类型标签（兼容旧接口）

    检查常见的 Mixin 类型，生成人类可读的标签列表。
    包含角色细分（Scorer(Single)/Scorer(Multi)）、可训练、支持分批推理、
    监督类型等维度。

    Args:
        cls (type): 目标类

    Returns:
        list[str]: 类型标签列表，如 ``["Scorer(Single)", "可训练", "支持分批推理"]``
    """
    tags = []

    # ---- 检测域角色 ----
    # Detector 通过类名后缀判断（实际 Detector 类不继承 BaseDetector）
    cls_name = cls.__name__ if hasattr(cls, '__name__') else ''
    if cls_name.endswith('Detector'):
        tags.append("Detector")
    else:
        try:
            from tsas.engine.operator.detection.base import (
                BasePredictorMixin, BaseScorerMixin, BaseDeciderMixin,
                SingleScorerMixin, MultiScorerMixin,
            )
            if issubclass(cls, SingleScorerMixin):
                tags.append("Scorer(Single)")
            elif issubclass(cls, MultiScorerMixin):
                tags.append("Scorer(Multi)")
            elif issubclass(cls, BaseScorerMixin):
                tags.append("Scorer")
            elif issubclass(cls, BaseDeciderMixin):
                tags.append("Decider")
            elif issubclass(cls, BasePredictorMixin):
                tags.append("Predictor")
        except ImportError:
            pass

    # ---- 可训练 ----
    try:
        from tsas.engine.operator.base import LearnableOperatorMixin
        if issubclass(cls, LearnableOperatorMixin):
            tags.append("可训练")
    except ImportError:
        pass

    # ---- 支持分批推理 ----
    try:
        from tsas.engine.operator.base import BatchRunNumericOperatorMixin
        if issubclass(cls, BatchRunNumericOperatorMixin):
            tags.append("支持分批推理")
    except ImportError:
        pass

    # ---- 监督类型 ----
    try:
        from tsas.engine.operator.base import (
            SupervisedNumericOperatorMixin,
            UnsupervisedNumericOperatorMixin,
        )
        if issubclass(cls, SupervisedNumericOperatorMixin):
            tags.append("有监督")
        elif issubclass(cls, UnsupervisedNumericOperatorMixin):
            tags.append("无监督")
    except ImportError:
        pass

    # ---- 特征算子 ----
    try:
        from tsas.engine.operator.feature.construction.base import BaseFeatureMixin
        if issubclass(cls, BaseFeatureMixin):
            tags.append("特征算子")
    except ImportError:
        pass

    # ---- 评价指标 ----
    try:
        from tsas.engine.operator.evaluation.base import BaseMetricOperator
        if issubclass(cls, BaseMetricOperator):
            tags.append("评价指标")
    except ImportError:
        pass

    return tags


# ============================================================================
# 内部辅助函数 — 文本提取
# ============================================================================

def _extract_summary(cls: type) -> str:
    """
    从类 docstring 中提取一行简介

    取 Docstring 的第一个非空行，去除首尾空白。
    如果没有 docstring，返回 "(无描述)"。

    Args:
        cls (type): 目标类

    Returns:
        str: 一行简介文本
    """
    doc = cls.__doc__
    if not doc:
        return "(无描述)"

    for line in doc.strip().splitlines():
        stripped = line.strip()
        if stripped:
            return stripped

    return "(无描述)"


def _extract_description(cls: type) -> str:
    """
    从类 docstring 中提取完整功能描述

    提取 docstring 从开头到第一个空行或者 section 标记之前的所有文本行。
    已包含 Input: 和 Output: section 标记以避免误提取。

    Args:
        cls (type): 目标类

    Returns:
        str: 多行功能描述文本
    """
    doc = cls.__doc__
    if not doc:
        return ""

    lines = doc.strip().splitlines()
    desc_lines = []

    # 段落 section 标记（含新增的 Input/Output）
    section_markers = {
        'Attributes:', 'Args:', 'Returns:', 'Raises:', 'Note:', 'Notes:',
        'Input:', 'Output:',
        '示例', '泛型参数:', '校验规则:', '数据流:', '核心', '训练阶段:',
        '推理阶段:', '输出:', '注意:',
    }

    for line in lines:
        stripped = line.strip()

        # 遇到 section 标记停止
        if any(stripped.startswith(marker) for marker in section_markers):
            break

        # 遇到空行时，如果已有内容则停止（只取第一段）
        if not stripped and desc_lines:
            break

        if stripped:
            desc_lines.append(stripped)

    return ' '.join(desc_lines)


def _extract_docstring_section(doc: str | None, section_name: str, *, multiline: bool = False) -> str:
    """
    从 docstring 中提取指定 section 的内容

    解析 Google 风格的 docstring，查找 ``{section_name}:`` 开头的 section，
    提取其下缩进的文本内容（到下一个 section 或空行前截止）。

    Args:
        doc (str | None): 原始 docstring 文本
        section_name (str): section 名称（不含冒号），如 ``"Input"`` 或 ``"Output"``
        multiline (bool): 是否保留原始换行结构。默认 ``False`` 用空格连接所有行
            （向后兼容）；``True`` 时用 ``\\n`` 连接，保留多行结构。
            多变量场景（如 Input 段含 ``x_real: ...`` 和 ``x_pred: ...`` 两行）
            需传 ``True``，否则两行被合并为一行导致变量解析失败。

    Returns:
        str: section 内容文本，未找到时返回空字符串
    """
    if not doc:
        return ""

    lines = doc.strip().splitlines()
    in_section = False
    content_lines: list[str] = []

    for line in lines:
        stripped = line.strip()

        # 检测目标 section 开始
        if stripped.startswith(f"{section_name}:"):
            in_section = True
            # 提取同行冒号后面的内容（如果有）
            after_colon = stripped[len(f"{section_name}:"):].strip()
            if after_colon:
                content_lines.append(after_colon)
            continue

        if in_section:
            # 遇到新的 section 标记则停止
            if stripped and not stripped.startswith(' ') and stripped.endswith(':'):
                break
            # 检测其他常见 section 标记（防止误读取）
            if any(stripped.startswith(m) for m in
                   ('Args:', 'Returns:', 'Raises:', 'Attributes:',
                    'Note:', 'Notes:', 'Input:', 'Output:')):
                break
            # 收集内容行
            if stripped:
                content_lines.append(stripped)
            elif content_lines:
                # 空行表示 section 结束
                break

    # 多行模式下用 \n 连接，保留原始换行结构（用于 Input/Output 段多变量解析）
    # 单行模式下用空格连接（保持向后兼容，用于 generate_detail 的其他段提取）
    if multiline:
        return '\n'.join(content_lines)
    return ' '.join(content_lines)


# ============================================================================
# 内部辅助函数 — 类型对象展开与推断
# ============================================================================

def _format_type_full(annotation) -> str:
    """将类型对象展开为全限定名字符串。

    支持递归展开的类型展开规则：

    - ``Annotated[X, ...]`` → 递归展开 ``X``
    - ``Union[A, B]`` / ``A | B``（Python 3.10+）→ 各部分分别展开后用 ``" | "`` 连接
    - ``pd.DataFrame`` → ``"pandas.DataFrame"``
    - ``np.ndarray``（含带参数形式） → ``"numpy.ndarray"``
    - ``tuple[T1, T2]`` → ``"tuple[T1_full, T2_full]"``
    - ``list[T]`` → ``"list[T_full]"``
    - ``dict[K, V]`` → ``"dict[K_full, V_full]"``
    - 基础标量类型（``int`` / ``float`` / ``str`` / ``bool`` / ``bytes``）→ 类型名原样
    - ``BaseModel`` 子类 → 类名
    - ``None`` / ``NoneType`` → ``"None"``
    - 其他类型 → ``__name__`` 或 ``str(annotation)``

    本函数是纯类型对象分析，不依赖任何算子基类判断，可对任意 ``typing`` 对象使用。

    Args:
        annotation (Any): 类型注解对象，可以是 ``type``、``typing.Union``、
            ``typing.Annotated``、Generic alias（如 ``list[int]``）、
            ``types.UnionType``（Python 3.10+）等，也可以是 ``None``。

    Returns:
        str: 全限定名格式的类型字符串，永远是非空字符串。
    """
    # 处理 None / NoneType
    if annotation is None or annotation is type(None):
        return "None"

    # 处理 Annotated（如 NumericData、ArrayN）
    # Annotated 对象有 __metadata__ 属性，__origin__ 指向实际类型
    if hasattr(annotation, '__metadata__'):
        return _format_type_full(annotation.__origin__)

    origin = get_origin(annotation)

    # 处理 Union（typing.Union 或 Python 3.10+ 的 types.UnionType）
    is_union = (
            origin is typing.Union
            or (hasattr(types, 'UnionType') and isinstance(annotation, types.UnionType))
    )
    if is_union:
        parts = [_format_type_full(arg) for arg in get_args(annotation)]
        return " | ".join(parts)

    # 处理带参数的具体类型（origin 非 None）
    if origin is not None:
        # ndarray 带参数（如 np.ndarray[Any, dtype]）→ numpy.ndarray
        if origin is np.ndarray or (isinstance(origin, type) and issubclass(origin, np.ndarray)):
            return "numpy.ndarray"
        # tuple[T1, T2, ...]
        if origin is tuple:
            parts = [_format_type_full(arg) for arg in get_args(annotation)]
            return f"tuple[{', '.join(parts)}]"
        # list[T]
        if origin is list:
            parts = [_format_type_full(arg) for arg in get_args(annotation)]
            return f"list[{', '.join(parts)}]"
        # dict[K, V]
        if origin is dict:
            parts = [_format_type_full(arg) for arg in get_args(annotation)]
            return f"dict[{', '.join(parts)}]"
        # 其他带参数类型，用 origin 的类名
        if hasattr(origin, '__name__'):
            return origin.__name__
        return str(origin)

    # 处理不带参数的具体类型
    if annotation is pd.DataFrame:
        return "pandas.DataFrame"
    if isinstance(annotation, type) and issubclass(annotation, np.ndarray):
        return "numpy.ndarray"
    if annotation is int:
        return "int"
    if annotation is float:
        return "float"
    if annotation is str:
        return "str"
    if annotation is bool:
        return "bool"
    if annotation is bytes:
        return "bytes"

    # BaseModel 子类 → 类名
    if isinstance(annotation, type) and issubclass(annotation, BaseModel):
        return annotation.__name__

    # 其他类型，用 __name__ 或 str()
    if hasattr(annotation, '__name__'):
        return annotation.__name__
    return str(annotation)


def _is_basemodel_subtype(annotation) -> bool:
    """判断注解是否是 ``BaseModel`` 子类。

    用于 CLI Help 渲染时决定是否需要追加 ``**结构**：`` 字段表
    （仅 BaseModel 子类有 ``model_fields`` 可供解析）。

    Args:
        annotation (Any): 任意类型注解对象。可以是 ``type``、联合类型、
            标量、``None`` 等。

    Returns:
        bool: 当 ``annotation`` 是 ``BaseModel`` 子类（含 BaseModel 本身）时
        返回 ``True``；其他情况（联合类型、标量、``None``、非类对象等）返回 ``False``。
    """
    return isinstance(annotation, type) and issubclass(annotation, BaseModel)


def _simplify_output_type(output_type):
    """从 ``T | tuple[T, EO]`` 形式的联合类型中提取主输出 ``T``。

    ``NumericOperator`` 等的输出类型形如 ``NumericData | tuple[NumericData, EO]``，
    表示"主输出 NumericData 或 (主输出, 附加输出 EO)"两种形态。本函数从这种联合类型中
    提取主输出类型，去除 ``tuple`` 分支的噪音。

    本函数为纯类型对象分析，不依赖任何基类判断
    （不检查 ``issubclass(cls, NumericOperator)``），完全基于类型对象的结构特征。

    规则：

    - 如果 ``output_type`` 是 ``Union`` 且 args 中含 ``tuple`` 形式，
      且非 tuple 部分只有一个 → 返回非 tuple 部分（视为主输出）
    - 其他情况（非 Union、Union 中无 tuple、多个非 tuple arg） → 原样返回

    Args:
        output_type (Any): 输出类型对象。可以是 ``Union[T, tuple[T, EO]]``、
            ``T | tuple[T, EO]``（Python 3.10+）、单一类型、``None`` 等。

    Returns:
        Any: 简化后的主输出类型对象。无法简化时（不符合 ``T | tuple[T, EO]``
        模式）原样返回输入；``None`` 输入返回 ``None``。
    """
    if output_type is None:
        return None

    origin = get_origin(output_type)
    is_union = (
            origin is typing.Union
            or (hasattr(types, 'UnionType') and isinstance(output_type, types.UnionType))
    )

    # 非 Union，原样返回
    if not is_union:
        return output_type

    args = get_args(output_type)
    non_tuple_args = [a for a in args if get_origin(a) is not tuple]
    tuple_args = [a for a in args if get_origin(a) is tuple]

    # 仅当"含 tuple 且非 tuple 部分唯一"时简化
    if tuple_args and len(non_tuple_args) == 1:
        return non_tuple_args[0]

    return output_type


def _split_input_types(input_type) -> list[str] | None:
    """将输入类型对象拆解为每个位置的类型字符串列表。

    用于 Input 段渲染时按位置匹配 docstring 中声明的多个变量名与对应的类型：

    - ``tuple[T1, T2]`` → ``[T1_full, T2_full]``（多变量场景，逐元素展开）
    - 其他类型（含 Union 联合类型如 ``NumericData``） → ``[T_full]``（单变量场景）
    - ``None`` → ``None``（无类型信息）

    Args:
        input_type (Any): 输入类型对象。典型形态包括 ``tuple[T1, T2, ...]``、
            单一类型、联合类型、``None`` 等。

    Returns:
        list[str] | None: 各位置类型的全限定名字符串列表。
            ``tuple`` 类型返回与元素数对应的列表；单一类型返回单元素列表；
            ``None`` 输入返回 ``None``。
    """
    if input_type is None:
        return None
    # tuple 类型按元素拆解（典型场景：evaluation 的 tuple[ndarray, ndarray]）
    if get_origin(input_type) is tuple:
        return [_format_type_full(arg) for arg in get_args(input_type)]
    # 其他类型作为单变量处理（不展开 Union，保留 "T1 | T2" 完整形式）
    return [_format_type_full(input_type)]


# ============================================================================
# 内部辅助函数 — docstring 变量解析
# ============================================================================

# 变量行匹配模式：变量名 (可选类型): 描述
# - 变量名：字母/数字/下划线
# - 可选类型：(...) 形式，开发者写时忽略
# - 分隔符：英文冒号或中文冒号
# - 描述：任意文本
_VARIABLE_LINE_PATTERN = re.compile(
    r'^(\w+)\s*(?:\([^)]*\))?\s*[:：]\s*(.*)$'
)


def _parse_variables(desc_text: str) -> list[tuple[str, str]]:
    """从 docstring ``Input``/``Output`` 段文本中解析变量列表。

    按行解析，每行可能是：

    - **变量行**：匹配 ``变量名 (可选类型): 描述`` 模式，返回 ``(变量名, 描述)``。
      开发者写在括号中的类型部分会被忽略（CLI Help 使用泛型自动推断出的类型，
      不采纳手写类型，保证一致性）。
    - **纯描述行**：不匹配上述模式，返回 ``("", 整行内容)``，变量名留空。

    空行被跳过，不参与解析。

    示例匹配：

    - ``"x: 特征矩阵"`` → ``("x", "特征矩阵")``
    - ``"x (DataFrame | ndarray): 特征矩阵"`` → ``("x", "特征矩阵")``
      （类型部分被忽略）
    - ``"x：特征矩阵"``（中文冒号）→ ``("x", "特征矩阵")``
    - ``"二维时序数据"`` → ``("", "二维时序数据")``

    Args:
        desc_text (str): docstring ``Input``/``Output`` 段的文本内容。
            可以是空字符串或 ``None``（视为空）。

    Returns:
        list[tuple[str, str]]: 变量列表，每个元素为 ``(变量名, 描述)`` 元组。
            变量名为空字符串表示该行是纯描述行（不匹配变量模式）。
            空输入返回空列表。
    """
    if not desc_text:
        return []
    variables: list[tuple[str, str]] = []
    for line in desc_text.split('\n'):
        stripped = line.strip()
        if not stripped:
            # 跳过空行
            continue
        match = _VARIABLE_LINE_PATTERN.match(stripped)
        if match:
            # 变量行：提取变量名和描述（忽略括号中开发者写的类型）
            name = match.group(1)
            description = match.group(2).strip()
            variables.append((name, description))
        else:
            # 纯描述行，变量名留空
            variables.append(('', stripped))
    return variables


# ============================================================================
# 内部辅助函数 — 格式化和表格
# ============================================================================

def _display_width(s: str) -> int:
    """
    计算字符串的终端显示宽度（CJK-aware）

    东亚全角字符（Wide/Fullwidth）计为 2，其余字符计为 1。

    Args:
        s (str): 目标字符串

    Returns:
        int: 显示宽度
    """
    width = 0
    for ch in s:
        eaw = unicodedata.east_asian_width(ch)
        if eaw in ('W', 'F'):
            width += 2
        else:
            width += 1
    return width


def _pad_cell(s: str, target_width: int) -> str:
    """
    用空格将字符串填充到目标显示宽度

    Args:
        s (str): 原始字符串
        target_width (int): 目标显示宽度

    Returns:
        str: 填充后的字符串
    """
    current = _display_width(s)
    if current >= target_width:
        return s
    return s + ' ' * (target_width - current)


def _build_aligned_table(headers: list[str], rows: list[list[str]]) -> str:
    """
    生成 CJK-aware 列对齐的 Markdown 表格

    计算每列的最大显示宽度（考虑东亚全角字符），用空格填充对齐。

    Args:
        headers (list[str]): 表头列表
        rows (list[list[str]]): 数据行列表，每行为单元格列表

    Returns:
        str: 对齐后的 Markdown 表格文本
    """
    if not headers:
        return ""

    # 计算每列的最大显示宽度
    col_widths = [_display_width(h) for h in headers]
    for row in rows:
        for i, cell in enumerate(row):
            if i < len(col_widths):
                col_widths[i] = max(col_widths[i], _display_width(cell))

    # 构建对齐后的行
    def _format_row(cells: list[str]) -> str:
        padded = [_pad_cell(cells[i], col_widths[i]) for i in range(len(headers))]
        return "| " + " | ".join(padded) + " |"

    lines = []
    # 表头行
    lines.append(_format_row(headers))
    # 分隔行
    separators = ['-' * w for w in col_widths]
    lines.append("| " + " | ".join(separators) + " |")
    # 数据行
    for row in rows:
        lines.append(_format_row(row))

    return '\n'.join(lines)


def _format_type(annotation) -> str:
    """
    格式化类型注解为人类可读字符串

    对 Enum 展示成员值列表，对 Literal 展示字面量。

    Args:
        annotation: 类型注解对象

    Returns:
        str: 格式化后的类型字符串
    """
    if annotation is None:
        return "Any"

    # Enum 类型
    if inspect.isclass(annotation) and issubclass(annotation, enum.Enum):
        values = [str(e.value) for e in annotation]
        return f"enum({', '.join(values)})"

    # Literal 类型
    if get_origin(annotation) is Literal:
        args = get_args(annotation)
        values = [str(a) for a in args]
        return f"literal({', '.join(values)})"

    # 基本类型
    if annotation is int:
        return "int"
    elif annotation is float:
        return "float"
    elif annotation is str:
        return "str"
    elif annotation is bool:
        return "bool"

    # 其他复杂类型（list, dict, Optional 等）
    origin = get_origin(annotation)
    if origin is not None:
        args = get_args(annotation)
        if args:
            args_str = ', '.join(_format_type(a) for a in args)
            origin_name = getattr(origin, '__name__', str(origin))
            return f"{origin_name}[{args_str}]"
        return str(origin)

    # 直接使用类名
    if hasattr(annotation, '__name__'):
        return annotation.__name__

    return str(annotation)


def _extract_constraints(field_info) -> str:
    """
    从 field_info 中提取值域和候选值约束

    检查 metadata 中的 Ge/Le/Gt/Lt 以及 annotation 中的 Enum/Literal。

    Args:
        field_info: Pydantic FieldInfo 对象

    Returns:
        str: 约束描述字符串，如 ``[1, 20]`` 或 ``euclidean, manhattan``
    """
    parts = []

    # ---- 从 metadata 提取数值边界 ----
    low = None
    high = None
    low_inclusive = True
    high_inclusive = True

    for meta in field_info.metadata:
        if isinstance(meta, Ge):
            low = meta.ge
            low_inclusive = True
        elif isinstance(meta, Gt):
            low = meta.gt
            low_inclusive = False
        elif isinstance(meta, Le):
            high = meta.le
            high_inclusive = True
        elif isinstance(meta, Lt):
            high = meta.lt
            high_inclusive = False

    if low is not None or high is not None:
        # 无穷侧必须用开区间：low 为 None（负无穷）时左括号为 "("，
        # high 为 None（正无穷）时右括号为 ")"
        low_bracket = "[" if (low_inclusive and low is not None) else "("
        high_bracket = "]" if (high_inclusive and high is not None) else ")"
        low_str = str(low) if low is not None else "-∞"
        high_str = str(high) if high is not None else "+∞"
        parts.append(f"{low_bracket}{low_str}, {high_str}{high_bracket}")

    # ---- 从 annotation 提取候选值 ----
    annotation = field_info.annotation
    if annotation is not None:
        if inspect.isclass(annotation) and issubclass(annotation, enum.Enum):
            values = [str(e.value) for e in annotation]
            parts.append(', '.join(values))
        elif get_origin(annotation) is Literal:
            values = [str(a) for a in get_args(annotation)]
            parts.append(', '.join(values))

    return ' | '.join(parts) if parts else "-"


def _extract_model_fields_description(model_cls: type[BaseModel]) -> str:
    """
    从 Pydantic BaseModel 的字段中生成简要描述表格

    用于展示附加输出（ExtraOutput）等结构的字段说明。
    说明列从 ``Field(description=...)`` 中提取。

    Args:
        model_cls (type[BaseModel]): Pydantic BaseModel 子类

    Returns:
        str: Markdown 格式的字段描述表格（CJK-aware 列对齐）
    """
    headers = ["字段名", "类型", "说明"]
    rows = []

    for field_name, field_info in model_cls.model_fields.items():
        type_str = _format_type(field_info.annotation)
        desc = field_info.description or ""
        rows.append([field_name, type_str, desc])

    return _build_aligned_table(headers, rows)


# ============================================================================
# 内部辅助函数 — Input/Output 段渲染
# ============================================================================

def _render_input_section(cls: type) -> list[str]:
    """渲染详情模式的"### 输入"段。

    整体流程：

    1. 从 ``cls._input_type`` 提取输入类型对象
    2. 用 ``_split_input_types`` 拆解为类型字符串列表（``tuple`` 拆解为多元素）
    3. 从 docstring ``Input:`` 段解析变量名与描述列表（忽略开发者写的类型部分）
    4. 按变量数与类型数匹配渲染：

       - 变量数 == 类型数：每行 ``"变量名 (类型): 描述"``（标准形式）
       - 变量数 ≠ 类型数：降级为单变量 ``"x (完整类型): 描述"``，并 ``logger.warning``
       - 无变量：显示 ``"(类型): 描述"`` 或 ``"(类型)"`` 或 docstring 纯文本或 ``"（无）"``

    5. 主输入是 ``BaseModel`` 子类时，追加 ``"**结构**：" + 字段表``
       （罕见场景，对称于主输出字段表）

    Args:
        cls (type): 算子类。期望具有 ``_input_type`` 类属性（由 ``__init_subclass__``
            自动填充）和可选的 docstring ``Input:`` 段。

    Returns:
        list[str]: 渲染后的 Markdown 文本行列表，含 ``"### 输入"`` 标题与结尾空行。
            空内容时降级为 ``"（无）"`` 行以保持向后兼容。
    """
    lines: list[str] = []
    lines.append("### 输入")
    lines.append("")

    # 步骤 1-2: 提取输入类型并拆解为各位置的类型字符串列表
    input_type = getattr(cls, '_input_type', None)
    type_list = _split_input_types(input_type)  # None 或 list[str]

    # 步骤 3: 解析 docstring Input 段中的变量名与描述
    # multiline=True 保留原始换行，支持多变量场景（如 x_real/x_pred）
    input_desc = _extract_docstring_section(cls.__doc__, "Input", multiline=True)
    variables = _parse_variables(input_desc)

    # 步骤 4: 按 (类型数, 变量数) 的不同组合渲染
    if type_list is None:
        # 分支 A: 无类型信息（罕见，说明泛型提取失败）
        if not variables:
            # 既无类型也无变量，原样显示 docstring 或"（无）"占位
            lines.append(input_desc if input_desc else "（无）")
        else:
            # 仅有变量名 + 描述（无类型），按 "变量名: 描述" 格式渲染
            for name, desc in variables:
                if name:
                    lines.append(f"{name}: {desc}" if desc else name)
                else:
                    lines.append(desc)
    else:
        # 分支 B: 有类型信息
        full_type_str = _format_type_full(input_type)
        if not variables:
            # 无变量场景（docstring 缺失 Input 段或仅含空行）
            if len(type_list) == 1:
                # 单一类型 → "(类型): 描述" 或 "(类型)"
                if input_desc:
                    lines.append(f"({type_list[0]}): {input_desc}")
                else:
                    lines.append(f"({type_list[0]})")
            else:
                # 多类型但无变量名（如 tuple 输入且 docstring 缺失），
                # 退而求其次显示完整 tuple 类型
                if input_desc:
                    lines.append(f"({full_type_str}): {input_desc}")
                else:
                    lines.append(f"({full_type_str})")
        elif len(variables) == len(type_list):
            # 变量数与类型数匹配 → 逐行渲染 "变量名 (类型): 描述"
            for (name, desc), type_str in zip(variables, type_list):
                if name:
                    lines.append(f"{name} ({type_str}): {desc}" if desc else f"{name} ({type_str})")
                else:
                    # 变量名为空（纯描述行），用 "(类型): 描述" 格式
                    lines.append(f"({type_str}): {desc}" if desc else f"({type_str})")
        else:
            # 变量数与类型数不匹配 → 降级为单变量 + warning
            logger.warning(
                f"{cls.__name__} 的 docstring Input 段变量数 ({len(variables)}) "
                f"与输入类型元素数 ({len(type_list)}) 不匹配，降级为单变量显示"
            )
            if len(variables) == 1:
                # 单变量场景：用完整类型字符串包裹
                name, desc = variables[0]
                if name:
                    lines.append(f"{name} ({full_type_str}): {desc}" if desc else f"{name} ({full_type_str})")
                else:
                    lines.append(f"({full_type_str}): {desc}" if desc else f"({full_type_str})")
            else:
                # 多变量 + 类型数不匹配，无法逐行匹配类型，仅在首行显示完整类型
                lines.append(f"({full_type_str})")
                for name, desc in variables:
                    if name:
                        lines.append(f"{name}: {desc}" if desc else name)
                    else:
                        lines.append(desc)

    lines.append("")

    # 步骤 5: 主输入是 BaseModel 子类时，追加字段表（罕见场景）
    if _is_basemodel_subtype(input_type):
        lines.append("**结构**：")
        lines.append("")
        struct_table = _extract_model_fields_description(input_type)
        if struct_table:
            lines.append(struct_table)
            lines.append("")

    return lines


def _render_output_section(cls: type) -> list[str]:
    """渲染详情模式的"### 主输出"段。

    整体流程：

    1. 从 ``cls._output_type`` 提取输出类型对象
    2. 用 ``_simplify_output_type`` 简化联合类型 ``T | tuple[T, EO]`` 为 ``T``
    3. 标题渲染：

       - 简化后类型非 ``None``：``"### 主输出 (类型字符串)"``
       - 类型为 ``None``：``"### 主输出"``

    4. 内容渲染（合并显示策略）：

       - docstring ``Output`` 段非空：先展示描述（语义/形状说明）
       - 简化后类型是 ``BaseModel`` 子类：追加 ``"**结构**："`` + 字段表
       - 两者皆空：显示 ``"（无）"``（向后兼容）

    Args:
        cls (type): 算子类。期望具有 ``_output_type`` 类属性（由 ``__init_subclass__``
            自动填充）和可选的 docstring ``Output:`` 段。

    Returns:
        list[str]: 渲染后的 Markdown 文本行列表，含标题与结尾空行。
            空内容时降级为 ``"（无）"`` 行以保持向后兼容。
    """
    lines: list[str] = []

    # 步骤 1-2: 提取并简化输出类型（去除 T | tuple[T, EO] 中的 tuple 分支噪音）
    output_type_raw = getattr(cls, '_output_type', None)
    output_type = _simplify_output_type(output_type_raw)

    # 步骤 3: 确定标题（类型存在时附在标题括号中）
    if output_type is not None:
        type_str = _format_type_full(output_type)
        lines.append(f"### 主输出 ({type_str})")
    else:
        lines.append("### 主输出")
    lines.append("")

    # 步骤 4: 合并显示 docstring 描述 + BaseModel 字段表
    # multiline=True 保留原始换行（Output 段较少多变量，但保持与 Input 一致）
    output_desc = _extract_docstring_section(cls.__doc__, "Output", multiline=True)
    has_struct_table = _is_basemodel_subtype(output_type)

    if output_desc:
        # docstring 提供的语义/形状说明（适用于所有算子）
        lines.append(output_desc)
        lines.append("")
    if has_struct_table:
        # BaseModel 输出提供的结构化字段表（仅 evaluation 算子的 MR 等场景）
        lines.append("**结构**：")
        lines.append("")
        struct_table = _extract_model_fields_description(output_type)
        if struct_table:
            lines.append(struct_table)
            lines.append("")
    if not output_desc and not has_struct_table:
        # 都无内容，向后兼容显示"（无）"
        lines.append("（无）")
        lines.append("")

    return lines
