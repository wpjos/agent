# -*- coding: utf-8 -*-

"""
HPO 搜索空间声明与提取模块

基于 Pydantic 原生约束机制，提供零侵入的超参数搜索空间声明和提取能力。

核心设计:
    绝大多数场景下，搜索范围由 Pydantic 原生 Field(ge/le/gt/lt)
    和 Enum/Literal 类型注解直接表达，无需额外标注。

    仅在需要 log 尺度采样、非1步长等 Pydantic 原生无法表达的
    分布参数时，通过 ``Annotated`` 注入 ``SearchHint`` 标记，
    该标记存储在 ``field_info.metadata`` 中（Pydantic 官方扩展通道）。

提取来源:
    1. ``field_info.metadata`` 中 ``annotated_types.Ge/Le/Gt/Lt`` → low/high
    2. ``field_info.annotation`` 中 Enum 子类 → choices（取各成员 .value）
    3. ``field_info.annotation`` 中 Literal → choices（取所有字面量）
    4. ``field_info.metadata`` 中 SearchHint → log/step

用法示例::

    from typing import Annotated
    from pydantic import BaseModel, Field
    from tsas.engine.hpo import SearchHint, extract_search_space

    # 95% 场景 — 零标注，原生 Field 即可
    class MyConfig(BaseModel):
        n_neighbors: int = Field(default=5, ge=1, le=20)
        percentile: float = Field(default=95.0, ge=50.0, le=99.9)
        metric: Literal["euclidean", "manhattan"] = "euclidean"

    # 5% 场景 — 需要 log/step 时通过 Annotated 注入 SearchHint
    class AdvancedConfig(BaseModel):
        learning_rate: Annotated[float, Field(default=0.001, ge=1e-5, le=1e-1),
                                 SearchHint(log=True)]

    # 提取
    space = extract_search_space(MyConfig)
    # → {"n_neighbors": {"type":"int","low":1,"high":20},
    #     "percentile": {"type":"float","low":50.0,"high":99.9},
    #     "metric": {"type":"cat","choices":["euclidean","manhattan"]}}
"""

from __future__ import annotations

import enum
import inspect
from dataclasses import dataclass
from typing import Any, Literal, get_args, get_origin

from annotated_types import Ge, Gt, Le, Lt
from pydantic import BaseModel
from pydantic.fields import PydanticUndefined

from tsas.engine.operator.base import BaseOperator
from tsas.engine.operator.detection.base import (
    BaseDetector,
    BasePredictorMixin,
    BaseDeciderMixin,
)

__all__ = [
    'SearchHint',
    'extract_search_space',
    'extract_search_space_from_operator',
    'config_to_optuna_suggestions',
]


# ============================================================================
# SearchHint — 最小化 HPO 分布标记
# ============================================================================

@dataclass(frozen=True)
class SearchHint:
    """HPO 分布参数标记（仅用于 Pydantic Field 原生不支持的语义）。

    通过 ``Annotated`` 注入到 ``field_info.metadata`` 中，
    绝大多数场景不需要此标记——搜索范围直接由 Field(ge/le) 或
    Enum/Literal 类型注解表达。

    Attributes:
        log (bool): 是否使用对数尺度采样。
            - ``True``: Optuna ``suggest_float(..., log=True)``
            - ``False`` (默认): 均匀采样
        step (int | None): 整数型超参搜索步长。
            - 正整数: Optuna ``suggest_int(..., step=step)``
            - ``None`` (默认): 默认步长为 1
    """
    log: bool = False
    """是否使用对数尺度采样，对应 Optuna suggest_float(..., log=True)"""
    step: int | None = None
    """整数型超参搜索步长，对应 Optuna suggest_int(..., step=step)，None 时默认步长为 1"""


# ============================================================================
# 搜索空间提取 — 从 Pydantic Config 类
# ============================================================================

def extract_search_space(config_cls: type[BaseModel]) -> dict[str, dict]:
    """从 Pydantic Config 类的原生约束中提取搜索空间。

    遍历 Config 类的所有字段，从以下来源提取搜索空间信息：
        1. ``field_info.metadata`` 中的 ``annotated_types`` 约束 → low/high
        2. ``field_info.annotation`` 中的 Enum/Literal → choices
        3. ``field_info.metadata`` 中的 SearchHint → log/step

    自动推断类型：
        - annotation 为 int → ``"int"``，映射为 ``suggest_int``
        - annotation 为 float → ``"float"``，映射为 ``suggest_float``
        - 有 choices → ``"cat"``，映射为 ``suggest_categorical``

    Args:
        config_cls (type[BaseModel]): Pydantic Config 类

    Returns:
        dict[str, dict]: 字段名到搜索空间信息的映射。
            每个搜索空间字典包含以下可选键：
            - ``"type"`` (str): ``"int"`` / ``"float"`` / ``"cat"``
            - ``"low"`` (int | float): 数值型下界
            - ``"high"`` (int | float): 数值型上界
            - ``"choices"`` (list): 离散型候选值列表
            - ``"log"`` (bool): 是否对数尺度采样（仅 float 类型）
            - ``"step"`` (int): 步长（仅 int 类型，None 时默认 1）
            - ``"default"`` (Any): 默认值

    Raises:
        TypeError: config_cls 不是 BaseModel 的子类时

    示例::

        class DemoConfig(BaseModel):
            n: int = Field(default=5, ge=1, le=20)
            p: float = Field(default=95.0, ge=50.0, le=99.9)

        extract_search_space(DemoConfig)
        # {"n": {"type":"int","low":1,"high":20,"default":5},
        #  "p": {"type":"float","low":50.0,"high":99.9,"default":95.0}}
    """
    if not (inspect.isclass(config_cls) and issubclass(config_cls, BaseModel)):
        raise TypeError(f"config_cls 必须是 BaseModel 的子类，当前类型为 {type(config_cls)}")

    result: dict[str, dict] = {}

    for field_name, field_info in config_cls.model_fields.items():
        space: dict[str, Any] = {}

        # ---- 步骤1: 从 metadata 中提取数值边界 ----
        # Field(ge=1, le=20) → _collect_metadata → annotated_types.Ge(1) / Le(20)
        for meta in field_info.metadata:
            if isinstance(meta, (Ge, Gt)):
                # Ge.ge / Gt.gt: 获取下边界值
                space["low"] = meta.ge if isinstance(meta, Ge) else meta.gt
            elif isinstance(meta, (Le, Lt)):
                # Le.le / Lt.lt: 获取上边界值
                space["high"] = meta.le if isinstance(meta, Le) else meta.lt

        # ---- 步骤2: 从 annotation 中提取候选值 ----
        # Enum 子类 → 取各成员 .value
        # Literal[...] → 取 get_args
        annotation = field_info.annotation
        if annotation is not None:
            if inspect.isclass(annotation) and issubclass(annotation, enum.Enum):
                # 枚举类型: 提取所有成员的 .value 作为候选值
                space["choices"] = [e.value for e in annotation]
            elif get_origin(annotation) is Literal:
                # Literal 类型: 提取所有字面量
                space["choices"] = list(get_args(annotation))

        # ---- 步骤3: 从 metadata 中提取 SearchHint ----
        for meta in field_info.metadata:
            if isinstance(meta, SearchHint):
                if meta.log:
                    space["log"] = True
                if meta.step is not None:
                    space["step"] = meta.step
                break  # 一个字段只取一个 SearchHint

        # ---- 步骤4: 推断类型 ----
        if "choices" in space:
            # 有候选值为分类搜索
            space["type"] = "cat"
        elif annotation is int:
            # 仅在有边界约束时才视为可搜索字段
            if "low" not in space and "high" not in space:
                continue
            space["type"] = "int"
            # 默认为1步长
            if "step" not in space:
                space["step"] = 1
        elif annotation is float:
            # 仅在有边界约束时才视为可搜索字段
            if "low" not in space and "high" not in space:
                continue
            space["type"] = "float"
        else:
            # 无法推断类型，跳过
            continue

        # ---- 步骤5: 补入默认值 ----
        # 如果字段有默认值（非必需字段），记录之
        # 排除 PydanticUndefined（无默认值）和 Ellipsis（…）
        default = field_info.default
        if default is not None and default is not PydanticUndefined and default != ...:  # type: ignore[operator]
            space["default"] = default

        if space:
            result[field_name] = space

    return result


# ============================================================================
# 搜索空间提取 — 从算子实例（支持 Composite 递归）
# ============================================================================

def extract_search_space_from_operator(
    operator: BaseOperator, *, prefix: str = ""
) -> dict[str, dict]:
    """从算子实例递归提取搜索空间（支持 Composite 子算子嵌套）。

    普通算子（如 KNNDetector）: 直接提取自身 Config 的搜索空间。
    Composite 算子: 递归遍历 ``operators`` 列表，为每个子算子
    的 Config 字段添加层级前缀。

    前缀命名规则:
        - Predictor (第0位): ``predictor.``
        - Scorer (第i位): ``scorer_{i}.``
        - Decider (最后位): ``decider.``

    这保证了 Composite 内部各组件的可搜索参数不会冲突。

    Args:
        operator (BaseOperator): 算子实例。
            - 普通 Detector/Scorer: 直接提取 Config
            - CompositeDetector/CompositeScorer: 递归遍历子算子
        prefix (str): 当前递归层级的前缀，内部使用

    Returns:
        dict[str, dict]: 字段名（含前缀）到搜索空间信息的映射。
            格式同 ``extract_search_space``。

    示例::

        comp = CompositeDetector(operators=[
            PCAPredictor(config=PCAPredictorConfig(n_components=Field(default=5, ge=1, le=50))),
            ResidualScorer(config=ResidualScorerConfig(metric=Literal["mse","mae"])),
            PercentileDecider(config=PercentileDeciderConfig(percentile=Field(default=95.0, ge=50.0, le=99.9))),
        ])
        extract_search_space_from_operator(comp)
        # {"predictor.n_components": {"type":"int","low":1,"high":50},
        #  "scorer_0.metric":       {"type":"cat","choices":["mse","mae"]},
        #  "decider.percentile":    {"type":"float","low":50.0,"high":99.9}}
    """
    result: dict[str, dict] = {}

    # ---- 普通算子: 从 Config 类提取 ----
    config_type = operator._config_type
    if config_type is not None:
        raw = extract_search_space(config_type)
        for field_name, space in raw.items():
            full_key = f"{prefix}{field_name}" if prefix else field_name
            result[full_key] = space

    # ---- Composite 算子: 递归遍历子算子列表 ----
    if hasattr(operator, '_operators'):
        # CompositeDetector/CompositeScorer 的 operators 列表
        sub_operators = operator._operators

        for i, sub_op in enumerate(sub_operators):
            # 根据子算子类型确定前缀
            if isinstance(sub_op, BasePredictorMixin):
                sub_prefix = "predictor."
            elif isinstance(sub_op, BaseDeciderMixin):
                # Decider 在 Composite 中永远是最后一位
                sub_prefix = "decider."
            else:
                # Scorer: 按序号命名
                # 需要计算这是第几个 Scorer
                scorer_idx = sum(
                    1 for o in sub_operators[:i]
                    if not isinstance(o, (BasePredictorMixin, BaseDeciderMixin))
                )
                sub_prefix = f"scorer_{scorer_idx}."

            # 递归提取（子算子本身也可能是 Composite）
            sub_result = extract_search_space_from_operator(
                sub_op, prefix=prefix + sub_prefix
            )
            result.update(sub_result)

    return result


# ============================================================================
# 搜索空间 → Optuna 采样建议映射
# ============================================================================

def config_to_optuna_suggestions(
    trial,
    search_space: dict[str, dict],
    *,
    params: list[str] | None = None,
) -> dict[str, Any]:
    """将搜索空间映射为 Optuna trial.suggest_* 调用。

    根据搜索空间字典中的 ``type`` 字段选择合适的采样方法：
        - ``"int"``: ``trial.suggest_int(name, low, high, step=...)``
        - ``"float"``: ``trial.suggest_float(name, low, high, log=...)``
        - ``"cat"``: ``trial.suggest_categorical(name, choices)``

    Args:
        trial (optuna.Trial): Optuna Trial 对象
        search_space (dict[str, dict]): 搜索空间字典，
            格式同 ``extract_search_space`` 返回值
        params (list[str] | None): 需要采样的参数名列表。
            ``None`` (默认) 时采样搜索空间中所有参数

    Returns:
        dict[str, Any]: 参数名到采样值的映射。
            可直接用于传入算子构造函数的 **kwargs

    Raises:
        ValueError: 搜索空间中类型未知或缺少必要参数时

    示例::

        import optuna
        study = optuna.create_study(direction="maximize")
        def objective(trial):
            params = config_to_optuna_suggestions(trial, search_space)
            detector = KNNDetector(**params)
            ...
    """
    import optuna  # 延迟导入，避免非HPO场景下的依赖

    if params is None:
        param_names = list(search_space.keys())
    else:
        param_names = params

    suggestions: dict[str, Any] = {}

    for param_name in param_names:
        if param_name not in search_space:
            continue

        space = search_space[param_name]
        space_type = space.get("type")

        if space_type == "cat":
            # 离散型: trial.suggest_categorical
            choices = space.get("choices", [])
            if not choices:
                raise ValueError(f"分类型参数 {param_name} 缺少 choices")
            # Optuna 的 suggest_categorical 不接受 '.' 字符作为参数名，
            # 需要使用安全名称
            safe_name = param_name.replace(".", "_")
            suggestions[param_name] = trial.suggest_categorical(safe_name, choices)

        elif space_type == "int":
            # 整数型: trial.suggest_int
            low = space.get("low")
            high = space.get("high")
            step = space.get("step", 1)

            if low is None or high is None:
                # 缺少边界时使用默认值
                if "default" in space:
                    suggestions[param_name] = space["default"]
                    continue
                raise ValueError(f"整数型参数 {param_name} 缺少 low 或 high")

            safe_name = param_name.replace(".", "_")
            suggestions[param_name] = trial.suggest_int(
                safe_name, int(low), int(high), step=int(step)
            )

        elif space_type == "float":
            # 连续型: trial.suggest_float
            low = space.get("low")
            high = space.get("high")
            log = space.get("log", False)

            if low is None or high is None:
                if "default" in space:
                    suggestions[param_name] = space["default"]
                    continue
                raise ValueError(f"浮点型参数 {param_name} 缺少 low 或 high")

            safe_name = param_name.replace(".", "_")
            suggestions[param_name] = trial.suggest_float(
                safe_name, float(low), float(high), log=log
            )

        elif space_type is None:
            # 类型未推断出来，使用默认值
            if "default" in space:
                suggestions[param_name] = space["default"]
        else:
            raise ValueError(f"未知的搜索空间类型: {space_type} (参数 {param_name})")

    return suggestions
