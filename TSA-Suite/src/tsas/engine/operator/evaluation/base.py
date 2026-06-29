# -*- coding: utf-8 -*-

"""
评价指标算子基类定义模块

基于 BaseOperator 扩展，提供评价指标算子的基础类型。
新增 scores() 方法供 HPO 单目标/多目标优化统一调用。

核心组件:
    - MR: 指标结果类型泛型变量，bound 为 Union[float, BaseModel]
    - MC: 实例参数类型泛型变量，bound 为 Union[BaseMetricConfig, None]
    - BaseMetricConfig: 评价指标算子配置基类，提供 main_scores 字段
    - BaseMetricOperator: 评价指标算子基类，无状态纯函数

设计要点:
    - 指标结果类型 MR 支持 float 和 Pydantic BaseModel 两种形态
    - scores() 方法按 main_scores 配置从 run 结果中提取命名标量字典
    - main_scores 为 None 时 scores() 返回 None，此时应直接使用 run() 获取完整结果
    - float 类型 MR 的路径统一使用 "_" 占位符
    - 多层继承场景下通过 __init_subclass__ 自动从 MC 泛型参数提取 Config 类型

使用示例::

    from pydantic import BaseModel
    from tsas.engine.operator.evaluation.base import (
        BaseMetricConfig, BaseMetricOperator,
    )

    # 简单标量指标（MR=float）
    class F1Config(BaseMetricConfig):
        main_scores: dict[str, str] | None = {"f1": "_"}

    class F1MetricOp(BaseMetricOperator[tuple[np.ndarray, np.ndarray], float, F1Config, None]):
        def _run(self, x, *, params):
            y_truth, y_scores = x
            return 0.85  # float

    op = F1MetricOp()
    op.run(x)     # -> 0.85
    op.scores(x)  # -> {"f1": 0.85}

    # 结构化指标（MR=BaseModel）
    class BinaryResult(BaseModel):
        f1: float
        far: float

    class BinaryConfig(BaseMetricConfig):
        positive_label: int = 1
        main_scores: dict[str, str] | None = {"f1": "f1", "far": "far"}

    class BinaryOp(BaseMetricOperator[tuple[np.ndarray, np.ndarray], BinaryResult, BinaryConfig, None]):
        def _run(self, x, *, params):
            return BinaryResult(f1=0.85, far=0.12)

    op = BinaryOp()
    op.scores(x)  # -> {"f1": 0.85, "far": 0.12}
"""

from abc import ABCMeta
from typing import TypeVar, Generic, Union

from pydantic import BaseModel, ConfigDict

from tsas.engine.operator.base import BaseOperator, I, RP

__all__ = [
    'MR',
    'MC',
    'BaseMetricConfig',
    'BaseMetricOperator',
]

# ============================================================================
# 泛型类型变量
# ============================================================================

MR = TypeVar("MR", bound=Union[float, BaseModel])
"""指标结果类型泛型变量"""


# ============================================================================
# 配置基类
# ============================================================================

class BaseMetricConfig(BaseModel):
    """评价指标算子配置基类

    所有评价指标算子的 Config 应继承此类，提供统一的 ``main_scores`` 字段。
    子类可重写 ``main_scores`` 的默认值以指定默认提取的指标组合。
    配置实例为 frozen 模式，创建后不可修改。

    Attributes:
        main_scores (dict[str, str] | None): 主评分路径映射。
            键为指标名称（如 ``"f1"``），值为 MR 属性路径（如 ``"f1"`` 或 ``"macro.f1"``）。
            - ``None``（默认）: scores() 返回 None，应直接使用 run() 获取完整结果
            - 非 None: scores() 按 {指标名称: MR属性路径} 从 MR 中提取标量
            - float 类型 MR 的路径统一使用 ``"_"`` 占位符
    """
    model_config = ConfigDict(frozen=True)

    main_scores: dict[str, str] | None = None
    """主评分路径映射：{指标名称: MR属性路径}，None 时 scores() 返回 None"""


MC = TypeVar("MC", bound=Union[BaseMetricConfig, None])
"""评价指标算子实例参数类型泛型变量"""


# ============================================================================
# 评价指标算子基类
# ============================================================================

class BaseMetricOperator(
    BaseOperator[I, MR, MC, RP],
    Generic[I, MR, MC, RP],
    metaclass=ABCMeta,
):
    """评价指标算子基类 — 无状态纯函数

    继承 BaseOperator 的参数管理、名称标识、持久化和 run 模板方法。
    不需要训练能力，不使用 LearnableOperatorMixin。

    新增 scores() 方法：按 main_scores 配置从 run 结果中提取命名标量字典，
    供 HPO 单目标/多目标优化统一使用。

    子类仅需实现 ``_run`` 方法完成指标计算，BaseMetricOperator 负责：
        - 从 MC 泛型参数自动提取 Config 类型（通过 __init_subclass__）
        - 验证 Config 必须是 BaseMetricConfig 的子类
        - 按 main_scores 配置从结果中提取命名标量字典（通过 scores）

    泛型参数:
        I (type): 输入类型（如 tuple[np.ndarray, np.ndarray] 或 np.ndarray）
        MR (type): 指标结果类型（float 或 Pydantic BaseModel 子类）
        MC (type): 实例参数类型（bound Union[BaseMetricConfig, None]）
        RP (type): 运行参数类型
    """

    def __init_subclass__(cls, **kwargs):
        """子类定义时的钩子，从 MC 泛型参数提取 Config 类型并验证

        完成一件事：

        - 使用 ``BaseOperator._extract_type_from_typevar`` 从 MC 泛型参数提取
          Config 类型，然后验证其必须是 ``BaseMetricConfig`` 的子类。

        注意：``_output_type`` 的提取**不需要**在此处定制。``BaseOperator.
        __init_subclass__`` 中升级后的 ``_extract_type_from_typevar`` 已支持
        多层泛型追踪，能自动反查 ``BaseMetricOperator`` 中 ``MR`` 等价于
        ``BaseOperator`` 的 ``O``，从而在 ``BaseOperator.__init_subclass__`` 中
        直接完成 ``_output_type`` 的提取。

        Args:
            **kwargs (Any): 传递给父类 __init_subclass__ 的关键字参数

        Raises:
            TypeError: 当 Config 类型不是 BaseMetricConfig 的子类时抛出
        """
        super().__init_subclass__(**kwargs)
        # BaseOperator.__init_subclass__ 通过 C TypeVar 提取 _config_type，
        # 但 BaseMetricOperator 使用 MC 替换了 C，多层追踪能正确提取 MC 对应类型
        if cls._config_type is None:
            extracted = BaseOperator._extract_type_from_typevar(cls, BaseMetricOperator, MC)
            cls._config_type = extracted if isinstance(extracted, type) and issubclass(extracted, BaseModel) else None
        # 运行时验证：Config 必须是 BaseMetricConfig 的子类
        if cls._config_type is not None and not issubclass(cls._config_type, BaseMetricConfig):
            raise TypeError(
                f"{cls.__name__} 的 Config 类型必须是 BaseMetricConfig 的子类，"
                f"但当前是 {cls._config_type.__name__}"
            )

    def scores(self, x: I, *, params: RP | None = None, **kwargs) -> dict[str, float] | None:
        """计算指标并按 main_scores 映射提取命名标量字典

        当 config 为 None 或 config.main_scores 为 None 时返回 None，
        此时应直接使用 run() 获取完整结果。
        当 config.main_scores 非 None 时，先调用 run() 获取完整结果，
        再按 {指标名称: MR属性路径} 映射提取标量值。

        HPO 集成示例::

            # 单目标优化
            scores = op.scores(x)
            objective = scores["f1"]

            # 多目标优化（Pareto 前沿）
            scores = op.scores(x)
            return scores["f1"], -scores["far"]

        Args:
            x (I): 输入数据
            params (RP | None): 运行参数，默认为 None
            **kwargs (Any): 运行参数键值对覆盖

        Returns:
            dict[str, float] | None:
                - config 为 None 或 config.main_scores 为 None 时，返回 None
                - config.main_scores 非 None 时，返回 {指标名称: 标量值} 字典
        """
        # config 或 main_scores 为空时无法提取，直接返回 None
        if self.config is None or self.config.main_scores is None:
            return None
        # 调用 run() 获取完整指标结果，再按映射提取标量
        result = self.run(x, params=params, **kwargs)
        return self._extract_scores(result, self.config.main_scores)

    def _extract_scores(self, result: MR, main_scores: dict[str, str]) -> dict[str, float]:
        """从指标结果中按 main_scores 映射提取标量字典

        遍历 main_scores 中的每个 {指标名称: 路径} 对，通过 _resolve_path
        从 result 中提取对应属性值，并转换为 float。

        Args:
            result (MR): _run 返回的指标结果（float 或 Pydantic BaseModel 实例）
            main_scores (dict[str, str]): {指标名称: MR属性路径} 映射

        Returns:
            dict[str, float]: {指标名称: 标量值} 字典
        """
        return {name: float(self._resolve_path(result, path))
                for name, path in main_scores.items()}

    @staticmethod
    def _resolve_path(obj, path: str):
        """按路径从对象中提取属性值

        支持两种路径模式:
            - ``"_"``: 直接返回 obj 本身，适用于 float 类型 MR
            - 点分属性名: 逐层 getattr 提取，如 ``"macro.f1"`` 提取 obj.macro.f1

        Args:
            obj (Any): 结果对象（float 或 Pydantic BaseModel 实例）
            path (str): 属性路径（如 ``"f1"``、``"macro.f1"``、``"_"``）

        Returns:
            Any: 提取到的属性值

        Raises:
            AttributeError: 当路径中的属性不存在时抛出
        """
        if path == "_":
            return obj
        for attr in path.split('.'):
            obj = getattr(obj, attr)
        return obj
