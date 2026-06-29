# -*- coding: utf-8 -*-

"""
特征构造算子基类定义模块

提供特征构造算子的完整类层次结构，基于 BaseOperator / LearnableOperatorMixin 扩展，
支持列维度（单列独立 / 多列联合）和行维度（逐行映射 / 滑动窗口）的正交组合。

类层次结构::

    第1层：Feature 基础混入
      BaseFeatureMixin                    ← Feature 独有逻辑

    第2层：Feature 基类（混入 + Operator）
      BaseFeature                         ← 不可训练特征构造基类
      LearnableFeature                    ← 可训练特征构造基类

    第3层：模式混入（列关系 + 行关系）
      IndependentFeatureMixin             ← 单列独立型
      JointFeatureMixin                   ← 多列联合型
      MapFeatureMixin                     ← 逐行映射
      WindowFeatureMixin                  ← 滑动窗口

    第4层：8个编排基类（开发者直接继承）
      IndependentMapFeature               ← 单列独立 + 逐行映射
      IndependentWindowFeature            ← 单列独立 + 滑动窗口
      JointMapFeature                     ← 多列联合 + 逐行映射
      JointWindowFeature                  ← 多列联合 + 滑动窗口
      LearnableIndependentMapFeature      ← 可训练 + 单列独立 + 逐行映射
      LearnableIndependentWindowFeature   ← 可训练 + 单列独立 + 滑动窗口
      LearnableJointMapFeature            ← 可训练 + 多列联合 + 逐行映射
      LearnableJointWindowFeature         ← 可训练 + 多列联合 + 滑动窗口

使用示例::

    from pydantic import BaseModel, ConfigDict, Field
    from tsas.engine.operator.feature.construction.base import (
        IndependentMapFeature, BaseFeatureConfig,
    )

    class SquareConfig(BaseFeatureConfig):
        pass  # 仅需 input_columns

    class SquareFeature(IndependentMapFeature[SquareConfig, None]):
        @staticmethod
        def compute(x, *, state=None, **params):
            return x ** 2

        def _name_output_column(self, input_col, output_val):
            return self._make_output_column_name(input_col, "square")
"""

from abc import abstractmethod, ABC, ABCMeta
from enum import StrEnum
from typing import TypeVar, Generic, Union, Self

import numpy as np
import pandas as pd
from pydantic import BaseModel, ConfigDict, Field

from tsas.engine.operator.base import LearnableOperatorMixin, C, NumericOperator, NumericData, \
    DataFrameMeta

__all__ = [
    # 枚举
    'Alignment',
    'Padding',
    # Config 基类
    'BaseFeatureConfig',
    'WindowFeatureConfig',
    # 第1层：Feature 基础混入
    'BaseFeatureMixin',
    # 第2层：Feature 基类
    'BaseFeature',
    'LearnableFeature',
    # 第3层：模式混入
    'IndependentFeatureMixin',
    'JointFeatureMixin',
    'MapFeatureMixin',
    'WindowFeatureMixin',
    # 第4层：编排基类
    'IndependentMapFeature',
    'IndependentWindowFeature',
    'JointMapFeature',
    'JointWindowFeature',
    'LearnableIndependentMapFeature',
    'LearnableIndependentWindowFeature',
    'LearnableJointMapFeature',
    'LearnableJointWindowFeature',
]


# ============================================================================
# 枚举类型
# ============================================================================

class Alignment(StrEnum):
    """窗口对齐方式枚举

    定义滑动窗口结果与原始时间序列的对齐策略。
    不同的对齐方式决定了窗口计算结果在原始序列中的时间对应关系。

    Attributes:
        LEFT: 左对齐，窗口起始位置与结果位置对齐。
            结果[i] 对应输入 x[i : i+w]，即窗口从第 i 行开始向右延伸。
        RIGHT: 右对齐（默认），窗口结束位置与结果位置对齐。
            结果[i] 对应输入 x[i-w+1 : i+1]，即窗口到第 i 行结束。
    """
    LEFT = "left"
    RIGHT = "right"


class Padding(StrEnum):
    """预定义填充模式枚举

    定义滑动窗口边界的填充策略，用于在不截断输出的情况下处理边界数据。
    填充模式仅在 ``WindowFeatureConfig.padding`` 为枚举值时生效；
    当 ``padding`` 为 ``None`` 时不填充，为数字时使用固定值填充。

    Attributes:
        EDGE: 边界值填充，用数据的首行（左对齐）或末行（右对齐）重复填充。
        NAN: NaN 填充，用 ``float('nan')`` 填充边界位置。
        REFLECT: 镜像填充，以数据边界为镜面翻转数据进行填充。
        RING: 循环填充（首尾相接），用数据对端的内容填充边界。
    """
    EDGE = "edge"
    NAN = "nan"
    REFLECT = "reflect"
    RING = "ring"


# ============================================================================
# Config 基类
# ============================================================================

class BaseFeatureConfig(BaseModel):
    """特征算子配置基类

    所有特征构造算子的 Config 应继承此类，提供统一的 ``input_columns`` 字段。
    使用 Pydantic BaseModel 实现，支持字段校验和不可变约束（``frozen=True``）。

    Attributes:
        input_columns (list[str]): 输入列名列表，至少包含一列。
            由子类在实例化时指定，决定算子从输入数据中选取哪些列参与计算。
    """
    model_config = ConfigDict(frozen=True)

    input_columns: list[str] = Field(
        min_length=1,
        description="输入列名列表，至少包含一列；决定算子从输入数据中选取哪些列参与计算",
    )


class WindowFeatureConfig(BaseFeatureConfig):
    """滑动窗口模式特征算子配置基类

    在 ``BaseFeatureConfig`` 基础上增加窗口大小、填充模式和对齐方式三个字段，
    供 ``WindowFeatureMixin`` 及其派生类使用。

    Attributes:
        window_size (int): 滑动窗口大小，必须为正整数（``gt=0``）。
            决定每次 ``compute`` 调用接收的数据行数。
        padding (Padding | float | int | None): 填充策略，默认为 ``None``（不填充）。
            - ``None``: 不填充，输出行数 = 输入行数 - window_size + 1
            - ``Padding`` 枚举: 使用预定义填充模式（EDGE / NAN / REFLECT / RING）
            - ``float | int``: 使用指定数值填充
        alignment (Alignment): 窗口对齐方式，默认为右对齐（``Alignment.RIGHT``）。
            右对齐时结果[i] 对应输入 x[i-w+1 : i+1]，
            左对齐时结果[i] 对应输入 x[i : i+w]。
    """

    window_size: int = Field(
        gt=0,
        description="滑动窗口大小，必须为正整数；决定每次 compute 调用接收的数据行数",
    )

    padding: Padding | float | int | None = Field(
        default=None,
        description=(
            "填充模式：None（不填充，输出长度减少）/ "
            "Padding.EDGE（边界值填充）/ Padding.NAN（NaN 填充）/ "
            "Padding.REFLECT（镜像填充）/ Padding.RING（循环填充）/ "
            "float|int（指定数值填充）"
        ),
    )

    alignment: Alignment = Field(
        default=Alignment.RIGHT,
        description=(
            "窗口对齐方式：RIGHT（右对齐，结果[i] 对应 x[i-w+1:i+1]）/ "
            "LEFT（左对齐，结果[i] 对应 x[i:i+w]）"
        ),
    )


# ============================================================================
# 第1层：Feature 基础混入
# ============================================================================

FS = TypeVar("FS", bound=Union[BaseModel, None])
"""特征训练状态（Feature State）类型泛型

- 绑定为 ``BaseModel | None``，即 Pydantic 模型子类或 ``None``。
- ``None`` 表示无状态（不可训练算子），
- ``BaseModel`` 子类表示有状态（可训练算子训练后保存的参数）。
"""


class BaseFeatureMixin(Generic[FS], metaclass=ABCMeta):
    """特征构造算子的基础混入类

    承载 Feature 相比 Operator 独有的公共逻辑，是整个特征构造类层次的第1层。
    该混入类不继承任何 Operator 基类，通过多重继承被第2层基类（BaseFeature / LearnableFeature）组合使用。

    职责概览:
        - **输入校验**:
          ``_validate_dataframe_input`` / ``_validate_ndarray_input`` 校验输入数据的列完整性。
        - **数据筛选**:
          ``_filter_data`` 按 ``input_columns`` 提取所需列并调整顺序。
        - **特征计算接口**:
          ``compute`` 抽象静态方法（子类必须实现），定义逐行或逐窗口的计算逻辑。
        - **状态训练接口**:
          ``train`` 静态方法（可选覆写），用于可训练算子的状态学习。
        - **参数注入**:
          ``_get_compute_params`` / ``_get_train_params`` 供子类向 ``compute`` / ``train`` 传递额外参数。
        - **输出列命名**:
          ``_make_output_column_name`` 提供默认的输出列名拼接策略。
        - **状态属性**:
          ``state`` 属性默认返回 ``None``，由 ``LearnableFeature`` 覆盖返回训练后的状态。

    泛型参数:
        FS: 特征训练状态类型，绑定 ``BaseModel | None``，无状态算子传入 ``None``。
    """

    @property
    def state(self) -> FS | None:
        """特征训练状态属性

        无状态算子默认返回 ``None``；有状态算子（如 ``LearnableFeature``）
        会在训练后覆盖此属性，返回训练学习到的状态对象。

        Returns:
            FS | None: 训练后的状态对象，无状态算子恒为 ``None``。
        """
        return None

    # region 直接模板方法

    def _validate_dataframe_input(self, x: pd.DataFrame, params: None) -> None:
        """DataFrame 输入的列完整性校验

        检查输入 DataFrame 是否包含 ``config.input_columns`` 中指定的全部列名。
        若存在缺失列则抛出 ``ValueError``。

        Args:
            x (pd.DataFrame): 输入数据。
            params (None): 运行参数（当前未使用，预留扩展）。

        Raises:
            ValueError: 输入 DataFrame 缺少 ``input_columns`` 中指定的列。
        """
        missing = set(self.config.input_columns) - set(x.columns)
        if missing:
            raise ValueError(f"输入数据缺少以下列: {missing}")

    def _validate_ndarray_input(self, x: np.ndarray, params: None) -> None:
        """ndarray 输入的列数校验

        检查输入 ndarray 的列维度是否满足 ``config.input_columns`` 要求的列数。
        一维数组被视为单列数据。

        Args:
            x (np.ndarray): 输入数据，支持一维或二维数组。
            params (None): 运行参数（当前未使用，预留扩展）。

        Raises:
            ValueError: 输入数据的列数少于 ``input_columns`` 要求的列数。
        """
        require_column_count = len(self.config.input_columns)
        if x.ndim == 1:
            if require_column_count > 1:
                raise ValueError(f"输入数据列数要求至少为 {require_column_count}，但实际为 1")
        elif x.ndim > 1:
            if x.shape[1] < require_column_count:
                raise ValueError(f"输入数据列数要求至少为 {require_column_count}，但实际为 {x.shape[1]}")

    def _filter_data(self, x: NumericData, params: None) -> NumericData:
        """按 ``input_columns`` 提取所需列并调整顺序

        对于 DataFrame 输入，按照 ``config.input_columns`` 提取并重排列；
        对于 ndarray 输入，一维数据原样返回，二维数据按列数截取前 N 列；
        其他类型抛出 ``TypeError``。

        Args:
            x (NumericData): 输入数据，支持 ``pd.DataFrame`` 或 ``np.ndarray``。
            params (None): 运行参数（当前未使用，预留扩展）。

        Returns:
            NumericData: 筛选后的数据，类型与输入一致。

        Raises:
            TypeError: 输入数据类型不是 ``pd.DataFrame`` 或 ``np.ndarray``。
        """
        if isinstance(x, pd.DataFrame):
            return x[self.config.input_columns]
        elif isinstance(x, np.ndarray):
            return x if x.ndim == 1 else x[:, :len(self.config.input_columns)]
        else:
            raise TypeError(f"数据类型必须是 pd.DataFrame 或 np.ndarray，但当前是 {type(x)}")

    # endregion

    # region 间接模板方法

    def _make_output_column_name(self, source_col: str, feature_name: str, value_name: str | None = None) -> str:
        """生成输出列名的模板方法

        默认命名规则:
            - 无 ``value_name`` 时: ``{源列名}_{特征名}``
            - 有 ``value_name`` 时: ``{源列名}_{特征名}_{特征值名}``

        子类可覆写此方法以自定义命名策略。

        Args:
            source_col (str): 源列名称。
            feature_name (str): 特征名称标识。
            value_name (str | None): 特征值名称，可选。当 ``compute`` 返回多维结果时，
                用于区分不同输出列。

        Returns:
            str: 拼接后的输出列名。
        """
        if value_name:
            return "_".join((source_col, feature_name, value_name))
        else:
            return "_".join((source_col, feature_name))

    @staticmethod
    @abstractmethod
    def compute(x: np.ndarray, *, state: FS | None = None, **params) -> float | np.ndarray:
        """特征计算抽象静态方法（子类必须实现）

        无状态与有状态算子的统一计算接口。在 Map 模式下，
        ``x`` 为单列或多列的全部行数据；在 Window 模式下，
        ``x`` 为单列或多列的窗口切片数据（行数 = window_size）。

        Args:
            x (np.ndarray): 输入数据数组。
            state (FS | None): 训练后的状态对象，无状态算子为 ``None``。
                该参数为 keyword-only，由框架在调用时自动注入。
            **params: 由 ``_get_compute_params`` 返回的额外计算参数。

        Returns:
            float | np.ndarray: Map 模式下返回与输入行数一致的数组；
                Window 模式下返回单个数值（float）。
        """
        ...

    @staticmethod
    def train(x: np.ndarray, **params) -> FS | None:
        """状态训练静态方法（可选覆写）

        默认实现返回 ``None``（无状态）。可训练算子（LearnableFeature 派生类）
        应覆写此方法，基于训练数据学习并返回状态对象。
        学习结果通过 ``state`` 属性暴露，供 ``compute`` 在推理时使用。

        Args:
            x (np.ndarray): 训练数据数组。
            **params: 由 ``_get_train_params`` 返回的额外训练参数。

        Returns:
            FS | None: 训练后的状态对象（BaseModel 子类），
                无状态算子默认返回 ``None``。
        """
        return None

    def _get_compute_params(self):
        """获取传递给 ``compute`` 的额外参数

        子类可覆写此方法以向 ``compute`` 注入运行时参数
        （如窗口大小、阈值、归一化因子等）。
        默认返回空字典，表示 ``compute`` 仅接收必要的 ``x`` 和 ``state``。

        Returns:
            dict: 额外计算参数键值对。
        """
        return {}

    def _get_train_params(self):
        """获取传递给 ``train`` 的额外参数

        子类可覆写此方法以向 ``train`` 注入训练时参数
        （如采样率、模型超参数等）。
        默认返回空字典，表示 ``train`` 仅接收必要的训练数据 ``x``。

        Returns:
            dict: 额外训练参数键值对。
        """
        return {}

    # endregion


# ============================================================================
# 第2层：Feature 基类
# ============================================================================

class BaseFeature(BaseFeatureMixin[None], NumericOperator[None, C, None], Generic[C], ABC):
    """不可训练特征构造基类

    将 ``BaseFeatureMixin`` 的特征构造逻辑与 ``NumericOperator`` 的算子框架结合。
    此类固定 ``FS=None``（无状态），即不支持训练。
    子类不应直接继承此类，而应继承第4层的编排基类（如 ``IndependentMapFeature``）。

    泛型参数:
        C: Config 类型，应继承 ``BaseFeatureConfig`` 或 ``WindowFeatureConfig``。
    """
    pass


class LearnableFeature(BaseFeatureMixin[FS], LearnableOperatorMixin[NumericData, None, None],
                       NumericOperator[None, C, None], Generic[C, FS], ABC):
    """可训练特征构造基类

    将 ``BaseFeatureMixin`` 的特征构造逻辑与 ``LearnableOperatorMixin`` 的可训练算子框架结合。
    子类不应直接继承此类，而应继承第4层的编排基类（如 ``LearnableIndependentMapFeature``）。

    训练流程:
        1. 调用 ``fit`` 触发训练，内部执行 ``_fit`` 模板方法。
        2. ``_fit`` 接收完整的输入 DataFrame（不经过 Independent/Joint 的列遍历编排），
           开发者在 ``_fit`` 中自行处理列选取和状态学习逻辑。
        3. ``train`` 静态方法用于学习状态，学习结果保存在 ``_state`` 中，
           通过 ``state`` 属性暴露，供 ``compute`` 方法在推理时使用。

    泛型参数:
        C: Config 类型，应继承 ``BaseFeatureConfig`` 或 ``WindowFeatureConfig``。
        FS: 状态类型，绑定 ``BaseModel | None``。实际应为 ``BaseModel`` 子类，
            训练后的状态对象将赋值给 ``_state``。
    """

    def __init__(self, **kwargs):
        """初始化可训练特征算子

        调用父类初始化器，并将内部状态 ``_state`` 设为 ``None``。
        训练完成后 ``_state`` 会被 ``train`` 返回值覆盖。

        Args:
            **kwargs: 传递给父类初始化器的关键字参数。
        """
        super().__init__(**kwargs)
        self._state: FS | None = None

    @property
    def state(self) -> FS | None:
        """训练后的状态属性

        覆盖 ``BaseFeatureMixin.state`` 的默认实现，返回 ``_state`` 而非 ``None``。
        训练前为 ``None``，训练后为由 ``train`` 返回的状态对象。

        Returns:
            FS | None: 训练后的状态对象，训练前为 ``None``。
        """
        return self._state

    def fit(self, x: NumericData, *, params: None = None, **kwargs) -> Self:
        """训练特征算子

        对 ``LearnableOperatorMixin.fit`` 的适配封装，
        将标签参数固定为 ``None``（特征构造无需标签数据）。

        Args:
            x (NumericData): 训练输入数据。
            params (None): 运行参数（当前未使用，预留扩展）。
            **kwargs: 传递给父类 ``fit`` 的额外关键字参数。

        Returns:
            Self: 训练后的算子实例（支持链式调用）。
        """
        return super().fit(x, None, params=params, **kwargs)

    def _fit(self, x: NumericData, y: None = None, *, params: None = None) -> None:
        """可训练特征算子的训练模板方法

        执行完整的训练管线：输入校验 → 列筛选 → 数据解包 → 核心训练 → 标记已训练。

        Args:
            x (NumericData): 训练输入数据。
            y (None): 标签数据（特征构造不使用，恒为 ``None``）。
            params (None): 运行参数（当前未使用，预留扩展）。
        """
        # 步骤1：验证输入数据类型和合法性
        self._validate_input(x, params)
        # 步骤2：按需进行列名筛选和顺序调整
        data = self._filter_data(x, params)
        # 步骤3：按输入类型解包为 (元信息, ndarray)，DataFrame 输入时记录元信息用于后续回包
        meta, data = self._unwrap_data(data, params)
        # 步骤4：执行核心训练逻辑，将 train 返回的状态保存到 _state
        self._state = self.train(data, **self._get_train_params())
        # 步骤5：标记算子已完成训练
        self._fitted = True
        return


# ============================================================================
# 第3层：模式混入
# ============================================================================

class IndependentFeatureMixin(BaseFeatureMixin[FS], metaclass=ABCMeta):
    """单列独立型特征混入

    表示输出中每列只与输入中的一列相关的语义约定。所有 ``input_columns`` 对应的列
    作为一个完整的 ndarray 传入 ``compute``，**不会按列拆分逐个调用**。

    **实现约定**:

        列间的独立性由 ``compute`` 方法自行保证（推荐利用 NumPy 的 ``axis`` 参数
        沿列方向进行独立计算）。框架负责根据输入列数和输出列数自动分组命名
        （每组列数 = 总输出列数 / 输入列数）。

    例如，若 ``input_columns`` 有 3 列，``compute`` 每列返回 2 个值，
    则输出共 6 列，前 2 列对应第 1 个输入列，中间 2 列对应第 2 个输入列，依此类推。
    """

    def _name_output_columns(self, output_data: np.ndarray, meta: DataFrameMeta | None, params: None) -> list[str]:
        """为独立型特征输出数据生成列名列表

        将输出数据按输入列数均分为若干组，每组调用 ``_name_output_column`` 生成列名。
        要求输出列数必须是输入列数的整数倍，否则抛出 ``ValueError``。

        Args:
            output_data (np.ndarray): 输出数据数组，二维时按列分组，一维时按元素分组。
            meta (DataFrameMeta | None): 输入 DataFrame 的元信息（当前未使用，预留扩展）。
            params (None): 运行参数（当前未使用，预留扩展）。

        Returns:
            list[str]: 与输出列数对应的列名列表。

        Raises:
            ValueError: 输出列数不是输入列数的整数倍。
        """
        # 获取输入列名列表和原始输入列数
        input_cols = self.config.input_columns
        n_inputs = len(input_cols)

        # 判断输出是一维还是二维，并获取总输出列数（或元素数）
        is_matrix = output_data.ndim > 1
        n_outputs = output_data.shape[1] if is_matrix else output_data.shape[0]

        # 校验：在 Independent 模式下，每个输入列产生的输出维度必须一致
        if n_outputs % n_inputs != 0:
            raise ValueError(
                f"输出列数({n_outputs})必须是输入列数({n_inputs})的整数倍，"
                f"但 {n_outputs} % {n_inputs} = {n_outputs % n_inputs}"
            )

        # 计算平均每个输入列生成的输出特征数量
        m = n_outputs // n_inputs
        result = []
        for i, col in enumerate(input_cols):
            # 计算当前输入列对应的输出数据切片索引
            # 假设输出数组是按输入列顺序排列的 [InCol1_Out1, InCol1_Out2, InCol2_Out1, ...]
            start = i * m
            end = start + m

            # 提取该输入列对应的输出数据块
            if is_matrix:
                group_values = output_data[:, start:end]
            else:
                group_values = output_data[start:end]

            # 调用子类实现的命名逻辑，生成该组对应的列名
            result.append(self._name_output_column(col, group_values))

        return result

    @abstractmethod
    def _name_output_column(self, input_col: str, output_val: float | np.ndarray) -> str:
        """为单个输入列的输出结果生成列名（子类必须实现）

        子类通常调用 ``_make_output_column_name`` 实现命名逻辑。

        Args:
            input_col (str): 源输入列名称。
            output_val (float | np.ndarray): 该输入列对应的输出值，
                用于判断输出维度并决定是否需要多维命名。

        Returns:
            str: 生成的输出列名。
        """
        ...


class JointFeatureMixin(BaseFeatureMixin[FS], metaclass=ABCMeta):
    """多列联合型特征混入

    输出中每列都可能与输入中的多个列相关。

    与 ``IndependentFeatureMixin`` 不同，此模式下框架一次性将 ``input_columns`` 
    对应的多列数据提取为二维数组，整体传入 ``compute`` 静态方法进行联合计算。
    此混入类不限制输出列名，由 ``NumericOperator`` 的默认机制处理（直接使用 compute 的返回值）。

    适用场景:
        - 列间差值、比值、相关性分析。
        - 跨多维指标的聚类、降维等。
    """
    pass


class MapFeatureMixin(BaseFeatureMixin[FS], metaclass=ABCMeta):
    """逐行映射模式混入

    定义一行输入对应一行输出（1:1 映射）的特征处理流程。

    本混入类实现了 ``_run_data`` 方法，它将全部筛选后的 NumPy 数据数组
    直接传递给 ``compute`` 静态方法，不涉及窗口移动。

    特点:
        - 逐元素或逐行独立变换。
        - 输出行数与输入行数始终一致。
        - 无需处理填充（Padding）和对齐（Alignment）。
    """

    def _run_data(self, x: np.ndarray, params: None, idx: pd.Index | None = None) -> np.ndarray:
        """逐行映射模式的核心数据处理方法

        Args:
            x (np.ndarray): 输入数据数组（全部行），二维或一维。
            params (None): 运行参数（保留参数，当前传 None）。

        Returns:
            np.ndarray: ``compute`` 的计算结果，行数与输入一致。
        """
        # 获取子类注入的额外计算参数
        compute_params = self._get_compute_params()
        # 直接全量数据传入 compute
        return self.compute(x, state=self.state, **compute_params)


class WindowFeatureMixin(BaseFeatureMixin[FS], metaclass=ABCMeta):
    """滑动窗口模式混入

    多行输入对应一行输出。提供滑动窗口遍历、填充（padding）和对齐（alignment）
    的通用处理逻辑。窗口参数通过 ``WindowFeatureConfig`` 配置。

    核心流程:
        1. ``_adjust_data``: 根据 padding 和 alignment 对输入数据进行边界填充。
        2. ``_run_data``: 以窗口大小为步长滑动遍历数据，逐窗口调用 ``compute``。
        3. ``_adjust_index``: 根据是否填充及对齐方式调整输出索引与原始索引的对应关系。
    """

    def _adjust_data(self, x: np.ndarray, params: None, idx: pd.Index | None = None) -> np.ndarray:
        """根据 padding + alignment 对 ndarray 进行边界填充

        当 ``config.padding`` 为 ``None`` 时不填充，直接返回原数据；
        否则根据填充模式（Padding 枚举或数值）和对齐方式在数据边界添加填充行，
        使滑动窗口输出行数与输入行数一致。

        Args:
            x (np.ndarray): 原始输入 ndarray（通常为二维，列数为 input_columns 长度）。
            params (None): 运行参数（保留参数，当前传 None）。

        Returns:
            np.ndarray: 填充后的 ndarray；padding 为 None 时直接返回原数据。

        Raises:
            TypeError: 填充模式类型无效（非 Padding 枚举也非数字类型）。
        """
        # 获取配置中的填充策略
        padding = self.config.padding
        if padding is None:
            # 不进行任何填充，窗口计算后行数将减少
            return x

        # 计算需要填充的总行数：为了使输出行数=输入行数，填充行数应为 window_size - 1
        pad_width = self.config.window_size - 1
        alignment = self.config.alignment

        # 1. 处理预定义的填充模式（枚举类型）
        if isinstance(padding, Padding):
            if padding == Padding.EDGE:
                return self._pad_edge(x, pad_width, alignment)
            elif padding == Padding.NAN:
                return self._pad_value(x, pad_width, alignment, float('nan'))
            elif padding == Padding.REFLECT:
                return self._pad_reflect(x, pad_width, alignment)
            elif padding == Padding.RING:
                return self._pad_ring(x, pad_width, alignment)

        # 2. 处理数值填充（固定值）
        elif isinstance(padding, (int, float)):
            # 统一转为 float 后进行填充
            return self._pad_value(x, pad_width, alignment, float(padding))

        # 类型错误抛出
        raise TypeError(f"填充模式类型无效，应为 Padding 枚举或数字类型，当前类型: {type(padding)}")

    def _run_data(self, x: np.ndarray, params: None, idx: pd.Index | None = None) -> np.ndarray:
        """对数据应用滑动窗口，逐窗口调用 ``compute`` 并收集结果

        输入数据应已根据填充配置进行了预处理（由 ``_adjust_data`` 处理）。
        本方法以步长为 1 滑动遍历数据，每次截取连续 ``window_size`` 行传入 ``compute``。

        Args:
            x (np.ndarray): 输入数据（可能包含填充行）。
            params (None): 运行参数（保留参数，当前传 None）。

        Returns:
            np.ndarray: 窗口特征计算结果数组，每行对应一个窗口的计算结果。
        """
        # 获取窗口大小和子类计算参数
        window_size = self.config.window_size
        n = len(x)
        compute_params = self._get_compute_params()

        # 计算有效窗口的总数量
        # 若已填充，则 num_windows == 原始输入行数
        # 若未填充，则 num_windows == 原始输入行数 - window_size + 1
        num_windows = n - window_size + 1

        # 遍历所有窗口进行计算
        rows = []
        for i in range(num_windows):
            # 获取当前窗口的切片数据 [i, i+window_size)
            window_slice = x[i: i + window_size]
            # 调用静态 compute 方法执行核心特征提取逻辑
            row_result = self.compute(window_slice, state=self.state, **compute_params)
            rows.append(row_result)

        # 将结果列表转换为 NumPy 数组返回
        return np.array(rows)

    def _adjust_index(self, output_data: np.ndarray, meta: DataFrameMeta | None, params: None) -> pd.Index:
        """根据 padding 和 alignment 调整输出索引

        当使用填充时，输出行数与输入行数一致，索引保持不变；
        当不填充时，输出行数少于输入行数，需要根据对齐方式截取索引：
            - 右对齐：取原始索引的后 ``num_windows`` 个（窗口末端对齐）。
            - 左对齐：取原始索引的前 ``num_windows`` 个（窗口起始对齐）。

        Args:
            output_data (np.ndarray): 输出值数组（用于推断输出行数）。
            meta (DataFrameMeta | None): 输入 DataFrame 的元信息，包含原始索引。
            params (None): 运行参数（当前未使用，预留扩展）。

        Returns:
            pd.Index: 调整后的输出索引。
        """
        index = meta.index
        # 有填充时输出行数 = 输入行数，索引保持不变
        if self.config.padding is not None:
            return index

        # 无填充时需要截取索引
        window_size = self.config.window_size
        num_windows = len(output_data)

        if self.config.alignment == Alignment.RIGHT:
            # 右对齐：结果[i] 对应输入 x[i-w+1 : i+1]，取索引的后 num_windows 个
            return index[window_size - 1:]
        else:
            # 左对齐：结果[i] 对应输入 x[i : i+w]，取索引的前 num_windows 个
            return index[:num_windows]

    @staticmethod
    def _pad_value(data: np.ndarray, pad_width: int, alignment: Alignment, fill_value: float) -> np.ndarray:
        """用固定数值填充数据边界

        生成 ``pad_width`` 行以 ``fill_value`` 填充的数组，根据对齐方式拼接在数据的前方或后方。

        Args:
            data (np.ndarray): 原始输入数据（二维数组）。
            pad_width (int): 填充行数。
            alignment (Alignment): 对齐方式，决定填充位置。
            fill_value (float): 填充值。

        Returns:
            np.ndarray: 填充后的数据。
        """
        # 生成填充行块：维度为 (pad_width, 列数)
        pad_rows = np.full((pad_width, data.shape[1]), fill_value, dtype=data.dtype)

        if alignment == Alignment.RIGHT:
            # 右对齐：计算结果对应窗口末端。为了使结果[i]对应x[i]，需在x前方（上方）填充。
            return np.concatenate([pad_rows, data], axis=0)
        else:
            # 左对齐：计算结果对应窗口起始。为了使结果[i]对应x[i]，需在x后方（下方）填充。
            return np.concatenate([data, pad_rows], axis=0)

    @staticmethod
    def _pad_edge(data: np.ndarray, pad_width: int, alignment: Alignment) -> np.ndarray:
        """用边界值填充

        使用数据的首行（针对右对齐）或末行（针对左对齐）作为填充值进行重复填充。

        Args:
            data (np.ndarray): 原始输入数据。
            pad_width (int): 填充行数。
            alignment (Alignment): 对齐方式。

        Returns:
            np.ndarray: 填充后的数据。
        """
        if alignment == Alignment.RIGHT:
            # 获取首行数据并保持二维维度
            edge_row = data[[0]]
            # 重复生成填充块
            pad_rows = np.repeat(edge_row, pad_width, axis=0)
            return np.concatenate([pad_rows, data], axis=0)
        else:
            # 获取末行数据
            edge_row = data[[-1]]
            pad_rows = np.repeat(edge_row, pad_width, axis=0)
            return np.concatenate([data, pad_rows], axis=0)

    @staticmethod
    def _pad_reflect(data: np.ndarray, pad_width: int, alignment: Alignment) -> np.ndarray:
        """镜像填充

        以边界行为镜面，将数据向外翻转填充。
        注意：此方法要求数据长度至少要大于填充宽度。

        Args:
            data (np.ndarray): 原始输入数据。
            pad_width (int): 填充行数。
            alignment (Alignment): 对齐方式。

        Returns:
            np.ndarray: 填充后的数据。

        Raises:
            ValueError: 如果数据行数不足以支撑镜像填充。
        """
        n = len(data)
        if pad_width >= n:
            raise ValueError(
                f"镜像填充要求 pad_width({pad_width}) < 数据长度({n})，"
                f"即 window_size({pad_width + 1}) <= 数据长度({n})"
            )

        if alignment == Alignment.RIGHT:
            # 在头部填充：取 (0, pad_width] 范围的数据并翻转，以 data[0] 为中心对称
            pad_part = data[1: pad_width + 1][::-1]
            return np.concatenate([pad_part, data], axis=0)
        else:
            # 在尾部填充：取 [n-1-pad_width, n-1) 范围的数据并翻转，以 data[-1] 为中心对称
            pad_part = data[-(pad_width + 1): -1][::-1]
            return np.concatenate([data, pad_part], axis=0)

    @staticmethod
    def _pad_ring(data: np.ndarray, pad_width: int, alignment: Alignment) -> np.ndarray:
        """循环填充（首尾相接）

        将数据视为环形序列，用数据对端（另一侧）的内容进行填充。

        Args:
            data (np.ndarray): 原始输入数据。
            pad_width (int): 填充行数。
            alignment (Alignment): 对齐方式。

        Returns:
            np.ndarray: 填充后的数据。

        Raises:
            ValueError: 如果数据行数不足以支撑循环填充。
        """
        n = len(data)
        if pad_width >= n:
            raise ValueError(
                f"循环填充要求 pad_width({pad_width}) < 数据长度({n})，"
                f"即 window_size({pad_width + 1}) <= 数据长度({n})"
            )

        if alignment == Alignment.RIGHT:
            # 在头部填充：取序列末尾的 pad_width 行
            pad_part = data[-pad_width:]
            return np.concatenate([pad_part, data], axis=0)
        else:
            # 在尾部填充：取序列开头的 pad_width 行
            pad_part = data[:pad_width]
            return np.concatenate([data, pad_part], axis=0)


# ============================================================================
# 第4层：编排基类（开发者直接继承）
# ============================================================================

# ---- 不可训练 × 4 ----

class IndependentMapFeature(IndependentFeatureMixin, MapFeatureMixin, BaseFeature[C], Generic[C]):
    """单列独立 + 逐行映射特征

    组合 ``IndependentFeatureMixin`` 与 ``MapFeatureMixin``，对 ``input_columns`` 中的每列
    独立调用 ``compute``，传入单列 NumPy 数组（全部行），输出行数与输入行数一致。

    子类需实现:
        - ``compute``: 定义逐行映射的特征计算逻辑。
        - ``_name_output_column``: 定义输出列名生成策略。

    泛型参数:
        C: Config 类型，应继承 ``BaseFeatureConfig``。
    """
    pass


class IndependentWindowFeature(IndependentFeatureMixin, WindowFeatureMixin, BaseFeature[C], Generic[C]):
    """单列独立 + 滑动窗口特征

    组合 ``IndependentFeatureMixin`` 与 ``WindowFeatureMixin``，对 ``input_columns`` 中的每列
    独立应用滑动窗口，逐窗口调用 ``compute``，传入单列 NumPy 数组（窗口大小行数），
    ``compute`` 返回单个数值（float）。

    子类需实现:
        - ``compute``: 定义滑动窗口的特征计算逻辑，返回单个数值。
        - ``_name_output_column``: 定义输出列名生成策略。

    泛型参数:
        C: Config 类型，应继承 ``WindowFeatureConfig``。
    """
    pass


class JointMapFeature(JointFeatureMixin, MapFeatureMixin, BaseFeature[C], Generic[C]):
    """多列联合 + 逐行映射特征

    组合 ``JointFeatureMixin`` 与 ``MapFeatureMixin``，将 ``input_columns`` 对应的多列数据
    整体传入 ``compute``（全部行），输出行数与输入行数一致。
    适用于需要跨列信息交互的逐行特征（如列间差值、比值等）。

    子类需实现:
        - ``compute``: 定义多列联合的逐行映射计算逻辑。

    泛型参数:
        C: Config 类型，应继承 ``BaseFeatureConfig``。
    """
    pass


class JointWindowFeature(JointFeatureMixin, WindowFeatureMixin, BaseFeature[C], Generic[C]):
    """多列联合 + 滑动窗口特征

    组合 ``JointFeatureMixin`` 与 ``WindowFeatureMixin``，将 ``input_columns`` 对应的多列数据
    应用滑动窗口，逐窗口调用 ``compute``，传入多列 NumPy 数组（窗口大小行数），
    ``compute`` 返回单个数值（float）。适用于需要跨列信息交互的窗口特征。

    子类需实现:
        - ``compute``: 定义多列联合的滑动窗口计算逻辑，返回单个数值。

    泛型参数:
        C: Config 类型，应继承 ``WindowFeatureConfig``。
    """
    pass


# ---- 可训练 × 4 ----

class LearnableIndependentMapFeature(IndependentFeatureMixin, MapFeatureMixin, LearnableFeature[C, FS], Generic[C, FS]):
    """可训练 + 单列独立 + 逐行映射特征

    组合 ``IndependentFeatureMixin``、``MapFeatureMixin`` 与 ``LearnableFeature``。
    ``_fit`` 接收完整 DataFrame，开发者自行处理列选取和状态学习；
    ``compute`` 对每列独立调用，传入单列 NumPy 数组（全部行），
    通过 ``state`` 参数接收训练后的状态。

    子类需实现:
        - ``compute``: 定义逐行映射的特征计算逻辑，接收 ``state`` 参数。
        - ``train``: 定义状态学习逻辑，返回训练后的状态对象。
        - ``_name_output_column``: 定义输出列名生成策略。

    泛型参数:
        C: Config 类型，应继承 ``BaseFeatureConfig``。
        FS: 状态类型，绑定 ``BaseModel | None``，实际应为 ``BaseModel`` 子类。
    """
    pass


class LearnableIndependentWindowFeature(IndependentFeatureMixin, WindowFeatureMixin, LearnableFeature[C, FS],
                                        Generic[C, FS]):
    """可训练 + 单列独立 + 滑动窗口特征

    组合 ``IndependentFeatureMixin``、``WindowFeatureMixin`` 与 ``LearnableFeature``。
    ``_fit`` 接收完整 DataFrame，开发者自行处理列选取和状态学习；
    ``compute`` 对每列独立应用滑动窗口，逐窗口调用，
    通过 ``state`` 参数接收训练后的状态。

    子类需实现:
        - ``compute``: 定义滑动窗口的特征计算逻辑，返回单个数值，接收 ``state`` 参数。
        - ``train``: 定义状态学习逻辑，返回训练后的状态对象。
        - ``_name_output_column``: 定义输出列名生成策略。

    泛型参数:
        C: Config 类型，应继承 ``WindowFeatureConfig``。
        FS: 状态类型，绑定 ``BaseModel | None``，实际应为 ``BaseModel`` 子类。
    """
    pass


class LearnableJointMapFeature(JointFeatureMixin, MapFeatureMixin, LearnableFeature[C, FS], Generic[C, FS]):
    """可训练 + 多列联合 + 逐行映射特征

    组合 ``JointFeatureMixin``、``MapFeatureMixin`` 与 ``LearnableFeature``。
    ``_fit`` 接收完整 DataFrame，开发者自行处理列选取和状态学习；
    ``compute`` 接收多列 NumPy 数组（全部行），通过 ``state`` 参数接收训练后的状态。

    子类需实现:
        - ``compute``: 定义多列联合的逐行映射计算逻辑，接收 ``state`` 参数。
        - ``train``: 定义状态学习逻辑，返回训练后的状态对象。

    泛型参数:
        C: Config 类型，应继承 ``BaseFeatureConfig``。
        FS: 状态类型，绑定 ``BaseModel | None``，实际应为 ``BaseModel`` 子类。
    """
    pass


class LearnableJointWindowFeature(JointFeatureMixin, WindowFeatureMixin, LearnableFeature[C, FS], Generic[C, FS]):
    """可训练 + 多列联合 + 滑动窗口特征

    组合 ``JointFeatureMixin``、``WindowFeatureMixin`` 与 ``LearnableFeature``。
    ``_fit`` 接收完整 DataFrame，开发者自行处理列选取和状态学习；
    ``compute`` 接收多列 NumPy 数组，逐窗口调用，通过 ``state`` 参数接收训练后的状态。

    子类需实现:
        - ``compute``: 定义多列联合的滑动窗口计算逻辑，返回单个数值，接收 ``state`` 参数。
        - ``train``: 定义状态学习逻辑，返回训练后的状态对象。

    泛型参数:
        C: Config 类型，应继承 ``WindowFeatureConfig``。
        FS: 状态类型，绑定 ``BaseModel | None``，实际应为 ``BaseModel`` 子类。
    """
    pass
