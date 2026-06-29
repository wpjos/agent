# -*- coding: utf-8 -*-

"""
CICADA 异常检测算子

基于 CICADA（Continual Learning via Incremental Component Adaptive Architecture）
重构误差的异常检测。CICADA 采用 Mixture-of-Experts + MAML 元学习 + 动态架构扩展，
支持多种异构专家编码器（GradPCA, GradKPCA, GradSFA, MLP 等）的自适应组合。

包含:
    - CICADAPredictor: CICADA 重构型预测器，训练后输出重构值
    - CICADAScorer: CICADA 异常评分器，组合 CICADAPredictor 与 ResidualScorer
      产出 1D 异常分数及逐变量重构值/分数

示例用法::

    # 训练 + 推理
    predictor = CICADAPredictor(config=CICADAPredictorConfig(name=["MLP"], win_size=10, num_channels=3, epochs=5))
    predictor.fit(train_data)
    recon = predictor.run(test_data)

    # 训练 + 异常评分（重构残差）
    scorer = CICADAScorer(config=CICADAScorerConfig(name=["MLP"], win_size=10, num_channels=3, epochs=5, metric="mse"))
    scorer.fit(train_data)
    scores, eo = scorer.run(test_data)
    # scores: shape (n_samples,) 的 1D 异常分数
    # eo.feature_recon: shape (n_samples, n_vars) 的逐变量重构值
    # eo.feature_scores: shape (n_samples, n_vars) 的逐变量残差分数
"""

import json
from enum import Enum
from pathlib import Path

import numpy as np
import pandas as pd
from pydantic import BaseModel, ConfigDict, Field

from tsas.engine.operator.base import NumericOperator, UnsupervisedNumericOperatorMixin
from tsas.engine.operator.detection.base import BasePredictorMixin, SingleScorerMixin
from tsas.engine.operator.detection.residual_scorer import ResidualMetric, ResidualScorer, ResidualScorerConfig

__all__ = [
    'CICADANormalization',
    'CICADAInferMode',
    'CICADAPredictorConfig',
    'CICADAPredictor',
    'CICADAScorerConfig',
    'CICADAScorerExtraOutput',
    'CICADAScorer',
]


class CICADANormalization(str, Enum):
    """CICADA 归一化方式枚举

    定义 CICADA 编码器内部使用的归一化方法。

    Attributes:
        NONE: 不做归一化
        BATCH_NORM: 批量归一化（Batch Normalization）
        LAYER_NORM: 层归一化（Layer Normalization）
    """
    NONE = "None"
    """不做归一化"""
    BATCH_NORM = "BatchNorm"
    """批量归一化（Batch Normalization）"""
    LAYER_NORM = "LayerNorm"
    """层归一化（Layer Normalization）"""


class CICADAInferMode(str, Enum):
    """CICADA 推理模式枚举

    定义 CICADA 模型推理时使用的工作模式。

    Attributes:
        OFFLINE: 离线推理模式，使用完整的训练后模型进行推理
        ONLINE: 在线推理模式，支持推理时的快速适应
    """
    OFFLINE = "offline"
    """离线推理模式"""
    ONLINE = "online"
    """在线推理模式"""


class CICADAPredictorConfig(BaseModel):
    """
    CICADA 预测器实例参数

    覆盖 CICADA 构造函数的全部参数。``num_channels`` 为 ``None`` 时自动从训练数据推断。

    Attributes:
        name (list[str]): 专家模型名称列表，如 ``["GradPCA", "GradKPCA"]``
        win_size (int): 滑动窗口长度，必须大于 0
        stride (int): 训练滑动步长，必须大于 0
        num_channels (int | None): 输入特征维度；``None`` 时从训练数据自动推断
        batch_size (int): 训练批大小，必须大于 0
        epochs (int): 训练轮数，必须大于 0
        latent_space_size (int): 隐空间维度，必须大于 0
        n_components (str | int): 降维分量数，``"auto"`` 为自动选择
        normalization (CICADANormalization): 归一化方式，默认不做归一化
        ar_order (int): 自回归阶数，必须大于 0
        attn_bucket_heads (int): 桶注意力头数，必须大于 0
        decoder_all_heads (int): 全局注意力头数，必须大于 0
        forward_expansion (int): FFN 扩展因子，必须大于 0
        train_init_meta_lr (float): 训练初始元学习率，必须大于 0
        test_meta_lr (float): 测试元学习率，必须大于 0
        meta_split_threshold (float): 专家分裂阈值，必须大于 0
        lr_split_factor (float): 分裂学习率因子，必须大于 1.0
        ml_lambda (float): 专家损失权重，必须大于 0
        penalty_rate (float): 元学习率正则化系数，不小于 0
        lr (float): 基础学习率，必须大于 0
        ttlr (float): 测试时学习率，必须大于 0
        gamma (float): 衰减因子，必须大于 0 且小于 1
        adaptive_add (bool): 是否动态扩展专家
        epoch_add (int): 扩展检查间隔（轮数），必须大于 0
        close_epochs (int): 停止扩展提前量（轮数），不小于 0
        valid_size (float | None): 验证集比例；``None`` 表示不划分，取值范围 [0, 1)
        shuffle (bool): 是否打乱训练数据
        infer_mode (CICADAInferMode): 推理模式
        th (float): 异常检测百分位阈值，必须大于 0 且不大于 1
    """

    model_config = ConfigDict(frozen=True)

    # -- 专家配置 --
    name: list[str] = Field(
        default=["GradPCA", "GradKPCA", "GradFreKPCA", "GradSubPCA"],
        description="专家模型名称列表",
    )

    # -- 窗口 / 数据形状 --
    win_size: int = Field(default=5, ge=1, description="滑动窗口长度")
    stride: int = Field(default=1, ge=1, description="训练滑动步长")
    num_channels: int | None = Field(
        default=None,
        description="输入特征维度；None 时从训练数据自动推断",
    )

    # -- 训练 --
    batch_size: int = Field(default=256, ge=1, description="训练批大小")
    epochs: int = Field(default=60, ge=1, description="训练轮数")

    # -- 编码器架构 --
    latent_space_size: int = Field(default=128, ge=1, description="隐空间维度")
    n_components: str | int = Field(
        default="auto",
        description="降维分量数，'auto' 为自动选择",
    )
    normalization: CICADANormalization = Field(
        default=CICADANormalization.NONE,
        description="归一化方式: 'None'(不做归一化)、'BatchNorm'(批量归一化)、'LayerNorm'(层归一化)",
    )
    ar_order: int = Field(default=2, ge=1, description="自回归阶数")

    # -- 注意力 --
    attn_bucket_heads: int = Field(default=4, ge=1, description="桶注意力头数")
    decoder_all_heads: int = Field(default=8, ge=1, description="全局注意力头数")
    forward_expansion: int = Field(default=4, ge=1, description="FFN 扩展因子")

    # -- 元学习 --
    train_init_meta_lr: float = Field(default=1e-4, gt=0.0, description="训练初始元学习率")
    test_meta_lr: float = Field(default=1e-3, gt=0.0, description="测试元学习率")
    meta_split_threshold: float = Field(default=5e-4, gt=0.0, description="专家分裂阈值")
    lr_split_factor: float = Field(default=1.414, gt=1.0, description="分裂学习率因子")
    ml_lambda: float = Field(default=10.0, gt=0.0, description="专家损失权重")
    penalty_rate: float = Field(default=0.01, ge=0.0, description="元学习率正则化系数")

    # -- 优化器 --
    lr: float = Field(default=1e-3, gt=0.0, description="基础学习率")
    ttlr: float = Field(default=1e-3, gt=0.0, description="测试时学习率")
    gamma: float = Field(default=0.99, gt=0.0, lt=1.0, description="衰减因子")

    # -- 动态扩展 --
    adaptive_add: bool = Field(default=True, description="是否动态扩展专家")
    epoch_add: int = Field(default=10, ge=1, description="扩展检查间隔（轮数）")
    close_epochs: int = Field(default=20, ge=0, description="停止扩展提前量（轮数）")

    # -- 其他 --
    valid_size: float | None = Field(
        default=None, ge=0.0, lt=1.0,
        description="验证集比例；None 表示不划分",
    )
    shuffle: bool = Field(default=False, description="是否打乱训练数据")

    # -- 推理 --
    infer_mode: CICADAInferMode = Field(default=CICADAInferMode.OFFLINE, description="推理模式")
    th: float = Field(default=0.98, gt=0.0, le=1.0, description="异常检测百分位阈值")


class CICADAPredictor(UnsupervisedNumericOperatorMixin[None],
                      BasePredictorMixin[None, CICADAPredictorConfig, None],
                      NumericOperator[None, CICADAPredictorConfig, None]):
    """
    CICADA 重构型预测器

    基于 CICADA 算法的重构型预测器。训练阶段通过 Mixture-of-Experts + MAML 元学习
    学习数据的重构表示，推理阶段输出与输入同维度的重构值。

    内部数据流::

        输入 x → CICADA.fit(x) → 训练 MoE 模型
        输入 x → CICADA.reconstruct(x) → 重构值 x_pred（与 x 同形状）

    核心逻辑:
        - ``_fit_data``: 创建并训练 CICADA 模型，自动推断 num_channels
        - ``_run_data``: 调用 CICADA reconstruct 返回重构值

    注意:
        - CICADA 包为延迟导入（``from cicada import CICADA``），使用前需确保已安装 cicada-ad
        - ``num_channels`` 在 Config 中为 ``None`` 时自动从训练数据推断
        - 输入数据长度需 >= ``win_size``，否则无法创建滑动窗口
        - 输入数据会被自动转换为 float32 以适配 CICADA 内部要求

    Input:
        x: 二维时序数据，形状 (n_samples, n_features)，每列为一个特征通道

    Output:
        重构值矩阵，与输入同形状

    泛型参数:
        - EO: ``None``（无附加输出）
        - C: :class:`CICADAPredictorConfig`
        - RP: ``None``（无运行参数）
        - FP: ``None``（无训练参数）

    Attributes:
        _model (cicada.CICADA | None): CICADA 模型实例，训练后非 None
        _num_channels_detected (int | None): 从训练数据推断的特征维度；
            仅当 ``config.num_channels`` 为 ``None`` 且已完成训练时有值
    """

    _MODEL_FILE = "cicada_model.pt"
    _META_FILE = "cicada_meta.json"

    @classmethod
    def name(cls) -> str:
        return "cicada_predictor"

    @classmethod
    def version(cls) -> tuple[int, ...]:
        """返回算子版本号

        Returns:
            tuple[int, ...]: 版本号三元组 ``(1, 0, 0)``
        """
        return (1, 0, 0)

    def __init__(self, *, oid: str | None = None, config: CICADAPredictorConfig | None = None, **kwargs):
        """
        初始化 CICADA 预测器

        Args:
            oid (str | None): 算子实例唯一标识后缀，缺省自动生成
            config (CICADAPredictorConfig | None): 类型化实例参数，优先级高于键值对参数
            **kwargs: 实例参数键值对，最终用于构造 :class:`CICADAPredictorConfig`；
                常用项包括 ``name``、``win_size``、``num_channels``、``epochs`` 等
        """
        super().__init__(oid=oid, config=config, **kwargs)
        self._model = None
        self._num_channels_detected: int | None = None

    # ------------------------------------------------------------------
    # 校验
    # ------------------------------------------------------------------

    def _validate_ndarray_input(self, x: np.ndarray, params) -> None:
        """
        校验 ndarray 输入的维度和行数

        CICADA 要求输入为 2D 数组且行数不小于 ``win_size``，
        否则无法构建滑动窗口进行重构。

        Args:
            x (np.ndarray): 输入 ndarray
            params (None): 无运行参数

        Raises:
            ValueError: 输入非 2D 时，或行数不足 ``win_size`` 时
        """
        if x.ndim != 2:
            raise ValueError(f"CICADAPredictor 要求 2D 输入，收到 {x.ndim}D")
        if x.shape[0] < self.config.win_size:
            raise ValueError(
                f"CICADAPredictor 要求输入行数 >= win_size={self.config.win_size}，"
                f"收到 {x.shape[0]} 行"
            )

    # ------------------------------------------------------------------
    # 训练
    # ------------------------------------------------------------------

    def _fit_data(self, x: np.ndarray, params: None) -> None:
        """
        训练 CICADA 重构模型

        创建 CICADA 实例并执行训练。训练流程包括：
        1. 校验输入维度与行数（``_validate_ndarray_input`` 的补充校验）
        2. 推断 ``num_channels``（Config 中为 ``None`` 时从数据自动推断）
        3. 将 Config 全部参数透传构造 CICADA 模型
        4. 将输入转为 float32 后调用 ``CICADA.fit`` 完成训练

        Args:
            x (np.ndarray): 训练数据，形状 ``(n_samples, n_features)``
            params (None): 无训练参数

        Raises:
            ValueError: 输入非 2D 或行数不足 ``win_size`` 时
        """
        try:
            from cicada import CICADA  # noqa: lazy import
        except ImportError:
            raise RuntimeError("请先安装 cicada-ad 包") from None

        # 校验输入维度（fit 管线不做此检查，需在 _fit_data 内部校验）
        if x.ndim != 2:
            raise ValueError(f"CICADAPredictor 要求 2D 输入，收到 {x.ndim}D")
        if x.shape[0] < self.config.win_size:
            raise ValueError(
                f"CICADAPredictor 要求输入行数 >= win_size={self.config.win_size}，"
                f"收到 {x.shape[0]} 行"
            )

        # 推断 num_channels：Config 中为 None 时取训练数据的特征维度
        num_channels = (
            self.config.num_channels
            if self.config.num_channels is not None
            else x.shape[1]
        )
        self._num_channels_detected = num_channels

        # 将 Config 全部参数透传构造 CICADA 模型实例
        self._model = CICADA(
            name=list(self.config.name),
            win_size=self.config.win_size,
            stride=self.config.stride,
            num_channels=num_channels,
            batch_size=self.config.batch_size,
            epochs=self.config.epochs,
            latent_space_size=self.config.latent_space_size,
            n_components=self.config.n_components,
            train_init_meta_lr=self.config.train_init_meta_lr,
            test_meta_lr=self.config.test_meta_lr,
            meta_split_threshold=self.config.meta_split_threshold,
            gamma=self.config.gamma,
            normalization=self.config.normalization,
            ar_order=self.config.ar_order,
            attn_bucket_heads=self.config.attn_bucket_heads,
            decoder_all_heads=self.config.decoder_all_heads,
            forward_expansion=self.config.forward_expansion,
            lr_split_factor=self.config.lr_split_factor,
            lr=self.config.lr,
            ttlr=self.config.ttlr,
            ml_lambda=self.config.ml_lambda,
            penalty_rate=self.config.penalty_rate,
            adaptive_add=self.config.adaptive_add,
            epoch_add=self.config.epoch_add,
            close_epochs=self.config.close_epochs,
            valid_size=self.config.valid_size,
            shuffle=self.config.shuffle,
            infer_mode=self.config.infer_mode,
            th=self.config.th,
        )

        # CICADA 内部要求 float32 输入，此处做类型转换以兼容 float64 等输入
        x_float32 = x.astype(np.float32) if x.dtype != np.float32 else x
        self._model.fit(x_float32)

    # ------------------------------------------------------------------
    # 推理
    # ------------------------------------------------------------------

    def _run_data(self, x: np.ndarray, params: None, idx: pd.Index | None = None) -> np.ndarray:
        """
        执行 CICADA 重构推理

        将输入数据转为 float32 后调用 ``CICADA.reconstruct``，
        返回与输入同形状的重构值。

        Args:
            x (np.ndarray): 输入数据，形状 ``(n_samples, n_features)``
            params (None): 无运行参数
            idx (pd.Index | None): 输入数据的行索引（当前未直接使用，由基类透传）

        Returns:
            np.ndarray: 重构值，形状与 ``x`` 相同
        """
        # CICADA 内部要求 float32 输入
        x_float32 = x.astype(np.float32) if x.dtype != np.float32 else x
        return self._model.reconstruct(x_float32)

    # ------------------------------------------------------------------
    # 持久化
    # ------------------------------------------------------------------

    def _save_fit_state(self, path: Path) -> None:
        """
        保存训练状态：训练参数 + CICADA 模型权重 + 元信息

        在基类保存 last_fit_params 的基础上，额外持久化：
        - CICADA 模型权重（``cicada_model.pt``，通过 ``torch.save`` 序列化）
        - 元信息文件（``cicada_meta.json``），记录 ``num_channels_detected``

        Args:
            path (Path): 目标目录路径
        """
        import torch  # noqa: lazy import

        super()._save_fit_state(path)

        # 保存 CICADA 模型权重（仅在训练后存在）
        if self._model is not None:
            torch.save(self._model, path / self._MODEL_FILE)

        # 保存元信息：自动推断的 num_channels 等运行时状态
        meta = {"num_channels_detected": self._num_channels_detected}
        (path / self._META_FILE).write_text(json.dumps(meta), encoding="utf-8")

    def _load_fit_state(self, path: Path) -> None:
        """
        恢复训练状态：训练参数 + CICADA 模型权重 + 元信息

        从磁盘恢复 CICADA 模型权重和 ``num_channels_detected`` 等运行时状态，
        并将 ``_fitted`` 标记为 True。

        Args:
            path (Path): 源目录路径
        """
        import torch  # noqa: lazy import

        super()._load_fit_state(path)

        # 恢复 CICADA 模型权重
        model_file = path / self._MODEL_FILE
        if model_file.exists():
            self._model = torch.load(model_file, weights_only=False)

        # 恢复元信息
        meta_file = path / self._META_FILE
        if meta_file.exists():
            meta = json.loads(meta_file.read_text(encoding="utf-8"))
            self._num_channels_detected = meta.get("num_channels_detected")

        # 模型权重恢复成功时，同步训练状态标志
        if self._model is not None:
            self._fitted = True


# ============================================================================
# CICADA 异常评分器
# ============================================================================


class CICADAScorerConfig(CICADAPredictorConfig):
    """
    CICADA 评分器实例参数

    继承 :class:`CICADAPredictorConfig` 的全部字段（覆盖 CICADA 模型的所有超参），
    并额外引入 ``metric`` 字段用于控制内部 :class:`ResidualScorer` 的残差度量方式。

    通过类继承复用 Predictor 配置，可保证两个算子的 CICADA 超参语义和默认值始终一致；
    新增字段仅一项，避免代码重复。

    Attributes:
        metric (Literal["mse", "mae"]): 残差计算方式。``"mse"`` 为均方误差（平方残差），
            ``"mae"`` 为平均绝对误差（绝对残差），默认 ``"mse"``。
            其余字段含义参见 :class:`CICADAPredictorConfig`
    """

    # 显式声明 model_config：保持 frozen 不可变
    model_config = ConfigDict(frozen=True)

    # -- 评分配置 --
    metric: ResidualMetric = Field(
        default=ResidualMetric.MSE,
        description="残差计算方式: 'mse' 为均方误差, 'mae' 为平均绝对误差",
    )


class CICADAScorerExtraOutput(BaseModel):
    """
    CICADA 评分器附加输出

    扁平化结构，参考 :class:`tsas.engine.operator.detection.xihe.XiHeGammaScorerExtraOutput`
    的设计风格。直接展开关键中间量为顶层字段，便于下游直接消费，不再嵌套子 EO。

    Attributes:
        feature_recon (np.ndarray): 逐变量重构值，形状 ``(n_samples, n_vars)``，
            来自内部 :class:`CICADAPredictor` 的 ``reconstruct`` 输出
        feature_scores (np.ndarray): 逐变量异常分数，形状 ``(n_samples, n_vars)``，
            为根据 ``metric`` 计算的逐变量残差
    """

    # 允许 BaseModel 容纳 numpy ndarray
    model_config = ConfigDict(arbitrary_types_allowed=True)

    feature_recon: np.ndarray = Field(description="逐变量重构值，形状为 (n_samples, n_vars)，列顺序与输入一致")
    """逐变量重构值，形状 (n_samples, n_vars)"""

    feature_scores: np.ndarray = Field(description="逐变量异常分数，形状为 (n_samples, n_vars)，由重构值与原始输入按 metric 计算得到")
    """逐变量异常分数，形状 (n_samples, n_vars)"""


class CICADAScorer(SingleScorerMixin[None],
                   UnsupervisedNumericOperatorMixin[None],
                   NumericOperator[CICADAScorerExtraOutput, CICADAScorerConfig, None]):
    """
    CICADA 异常评分器 — 组合 CICADAPredictor + ResidualScorer

    基于 CICADA 重构误差的异常评分器。沿用 :class:`PCAScorer` 同款 "Predictor + ResidualScorer"
    组合范式：

    内部数据流::

        输入 x → CICADAPredictor.run(x) → 重构值 x_pred
              → ResidualScorer.run((x, x_pred)) → (1D 分数, ResidualScorerExtraOutput)
              → 输出: (1D 异常分数, CICADAScorerExtraOutput)

    训练阶段仅训练 :class:`CICADAPredictor`（:class:`ResidualScorer` 为 BiNumericOperator
    无需训练）。

    Input:
        x: 二维时序数据，形状 (n_samples, n_features)，每列为一个特征通道

    Output:
        异常分数，形状 (n_samples,)，值越大越异常。
        分数由内部 CICADAPredictor 的重构值与原始输入的残差经 ResidualScorer 聚合得到

    泛型参数:
        - EO: :class:`CICADAScorerExtraOutput`（附加输出由 ``_eo_type`` 自动渲染）
        - C: :class:`CICADAScorerConfig`
        - RP: ``None``（无运行参数）
        - FP: ``None``（无训练参数）

    Attributes:
        _predictor (CICADAPredictor): 内部持有的 CICADA 重构预测器
        _scorer (ResidualScorer): 内部持有的残差评分器
    """

    # 持久化子目录名
    _PREDICTOR_DIR = "predictor"

    @classmethod
    def name(cls) -> str:
        """
        返回算子名称标识

        Returns:
            str: 固定返回 ``"cicada_scorer"``
        """
        return "cicada_scorer"

    @classmethod
    def version(cls) -> tuple[int, ...]:
        """返回算子版本号

        Returns:
            tuple[int, ...]: 版本号三元组 ``(1, 0, 0)``
        """
        return (1, 0, 0)

    def __init__(self, *, oid: str | None = None,
                 config: CICADAScorerConfig | None = None, **kwargs):
        """
        初始化 CICADA 评分器

        从 Scorer 配置中剥离 ``metric`` 字段后，剩余字段全部交给
        :class:`CICADAPredictor`；同时根据 ``metric`` 构造
        :class:`ResidualScorer`。

        Args:
            oid (str | None): 算子实例唯一标识后缀，缺省自动生成
            config (CICADAScorerConfig | None): 类型化实例参数，优先级高于键值对参数
            **kwargs: 透传给基类的实例参数键值对
        """
        super().__init__(oid=oid, config=config, **kwargs)

        # 从 Scorer 配置中剥离 metric 字段，剩余字段全部交给 CICADAPredictor
        predictor_kwargs = self.config.model_dump(exclude={"metric"})
        self._predictor: CICADAPredictor = CICADAPredictor(
            config=CICADAPredictorConfig(**predictor_kwargs)
        )
        """内部 CICADAPredictor，承担模型训练与重构推理"""

        self._scorer: ResidualScorer = ResidualScorer(
            config=ResidualScorerConfig(metric=self.config.metric)
        )
        """内部 ResidualScorer，根据 metric 计算逐变量残差并聚合为 1D 异常分数"""

    # ------------------------------------------------------------------
    # 训练
    # ------------------------------------------------------------------

    def _fit_data(self, x: np.ndarray, params: None) -> None:
        """
        训练 CICADA 评分器

        仅训练内部 :class:`CICADAPredictor`（:class:`ResidualScorer` 是双输入无状态算子，
        无需训练）。

        Args:
            x (np.ndarray): 训练数据，形状 ``(n_samples, n_features)``
            params (None): 无训练参数
        """
        # 委托给 Predictor 进行实际训练
        self._predictor.fit(x)

    # ------------------------------------------------------------------
    # 推理
    # ------------------------------------------------------------------

    def _run_data(self, x: np.ndarray, params: None,
                  idx: pd.Index | None = None) -> tuple[np.ndarray, CICADAScorerExtraOutput]:
        """
        计算 CICADA 重构误差异常分数

        执行流程：
        1. 调用 :class:`CICADAPredictor` 获取重构值 ``x_pred``
        2. 将 ``(x, x_pred)`` 喂入 :class:`ResidualScorer`，得到 1D 异常分数与逐变量残差
        3. 将逐变量重构值与逐变量分数展开打包到 :class:`CICADAScorerExtraOutput`

        Args:
            x (np.ndarray): 输入数据，形状 ``(n_samples, n_features)``
            params (None): 无运行参数
            idx (pd.Index | None): 输入数据的行索引

        Returns:
            tuple[np.ndarray, CICADAScorerExtraOutput]:
                - scores (np.ndarray): 1D 异常分数，形状 ``(n_samples,)``
                - eo (CICADAScorerExtraOutput): 附加输出，含逐变量重构值与残差分数
        """
        # 步骤 1：CICADA 重构
        x_pred = self._predictor.run(x)

        # 步骤 2：残差打分
        scores, residual_eo = self._scorer.run((x, x_pred))

        # 步骤 3：构造扁平化 EO
        eo = CICADAScorerExtraOutput(
            feature_recon=np.asarray(x_pred),
            feature_scores=np.asarray(residual_eo.scores),
        )
        return scores.ravel(), eo

    # ------------------------------------------------------------------
    # 持久化
    # ------------------------------------------------------------------

    def _save_fit_state(self, path: Path) -> None:
        """
        保存训练状态：训练参数 + 子组件状态

        在基类保存 last_fit_params 的基础上，委托 :class:`CICADAPredictor` 的
        ``save`` 将 torch 模型权重和元信息持久化到 ``predictor/`` 子目录。
        :class:`ResidualScorer` 无状态，无需保存。

        Args:
            path (Path): 目标目录路径
        """
        super()._save_fit_state(path)
        # 委托内部 Predictor 落盘 torch 权重到 predictor/ 子目录
        self._predictor.save(path / self._PREDICTOR_DIR)

    def _load_fit_state(self, path: Path) -> None:
        """
        恢复训练状态：训练参数 + 子组件状态

        从 ``predictor/`` 子目录恢复已训练的 :class:`CICADAPredictor` 实例，
        覆盖 ``__init__`` 中创建的占位实例，并将 ``_fitted`` 标记为 True。

        Args:
            path (Path): 源目录路径
        """
        super()._load_fit_state(path)
        # 恢复 Predictor（含 torch 权重与 num_channels_detected 等内部状态）
        self._predictor = CICADAPredictor.load(path / self._PREDICTOR_DIR)
        # Predictor 已就绪时，同步评分器的训练态
        if self._predictor.is_fitted:
            self._fitted = True
