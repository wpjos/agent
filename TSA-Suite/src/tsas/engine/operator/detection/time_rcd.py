# -*- coding: utf-8 -*-

"""
Time-RCD 零样本异常检测算子（Scorer / Predictor）

将 ``bq_rcd`` 的零样本推理接口封装为 TSA-Suite 算子：

- :class:`TimeRCDScorer`（SingleScorer / 第 2 层）封装
  ``TimeRCDPretrainTester.zero_shot``，输出形状 ``(N,)`` 的逐时刻异常分数；
- :class:`TimeRCDPredictor`（重构型 Predictor / 第 1 层）封装
  ``TimeRCDPretrainTester.zero_shot_reconstruct``，输出与输入同形状的重建序列。

与 cicada 等"学统计量"算子的本质差异：
    - Time-RCD 是预训练零样本模型，``fit()`` **不更新权重、不学统计量**，
      仅完成"加载预训练 checkpoint + 推断特征维度"的资源准备。
    - ``run()`` 每次对当次输入做一次 Z-score（走 ndarray 默认路径），
      不持久化任何归一化参数，与原版 ``TimeRCDDataset(normalize=True)`` 行为一致。
    - ``save/load`` 仅持久化少量元信息（特征维度、checkpoint 路径），
      模型权重靠 HuggingFace Hub 缓存恢复。

模式切换以 **运行参数** 暴露：

- ``TimeRCDScorerRunParams.score_form``：``"prob"``（默认）/``"logit"``
- ``TimeRCDPredictorRunParams.denormalize``：``True``（默认，回到原始量纲）/``False``

示例用法::

    # Scorer
    scorer = TimeRCDScorer(win_size=200, batch_size=2)
    scorer.fit(train_data)
    scores = scorer.run(test_data)                    # 默认 prob
    logits = scorer.run(test_data, score_form="logit")

    # Predictor
    predictor = TimeRCDPredictor(win_size=200, batch_size=2)
    predictor.fit(train_data)
    recon = predictor.run(test_data)                  # 默认反归一化
    recon_norm = predictor.run(test_data, denormalize=False)
"""

import json
import warnings
from pathlib import Path
from typing import Literal, Self

import numpy as np
import pandas as pd
from pydantic import BaseModel, ConfigDict, Field

from tsas.engine.operator.base import (
    NumericOperator,
    UnsupervisedNumericOperatorMixin,
)
from tsas.engine.operator.detection.base import BasePredictor, SingleScorerMixin

__all__ = [
    "TimeRCDScorerConfig",
    "TimeRCDScorerRunParams",
    "TimeRCDScorer",
    "TimeRCDPredictorConfig",
    "TimeRCDPredictorRunParams",
    "TimeRCDPredictor",
]


# ============================================================================
# 共享辅助：从 Config 构造 TimeRCDPretrainTester（避免 Scorer/Predictor 重复实现）
# ============================================================================


def _build_tester(num_features: int, config: "TimeRCDScorerConfig | TimeRCDPredictorConfig"):
    """根据已知 num_features 与算子 config 构造 ``TimeRCDPretrainTester``。

    Args:
        num_features: 输入通道数（必须已确定，构造 ``TimeSeriesConfig`` 需要）。
        config: Scorer/Predictor 的实例参数，提供 ``win_size`` / ``batch_size``
            / ``patch_size`` / ``checkpoint`` 等字段。

    Returns:
        已加载默认（或指定）权重、可直接推理的 ``TimeRCDPretrainTester`` 实例。
    """
    try:
        from bq_rcd.time_rcd import TimeRCDPretrainTester
        from bq_rcd.time_rcd.time_rcd_config import TimeRCDConfig, TimeSeriesConfig
    except ImportError:
        raise RuntimeError("请先安装 bq-rcd 包") from None

    ts_config = TimeSeriesConfig(
        patch_size=config.patch_size,
        num_features=num_features,
    )
    rcd_config = TimeRCDConfig(
        ts_config=ts_config,
        batch_size=config.batch_size,
        win_size=config.win_size,
    )
    return TimeRCDPretrainTester(
        checkpoint_path=config.checkpoint,
        config=rcd_config,
    )


# ============================================================================
# Scorer
# ============================================================================


class TimeRCDScorerConfig(BaseModel):
    """Time-RCD 评分器实例参数

    Attributes:
        win_size: 滑动窗口长度（推理用非重叠窗口拼分数）
        batch_size: 推理批大小
        patch_size: patch 大小，须与 checkpoint 匹配（HF 上的官方权重为 16）
        num_features: 输入特征通道数；None 时由 fit 阶段自动从数据推断
        checkpoint: 本地 checkpoint 路径；None 时自动从 HF Hub 下载并缓存
    """

    model_config = ConfigDict(frozen=True)

    win_size: int = Field(default=5000, gt=0, description="滑动窗口长度")
    batch_size: int = Field(default=64, gt=0, description="推理批大小")
    patch_size: int = Field(default=16, gt=0, description="patch 大小，须匹配 checkpoint")
    num_features: int | None = Field(
        default=None,
        description="输入特征通道数；None 时 fit 阶段自动推断",
    )
    checkpoint: str | None = Field(
        default=None,
        description="checkpoint 路径；None 时自动从 HF Hub 下载",
    )


class TimeRCDScorerRunParams(BaseModel):
    """Time-RCD 评分器运行参数

    Attributes:
        score_form: 异常分数输出形式：
            - ``"prob"``（默认）：softmax 后的异常概率 ∈ [0, 1]
            - ``"logit"``：logit_1 - logit_0，取值 (-∞, +∞)
    """

    score_form: Literal["prob", "logit"] = Field(
        default="prob",
        description="输出分数形式：prob=softmax 概率 / logit=log-odds 差",
    )


class TimeRCDScorer(
    SingleScorerMixin[TimeRCDScorerRunParams],
    UnsupervisedNumericOperatorMixin[None],
    NumericOperator[None, TimeRCDScorerConfig, TimeRCDScorerRunParams],
):
    """Time-RCD 零样本异常分数评分器

    基于 ``bq_rcd.time_rcd.TimeRCDPretrainTester`` 的零样本异常检测：

    - ``_fit_data``: 推断 num_features（None 时取 ``x.shape[1]``）→ 构造 tester
      （触发 HF Hub 下载或本地 checkpoint 解析）。**不更新权重、不学统计量**。
    - ``_run_data``: 直接将 ndarray 传给 ``tester.zero_shot``，由其内部对当次
      输入做一次 Z-score。``score_form`` 通过 RunParams 在每次 ``run()`` 时切换，
      默认 ``"prob"``。

    输出:
        - 异常分数 ndarray，形状 ``(N,)``，其中 N == 输入长度。
        - DataFrame 输入时，输出列名为 ``["score"]``。

    泛型参数:
        - EO: None（无附加输出）
        - C: TimeRCDScorerConfig
        - RP: TimeRCDScorerRunParams
    """

    _META_FILE = "time_rcd_meta.json"

    @classmethod
    def name(cls) -> str:
        return "time_rcd_scorer"

    def __init__(
        self,
        *,
        oid: str | None = None,
        config: TimeRCDScorerConfig | None = None,
        **kwargs,
    ) -> None:
        """初始化 Time-RCD 评分器。

        Args:
            oid: 算子实例唯一标识后缀
            config: 类型化实例参数
            **kwargs: 透传给 Config 的字段（win_size、batch_size、num_features 等）
        """
        super().__init__(oid=oid, config=config, **kwargs)
        self._tester = None
        self._num_features_detected: int | None = None
        self._checkpoint_path_resolved: str | None = None

    # ------------------------------------------------------------------
    # 校验
    # ------------------------------------------------------------------

    def _validate_ndarray_input(self, x: np.ndarray, params: TimeRCDScorerRunParams | None) -> None:
        if x.ndim != 2:
            raise ValueError(
                f"TimeRCDScorer 要求 2D 输入（n_samples, n_features），收到 {x.ndim}D",
            )
        if x.shape[0] < self.config.win_size:
            raise ValueError(
                f"TimeRCDScorer 要求输入行数 >= win_size={self.config.win_size}，"
                f"收到 {x.shape[0]} 行",
            )

    # ------------------------------------------------------------------
    # 训练（实为加载预训练 checkpoint + 推断维度，不更新权重）
    # ------------------------------------------------------------------

    def _fit_data(self, x: np.ndarray, params: None) -> None:
        """加载预训练权重 + 推断特征维度。

        本方法名为 ``_fit_data`` 仅为满足 TSA-Suite 框架"必须先 fit 再 run"的契约。
        Time-RCD 是预训练零样本模型，**不会**在此处更新模型权重，
        也**不会**学习任何归一化统计量。

        Args:
            x: 训练/校准数据，形状 (n_samples, n_features)
            params: 无训练参数
        """
        if x.ndim != 2:
            raise ValueError(
                f"TimeRCDScorer 要求 2D 输入（n_samples, n_features），收到 {x.ndim}D",
            )
        if x.shape[0] < self.config.win_size:
            raise ValueError(
                f"TimeRCDScorer 要求输入行数 >= win_size={self.config.win_size}，"
                f"收到 {x.shape[0]} 行",
            )

        num_features = (
            self.config.num_features
            if self.config.num_features is not None
            else x.shape[1]
        )
        self._num_features_detected = num_features
        self._tester = _build_tester(num_features, self.config)
        self._checkpoint_path_resolved = self._tester.checkpoint_path

    # ------------------------------------------------------------------
    # 推理
    # ------------------------------------------------------------------

    def _run_data(
        self,
        x: np.ndarray,
        params: TimeRCDScorerRunParams | None,
        idx: pd.Index | None = None,
    ) -> np.ndarray:
        """对输入做零样本异常分数推理。

        每次调用都会让 ``zero_shot`` 内部对当次输入做一次 Z-score，
        不复用任何"训练时"的统计量。

        Args:
            x: 输入数据，形状 (n_samples, n_features)
            params: 运行参数，``score_form`` 默认为 ``"prob"``
            idx: DataFrame 输入时的行索引（此处未使用）

        Returns:
            np.ndarray: 异常分数，形状 (n_samples,)
        """
        score_form = self._resolve_param(params, "score_form", default="prob")
        x_float32 = x.astype(np.float32) if x.dtype != np.float32 else x
        with warnings.catch_warnings():
            # zero_shot 对 ndarray 输入会发 UserWarning 提示"将临时 fit 一组 Z-score"，
            # 这正是我们想要的零样本默认行为，无需冒泡到用户。
            warnings.simplefilter("ignore", UserWarning)
            scores = self._tester.zero_shot(x_float32, score_form=score_form)
        return np.asarray(scores)

    # ------------------------------------------------------------------
    # 持久化（仅落元信息，模型权重靠 HF 缓存恢复）
    # ------------------------------------------------------------------

    def save(self, path: str | Path) -> None:
        """保存算子状态到目录（仅 config + 元信息）。

        Args:
            path: 目标目录路径
        """
        path = Path(path)
        super().save(path)

        meta = {
            "num_features_detected": self._num_features_detected,
            "checkpoint_path_resolved": self._checkpoint_path_resolved,
        }
        (path / self._META_FILE).write_text(
            json.dumps(meta, ensure_ascii=False),
            encoding="utf-8",
        )

    @classmethod
    def load(cls, path: str | Path, *, oid: str | None = None) -> Self:
        """从目录恢复算子（重建 tester，HF 命中缓存即秒级返回）。

        Args:
            path: 源目录路径
            oid: 算子实例唯一标识后缀

        Returns:
            恢复后的 TimeRCDScorer 实例
        """
        path = Path(path)
        instance = super().load(path, oid=oid)

        meta_file = path / cls._META_FILE
        if meta_file.exists():
            meta = json.loads(meta_file.read_text(encoding="utf-8"))
            instance._num_features_detected = meta.get("num_features_detected")
            instance._checkpoint_path_resolved = meta.get("checkpoint_path_resolved")

        if instance._num_features_detected is not None:
            instance._tester = _build_tester(
                instance._num_features_detected, instance.config,
            )
            # 优先尊重 config.checkpoint，否则回填 save 时解析过的路径
            if instance.config.checkpoint is None and instance._checkpoint_path_resolved:
                instance._checkpoint_path_resolved = instance._tester.checkpoint_path
            instance._fitted = True

        return instance


# ============================================================================
# Predictor（reconstruction）
# ============================================================================


class TimeRCDPredictorConfig(BaseModel):
    """Time-RCD 重建预测器实例参数

    字段含义与 :class:`TimeRCDScorerConfig` 一致：均为构造 backbone tester 所需的
    超参（win_size、batch_size、patch_size、num_features、checkpoint）。
    """

    model_config = ConfigDict(frozen=True)

    win_size: int = Field(default=5000, gt=0, description="滑动窗口长度")
    batch_size: int = Field(default=64, gt=0, description="推理批大小")
    patch_size: int = Field(default=16, gt=0, description="patch 大小，须匹配 checkpoint")
    num_features: int | None = Field(
        default=None,
        description="输入特征通道数；None 时 fit 阶段自动推断",
    )
    checkpoint: str | None = Field(
        default=None,
        description="checkpoint 路径；None 时自动从 HF Hub 下载",
    )


class TimeRCDPredictorRunParams(BaseModel):
    """Time-RCD 重建预测器运行参数

    Attributes:
        denormalize: 是否将重建结果反归一化回原始量纲。``True``（默认）等价于
            ``zero_shot_reconstruct(..., denormalize=True)``；``False`` 返回
            归一化空间下的重建。
    """

    denormalize: bool = Field(
        default=True,
        description="是否反归一化回原始量纲",
    )


class TimeRCDPredictor(
    UnsupervisedNumericOperatorMixin[None],
    BasePredictor[None, TimeRCDPredictorConfig, TimeRCDPredictorRunParams],
):
    """Time-RCD 零样本信号重建预测器

    走 ``reconstruction_head`` 的零样本重建：

    - ``_fit_data``: 与 Scorer 同样为"假 fit"，仅加载 checkpoint 并推断
      num_features，**不更新权重、不学统计量**。
    - ``_run_data``: 调用 ``tester.zero_shot_reconstruct``，输出与输入同维度的
      重建序列。``denormalize`` 通过 RunParams 切换，默认 ``True``。

    输出形状约定:
        - 输入 (N, C) → 输出 (N, C)，沿用输入列名（``BasePredictor`` 行为）。
        - bq_rcd 的 ``zero_shot_reconstruct`` 在 ``num_channels == 1`` 时会
          ravel 到 1D；这里统一 reshape 回 (N, 1) 以满足 NumericOperator 输出
          为 2D 的契约。

    泛型参数:
        - EO: None（无附加输出）
        - C: TimeRCDPredictorConfig
        - RP: TimeRCDPredictorRunParams
    """

    _META_FILE = "time_rcd_meta.json"

    @classmethod
    def name(cls) -> str:
        return "time_rcd_predictor"

    def __init__(
        self,
        *,
        oid: str | None = None,
        config: TimeRCDPredictorConfig | None = None,
        **kwargs,
    ) -> None:
        super().__init__(oid=oid, config=config, **kwargs)
        self._tester = None
        self._num_features_detected: int | None = None
        self._checkpoint_path_resolved: str | None = None

    # ------------------------------------------------------------------
    # 校验
    # ------------------------------------------------------------------

    def _validate_ndarray_input(self, x: np.ndarray, params: TimeRCDPredictorRunParams | None) -> None:
        if x.ndim != 2:
            raise ValueError(
                f"TimeRCDPredictor 要求 2D 输入（n_samples, n_features），收到 {x.ndim}D",
            )
        if x.shape[0] < self.config.win_size:
            raise ValueError(
                f"TimeRCDPredictor 要求输入行数 >= win_size={self.config.win_size}，"
                f"收到 {x.shape[0]} 行",
            )

    # ------------------------------------------------------------------
    # 训练（同 Scorer：仅加载 checkpoint + 推断维度）
    # ------------------------------------------------------------------

    def _fit_data(self, x: np.ndarray, params: None) -> None:
        if x.ndim != 2:
            raise ValueError(
                f"TimeRCDPredictor 要求 2D 输入（n_samples, n_features），收到 {x.ndim}D",
            )
        if x.shape[0] < self.config.win_size:
            raise ValueError(
                f"TimeRCDPredictor 要求输入行数 >= win_size={self.config.win_size}，"
                f"收到 {x.shape[0]} 行",
            )

        num_features = (
            self.config.num_features
            if self.config.num_features is not None
            else x.shape[1]
        )
        self._num_features_detected = num_features
        self._tester = _build_tester(num_features, self.config)
        self._checkpoint_path_resolved = self._tester.checkpoint_path

    # ------------------------------------------------------------------
    # 推理
    # ------------------------------------------------------------------

    def _run_data(
        self,
        x: np.ndarray,
        params: TimeRCDPredictorRunParams | None,
        idx: pd.Index | None = None,
    ) -> np.ndarray:
        denormalize = self._resolve_param(params, "denormalize", default=True)
        x_float32 = x.astype(np.float32) if x.dtype != np.float32 else x
        with warnings.catch_warnings():
            warnings.simplefilter("ignore", UserWarning)
            recon = self._tester.zero_shot_reconstruct(
                x_float32,
                denormalize=bool(denormalize),
            )
        recon = np.asarray(recon)
        # 单变量场景 bq_rcd 会 ravel 到 1D；这里 reshape 回 (N, 1) 保持 2D
        if recon.ndim == 1:
            recon = recon.reshape(-1, 1)
        return recon

    # ------------------------------------------------------------------
    # 持久化（仅落元信息）
    # ------------------------------------------------------------------

    def save(self, path: str | Path) -> None:
        path = Path(path)
        super().save(path)
        meta = {
            "num_features_detected": self._num_features_detected,
            "checkpoint_path_resolved": self._checkpoint_path_resolved,
        }
        (path / self._META_FILE).write_text(
            json.dumps(meta, ensure_ascii=False),
            encoding="utf-8",
        )

    @classmethod
    def load(cls, path: str | Path, *, oid: str | None = None) -> Self:
        path = Path(path)
        instance = super().load(path, oid=oid)

        meta_file = path / cls._META_FILE
        if meta_file.exists():
            meta = json.loads(meta_file.read_text(encoding="utf-8"))
            instance._num_features_detected = meta.get("num_features_detected")
            instance._checkpoint_path_resolved = meta.get("checkpoint_path_resolved")

        if instance._num_features_detected is not None:
            instance._tester = _build_tester(
                instance._num_features_detected, instance.config,
            )
            if instance.config.checkpoint is None and instance._checkpoint_path_resolved:
                instance._checkpoint_path_resolved = instance._tester.checkpoint_path
            instance._fitted = True

        return instance
