# -*- coding: utf-8 -*-

"""
羲和 Gamma 时间序列异常检测评分器模块

基于预训练的羲和 Gamma 模型（``pangu_xihe_gamma``）对时间序列数据进行逐变量异常评分。
模型通过滑动窗口对时间序列进行分段推理，返回每个窗口内各变量的异常分数和重构值，
再通过窗口重叠合并策略生成全局逐变量异常分数。

核心特点:
    - 基于预训练模型，无需训练（``fit``），只需 ``load_model`` 加载权重
    - 滑动窗口 + 批量推理 + 窗口重叠合并
    - 支持两种合并策略：``"heuristic"``（启发式：去极值后 mean+std）和 ``"mean"``（均值）
    - 支持 ``local_value_scale``：推理前 StandardScaler 标准化，重构值逆变换后返回
    - 主输出为 2D 逐变量异常分数 ``(n_samples, n_vars)``
    - 附加输出包含逐变量重构值和时间戳

数据流::

    DataFrame(time_index, var1, var2, ...)
      → _unwrap_data → ndarray + meta
        → _adjust_data（可选 StandardScaler 标准化）
          → _iter_batch_results → 滑窗 Dataset → DataLoader → model.predict()
            → 窗口重叠合并 → yield (batch_idx, (batch_scores, batch_eo))
          → _merge_batch_results → (all_scores, final_eo)
        → _validate_and_wrap_output → DataFrame + EO

使用示例::

    from tsas.engine.operator.detection.xihe import XiHeGammaScorer, XiHeGammaScorerConfig

    config = XiHeGammaScorerConfig(model_path="/path/to/model", device="cpu")
    scorer = XiHeGammaScorer(config=config)

    # 全量推理
    scores_df, eo = scorer.run(df)  # df 的索引为时间列

    # 流式分批推理
    for batch_result in scorer.batch_run(df):
        batch_scores_df, batch_eo = batch_result
        process(batch_scores_df, batch_eo)

主要组件:
    - XiHeGammaScorerConfig: 评分器实例参数
    - XiHeGammaScorerExtraOutput: 附加输出（重构值 + 时间戳）
    - XiHeGammaScorer: 羲和 Gamma 异常评分器
"""

import bisect
import math
import os.path
import time
from collections.abc import Generator
from enum import Enum
from pathlib import Path

import numpy as np
import pandas as pd
import torch
from loguru import logger
from pydantic import BaseModel, Field, ConfigDict
from sklearn.preprocessing import StandardScaler
from torch.utils.data import DataLoader
from tsas.basic.dataset.sliding_window import DataFrameSlidingWindowDataset
from tsas.engine.util.pt_helper import get_torch_device

from tsas.engine.operator.base import (
    BatchRunNumericOperatorMixin,
    NumericOperator,
)
from tsas.engine.operator.detection.base import MultiScorerMixin

__all__ = [
    'XiHeScoreMerge',
    'XiHeAlign',
    'XiHeGammaScorerConfig',
    'XiHeGammaScorerExtraOutput',
    'XiHeGammaScorer',
]


# ============================================================================
# 参数与输出定义
# ============================================================================


class XiHeScoreMerge(str, Enum):
    """羲和窗口重叠区域异常分数合并策略枚举

    定义滑动窗口推理时，重叠区域的异常分数如何合并为最终分数。

    Attributes:
        HEURISTIC: 启发式合并（去极值后 mean+std），兼顾鲁棒性和灵敏度
        MEAN: 均值合并，平滑策略
    """
    HEURISTIC = "heuristic"
    """启发式合并（去极值后 mean+std），兼顾鲁棒性和灵敏度"""
    MEAN = "mean"
    """均值合并，平滑策略"""


class XiHeAlign(str, Enum):
    """羲和窗口对齐方式枚举

    定义滑动窗口结果与原始时间序列的对齐策略。

    Attributes:
        LEFT: 左对齐，窗口起始位置与结果位置对齐
        RIGHT: 右对齐，窗口结束位置与结果位置对齐
    """
    LEFT = "left"
    """左对齐，窗口起始位置与结果位置对齐"""
    RIGHT = "right"
    """右对齐，窗口结束位置与结果位置对齐"""


class XiHeGammaScorerConfig(BaseModel):
    """羲和 Gamma 评分器实例参数

    Attributes:
        model_path(str | None): 预训练模型路径，为 None 时使用默认路径
        device(str | None): 推理设备标识，如 ``"cpu"``、``"cuda:0"``、``"npu:0"``，
            为 None 时自动选择
        score_merge(XiHeScoreMerge): 窗口重叠区域异常分数合并策略
        batch_size(int): 推理批大小，必须 >= 1
        step(int): 滑动窗口步长，必须 >= 1
        align(XiHeAlign): 窗口对齐方式
        local_value_scale(bool): 是否对输入值进行 StandardScaler 标准化，
            标准化后推理，重构值会逆变换后返回
    """
    model_path: str | None = Field(default=None, description="预训练模型路径，为 None 时使用默认内置预训练模型参数")
    device: str | None = Field(default=None, description='推理设备标识，"cpu"、``"cuda:0"、"npu:0"，为 None 时自动选择')
    score_merge: XiHeScoreMerge = Field(
        default=XiHeScoreMerge.HEURISTIC,
        description='窗口重叠区域异常分数合并策略，"heuristic" 为启发式（去极值后 mean+std），"mean" 为均值',
    )
    batch_size: int = Field(default=8, ge=1, description="推理批大小，必须 >= 1")
    step: int = Field(default=1, ge=1, description="滑动窗口步长，必须 >= 1")
    align: XiHeAlign = Field(
        default=XiHeAlign.LEFT,
        description='窗口对齐方式，"left" 或 "right"',
    )
    local_value_scale: bool = Field(default=False, description="是否对输入值进行 StandardScaler 标准化")


class XiHeGammaScorerExtraOutput(BaseModel):
    """羲和 Gamma 评分器附加输出

    Attributes:
        timestamp(list): 时间戳列表，与主输出行一一对应
        feature_recon(np.ndarray): 逐变量重构值，形状 ``(n_samples, n_vars)``
    """
    model_config = ConfigDict(arbitrary_types_allowed=True)

    timestamp: list = Field(description="时间戳列表，与主输出行一一对应；ndarray 输入时为样本位置索引")
    """时间戳列表"""
    feature_recon: np.ndarray = Field(description="逐变量重构值，形状为 (n_samples, n_vars)，列顺序与输入一致")
    """逐变量重构值 (n_samples, n_vars)"""


# ============================================================================
# 羲和 Gamma 评分器
# ============================================================================


# 羲和模型固定窗口大小
_XIHE_WINDOW_SIZE = 100


class XiHeGammaScorer(
    MultiScorerMixin[None],
    BatchRunNumericOperatorMixin,
    NumericOperator[XiHeGammaScorerExtraOutput, XiHeGammaScorerConfig, None],
):
    """羲和 Gamma 时间序列异常评分器

    基于预训练的羲和 Gamma 模型对时间序列数据进行逐变量异常评分。
    继承 ``MultiScorerMixin``（输出列名沿用输入列名）、``NumericBatchRunMixin``
    （提供 ``batch_run`` + ``_run_data`` 批处理框架）和 ``NumericOperator``。

    工作流程:
        1. 加载预训练模型（``load_model``）
        2. 输入 DataFrame（索引为时间列）或 ndarray
        3. 构建滑动窗口数据集 → DataLoader 分批推理
        4. 窗口重叠区域合并异常分数和重构值
        5. 返回 2D 逐变量异常分数 + 附加输出（重构值、时间戳）

    Input:
        x: 二维时序数据，形状 (n_samples, n_features)，每列为一个特征通道

    Output:
        逐变量异常分数，形状 (n_samples, n_features)，值越大越异常。
        重构值与时间戳信息由附加输出 ``XiHeGammaScorerExtraOutput`` 提供

    泛型参数:
        EO: XiHeGammaScorerExtraOutput（附加输出由 ``_eo_type`` 自动渲染）
        C: XiHeGammaScorerConfig
        RP: None（无运行参数，第一版所有参数通过 Config 传入）

    Attributes:
        _device(torch.device): 推理设备
        _model: 预训练时间序列预测器实例
        _model_path(str | None): 模型路径
        _value_scaler(StandardScaler | None): 当前推理使用的标准化器
            （仅在 ``local_value_scale=True`` 时非 None）
    """

    @classmethod
    def name(cls) -> str:
        """返回算子名称

        Returns:
            str: 固定返回 ``"xihe_gamma_scorer"``
        """
        return "xihe_gamma_scorer"

    @classmethod
    def version(cls) -> tuple[int, ...]:
        """返回算子版本号

        Returns:
            tuple[int, ...]: 版本号三元组 ``(1, 0, 0)``
        """
        return (1, 0, 0)

    def __init__(
            self,
            *,
            oid: str | None = None,
            config: XiHeGammaScorerConfig | None = None,
            **kwargs,
    ):
        """初始化羲和 Gamma 评分器

        如果 ``config.model_path`` 不为空，则自动加载模型。

        Args:
            oid(str | None): 算子标识符
            config(XiHeGammaScorerConfig | None): 实例参数配置
            **kwargs: 透传给基类的参数
        """
        super().__init__(oid=oid, config=config, **kwargs)
        # 设备管理：从 Config 读取 device 字符串，转为 torch.device
        device_str = self.config.device if self.config else None
        self._device: torch.device = get_torch_device(device=device_str)
        """推理设备"""
        self._model = None
        """预训练模型实例"""
        self._model_path: str | None = None
        """模型路径"""
        self._value_scaler: StandardScaler | None = None
        """当前推理使用的标准化器（运行期间临时状态）"""
        # 自动加载模型
        if self.config and self.config.model_path:
            self.load_model(self.config.model_path)

    def load_model(self, path: str | None = None) -> None:
        """加载预训练模型

        Args:
            path(str | None): 模型路径，为 None 时使用默认路径

        Raises:
            ImportError: pangu_xihe_gamma 未安装时
        """
        from pangu_xihe_gamma.infer_service.pangu_ts_predictor import TSPredictor
        if path:
            self._model_path = os.path.abspath(path)
            self._model = TSPredictor(self._model_path)
        else:
            self._model = TSPredictor()
            self._model_path = self._model.checkpoint_path

    def _can_run(self) -> None:
        """推理前置校验：模型必须已加载

        Raises:
            RuntimeError: 模型未加载时
        """
        if self._model is None:
            raise RuntimeError(
                f"{type(self).__name__} 模型未加载，请先调用 load_model() 或在 Config 中配置 model_path"
            )

    # ------------------------------------------------------------------
    # 数据预处理钩子
    # ------------------------------------------------------------------

    def _adjust_data(
            self, x: np.ndarray, params: None, idx: pd.Index | None = None,
    ) -> np.ndarray:
        """按需对输入数据进行 StandardScaler 标准化

        当 ``config.local_value_scale`` 为 True 时，对输入 ndarray 执行 fit_transform，
        并将 scaler 存储在 ``self._value_scaler`` 中，供输出时逆变换使用。

        Args:
            x(np.ndarray): 输入 ndarray，形状 ``(n_samples, n_vars)``
            params(None): 无运行参数
            idx(pd.Index | None): 行索引

        Returns:
            np.ndarray: 标准化后（或原样）的 ndarray
        """
        if self.config and self.config.local_value_scale:
            self._value_scaler = StandardScaler()
            return self._value_scaler.fit_transform(x)
        else:
            self._value_scaler = None
            return x

    # ------------------------------------------------------------------
    # NumericBatchRunMixin 抽象方法实现
    # ------------------------------------------------------------------

    def _iter_batch_results(
            self, x: np.ndarray, params: None, idx: pd.Index | None,
    ) -> Generator[
        tuple[pd.Index | np.ndarray, tuple[np.ndarray, XiHeGammaScorerExtraOutput]],
        None, None,
    ]:
        """逐批推理：滑窗 → DataLoader → 模型推理 → 窗口合并 → yield

        Args:
            x(np.ndarray): 预处理后的输入数据，形状 ``(n_samples, n_vars)``
            params(None): 无运行参数
            idx(pd.Index | None): 行索引

        Yields:
            tuple: ``(batch_index, (batch_scores, batch_eo))``
        """
        # 构建时间索引列表
        time_index = idx.tolist() if idx is not None else list(range(len(x)))
        # 读取配置参数
        step = self.config.step if self.config else 1
        align = self.config.align if self.config else "left"
        batch_size = self.config.batch_size if self.config else 8
        score_merge = self.config.score_merge if self.config else "heuristic"

        # 构建 DataFrame 供 DataFrameSlidingWindowDataset 使用
        n_vars = x.shape[1] if x.ndim == 2 else 1
        var_columns = [f"var_{i}" for i in range(n_vars)]
        time_column = "__time__"
        df = pd.DataFrame(x, columns=var_columns)
        df.insert(0, time_column, time_index)

        # 创建滑窗 Dataset
        dataset = DataFrameSlidingWindowDataset(
            data=df, time_column=time_column, var_columns=var_columns,
            step=step, align=align,
        )

        # 校验窗口大小
        window_size = dataset.window_size
        assert window_size == _XIHE_WINDOW_SIZE, (
            f"羲和模型仅支持窗口长度 {_XIHE_WINDOW_SIZE}，当前为 {window_size}"
        )
        assert 0 < step <= window_size, (
            f"滑动窗口步长必须大于 0 且不超过窗口长度，当前为 {step}"
        )

        # 选择合并函数
        if isinstance(score_merge, str) and score_merge.lower() == XiHeScoreMerge.MEAN.value:
            merge_scores_fn = self._merge_scores_mean
        else:
            merge_scores_fn = self._merge_scores_heuristic_np

        # 创建 DataLoader
        dataloader = DataLoader(
            dataset, batch_size=batch_size, shuffle=False,
            collate_fn=self._collate_batch,
        )

        # 初始化合并缓存
        unmerged_cache: dict = {}
        ready_time = None
        # 3D 合并缓冲区，用 NaN 填充
        merge_cache = (
            np.full((step * batch_size, n_vars, (math.ceil(window_size / step) - 1) + batch_size), np.nan),
            np.full((step * batch_size, n_vars, (math.ceil(window_size / step) - 1) + batch_size), np.nan),
        )

        for batch_time_list, batch_context in dataloader:
            # 调用模型推理
            req_data = {"data": batch_context}
            res_data = self._model.predict(req_data)["data"]
            # 及时释放内存
            batch_context.clear()
            req_data.clear()
            del batch_context, req_data

            # 逐窗口解包到待合并缓存
            t = time.time()
            logger.info("批次结果合并中...")
            for win_time_list, res_item in zip(batch_time_list, res_data):
                score_dict = res_item["anomaly_score"]
                recon_dict = res_item["reconstruction"]
                score_arr = np.column_stack([score_dict[col] for col in var_columns])
                recon_arr = np.column_stack([recon_dict[col] for col in var_columns])
                res_item.clear()
                del res_item
                first_time = win_time_list[0]
                unmerged_cache[first_time] = (win_time_list, score_arr, recon_arr)
                ready_time_idx = min(step, len(win_time_list)) - 1
                ready_time = win_time_list[ready_time_idx]
            res_data.clear()
            del res_data

            # 合并当前批次可合并的窗口
            batch_times, batch_score_arr, batch_recon_arr = self._batch_merge(
                var_columns, unmerged_cache, merge_cache, merge_scores_fn, ready_time,
            )
            logger.info("批次结果合并结束，耗时{}ms", (time.time() - t) * 1000)

            # 逆标准化重构值
            if self._value_scaler is not None:
                batch_recon_arr = self._value_scaler.inverse_transform(batch_recon_arr)

            # 构造批次索引和输出
            batch_idx = pd.Index(batch_times) if idx is not None else np.array(batch_times)
            batch_eo = XiHeGammaScorerExtraOutput(
                timestamp=batch_times,
                feature_recon=batch_recon_arr,
            )
            yield batch_idx, (batch_score_arr, batch_eo)

        # 合并所有剩余窗口
        if unmerged_cache:
            t = time.time()
            logger.info("剩余批次结果合并中...")
            # 剩余结果可能需要更大的合并缓冲区
            if window_size - step > step * batch_size:
                merge_cache = (
                    np.full((window_size - step, n_vars, len(unmerged_cache)), np.nan),
                    np.full((window_size - step, n_vars, len(unmerged_cache)), np.nan),
                )
            batch_times, batch_score_arr, batch_recon_arr = self._batch_merge(
                var_columns, unmerged_cache, merge_cache, merge_scores_fn,
            )
            logger.info("剩余批次结果合并结束，耗时{}ms", (time.time() - t) * 1000)

            # 逆标准化重构值
            if self._value_scaler is not None:
                batch_recon_arr = self._value_scaler.inverse_transform(batch_recon_arr)

            batch_idx = pd.Index(batch_times) if idx is not None else np.array(batch_times)
            batch_eo = XiHeGammaScorerExtraOutput(
                timestamp=batch_times,
                feature_recon=batch_recon_arr,
            )
            yield batch_idx, (batch_score_arr, batch_eo)

    def _merge_batch_results(
            self,
            batch_results: list[tuple[pd.Index | np.ndarray, tuple[np.ndarray, XiHeGammaScorerExtraOutput]]],
            params: None,
    ) -> tuple[np.ndarray, XiHeGammaScorerExtraOutput]:
        """合并所有批次结果为最终输出

        将各批次的异常分数和重构值纵向拼接，构造最终的 ``XiHeGammaScorerExtraOutput``。

        Args:
            batch_results(list): ``_iter_batch_results`` yield 的全部
                ``(batch_index, (batch_scores, batch_eo))`` 列表
            params(None): 无运行参数

        Returns:
            tuple[np.ndarray, XiHeGammaScorerExtraOutput]: 合并后的
                ``(all_scores, final_eo)``
        """
        all_scores = []
        all_recons = []
        all_times = []
        for batch_idx, (batch_scores, batch_eo) in batch_results:
            all_scores.append(batch_scores)
            all_recons.append(batch_eo.feature_recon)
            all_times.extend(batch_eo.timestamp)
        final_scores = np.vstack(all_scores)
        final_recons = np.vstack(all_recons)
        final_eo = XiHeGammaScorerExtraOutput(
            timestamp=all_times,
            feature_recon=final_recons,
        )
        return final_scores, final_eo

    # ------------------------------------------------------------------
    # _name_output_columns 由 MultiScorerMixin 提供（沿用输入列名）
    # ------------------------------------------------------------------

    # ------------------------------------------------------------------
    # 持久化
    # ------------------------------------------------------------------

    def save(self, path: str | Path) -> None:
        """持久化算子到指定目录

        保存 Config 和模型路径信息。

        Args:
            path(str | Path): 目标目录路径
        """
        super().save(path)

    # ------------------------------------------------------------------
    # 核心算法：窗口合并（从旧版原样迁移）
    # ------------------------------------------------------------------

    @staticmethod
    def _batch_merge(
            var_columns: list,
            unmerged_cache: dict,
            merge_cache: tuple,
            merge_scores_fn,
            ready_time=None,
    ) -> tuple[list, np.ndarray, np.ndarray]:
        """合并批次窗口结果

        将多个滑动窗口的预测结果进行合并，生成最终的异常评分和重构数据。
        根据 ``ready_time`` 参数确定哪些数据可以合并，哪些需要保留在缓冲区中等待后续处理。

        Args:
            var_columns(list): 变量列名列表
            unmerged_cache(dict): 未合并数据缓存，
                格式 ``{first_time: (time_list, score_arr, recon_arr)}``
            merge_cache(tuple): 3D 合并缓冲区 ``(score_cache, recon_cache)``
            merge_scores_fn(callable): 评分合并函数
            ready_time: 可合并时间点，该时间点之前的数据可以被合并，
                缺省为合并所有待合并数据

        Returns:
            tuple[list, np.ndarray, np.ndarray]: ``(times, score_arr, recon_arr)``
        """
        ready_windows = []
        # 按插入顺序遍历待合并缓存（Python 3.7+ dict 保持插入顺序）
        first_time_list = list(unmerged_cache.keys())
        for first_time in first_time_list:
            time_list, score_arr, recon_arr = unmerged_cache[first_time]
            last_time = time_list[-1]
            if ready_time is None or last_time <= ready_time:
                # 整个窗口都可合并
                ready_windows.append((time_list, score_arr, recon_arr))
                del unmerged_cache[first_time]
            elif first_time > ready_time:
                # 后续窗口都不需要再看
                break
            else:
                # 可合并时间点位于窗口内，需要切分
                ready_index = bisect.bisect_right(time_list, ready_time)
                ready_times = time_list[:ready_index]
                ready_scores = score_arr[:ready_index]
                ready_recons = recon_arr[:ready_index]
                ready_windows.append((ready_times, ready_scores, ready_recons))
                # 保留未满足合并条件的部分
                remain_times = time_list[ready_index:]
                remain_scores = score_arr[ready_index:]
                remain_recons = recon_arr[ready_index:]
                unmerged_cache[first_time] = (remain_times, remain_scores, remain_recons)
                # 及时释放内存
                time_list.clear()
                del time_list

        # 收集所有唯一时间点并建立索引
        all_times_set = set()
        for time_list, _, _ in ready_windows:
            all_times_set.update(time_list)
        all_times = sorted(all_times_set)
        del all_times_set
        time_to_idx = {t: i for i, t in enumerate(all_times)}
        n_points = len(all_times)
        n_vars = len(var_columns)
        n_windows = len(ready_windows)

        # 填充 3D 合并缓冲区
        score_3d, recon_3d = merge_cache
        for win_idx, (time_list, score_arr, recon_arr) in enumerate(ready_windows):
            first_time = time_list[0]
            last_time = time_list[-1]
            global_first_idx = time_to_idx[first_time]
            global_last_idx = time_to_idx[last_time]
            score_3d[:global_first_idx, :, win_idx] = np.nan
            recon_3d[:global_first_idx, :, win_idx] = np.nan
            score_3d[global_first_idx:global_last_idx + 1, :, win_idx] = score_arr
            recon_3d[global_first_idx:global_last_idx + 1, :, win_idx] = recon_arr
            score_3d[global_last_idx + 1:, :, win_idx] = np.nan
            recon_3d[global_last_idx + 1:, :, win_idx] = np.nan

        # 及时释放内存
        ready_windows.clear()
        del ready_windows

        # 向量化合并
        score_merged = merge_scores_fn(score_3d[:n_points, :, :n_windows])
        recon_merged = np.nanmean(recon_3d[:n_points, :, :n_windows], axis=2)

        return all_times, score_merged, recon_merged

    @staticmethod
    def _collate_batch(batch) -> tuple[list[list], list[dict]]:
        """DataLoader 批处理数据整理函数

        将 DataFrameSlidingWindowDataset 产出的一批样本整理成模型推理所需的格式。

        Args:
            batch: DataLoader 的一个批次，每个元素为 ``(time_series, value_dataframe)``

        Returns:
            tuple[list[list], list[dict]]: ``(time_lists, context_dicts)``
        """
        time_data = [item[0].to_list() for item in batch]
        context_data = [
            {"context": {col: item[1][col].to_list() for col in item[1].columns.to_list()}}
            for item in batch
        ]
        return time_data, context_data

    @staticmethod
    def _merge_scores_mean(merge_cache: np.ndarray) -> np.ndarray:
        """均值合并策略

        对 3D 分数缓冲区沿窗口轴取 nanmean。

        Args:
            merge_cache(np.ndarray): 形状 ``(n_points, n_vars, n_windows)``

        Returns:
            np.ndarray: 合并后分数，形状 ``(n_points, n_vars)``
        """
        return np.nanmean(merge_cache, axis=2)

    @staticmethod
    def _merge_scores_heuristic_np(merge_cache: np.ndarray) -> np.ndarray:
        """启发式合并策略

        对于每个 (点, 变量) 位置，舍弃 10% 极值后计算 mean + std。
        有效值不足 5 个时不去极值。

        Args:
            merge_cache(np.ndarray): 形状 ``(n_points, n_vars, n_windows)``

        Returns:
            np.ndarray: 合并后分数，形状 ``(n_points, n_vars)``
        """
        n_points, n_vars, n_windows = merge_cache.shape
        # 原地排序，NaN 会被排到最后
        merge_cache.sort(axis=2)
        # 计算每个位置的有效值数量
        valid_count = np.sum(~np.isnan(merge_cache), axis=2)
        # 去掉 10% 极值
        k = np.ceil(valid_count * 0.1).astype(np.int32)
        k[valid_count <= 5] = 0
        # 创建索引掩码
        indices = np.arange(n_windows)[None, None, :]
        lower_bound = k[:, :, None]
        upper_bound = (valid_count - k)[:, :, None]
        keep_mask = (indices >= lower_bound) & (indices < upper_bound) & ~np.isnan(merge_cache)
        del indices, lower_bound, upper_bound, k, valid_count
        merge_cache[~keep_mask] = np.nan
        del keep_mask
        # 计算 mean + std
        mean_vals = np.nanmean(merge_cache, axis=2)
        std_vals = np.nanstd(merge_cache, axis=2)
        return mean_vals + std_vals
