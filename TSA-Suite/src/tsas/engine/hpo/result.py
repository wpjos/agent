# -*- coding: utf-8 -*-

"""
HPO 优化结果数据模型。

提供 TrialInfo（单次试验记录）和 HPOResult（完整优化结果）
两个核心数据容器，支持 TopK 最佳结果提取和全量试验记录。

用法示例::

    result: HPOResult = trainer.fit(train_data)
    print(result.best_params)   # 最优参数
    print(result.best_score)    # 最优分数
    for trial in result.all_trials:
        print(trial.params, trial.scores)
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any

from tsas.engine.operator.base import BaseOperator

__all__ = [
    'TrialInfo',
    'HPOResult',
]


# ============================================================================
# TrialInfo — 单次试验记录
# ============================================================================

@dataclass
class TrialInfo:
    """单次超参数优化试验的记录。

    每次 Optuna trial 采样一组超参数并评估后，生成一条 TrialInfo。

    Attributes:
        number (int): 试验编号（从 0 起）
        params (dict[str, Any]): 该次试验采样的超参数组合
        scores (dict[str, float]): 指标名称到评估分数的映射。
            单目标时为单个键值对，多目标时为多个键值对
        operator (BaseOperator | None): 已训练好的算子实例。
            仅在需要保留模型时非空，否则为 None
    """
    number: int
    """试验编号（从 0 起）"""
    params: dict[str, Any]
    """该次试验采样的超参数组合"""
    scores: dict[str, float]
    """指标名称到评估分数的映射"""
    operator: BaseOperator | None = None
    """已训练好的算子实例，需要时非空"""

    @property
    def score(self) -> float:
        """获取主分数（scores 字典中的第一个值）。

        对于单目标优化，直接返回唯一指标分数。
        对于多目标优化，返回第一个指标的分数。

        Returns:
            float: 主分数值
        """
        if not self.scores:
            return float('-inf')
        return next(iter(self.scores.values()))

    @property
    def score_name(self) -> str:
        """获取主分数名称（scores 字典中的第一个键）。

        Returns:
            str: 主分数名称，scores 为空时返回空字符串
        """
        if not self.scores:
            return ''
        return next(iter(self.scores.keys()))


# ============================================================================
# HPOResult — 完整优化结果
# ============================================================================

@dataclass
class HPOResult:
    """HPO 优化过程的完整结果容器。

    包含最优 TopK 试验、全量试验记录以及原始 Optuna Study 对象。
    提供便捷属性支持直接获取最优参数、最优分数和最优算子。

    Attributes:
        best_trials (list[TrialInfo]): 按分数降序排列的最优 TopK 试验列表。
            列表第一个元素为全局最优
        all_trials (list[TrialInfo]): 按试验编号递增排列的所有试验记录
        top_k (int): 保留的最优试验数量
        directions (list[str]): 优化方向列表，每个元素为 ``"maximize"`` 或 ``"minimize"``
        search_space (dict[str, dict]): 此次优化使用的搜索空间
        metric_names (list[str]): 评估指标名称列表
    """
    best_trials: list[TrialInfo] = field(default_factory=list)
    """最优 TopK 试验列表，按分数从高到低排序"""
    all_trials: list[TrialInfo] = field(default_factory=list)
    """所有试验记录，按试验编号递增排列"""
    top_k: int = 1
    """保留的最优试验数量"""
    directions: list[str] = field(default_factory=lambda: ["maximize"])
    """优化方向列表"""
    search_space: dict[str, dict] = field(default_factory=dict)
    """此次优化使用的搜索空间"""
    metric_names: list[str] = field(default_factory=list)
    """评估指标名称列表"""

    # ---- 便捷属性 ----

    @property
    def best_params(self) -> dict[str, Any]:
        """获取全局最优试验的超参数组合。

        Returns:
            dict[str, Any]: 全局最优试验的超参数，
                无试验时返回空字典

        Raises:
            IndexError: best_trials 为空时
        """
        if not self.best_trials:
            raise IndexError("best_trials 为空，无法获取最优参数")
        return self.best_trials[0].params

    @property
    def best_score(self) -> dict[str, float]:
        """获取全局最优试验的评估分数。

        Returns:
            dict[str, float]: 最优试验的各指标分数，
                无试验时返回空字典

        Raises:
            IndexError: best_trials 为空时
        """
        if not self.best_trials:
            raise IndexError("best_trials 为空，无法获取最优分数")
        return self.best_trials[0].scores

    @property
    def best_score_value(self) -> float:
        """获取全局最优试验的主分数值。

        等效于 ``best_trials[0].score``。

        Returns:
            float: 最优试验的主分数值

        Raises:
            IndexError: best_trials 为空时
        """
        if not self.best_trials:
            raise IndexError("best_trials 为空，无法获取最优分数值")
        return self.best_trials[0].score

    @property
    def best_operator(self) -> BaseOperator | None:
        """获取全局最优试验对应的已训练算子实例。

        Returns:
            BaseOperator | None: 最优算子实例，未保留时为 None

        Raises:
            IndexError: best_trials 为空时
        """
        if not self.best_trials:
            raise IndexError("best_trials 为空，无法获取最优算子")
        return self.best_trials[0].operator

    def __repr__(self) -> str:
        """返回 HPOResult 的简洁字符串表示。

        Returns:
            str: 包含试验统计信息的字符串
        """
        n_best = len(self.best_trials)
        n_all = len(self.all_trials)
        best_str = ''
        if self.best_trials:
            best_str = f', best={self.best_trials[0].scores}'
        return (f'HPOResult(best_trials={n_best}/{n_all}'
                f', directions={self.directions}{best_str})')
