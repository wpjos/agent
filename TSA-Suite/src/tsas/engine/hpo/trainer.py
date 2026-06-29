# -*- coding: utf-8 -*-

"""
HPO 训练编排器。

HPOTrainer 是超参数优化的顶层编排组件，负责将搜索空间提取、
Optuna 优化循环、算子重建、数据切分、指标评估和结果收集串联为
完整的自动化 HPO 流程。

支持场景:
    - 基于算子的类（class）: 自动提取搜索空间，多次实例化
    - 基于算子实例（instance）: 支持 Composite 递归重建
    - 单目标优化: ``"maximize"`` 或 ``"minimize"``
    - 多目标优化: 混合 ``["maximize", "minimize"]``
    - 验证方式: 独立验证集 / 训练集内切分 / K-Fold 交叉验证
    - TopK 结果: 返回最优 K 组参数及对应的算子实例
    - 全量记录: 保留所有已尝试的参数组合与指标

用法示例::

    from tsas.engine.hpo import HPOTrainer
    from tsas.engine.operator.detection.knn import KNNDetector
    from tsas.engine.operator.evaluation.binary_classification import BinaryClassificationMetric

    trainer = HPOTrainer(KNNDetector, BinaryClassificationMetric(),
                         n_trials=50, top_k=3)
    result = trainer.fit(train_data, val_labels=val_labels, val_split=0.3)
    print(result.best_params)
    print(result.best_score)
"""

from __future__ import annotations

import inspect
import time
from typing import Any, Literal

import numpy as np

from tsas.engine.hpo.result import HPOResult, TrialInfo
from tsas.engine.hpo.search_hint import (
    config_to_optuna_suggestions,
    extract_search_space,
    extract_search_space_from_operator,
)
from tsas.engine.operator.base import BaseOperator
from tsas.engine.operator.detection.base import (
    BaseDeciderMixin,
    BasePredictorMixin,
)
from tsas.engine.operator.evaluation.base import BaseMetricOperator

__all__ = [
    'HPOTrainer',
]


# ============================================================================
# 内部工具：算子重建
# ============================================================================

def _rebuild_operator(operator: BaseOperator, params: dict[str, Any]) -> BaseOperator:
    """根据采样参数重建算子实例（支持 Composite 递归）。

    对于普通算子：从 params 中提取本 Config 需要的字段，
    创建新 Config 实例并重建算子。
    对于 Composite 算子：递归重建各子算子，
    再组装为新的 Composite 实例。

    Args:
        operator (BaseOperator): 当前算子实例（作为类型模板）
        params (dict[str, Any]): 采样的全部参数（含层级前缀）

    Returns:
        BaseOperator: 重建后的新算子实例

    Raises:
        ValueError: 无法推断算子类型时
    """
    cls = type(operator)
    config_type = operator._config_type

    # ---- 普通算子（非 Composite） ----
    if not hasattr(operator, '_operators') or operator._operators is None:
        # 提取本 Config 需要的字段
        if config_type is not None:
            field_names = set(config_type.model_fields.keys())
            # 从 params 中筛选本算子相关的字段
            config_kwargs = {k: v for k, v in params.items() if k in field_names}
        else:
            config_kwargs = {}

        # 重建算子
        if config_kwargs:
            new_config = config_type(**config_kwargs)
            return cls(config=new_config)
        else:
            return cls()

    # ---- Composite 算子：递归重建各子算子 ----
    operators = operator._operators
    new_operators = []
    for i, sub_op in enumerate(operators):
        # 推断子算子的前缀
        if isinstance(sub_op, BasePredictorMixin):
            sub_prefix = "predictor."
        elif isinstance(sub_op, BaseDeciderMixin):
            sub_prefix = "decider."
        else:
            # Scorer: 按序号命名
            scorer_idx = sum(
                1 for o in operators[:i]
                if not isinstance(o, (BasePredictorMixin, BaseDeciderMixin))
            )
            sub_prefix = f"scorer_{scorer_idx}."

        # 提取该子算子相关的参数（去掉前缀）
        sub_params = {}
        for key, value in params.items():
            if key.startswith(sub_prefix):
                sub_params[key[len(sub_prefix):]] = value

        # 递归重建子算子
        new_sub_op = _rebuild_operator(sub_op, sub_params)
        new_operators.append(new_sub_op)

    return cls(operators=new_operators)


def _resolve_validation_strategy(
    train_data: np.ndarray | list[np.ndarray],
    *,
    val_data: np.ndarray | list[np.ndarray] | None = None,
    val_split: float | None = None,
    cv_folds: int | None = None,
    random_seed: int = 42,
) -> list[tuple[np.ndarray, np.ndarray]]:
    """解析验证策略，返回 (训练集, 验证集) 对列表。

    支持三种验证方式：
        1. 独立验证集: val_data 非 None
        2. 训练集内切分: val_split 指定比例
        3. K-Fold 交叉验证: cv_folds 指定折数

    所有返回的训练集和验证集均为 np.ndarray 格式。

    Args:
        train_data (np.ndarray | list[np.ndarray]): 训练数据
        val_data (np.ndarray | list[np.ndarray] | None): 独立验证集
        val_split (float | None): 验证集切分比例，如 0.3 表示 30%
        cv_folds (int | None): 交叉验证折数
        random_seed (int): 随机种子

    Returns:
        list[tuple[np.ndarray, np.ndarray]]: (训练集, 验证集) 对列表。
            单次验证时列表长度为 1，K-Fold 时列表长度为 cv_folds
    """
    # 确保 train_data 是数组
    if isinstance(train_data, list):
        train_arr = np.array(train_data)
    else:
        train_arr = np.asarray(train_data)

    n_samples = len(train_arr)

    if val_data is not None:
        # ---- 策略1: 独立验证集 ----
        if isinstance(val_data, list):
            val_arr = np.array(val_data)
        else:
            val_arr = np.asarray(val_data)
        return [(train_arr, val_arr)]

    elif cv_folds is not None and cv_folds > 1:
        # ---- 策略2: K-Fold 交叉验证 ----
        rng = np.random.RandomState(random_seed)
        indices = rng.permutation(n_samples)
        fold_size = n_samples // cv_folds
        folds = []
        for i in range(cv_folds):
            val_start = i * fold_size
            val_end = (i + 1) * fold_size if i < cv_folds - 1 else n_samples
            val_idx = indices[val_start:val_end]
            train_idx = np.concatenate([indices[:val_start], indices[val_end:]])
            folds.append((train_arr[train_idx], train_arr[val_idx]))
        return folds

    elif val_split is not None and 0 < val_split < 1:
        # ---- 策略3: 训练集内切分 ----
        rng = np.random.RandomState(random_seed)
        indices = rng.permutation(n_samples)
        split_idx = int(n_samples * (1 - val_split))
        train_idx = indices[:split_idx]
        val_idx = indices[split_idx:]
        return [(train_arr[train_idx], train_arr[val_idx])]

    else:
        # 无验证集时，使用全量训练数据（仅训练不评估）
        # 实际调用时需要在 fit 中处理此情况
        return []


# ============================================================================
# HPOTrainer — 超参数优化编排器
# ============================================================================

class HPOTrainer:
    """超参数优化训练编排器。

    将算子、指标、搜索空间组合为 Optuna 优化流程，
    支持单目标和多目标优化、多种验证策略和 TopK 结果返回。

    Attributes:
        operator: 算子类或算子实例
        metric_op (BaseMetricOperator): 评估指标算子
        search_space (dict[str, dict] | None): 搜索空间字典。
            ``None`` 时自动从 operator 提取
        directions (list[str]): 优化方向列表，每个元素为 ``"maximize"`` 或 ``"minimize"``
        sampler (str): Optuna 采样器名称，可选 ``"tpe"`` (默认) / ``"random"`` / ``"grid"``
        n_trials (int): 最大试验次数
        time_limit (int | None): 最大优化时间（秒），``None`` 表示无限制
        pruning (bool): 是否启用 Optuna 剪枝（MedianPruner）
        pruning_n_startup_trials (int): 剪枝前的最少试验次数
        top_k (int): 返回的最优试验数量
        random_seed (int): 随机种子
    """

    def __init__(
        self,
        operator: type[BaseOperator] | BaseOperator,
        metric_op: BaseMetricOperator,
        *,
        search_space: dict[str, dict] | None = None,
        directions: str | list[str] = "maximize",
        sampler: Literal["tpe", "random", "grid"] = "tpe",
        n_trials: int = 100,
        time_limit: int | None = None,
        pruning: bool = False,
        pruning_n_startup_trials: int = 5,
        top_k: int | None = None,
        random_seed: int = 42,
    ):
        """初始化 HPOTrainer。

        Args:
            operator (type[BaseOperator] | BaseOperator): 待优化的算子类或实例。
                - 类: 每次 trial 用采样参数新实例化
                - 实例: 每次 trial 重建（支持 Composite 递归）
            metric_op (BaseMetricOperator): 评估指标算子实例
            search_space (dict[str, dict] | None): 搜索空间字典。
                ``None`` (默认) 时自动从 operator 提取
            directions (str | list[str]): 优化方向。
                - ``"maximize"``: 单目标最大化
                - ``"minimize"``: 单目标最小化
                - ``["maximize", "minimize"]``: 多目标混合
            sampler (str): Optuna 采样器。默认 ``"tpe"``
            n_trials (int): 最大试验次数。默认 100
            time_limit (int | None): 时间限制（秒）。``None`` 无限制
            pruning (bool): 是否启用剪枝。默认 ``False``
            pruning_n_startup_trials (int): 剪枝前最少试验数。默认 5
            top_k (int | None): 返回最优 TopK 结果。
                ``None`` (默认) 时单目标仅返回最优 1 个，
                多目标返回 Pareto 前沿
            random_seed (int): 随机种子。默认 42
        """
        self.operator = operator
        self.metric_op = metric_op

        # 搜索空间
        self._search_space = search_space

        # 规范化 directions
        if isinstance(directions, str):
            self.directions = [directions]
        else:
            self.directions = list(directions)

        self.sampler = sampler
        self.n_trials = n_trials
        self.time_limit = time_limit
        self.pruning = pruning
        self.pruning_n_startup_trials = pruning_n_startup_trials
        self.random_seed = random_seed

        # 确定 top_k
        if top_k is not None:
            self.top_k = top_k
        elif len(self.directions) > 1:
            self.top_k = 10  # 多目标时保留更多结果
        else:
            self.top_k = 1

        # 延迟导入 Optuna
        self._optuna = None

    @property
    def optuna(self):
        """延迟加载 Optuna 模块。

        避免非 HPO 使用场景下的强依赖。

        Returns:
            optuna 模块
        """
        if self._optuna is None:
            import optuna
            self._optuna = optuna
        return self._optuna

    # ------------------------------------------------------------------
    # 搜索空间解析
    # ------------------------------------------------------------------

    def _resolve_search_space(self) -> dict[str, dict]:
        """解析搜索空间（自动提取或使用用户指定的）。

        优先级:
            1. 用户传入的 search_space
            2. 从 operator 自动提取

        Returns:
            dict[str, dict]: 搜索空间字典
        """
        if self._search_space is not None:
            return self._search_space

        operator = self.operator
        if inspect.isclass(operator) and issubclass(operator, BaseOperator):
            # 类级算子: 从 Config 类提取
            config_type = operator._config_type
            if config_type is None:
                raise ValueError(f"算子 {operator.__name__} 未定义 _config_type")
            return extract_search_space(config_type)
        elif isinstance(operator, BaseOperator):
            # 实例级算子: 递归提取（支持 Composite）
            return extract_search_space_from_operator(operator)
        else:
            raise TypeError(f"不支持的算子类型: {type(operator)}")

    # ------------------------------------------------------------------
    # fit — 主优化入口
    # ------------------------------------------------------------------

    def fit(
        self,
        train_data: np.ndarray | list[np.ndarray],
        *,
        val_data: np.ndarray | list[np.ndarray] | None = None,
        val_labels: np.ndarray | None = None,
        val_split: float | None = None,
        cv_folds: int | None = None,
        callbacks: list[Callable] | None = None,
    ) -> HPOResult:
        """执行超参数优化。

        根据验证策略进行多轮试验，每轮试验采样一组超参数、
        训练算子并在验证集上评估，最终返回最优结果。

        Args:
            train_data (np.ndarray | list[np.ndarray]): 训练数据。
                形状通常为 (n_samples, n_features)
            val_data (np.ndarray | list[np.ndarray] | None): 独立验证集。
                ``None`` 时使用 val_split 或 cv_folds
            val_labels (np.ndarray | None): 验证集真实标签。
                用于指标评估。仅在 val_data 或 val_split 方式下有效
            val_split (float | None): 训练集内切分验证比例。
                如 ``0.3`` 表示 30% 作为验证集
            cv_folds (int | None): K-Fold 交叉验证折数。
                如 ``5`` 表示 5 折 CV
            callbacks (list[Callable] | None): 每轮试验结束后的回调函数列表，
                接收 ``(TrialInfo, n_trial, n_total)`` 参数

        Returns:
            HPOResult: 包含最优 TopK 试验和全量记录的优化结果

        Raises:
            ValueError: 搜索空间为空或参数无效时
            RuntimeError: Optuna 优化过程出错时
        """
        # ---- 1. 解析搜索空间 ----
        search_space = self._resolve_search_space()
        if not search_space:
            raise ValueError("搜索空间为空，请检查算子 Config 是否包含可搜索参数")

        # ---- 2. 解析验证策略 ----
        folds = _resolve_validation_strategy(
            train_data,
            val_data=val_data,
            val_split=val_split,
            cv_folds=cv_folds,
            random_seed=self.random_seed,
        )

        # ---- 3. 确定指标名称 ----
        metric_names = self._resolve_metric_names()

        # ---- 4. 存储 trial 记录 ----
        all_trials: list[TrialInfo] = []

        # ---- 5. 创建 Optuna Study ----
        study_kwargs: dict[str, Any] = dict(
            directions=self.directions,
        )

        # 设置采样器
        if self.sampler == "random":
            study_kwargs["sampler"] = self.optuna.samplers.RandomSampler(
                seed=self.random_seed
            )
        elif self.sampler == "grid":
            # Grid 采样器需要搜索空间信息
            study_kwargs["sampler"] = self.optuna.samplers.GridSampler(
                search_space=self._build_grid_search_space(search_space),
                seed=self.random_seed,
            )
        else:
            # 默认 TPE
            study_kwargs["sampler"] = self.optuna.samplers.TPESampler(
                seed=self.random_seed
            )

        # 设置剪枝
        if self.pruning:
            study_kwargs["pruner"] = self.optuna.pruners.MedianPruner(
                n_startup_trials=self.pruning_n_startup_trials,
            )

        study = self.optuna.create_study(**study_kwargs)

        # ---- 6. 定义目标函数 ----
        def objective(trial) -> float | list[float]:
            """单次 Optuna trial 的目标函数。

            采样参数 → 构建算子 → 训练验证 → 评估指标 → 返回分数。

            Args:
                trial (optuna.Trial): Optuna Trial 对象

            Returns:
                float | list[float]: 单目标返回单个分数，
                    多目标返回分数列表
            """
            # 6a. 采样超参数
            sampled_params = config_to_optuna_suggestions(
                trial, search_space
            )

            # 6b. 构建算子
            try:
                operator_instance = self._build_operator(sampled_params)
            except Exception as e:
                # 参数组合无效（Pydantic 验证失败等），跳过此 trial
                raise self.optuna.exceptions.TrialPruned(
                    f"参数组合无效: {e}"
                ) from e

            # 6c. 训练和验证
            all_scores: list[dict[str, float]] = []
            for fold_train, fold_val in folds:
                # 每折重建算子实例（避免重复 fit 报错）
                fold_operator = self._build_operator(sampled_params)
                # 训练
                try:
                    fold_operator.fit(fold_train)
                except Exception as e:
                    raise self.optuna.exceptions.TrialPruned(
                        f"训练失败: {e}"
                    ) from e

                # 推理
                try:
                    output = fold_operator.run(fold_val)
                    # run 返回 (result, extra) 或 result
                    if isinstance(output, tuple) and len(output) >= 1:
                        predictions = output[0]
                    else:
                        predictions = output
                except Exception as e:
                    raise self.optuna.exceptions.TrialPruned(
                        f"推理失败: {e}"
                    ) from e

                # 如果有 val_labels，用验证标签评估
                if val_labels is not None:
                    metric_input = (val_labels, predictions)
                else:
                    # 无验证标签时仅用预测做自评估
                    metric_input = predictions

                # 评估
                try:
                    fold_scores = self.metric_op.scores(metric_input)
                except Exception as e:
                    raise self.optuna.exceptions.TrialPruned(
                        f"指标评估失败: {e}"
                    ) from e

                all_scores.append(fold_scores)

            # 6d. 汇总多折分数（取平均）
            if len(all_scores) > 1:
                aggregated = {}
                for key in all_scores[0]:
                    vals = [s[key] for s in all_scores]
                    aggregated[key] = float(np.mean(vals))
            else:
                aggregated = all_scores[0]

            # 6e. 记录 trial
            trial_info = TrialInfo(
                number=trial.number,
                params=sampled_params,
                scores=dict(aggregated),
                operator=fold_operator,
            )
            all_trials.append(trial_info)

            # 6f. 返回分数
            if len(self.directions) == 1:
                # 单目标: 返回聚合分数字典的第一个值
                return float(next(iter(aggregated.values())))
            else:
                # 多目标: 返回各指标对应的值
                return [float(aggregated.get(name, float('-inf')))
                        for name in metric_names]

        # ---- 7. 执行优化 ----
        try:
            study.optimize(
                objective,
                n_trials=self.n_trials,
                timeout=self.time_limit,
                callbacks=callbacks,
                n_jobs=1,  # 单进程（避免复杂序列化问题）
            )
        except KeyboardInterrupt:
            # 允许用户中断，保留已完成的 trial
            pass

        # ---- 8. 排序和收集结果 ----
        # 按优化方向排序 all_trials
        if len(self.directions) == 1:
            # 单目标排序
            reverse = self.directions[0] == "maximize"
            all_trials.sort(key=lambda t: t.score, reverse=reverse)
        else:
            # 多目标: 按第一个方向排序
            reverse = self.directions[0] == "maximize"
            all_trials.sort(key=lambda t: t.score, reverse=reverse)

        # 取 TopK
        best_trials = all_trials[:self.top_k]

        # ---- 9. 构建结果 ----
        result = HPOResult(
            best_trials=best_trials,
            all_trials=all_trials,
            top_k=self.top_k,
            directions=self.directions,
            search_space=search_space,
            metric_names=metric_names,
        )

        return result

    # ------------------------------------------------------------------
    # 内部工具方法
    # ------------------------------------------------------------------

    def _build_operator(self, params: dict[str, Any]) -> BaseOperator:
        """根据采样参数构建算子实例。

        - 若 self.operator 是类：用 params 实例化
        - 若 self.operator 是实例：通过 _rebuild_operator 重建

        Args:
            params (dict[str, Any]): 采样参数

        Returns:
            BaseOperator: 新算子实例
        """
        operator = self.operator

        if inspect.isclass(operator) and issubclass(operator, BaseOperator):
            # 类级算子: 从 Config 类创建实例
            config_type = operator._config_type
            if config_type is not None:
                # 提取 Config 需要的字段
                field_names = set(config_type.model_fields.keys())
                config_kwargs = {k: v for k, v in params.items()
                                if k in field_names}
                config = config_type(**config_kwargs)
                return operator(config=config)
            else:
                return operator()
        elif isinstance(operator, BaseOperator):
            # 实例级算子: 重建
            return _rebuild_operator(operator, params)
        else:
            raise TypeError(f"不支持的算子类型: {type(operator)}")

    def _resolve_metric_names(self) -> list[str]:
        """推断指标名称列表。

        通过向 metric_op 传入空输入触发 scores() 方法，
        从中获取键名。若获取失败则使用默认名称。

        Returns:
            list[str]: 指标名称列表
        """
        try:
            # 尝试用一个小数组推断指标名称
            dummy = np.zeros((2,), dtype=float)
            sample_scores = self.metric_op.scores((dummy, dummy))
            return list(sample_scores.keys())
        except Exception:
            return [f"metric_{i}" for i in range(len(self.directions))]

    def _build_grid_search_space(
        self, search_space: dict[str, dict]
    ) -> dict[str, Any]:
        """将内部搜索空间转为 Optuna GridSampler 所需的搜索空间格式。

        GridSampler 接受候选值列表：
            - 整数: ``list[int]`` — 候选整数值列表
            - 浮点: ``list[float]`` — 候选浮点值列表
            - 分类: ``list`` — 候选值列表

        Args:
            search_space (dict[str, dict]): 内部搜索空间字典

        Returns:
            dict[str, Any]: Optuna GridSampler 兼容的搜索空间。
                每个键对应一个候选值列表
        """
        grid_space: dict[str, Any] = {}
        for name, space in search_space.items():
            safe_name = name.replace(".", "_")
            space_type = space.get("type")

            if space_type == "cat":
                # 分类型: 直接使用候选值列表
                grid_space[safe_name] = space.get("choices", [])
            elif space_type == "int":
                low = int(space.get("low", 0))
                high = int(space.get("high", 100))
                step = int(space.get("step", 1))
                # 生成候选整数值列表
                grid_space[safe_name] = list(range(low, high + 1, step))
            elif space_type == "float":
                low = float(space.get("low", 0.0))
                high = float(space.get("high", 1.0))
                # 为网格搜索生成合理的候选值
                # 在 [low, high] 区间均匀取 11 个点（含两端）
                num_points = 11
                step = (high - low) / (num_points - 1)
                grid_space[safe_name] = [
                    low + i * step for i in range(num_points)
                ]
        return grid_space
