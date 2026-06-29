# -*- coding: utf-8 -*-

"""
HPO 训练编排器单元测试

对应源文件：
- trainer.py: HPOTrainer, _rebuild_operator, _resolve_validation_strategy

测试范围：
- HPOTrainer 初始化参数
- _resolve_search_space（自动提取与手动指定）
- _resolve_validation_strategy（独立验证集 / val_split / cv_folds）
- _build_operator（类级 / 实例级）
- _rebuild_operator（普通算子 / Composite 递归重建）
- HPOTrainer.fit() 端到端集成测试（少量 trial）
"""

import inspect
from typing import Literal

import numpy as np
import pytest
from pydantic import BaseModel, Field

from tsas.engine.hpo.search_hint import (
    extract_search_space,
    extract_search_space_from_operator,
    config_to_optuna_suggestions,
)
from tsas.engine.hpo.trainer import (
    HPOTrainer,
    _rebuild_operator,
    _resolve_validation_strategy,
)
from tsas.engine.operator.base import BaseOperator
from tsas.engine.operator.detection.knn import KNNDetector, KNNDetectorConfig
from tsas.engine.operator.detection.zscore import ZScoreDetector, ZScoreDetectorConfig


# ============================================================================
# 测试用 Mock 指标
# ============================================================================

class _MockMetric:
    """Mock 指标算子，记录输入并返回固定分数"""

    def __init__(self, fixed_scores=None):
        self._fixed = fixed_scores or {"f1": 0.5}
        self.last_input = None

    def scores(self, inputs):
        self.last_input = inputs
        return dict(self._fixed)


# ============================================================================
# 公共测试数据
# ============================================================================

@pytest.fixture
def train_data():
    """训练数据（40x3, 标准正态分布）"""
    np.random.seed(42)
    return np.random.randn(40, 3)


@pytest.fixture
def val_data():
    """验证数据（10x3）"""
    np.random.seed(99)
    return np.random.randn(10, 3)


@pytest.fixture
def val_labels():
    """验证标签（10个，5个正常5个异常）"""
    np.random.seed(123)
    return np.array([0] * 5 + [1] * 5)


@pytest.fixture
def metric_op():
    """Mock 指标算子"""
    return _MockMetric({"f1": 0.85})


# ============================================================================
# _resolve_validation_strategy 测试
# ============================================================================

class TestResolveValidationStrategy:
    """测试 _resolve_validation_strategy 函数"""

    def test_independent_val_set(self):
        """
        目的：验证独立验证集策略
        输入：train_data(ndarray), val_data(ndarray)
        预期：返回 [(train_arr, val_arr)]
        """
        train = np.random.randn(50, 3)
        val = np.random.randn(20, 3)
        folds = _resolve_validation_strategy(train, val_data=val)
        assert len(folds) == 1
        train_fold, val_fold = folds[0]
        assert len(train_fold) == 50
        assert len(val_fold) == 20

    def test_val_split(self):
        """
        目的：验证训练集内切分策略
        输入：train_data, val_split=0.3
        预期：返回 [(train_subset, val_subset)]，比例为 7:3
        """
        train = np.random.randn(100, 3)
        folds = _resolve_validation_strategy(train, val_split=0.3, random_seed=42)
        assert len(folds) == 1
        train_fold, val_fold = folds[0]
        assert len(train_fold) == 70
        assert len(val_fold) == 30

    def test_cv_folds(self):
        """
        目的：验证 K-Fold 交叉验证策略
        输入：train_data(50个), cv_folds=5
        预期：返回 5 个 (train, val) 对，每折 10 个验证样本
        """
        train = np.random.randn(50, 3)
        folds = _resolve_validation_strategy(train, cv_folds=5, random_seed=42)
        assert len(folds) == 5
        for train_fold, val_fold in folds:
            assert len(val_fold) == 10
            assert len(train_fold) == 40

    def test_no_strategy_returns_empty(self):
        """
        目的：验证无验证策略时返回空列表
        输入：仅 train_data
        预期：返回 []
        """
        train = np.random.randn(50, 3)
        folds = _resolve_validation_strategy(train)
        assert folds == []

    def test_cv_folds_uneven_split(self):
        """
        目的：验证 CV 折数不能整除时的处理
        输入：train_data(23个), cv_folds=5
        预期：最后一折的验证集可能比其他折多或少
        """
        train = np.random.randn(23, 3)
        folds = _resolve_validation_strategy(train, cv_folds=5, random_seed=42)
        assert len(folds) == 5
        # 所有训练+验证样本总和应为 23
        for train_fold, val_fold in folds:
            assert len(train_fold) + len(val_fold) == 23

    def test_val_split_zero(self):
        """
        目的：验证 val_split=0（不切分）
        输入：train_data, val_split=0
        预期：返回空列表（0 < val_split < 1 才切分）
        """
        train = np.random.randn(50, 3)
        folds = _resolve_validation_strategy(train, val_split=0)
        assert folds == []

    def test_val_split_one(self):
        """
        目的：验证 val_split=1（全部验证集）
        输入：train_data, val_split=1
        预期：返回空列表（不在 (0,1) 范围内）
        """
        train = np.random.randn(50, 3)
        folds = _resolve_validation_strategy(train, val_split=1)
        assert folds == []


# ============================================================================
# HPOTrainer 初始化测试
# ============================================================================

class TestHPOTrainerInit:
    """测试 HPOTrainer 初始化"""

    def test_class_based_init(self):
        """
        目的：验证类级算子的 Trainer 初始化
        输入：HPOTrainer(KNNDetector, _MockMetric())
        预期：成功初始化，directions=["maximize"]
        """
        trainer = HPOTrainer(KNNDetector, _MockMetric())
        assert trainer.operator is KNNDetector
        assert trainer.directions == ["maximize"]
        assert trainer.n_trials == 100

    def test_instance_based_init(self):
        """
        目的：验证实例级算子的 Trainer 初始化
        输入：HPOTrainer(ZScoreDetector(), _MockMetric())
        预期：成功初始化
        """
        detector = ZScoreDetector()
        trainer = HPOTrainer(detector, _MockMetric())
        assert trainer.operator is detector

    def test_minimize_direction(self):
        """
        目的：验证 minimize 方向
        输入：directions="minimize"
        预期：directions=["minimize"]
        """
        trainer = HPOTrainer(KNNDetector, _MockMetric(), directions="minimize")
        assert trainer.directions == ["minimize"]

    def test_multi_objective_directions(self):
        """
        目的：验证多目标方向
        输入：directions=["maximize", "minimize"]
        预期：top_k 自动设为 10
        """
        trainer = HPOTrainer(KNNDetector, _MockMetric(),
                             directions=["maximize", "minimize"])
        assert trainer.directions == ["maximize", "minimize"]
        assert trainer.top_k == 10  # 多目标默认 top_k=10

    def test_custom_top_k(self):
        """
        目的：验证自定义 top_k
        输入：top_k=5
        预期：top_k=5
        """
        trainer = HPOTrainer(KNNDetector, _MockMetric(), top_k=5)
        assert trainer.top_k == 5

    def test_manual_search_space(self):
        """
        目的：验证手动指定搜索空间
        输入：search_space={"a": {"type": "int", "low": 1, "high": 10}}
        预期：_search_space 被设置
        """
        space = {"a": {"type": "int", "low": 1, "high": 10}}
        trainer = HPOTrainer(KNNDetector, _MockMetric(), search_space=space)
        assert trainer._search_space is space

    def test_resolve_search_space_auto_class(self):
        """
        目的：验证自动提取类级搜索空间
        输入：HPOTrainer(KNNDetector, ...)
        预期：自动提取 KNNDetectorConfig 的搜索空间
        """
        trainer = HPOTrainer(KNNDetector, _MockMetric())
        space = trainer._resolve_search_space()
        assert "n_neighbors" in space
        assert "percentile" in space

    def test_resolve_search_space_auto_instance(self):
        """
        目的：验证自动提取实例级搜索空间
        输入：HPOTrainer(ZScoreDetector(), ...)
        预期：自动提取 ZScoreDetector 的搜索空间
        """
        detector = ZScoreDetector()
        trainer = HPOTrainer(detector, _MockMetric())
        space = trainer._resolve_search_space()
        assert "threshold" in space

    def test_resolve_search_space_manual_priority(self):
        """
        目的：验证手动搜索空间优于自动提取
        输入：手动指定 search_space，算子自身也有搜索空间
        预期：使用手动指定的
        """
        custom = {"custom_param": {"type": "int", "low": 1, "high": 5}}
        trainer = HPOTrainer(KNNDetector, _MockMetric(), search_space=custom)
        space = trainer._resolve_search_space()
        assert space is custom

    def test_all_sampler_types(self):
        """
        目的：验证所有采样器类型可接受
        输入：sampler="tpe", "random", "grid"
        预期：均成功初始化
        """
        for s in ["tpe", "random", "grid"]:
            trainer = HPOTrainer(KNNDetector, _MockMetric(), sampler=s)
            assert trainer.sampler == s

    def test_pruning_enabled(self):
        """
        目的：验证剪枝参数
        输入：pruning=True, pruning_n_startup_trials=10
        预期：属性正确
        """
        trainer = HPOTrainer(KNNDetector, _MockMetric(),
                             pruning=True, pruning_n_startup_trials=10)
        assert trainer.pruning is True
        assert trainer.pruning_n_startup_trials == 10


# ============================================================================
# _build_operator & _rebuild_operator 测试
# ============================================================================

class TestBuildOperator:
    """测试算子构建/重建逻辑"""

    def test_build_class_based(self):
        """
        目的：验证从类构建算子
        输入：trainer._build_operator({"n_neighbors": 3, ...})
        预期：返回 KNNDetector 实例
        """
        trainer = HPOTrainer(KNNDetector, _MockMetric())
        op = trainer._build_operator({
            "n_neighbors": 3,
            "distance_metric": "manhattan",
            "score_method": "mean",
            "percentile": 90.0,
        })
        assert isinstance(op, KNNDetector)
        assert op.config.n_neighbors == 3
        assert op.config.distance_metric.value == "manhattan"

    def test_build_instance_based(self):
        """
        目的：验证从实例重建算子
        输入：trainer._build_operator({"threshold": 5.0})
        预期：返回新 ZScoreDetector 实例
        """
        detector = ZScoreDetector()
        trainer = HPOTrainer(detector, _MockMetric())
        op = trainer._build_operator({"threshold": 5.0})
        assert isinstance(op, ZScoreDetector)
        assert op.config.threshold == 5.0

    def test_rebuild_simple_operator(self):
        """
        目的：验证 _rebuild_operator 重建普通算子
        输入：_rebuild_operator(ZScoreDetector(), {"threshold": 7.0})
        预期：返回新 ZScoreDetector 实例，threshold=7.0
        """
        original = ZScoreDetector()
        rebuilt = _rebuild_operator(original, {"threshold": 7.0})
        assert isinstance(rebuilt, ZScoreDetector)
        assert rebuilt.config.threshold == 7.0
        assert rebuilt is not original

    def test_rebuild_composite(self):
        """
        目的：验证 _rebuild_operator 重建 Composite 算子
        输入：Composite 实例 + 带前缀的参数
        预期：子算子被重建
        """
        from tsas.engine.operator.detection.composite import CompositeDetector
        from tsas.engine.operator.detection.pca import PCAPredictor, PCAPredictorConfig
        from tsas.engine.operator.detection.residual_scorer import ResidualScorer
        from tsas.engine.operator.detection.percentile_decider import PercentileDecider

        comp = CompositeDetector(operators=[
            PCAPredictor(config=PCAPredictorConfig(n_components=5)),
            ResidualScorer(),
            PercentileDecider(),
        ])

        params = {
            "predictor.n_components": 10,
            "scorer_0.metric": "mae",
            "decider.percentile": 80.0,
        }

        rebuilt = _rebuild_operator(comp, params)
        assert isinstance(rebuilt, CompositeDetector)
        assert rebuilt._operators[0].config.n_components == 10
        assert rebuilt._operators[1].config.metric == "mae"
        assert rebuilt._operators[2].config.percentile == 80.0


# ============================================================================
# HPOTrainer.fit() 集成测试
# ============================================================================

class TestHPOTrainerFit:
    """测试 HPOTrainer.fit() 端到端流程"""

    def test_fit_with_val_data(self, train_data, val_data, val_labels, metric_op):
        """
        目的：验证独立验证集的 fit 流程
        输入：train_data, val_data, val_labels
        预期：返回 HPOResult，包含 best_trials 和 all_trials
        """
        trainer = HPOTrainer(
            ZScoreDetector,
            metric_op,
            n_trials=3,
            random_seed=42,
            top_k=1,
        )
        result = trainer.fit(
            train_data,
            val_data=val_data,
            val_labels=val_labels,
        )
        assert len(result.all_trials) == 3
        assert len(result.best_trials) == 1
        assert result.search_space is not None
        assert "threshold" in result.search_space

    def test_fit_with_val_split(self, train_data, val_labels, metric_op):
        """
        目的：验证训练集内切分的 fit 流程
        输入：train_data, val_split=0.3, val_labels
        预期：返回 HPOResult
        """
        trainer = HPOTrainer(
            ZScoreDetector,
            metric_op,
            n_trials=3,
            random_seed=42,
        )
        result = trainer.fit(
            train_data,
            val_labels=val_labels,
            val_split=0.3,
        )
        assert len(result.all_trials) == 3
        assert len(result.best_trials) >= 1

    def test_fit_best_params_valid(self, train_data, val_labels, metric_op):
        """
        目的：验证最优参数在搜索范围内
        输入：train_data, val_split, val_labels
        预期：best_params 的 threshold 在 [1.0, 10.0] 内
        """
        trainer = HPOTrainer(
            ZScoreDetector,
            metric_op,
            n_trials=5,
            random_seed=42,
        )
        result = trainer.fit(
            train_data,
            val_labels=val_labels,
            val_split=0.3,
        )
        best = result.best_params
        assert 1.0 <= best["threshold"] <= 10.0

    def test_fit_all_trials_recorded(self, train_data, val_labels, metric_op):
        """
        目的：验证所有 trial 都被记录
        输入：train_data, val_labels, val_split
        预期：all_trials 长度 = n_trials
        """
        n = 4
        trainer = HPOTrainer(
            ZScoreDetector,
            metric_op,
            n_trials=n,
            random_seed=42,
        )
        result = trainer.fit(
            train_data,
            val_labels=val_labels,
            val_split=0.3,
        )
        assert len(result.all_trials) == n

    def test_fit_returns_hpo_result(self, train_data, val_labels, metric_op):
        """
        目的：验证 fit 返回 HPOResult 类型
        输入：任意合法参数
        预期：返回 HPOResult 实例
        """
        from tsas.engine.hpo.result import HPOResult as HR
        trainer = HPOTrainer(ZScoreDetector, metric_op, n_trials=2, random_seed=42)
        result = trainer.fit(train_data, val_labels=val_labels, val_split=0.3)
        assert isinstance(result, HR)

    def test_fit_top_k(self, train_data, val_labels, metric_op):
        """
        目的：验证 top_k 返回正确数量的最优 trial
        输入：n_trials=5, top_k=3
        预期：best_trials 长度 = 3
        """
        trainer = HPOTrainer(
            ZScoreDetector,
            metric_op,
            n_trials=5,
            top_k=3,
            random_seed=42,
        )
        result = trainer.fit(
            train_data,
            val_labels=val_labels,
            val_split=0.3,
        )
        assert len(result.best_trials) == 3

    def test_fit_with_cv(self, train_data, val_labels, metric_op):
        """
        目的：验证 K-Fold CV 的 fit 流程
        输入：train_data, val_labels, cv_folds=3
        预期：返回 HPOResult
        """
        trainer = HPOTrainer(
            ZScoreDetector,
            metric_op,
            n_trials=2,
            random_seed=42,
        )
        result = trainer.fit(
            train_data,
            val_labels=val_labels,
            cv_folds=3,
        )
        assert len(result.all_trials) > 0

    def test_fit_empty_search_space_raises(self):
        """
        目的：验证搜索空间为空时抛出 ValueError
        输入：算子无搜索空间字段
        预期：抛出 ValueError
        """
        from tsas.engine.operator.detection.threshold_decider import ThresholdDecider
        trainer = HPOTrainer(ThresholdDecider, _MockMetric(), n_trials=1,
                             search_space={})
        with pytest.raises(ValueError, match="搜索空间为空"):
            trainer.fit(np.random.randn(10, 3), val_labels=np.array([0]*10),
                       val_split=0.3)


# ============================================================================
# _rebuild_operator 边界测试
# ============================================================================

class TestRebuildOperatorEdgeCases:
    """测试 _rebuild_operator 的边界情况"""

    def test_rebuild_operator_without_operators(self):
        """
        目的：验证 _operators 为空列表时正确处理
        输入：operator._operators = []，params 包含字段参数
        预期：返回新算子实例（按普通算子处理）
        """
        original = ZScoreDetector()
        # ZScoreDetector 无 _operators 属性或为 None，走普通算子分支
        rebuilt = _rebuild_operator(original, {"threshold": 5.0})
        assert isinstance(rebuilt, ZScoreDetector)
        assert rebuilt.config.threshold == 5.0

    def test_rebuild_operator_without_config_type(self):
        """
        目的：验证 _config_type 为 None 时的算子重建
        输入：传入空 params
        预期：使用默认构造
        """
        original = ZScoreDetector()
        # 传入空 params，走 config_kwargs={} / no config kwargs 分支
        rebuilt = _rebuild_operator(original, {})
        assert isinstance(rebuilt, ZScoreDetector)
        # 使用默认 config
        assert rebuilt.config.threshold == 3.0


# ============================================================================
# _resolve_validation_strategy 边界测试
# ============================================================================

class TestValidationStrategyListInput:
    """测试 _resolve_validation_strategy 的列表输入"""

    def test_train_data_as_list(self):
        """
        目的：验证 train_data 为 list 时的处理
        输入：list 格式的训练数据 + val_split
        预期：正确转换为 ndarray 并切分
        """
        train = [[1.0, 2.0], [3.0, 4.0], [5.0, 6.0], [7.0, 8.0]]
        folds = _resolve_validation_strategy(train, val_split=0.5, random_seed=42)
        assert len(folds) == 1
        assert len(folds[0][0]) == 2
        assert len(folds[0][1]) == 2

    def test_val_data_as_list(self):
        """
        目的：验证 val_data 为 list 时的处理
        输入：list 格式的验证数据
        预期：正确转换为 ndarray
        """
        train = np.random.randn(10, 2)
        val = [[1.0, 2.0], [3.0, 4.0]]
        folds = _resolve_validation_strategy(train, val_data=val)
        assert len(folds) == 1
        assert len(folds[0][1]) == 2


# ============================================================================
# _resolve_search_space 错误分支测试
# ============================================================================

class TestResolveSearchSpaceErrors:
    """测试 _resolve_search_space 的错误分支"""

    def test_no_config_type_raises(self):
        """
        目的：验证算子类无 _config_type 时抛出 ValueError
        输入：一个 _config_type=None 的算子类
        预期：抛出 ValueError
        """
        from tsas.engine.operator.base import BaseOperator

        # 创建一个没有泛型参数（因此 _config_type=None）的算子
        class _NoConfigOp(BaseOperator):
            @classmethod
            def name(cls):
                return "no_config"
            @classmethod
            def version(cls) -> tuple[int, ...]:
                return (1, 0, 0)
            def _run(self, x, *, params=None):
                return x

        trainer = HPOTrainer(_NoConfigOp, _MockMetric())
        with pytest.raises(ValueError, match="未定义 _config_type"):
            trainer._resolve_search_space()


# ============================================================================
# HPOTrainer.fit() 更多场景测试
# ============================================================================

class TestHPOTrainerFitMore:
    """测试 HPOTrainer.fit() 的更多场景"""

    def test_fit_with_random_sampler(self, train_data, val_labels, metric_op):
        """
        目的：验证 RandomSampler 的 fit 流程
        输入：sampler="random"
        预期：返回 HPOResult
        """
        trainer = HPOTrainer(
            ZScoreDetector,
            metric_op,
            sampler="random",
            n_trials=2,
            random_seed=42,
        )
        result = trainer.fit(
            train_data,
            val_labels=val_labels,
            val_split=0.3,
        )
        assert len(result.all_trials) == 2
        assert len(result.best_trials) >= 1

    def test_fit_with_pruning(self, train_data, val_labels, metric_op):
        """
        目的：验证启用剪枝的 fit 流程
        输入：pruning=True, n_trials=5
        预期：返回 HPOResult，无异常抛出
        """
        trainer = HPOTrainer(
            ZScoreDetector,
            metric_op,
            pruning=True,
            n_trials=3,
            random_seed=42,
        )
        result = trainer.fit(
            train_data,
            val_labels=val_labels,
            val_split=0.3,
        )
        assert len(result.all_trials) == 3

    def test_fit_multi_objective(self, train_data, val_labels):
        """
        目的：验证多目标优化的 fit 流程
        输入：directions=["maximize", "minimize"]
        预期：返回 HPOResult，metric_names 包含两个指标
        """
        mock = _MockMetric({"f1": 0.85, "loss": 0.15})
        trainer = HPOTrainer(
            ZScoreDetector,
            mock,
            directions=["maximize", "minimize"],
            n_trials=2,
            random_seed=42,
        )
        result = trainer.fit(
            train_data,
            val_labels=val_labels,
            val_split=0.3,
        )
        assert len(result.all_trials) == 2
        assert len(result.metric_names) >= 2


# ============================================================================
# _build_grid_search_space 测试
# ============================================================================

class TestBuildGridSearchSpace:
    """测试 _build_grid_search_space 方法"""

    def test_grid_space_int(self):
        """
        目的：验证 int 类型搜索空间转 GridSampler 候选值列表
        输入：int 搜索空间 {n: {type:"int", low:1, high:10, step:2}}
        预期：候选值列表为 [1, 3, 5, 7, 9]
        """
        trainer = HPOTrainer(KNNDetector, _MockMetric())
        space = {"n": {"type": "int", "low": 1, "high": 10, "step": 2}}
        grid = trainer._build_grid_search_space(space)
        assert "n" in grid
        assert grid["n"] == [1, 3, 5, 7, 9]

    def test_grid_space_float(self):
        """
        目的：验证 float 类型搜索空间转 GridSampler 候选值列表
        输入：float 搜索空间 {p: {type:"float", low:0.0, high:1.0}}
        预期：候选值列表包含 11 个等间隔浮点值（含两端）
        """
        trainer = HPOTrainer(KNNDetector, _MockMetric())
        space = {"p": {"type": "float", "low": 0.0, "high": 1.0}}
        grid = trainer._build_grid_search_space(space)
        assert "p" in grid
        assert len(grid["p"]) == 11
        assert grid["p"][0] == 0.0
        assert grid["p"][-1] == 1.0

    def test_grid_space_cat(self):
        """
        目的：验证 cat 类型搜索空间转 GridSampler 候选值列表
        输入：cat 搜索空间 {m: {type:"cat", choices:["a","b","c"]}}
        预期：候选值列表直接传递
        """
        trainer = HPOTrainer(KNNDetector, _MockMetric())
        space = {"m": {"type": "cat", "choices": ["a", "b", "c"]}}
        grid = trainer._build_grid_search_space(space)
        assert "m" in grid
        assert grid["m"] == ["a", "b", "c"]

    def test_grid_space_dot_name(self):
        """
        目的：验证含 '.' 的参数名被替换为 '_'
        输入："predictor.n_components" int 搜索空间
        预期：键名中 '.' 替换为 '_'，值为候选整数值列表
        """
        trainer = HPOTrainer(KNNDetector, _MockMetric())
        space = {"predictor.n_components": {"type": "int", "low": 1, "high": 50}}
        grid = trainer._build_grid_search_space(space)
        assert "predictor_n_components" in grid
        assert isinstance(grid["predictor_n_components"], list)


# ============================================================================
# _rebuild_operator 全覆盖测试
# ============================================================================

class TestRebuildOperatorFullCoverage:
    """测试 _rebuild_operator 的全部分支覆盖"""

    def test_rebuild_no_config_type_no_operators(self):
        """
        目的：验证 _rebuild_operator 在 _config_type=None 且无 _operators 时走默认构造
        输入：一个 _config_type=None 且无 _operators 属性的算子实例
        预期：返回该类型的新实例（使用默认构造）
        """
        from tsas.engine.operator.base import BaseOperator

        class _PlainOp(BaseOperator):
            """无 _config_type 的普通算子"""
            @classmethod
            def name(cls):
                return "plain_op"
            @classmethod
            def version(cls) -> tuple[int, ...]:
                return (1, 0, 0)
            def _run(self, x, *, params=None):
                return x

        original = _PlainOp()
        rebuilt = _rebuild_operator(original, {"irrelevant": 99})
        # 应该返回 _PlainOp 的新实例
        assert isinstance(rebuilt, _PlainOp)
        assert rebuilt is not original


# ============================================================================
# _resolve_search_space / _build_operator 类型错误测试
# ============================================================================

class TestOperatorTypeErrors:
    """测试 _resolve_search_space 和 _build_operator 的类型错误分支"""

    def test_resolve_search_space_bad_type(self):
        """
        目的：验证 _resolve_search_space 对非类非实例的算子抛出 TypeError
        输入：self.operator 被设为字符串
        预期：抛出 TypeError
        """
        trainer = HPOTrainer(KNNDetector, _MockMetric())
        trainer.operator = "not_an_operator"
        with pytest.raises(TypeError, match="不支持的算子类型"):
            trainer._resolve_search_space()

    def test_build_operator_bad_type(self):
        """
        目的：验证 _build_operator 对非类非实例的算子抛出 TypeError
        输入：self.operator 被设为整数
        预期：抛出 TypeError
        """
        trainer = HPOTrainer(KNNDetector, _MockMetric())
        trainer.operator = 12345
        with pytest.raises(TypeError, match="不支持的算子类型"):
            trainer._build_operator({})

    def test_build_operator_no_config_type(self):
        """
        目的：验证 _build_operator 对无 _config_type 的算子类走默认构造
        输入：一个 _config_type=None 的算子类
        预期：返回默认构造的实例
        """
        from tsas.engine.operator.base import BaseOperator

        class _NoCfgOp(BaseOperator):
            @classmethod
            def name(cls):
                return "no_cfg"
            @classmethod
            def version(cls) -> tuple[int, ...]:
                return (1, 0, 0)
            def _run(self, x, *, params=None):
                return x

        trainer = HPOTrainer(_NoCfgOp, _MockMetric())
        op = trainer._build_operator({"any": "value"})
        assert isinstance(op, _NoCfgOp)


# ============================================================================
# _resolve_metric_names 异常回退测试
# ============================================================================

class TestResolveMetricNamesFallback:
    """测试 _resolve_metric_names 的异常回退分支"""

    def test_metric_names_fallback_on_error(self):
        """
        目的：验证 scores() 抛异常时 _resolve_metric_names 使用默认名称
        输入：metric_op.scores() 抛出 RuntimeError
        预期：返回 ["metric_0", "metric_1"]（长度为 directions 的数量）
        """

        class _BadMetric:
            """scores() 会抛出异常的指标算子"""
            def scores(self, inputs):
                raise RuntimeError("模拟评估失败")

        trainer = HPOTrainer(
            KNNDetector, _BadMetric(),
            directions=["maximize", "minimize"],
        )
        names = trainer._resolve_metric_names()
        assert names == ["metric_0", "metric_1"]


# ============================================================================
# HPOTrainer.fit() 无 val_labels 测试
# ============================================================================

class TestFitWithoutValLabels:
    """测试 fit 不传 val_labels 的场景"""

    def test_fit_no_val_labels(self, train_data, val_data, metric_op):
        """
        目的：验证不传 val_labels 时，fit 仍能正常运行（自评估模式）
        输入：train_data, val_data，但不传 val_labels
        预期：返回 HPOResult，all_trials 记录完整
        """
        trainer = HPOTrainer(
            ZScoreDetector,
            metric_op,
            n_trials=2,
            random_seed=42,
        )
        result = trainer.fit(train_data, val_data=val_data)
        assert len(result.all_trials) == 2
        assert len(result.best_trials) >= 1


# ============================================================================
# HPOTrainer.fit() GridSampler 测试
# ============================================================================

class TestFitWithGridSampler:
    """测试 fit 使用 GridSampler 的流程"""

    def test_fit_with_grid_sampler(self, train_data, val_labels, metric_op):
        """
        目的：验证 GridSampler 的 fit 流程
        输入：sampler="grid"
        预期：返回 HPOResult，GridSampler 分支正确执行
        """
        trainer = HPOTrainer(
            ZScoreDetector,
            metric_op,
            sampler="grid",
            n_trials=2,
            random_seed=42,
        )
        result = trainer.fit(
            train_data,
            val_labels=val_labels,
            val_split=0.3,
        )
        assert len(result.all_trials) >= 1
        assert len(result.best_trials) >= 1
