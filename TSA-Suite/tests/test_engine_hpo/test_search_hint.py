# -*- coding: utf-8 -*-

"""
HPO 搜索空间声明与提取单元测试

对应源文件：
- search_hint.py: SearchHint, extract_search_space,
  extract_search_space_from_operator, config_to_optuna_suggestions

测试范围：
- SearchHint 基本功能（默认值/自定义）
- extract_search_space: int/float/Literal/Enum 类型的字段提取
- extract_search_space: SearchHint log/step 提取
- extract_search_space: 无搜索空间字段、非 BaseModel 错误
- extract_search_space_from_operator: 普通算子
- extract_search_space_from_operator: Composite 算子前缀
- config_to_optuna_suggestions: int/float/cat 采样（真实 Optuna 集成）
- config_to_optuna_suggestions: log/step 参数传递
- config_to_optuna_suggestions: 错误处理与默认值回退
"""

from enum import Enum
from typing import Annotated, Literal

import numpy as np
import pytest
from pydantic import BaseModel, Field

from tsas.engine.hpo.search_hint import (
    SearchHint,
    config_to_optuna_suggestions,
    extract_search_space,
    extract_search_space_from_operator,
)
from tsas.engine.operator.detection.base import BaseDetector
from tsas.engine.operator.detection.zscore import ZScoreDetector, ZScoreDetectorConfig
from tsas.engine.operator.detection.knn import KNNDetector, KNNDetectorConfig


# ============================================================================
# 测试用 Config 类
# ============================================================================

class _TestMetric(str, Enum):
    """测试用距离度量枚举"""
    EUCLIDEAN = "euclidean"
    COSINE = "cosine"


class _IntConfig(BaseModel):
    """仅有 int 字段的 Config"""
    n_neighbors: int = Field(default=5, ge=1, le=20)


class _FloatConfig(BaseModel):
    """仅有 float 字段的 Config"""
    percentile: float = Field(default=95.0, ge=50.0, le=99.9)


class _LiteralConfig(BaseModel):
    """Literal 字段的 Config"""
    metric: Literal["euclidean", "manhattan", "cosine"] = "euclidean"


class _EnumConfig(BaseModel):
    """Enum 字段的 Config"""
    metric: _TestMetric = Field(default=_TestMetric.EUCLIDEAN)


class _MixedConfig(BaseModel):
    """混合字段 Config"""
    n_neighbors: int = Field(default=5, ge=1, le=20)
    percentile: float = Field(default=95.0, ge=50.0, le=99.9)
    metric: Literal["euclidean", "manhattan"] = "euclidean"


class _SearchHintConfig(BaseModel):
    """带有 SearchHint 的 Config"""
    learning_rate: Annotated[float, Field(default=0.001, ge=1e-5, le=1e-1),
                             SearchHint(log=True)]
    batch_size: Annotated[int, Field(default=32, ge=8, le=256),
                          SearchHint(step=8)]


class _NoSearchFieldsConfig(BaseModel):
    """无搜索空间的 Config"""
    name: str = "test"
    version: int = 1


# ============================================================================
# SearchHint 测试
# ============================================================================

class TestSearchHint:
    """测试 SearchHint 数据类"""

    def test_default_values(self):
        """
        目的：验证 SearchHint 默认值
        输入：无参数构造 SearchHint()
        预期：log=False, step=None
        """
        hint = SearchHint()
        assert hint.log is False
        assert hint.step is None

    def test_custom_values(self):
        """
        目的：验证 SearchHint 自定义值
        输入：SearchHint(log=True, step=5)
        预期：log=True, step=5
        """
        hint = SearchHint(log=True, step=5)
        assert hint.log is True
        assert hint.step == 5

    def test_frozen(self):
        """
        目的：验证 SearchHint 不可变（frozen）
        输入：尝试修改 log 属性
        预期：抛出 FrozenInstanceError
        """
        hint = SearchHint(log=True)
        with pytest.raises(Exception):  # FrozenInstanceError or AttributeError
            hint.log = False  # type: ignore[misc]

    def test_partial_custom(self):
        """
        目的：验证仅设置 log 或仅设置 step
        输入：SearchHint(log=True), SearchHint(step=3)
        预期：分别 log=True/step=None, log=False/step=3
        """
        hint1 = SearchHint(log=True)
        assert hint1.log is True
        assert hint1.step is None

        hint2 = SearchHint(step=3)
        assert hint2.log is False
        assert hint2.step == 3


# ============================================================================
# extract_search_space 测试
# ============================================================================

class TestExtractSearchSpace:
    """测试 extract_search_space 函数"""

    # --- int 类型 ---

    def test_int_field(self):
        """
        目的：验证 int 字段的搜索空间提取
        输入：_IntConfig (n_neighbors: int, ge=1, le=20)
        预期：type="int", low=1, high=20, step=1, default=5
        """
        space = extract_search_space(_IntConfig)
        assert "n_neighbors" in space
        assert space["n_neighbors"]["type"] == "int"
        assert space["n_neighbors"]["low"] == 1
        assert space["n_neighbors"]["high"] == 20
        assert space["n_neighbors"]["step"] == 1
        assert space["n_neighbors"]["default"] == 5

    def test_int_field_gt_lt(self):
        """
        目的：验证使用 gt/lt（严格不等式）的 int 字段
        输入：Field(gt=0, lt=100) 的 int 字段
        预期：low=0, high=100（gt/lt 语义与 ge/le 提取一致）
        """

        class _GtLtConfig(BaseModel):
            count: int = Field(default=1, gt=0, lt=100)

        space = extract_search_space(_GtLtConfig)
        assert space["count"]["low"] == 0
        assert space["count"]["high"] == 100

    # --- float 类型 ---

    def test_float_field(self):
        """
        目的：验证 float 字段的搜索空间提取
        输入：_FloatConfig (percentile: float, ge=50.0, le=99.9)
        预期：type="float", low=50.0, high=99.9, default=95.0
        """
        space = extract_search_space(_FloatConfig)
        assert "percentile" in space
        assert space["percentile"]["type"] == "float"
        assert space["percentile"]["low"] == 50.0
        assert space["percentile"]["high"] == 99.9
        assert space["percentile"]["default"] == 95.0

    def test_float_field_no_default(self):
        """
        目的：验证无默认值的 float 字段
        输入：Field(ge=0.0, le=1.0) 无 default
        预期：无 default 键
        """

        class _NoDefaultConfig(BaseModel):
            alpha: float = Field(ge=0.0, le=1.0)

        space = extract_search_space(_NoDefaultConfig)
        assert "alpha" in space
        assert "default" not in space["alpha"]

    # --- Literal 类型 ---

    def test_literal_field(self):
        """
        目的：验证 Literal 字段的搜索空间提取
        输入：_LiteralConfig (metric: Literal["euclidean", "manhattan", "cosine"])
        预期：type="cat", choices=["euclidean", "manhattan", "cosine"]
        """
        space = extract_search_space(_LiteralConfig)
        assert "metric" in space
        assert space["metric"]["type"] == "cat"
        assert space["metric"]["choices"] == ["euclidean", "manhattan", "cosine"]

    # --- Enum 类型 ---

    def test_enum_field(self):
        """
        目的：验证 Enum 字段的搜索空间提取
        输入：_EnumConfig (metric: _TestMetric)
        预期：type="cat", choices=["euclidean", "cosine"]
        """
        space = extract_search_space(_EnumConfig)
        assert "metric" in space
        assert space["metric"]["type"] == "cat"
        assert space["metric"]["choices"] == ["euclidean", "cosine"]

    # --- SearchHint ---

    def test_search_hint_log(self):
        """
        目的：验证 SearchHint(log=True) 被正确提取
        输入：_SearchHintConfig 的 learning_rate 字段
        预期：type="float", log=True
        """
        space = extract_search_space(_SearchHintConfig)
        assert "learning_rate" in space
        assert space["learning_rate"]["type"] == "float"
        assert space["learning_rate"]["log"] is True
        assert space["learning_rate"]["low"] == 1e-5
        assert space["learning_rate"]["high"] == 1e-1

    def test_search_hint_step(self):
        """
        目的：验证 SearchHint(step=8) 被正确提取
        输入：_SearchHintConfig 的 batch_size 字段
        预期：type="int", step=8
        """
        space = extract_search_space(_SearchHintConfig)
        assert "batch_size" in space
        assert space["batch_size"]["type"] == "int"
        assert space["batch_size"]["step"] == 8

    # --- 混合字段 ---

    def test_mixed_config(self):
        """
        目的：验证混合字段 Config 全部正确提取
        输入：_MixedConfig (int + float + Literal)
        预期：3 个字段全部正确提取
        """
        space = extract_search_space(_MixedConfig)
        assert len(space) == 3
        assert space["n_neighbors"]["type"] == "int"
        assert space["percentile"]["type"] == "float"
        assert space["metric"]["type"] == "cat"

    # --- 边界和错误 ---

    def test_no_searchable_fields(self):
        """
        目的：验证无搜索字段的 Config 返回空字典
        输入：_NoSearchFieldsConfig (str name, int version)
        预期：返回空 dict
        """
        space = extract_search_space(_NoSearchFieldsConfig)
        assert space == {}

    def test_float_field_no_bounds_skipped(self):
        """
        目的：验证无边界约束的 float 字段被跳过（类似 int 的处理）
        输入：float 字段无 ge/le/gt/lt 约束
        预期：该字段不出现在搜索空间
        """

        class _FloatNoBounds(BaseModel):
            weight: float = Field(default=1.0)

        space = extract_search_space(_FloatNoBounds)
        assert "weight" not in space

    def test_not_a_base_model_raises(self):
        """
        目的：验证非 BaseModel 子类抛出 TypeError
        输入：普通类 class Foo: pass
        预期：抛出 TypeError
        """

        class _NotConfig:
            pass

        with pytest.raises(TypeError):
            extract_search_space(_NotConfig)

    # --- 真实算子 Config ---

    def test_zscore_scorer_config(self):
        """
        目的：验证 ZScoreScorerConfig 搜索空间提取
        输入：ZScoreScorerConfig
        预期：threshold 字段 float 类型，范围 [1.0, 10.0]
        """
        from tsas.engine.operator.detection.zscore import ZScoreScorerConfig
        space = extract_search_space(ZScoreScorerConfig)
        assert "threshold" in space
        assert space["threshold"]["type"] == "float"
        assert space["threshold"]["low"] == 1.0
        assert space["threshold"]["high"] == 10.0

    def test_knn_scorer_config(self):
        """
        目的：验证 KNNScorerConfig 搜索空间提取
        输入：KNNScorerConfig
        预期：3 个字段全部正确提取（n_neighbors, distance_metric, score_method）
        """
        from tsas.engine.operator.detection.knn import KNNScorerConfig
        space = extract_search_space(KNNScorerConfig)
        assert space["n_neighbors"]["type"] == "int"
        assert space["distance_metric"]["type"] == "cat"
        assert space["score_method"]["type"] == "cat"
        assert len(space["distance_metric"]["choices"]) == 2
        assert len(space["score_method"]["choices"]) == 3

    def test_knn_detector_config(self):
        """
        目的：验证 KNNDetectorConfig 搜索空间提取
        输入：KNNDetectorConfig
        预期：4 个字段，含 percentile float 字段
        """
        from tsas.engine.operator.detection.knn import KNNDetectorConfig
        space = extract_search_space(KNNDetectorConfig)
        assert "percentile" in space
        assert space["percentile"]["type"] == "float"
        assert space["percentile"]["low"] == 50.0
        assert space["percentile"]["high"] == 99.9


# ============================================================================
# extract_search_space_from_operator 测试
# ============================================================================

class TestExtractSearchSpaceFromOperator:
    """测试 extract_search_space_from_operator 函数"""

    def test_class_based_operator(self):
        """
        目的：验证从算子实例提取搜索空间
        输入：ZScoreDetector 实例
        预期：包含 threshold 字段
        """
        detector = ZScoreDetector()
        space = extract_search_space_from_operator(detector)
        assert "threshold" in space
        assert space["threshold"]["type"] == "float"

    def test_knn_detector_instance(self):
        """
        目的：验证从 KNNDetector 实例提取搜索空间
        输入：KNNDetector 实例
        预期：4 个字段全部存在
        """
        detector = KNNDetector()
        space = extract_search_space_from_operator(detector)
        assert len(space) == 4
        assert "n_neighbors" in space
        assert "distance_metric" in space
        assert "score_method" in space
        assert "percentile" in space

    def test_composite_operator(self):
        """
        目的：验证从 Composite 算子递归提取搜索空间
        输入：带 Predictor+Scorer+Decider 的 Composite
        预期：各字段带正确前缀
        """
        from tsas.engine.operator.detection.composite import CompositeDetector
        from tsas.engine.operator.detection.pca import PCAPredictor, PCAPredictorConfig
        from tsas.engine.operator.detection.residual_scorer import ResidualScorer
        from tsas.engine.operator.detection.percentile_decider import PercentileDecider

        comp = CompositeDetector(operators=[
            PCAPredictor(config=PCAPredictorConfig(n_components=5)),
            ResidualScorer(),
            PercentileDecider(config=None),
        ])
        space = extract_search_space_from_operator(comp)
        # PCAPredictorConfig 的 n_components 无 ge/le 约束，不会出现在搜索空间中
        # 仅有约束字段（Literal/Enum 或带 ge/le 的数值）才会被提取
        # ResidualScorer 的 metric 字段（Literal 类型）被正确提取
        assert "scorer_0.metric" in space
        # PercentileDecider 使用 Field(gt=0, lt=100)，搜索空间也被提取
        assert "decider.percentile" in space
        # 验证 Composite 递归提取正确识别了子算子角色
        assert len(space) >= 2

    def test_empty_config_operator(self):
        """
        目的：验证 ThresholdDecider 的搜索空间提取
        输入：ThresholdDecider 实例
        预期：包含 threshold 字段
        """
        from tsas.engine.operator.detection.threshold_decider import ThresholdDecider
        decider = ThresholdDecider()
        space = extract_search_space_from_operator(decider)
        # ThresholdDeciderConfig 有 threshold 字段
        assert "threshold" in space


# ============================================================================
# config_to_optuna_suggestions 测试
# ============================================================================

class TestConfigToOptunaSuggestions:
    """测试 config_to_optuna_suggestions 函数（真实 Optuna 集成）"""

    @pytest.fixture(autouse=True)
    def _setup_optuna(self):
        """导入 Optuna 模块"""
        import optuna
        self.optuna = optuna

    # ---- 成功采样测试 ----

    def test_suggest_int(self):
        """
        目的：验证 int 类型搜索空间的 Optuna 采样
        输入：int 搜索空间 {n: {type:"int", low:1, high:10}}
        预期：suggest_int 被调用，返回 [1, 10] 范围内整数
        """
        space = {"n": {"type": "int", "low": 1, "high": 10}}

        captured = {}
        def objective(trial):
            captured.update(config_to_optuna_suggestions(trial, space))
            return 0.0

        study = self.optuna.create_study(direction="maximize")
        study.optimize(objective, n_trials=1)

        assert "n" in captured
        assert isinstance(captured["n"], int)
        assert 1 <= captured["n"] <= 10

    def test_suggest_float(self):
        """
        目的：验证 float 类型搜索空间的 Optuna 采样
        输入：float 搜索空间 {p: {type:"float", low:0.0, high:1.0}}
        预期：suggest_float 被调用，返回 [0.0, 1.0] 范围内浮点数
        """
        space = {"p": {"type": "float", "low": 0.0, "high": 1.0}}

        captured = {}
        def objective(trial):
            captured.update(config_to_optuna_suggestions(trial, space))
            return 0.0

        study = self.optuna.create_study(direction="maximize")
        study.optimize(objective, n_trials=1)

        assert "p" in captured
        assert isinstance(captured["p"], float)
        assert 0.0 <= captured["p"] <= 1.0

    def test_suggest_float_log(self):
        """
        目的：验证 log=True 的 float 搜索空间
        输入：{lr: {type:"float", low:1e-5, high:1e-1, log:True}}
        预期：suggest_float(log=True) 被调用
        """
        space = {"lr": {"type": "float", "low": 1e-5, "high": 1e-1, "log": True}}

        captured = {}
        def objective(trial):
            captured.update(config_to_optuna_suggestions(trial, space))
            return 0.0

        study = self.optuna.create_study(direction="maximize")
        study.optimize(objective, n_trials=1)

        assert "lr" in captured
        assert 1e-5 <= captured["lr"] <= 1e-1

    def test_suggest_int_with_step(self):
        """
        目的：验证 step=4 的 int 搜索空间
        输入：{bs: {type:"int", low:8, high:64, step:4}}
        预期：suggest_int(step=4) 被调用，返回值可被 4 整除
        """
        space = {"bs": {"type": "int", "low": 8, "high": 64, "step": 4}}

        captured = {}
        def objective(trial):
            captured.update(config_to_optuna_suggestions(trial, space))
            return 0.0

        study = self.optuna.create_study(direction="maximize")
        study.optimize(objective, n_trials=1)

        assert "bs" in captured
        assert captured["bs"] % 4 == 0

    def test_suggest_categorical(self):
        """
        目的：验证 cat 类型搜索空间的 Optuna 采样
        输入：{m: {type:"cat", choices:["euclidean","manhattan"]}}
        预期：suggest_categorical 被调用，返回 choices 之一
        """
        space = {"m": {"type": "cat", "choices": ["euclidean", "manhattan"]}}

        captured = {}
        def objective(trial):
            captured.update(config_to_optuna_suggestions(trial, space))
            return 0.0

        study = self.optuna.create_study(direction="maximize")
        study.optimize(objective, n_trials=1)

        assert "m" in captured
        assert captured["m"] in ["euclidean", "manhattan"]

    def test_params_filter(self):
        """
        目的：验证 params 参数过滤功能
        输入：搜索空间包含 a,b,c，params=["a"]
        预期：只返回 a
        """
        space = {
            "a": {"type": "int", "low": 1, "high": 10},
            "b": {"type": "float", "low": 0.0, "high": 1.0},
            "c": {"type": "cat", "choices": ["x", "y"]},
        }

        captured = {}
        def objective(trial):
            captured.update(config_to_optuna_suggestions(trial, space, params=["a"]))
            return 0.0

        study = self.optuna.create_study(direction="maximize")
        study.optimize(objective, n_trials=1)

        assert set(captured.keys()) == {"a"}

    def test_params_none(self):
        """
        目的：验证 params=None 时返回所有参数
        输入：搜索空间包含 a,b，params=None
        预期：返回 a 和 b
        """
        space = {
            "a": {"type": "int", "low": 1, "high": 10},
            "b": {"type": "float", "low": 0.0, "high": 1.0},
        }

        captured = {}
        def objective(trial):
            captured.update(config_to_optuna_suggestions(trial, space))
            return 0.0

        study = self.optuna.create_study(direction="maximize")
        study.optimize(objective, n_trials=1)

        assert set(captured.keys()) == {"a", "b"}

    def test_composite_dot_name(self):
        """
        目的：验证含 '.' 的前缀名被正确处理
        输入：{"predictor.n_components": {type:"int", low:1, high:50}}
        预期：参数名中 '.' 被替换为 '_' 传递给 Optuna，返回的 key 保留原始前缀名
        """
        space = {"predictor.n_components": {"type": "int", "low": 1, "high": 50}}

        captured = {}
        def objective(trial):
            captured.update(config_to_optuna_suggestions(trial, space))
            return 0.0

        study = self.optuna.create_study(direction="maximize")
        study.optimize(objective, n_trials=1)

        assert "predictor.n_components" in captured
        assert isinstance(captured["predictor.n_components"], int)

    # ---- 默认值回退测试 ----

    def test_missing_low_high_uses_default(self):
        """
        目的：验证无 low/high 但有 default 时使用默认值
        输入：{x: {type:"int", default:5}}
        预期：返回 default=5，不调用 suggest_*
        """
        space = {"x": {"type": "int", "default": 5}}

        captured = {}
        def objective(trial):
            captured.update(config_to_optuna_suggestions(trial, space))
            return 0.0

        study = self.optuna.create_study(direction="maximize")
        study.optimize(objective, n_trials=1)

        assert captured["x"] == 5

    def test_none_type_uses_default(self):
        """
        目的：验证 type=None 但有 default 时使用默认值
        输入：{x: {default:42}}
        预期：返回 default=42
        """
        space = {"x": {"default": 42}}

        captured = {}
        def objective(trial):
            captured.update(config_to_optuna_suggestions(trial, space))
            return 0.0

        study = self.optuna.create_study(direction="maximize")
        study.optimize(objective, n_trials=1)

        assert captured["x"] == 42

    # ---- 错误处理测试 ----

    def test_missing_choices_raises(self):
        """
        目的：验证 cat 类型无 choices 抛出 ValueError
        输入：{x: {type:"cat"}} 无 choices
        预期：trial 状态为 FAIL
        """
        space = {"x": {"type": "cat"}}

        def objective(trial):
            config_to_optuna_suggestions(trial, space)
            return 0.0

        study = self.optuna.create_study(direction="maximize")
        # 使用 catch=(ValueError,) 让 Optuna 不向上抛异常
        study.optimize(objective, n_trials=1, catch=(ValueError,))

        # ValueError 被 Optuna 捕获，trial 标记为 FAIL
        assert study.trials[0].state == self.optuna.trial.TrialState.FAIL

    def test_unknown_type_raises(self):
        """
        目的：验证未知类型抛出 ValueError
        输入：{x: {type:"unknown"}}
        预期：trial 状态为 FAIL
        """
        space = {"x": {"type": "unknown", "choices": [1, 2, 3]}}

        def objective(trial):
            config_to_optuna_suggestions(trial, space)
            return 0.0

        study = self.optuna.create_study(direction="maximize")
        # 使用 catch=(ValueError,) 让 Optuna 不向上抛异常
        study.optimize(objective, n_trials=1, catch=(ValueError,))

        # ValueError 被 Optuna 捕获，trial 标记为 FAIL
        assert study.trials[0].state == self.optuna.trial.TrialState.FAIL


    def test_suggest_param_not_in_space(self):
        """
        目的：验证 params 包含不存在于搜索空间的参数名时被跳过
        输入：搜索空间包含 a，params=["a", "nonexistent"]
        预期：只返回 a，不存在的参数被跳过
        """
        space = {"a": {"type": "int", "low": 1, "high": 10}}

        captured = {}
        def objective(trial):
            captured.update(config_to_optuna_suggestions(
                trial, space, params=["a", "nonexistent"]
            ))
            return 0.0

        study = self.optuna.create_study(direction="maximize")
        study.optimize(objective, n_trials=1)

        assert set(captured.keys()) == {"a"}

    def test_suggest_float_missing_bounds_uses_default(self):
        """
        目的：验证 float 类型无 low/high 但有 default 时使用默认值
        输入：{x: {type:"float", default:3.14}}
        预期：返回 default=3.14
        """
        space = {"x": {"type": "float", "default": 3.14}}

        captured = {}
        def objective(trial):
            captured.update(config_to_optuna_suggestions(trial, space))
            return 0.0

        study = self.optuna.create_study(direction="maximize")
        study.optimize(objective, n_trials=1)

        assert captured["x"] == 3.14

    # ---- 缺失 bounds 且无 default 的错误测试 ----

    def test_suggest_int_missing_bounds_no_default(self):
        """
        目的：验证 int 类型无 low/high 且无 default 时抛出 ValueError
        输入：{x: {type:"int"}} 无 low/high/default
        预期：trial 状态为 FAIL（ValueError 被 Optuna catch 捕获）
        """
        space = {"x": {"type": "int"}}

        def objective(trial):
            config_to_optuna_suggestions(trial, space)
            return 0.0

        study = self.optuna.create_study(direction="maximize")
        study.optimize(objective, n_trials=1, catch=(ValueError,))

        assert study.trials[0].state == self.optuna.trial.TrialState.FAIL

    def test_suggest_float_missing_bounds_no_default(self):
        """
        目的：验证 float 类型无 low/high 且无 default 时抛出 ValueError
        输入：{x: {type:"float"}} 无 low/high/default
        预期：trial 状态为 FAIL（ValueError 被 Optuna catch 捕获）
        """
        space = {"x": {"type": "float"}}

        def objective(trial):
            config_to_optuna_suggestions(trial, space)
            return 0.0

        study = self.optuna.create_study(direction="maximize")
        study.optimize(objective, n_trials=1, catch=(ValueError,))

        assert study.trials[0].state == self.optuna.trial.TrialState.FAIL
