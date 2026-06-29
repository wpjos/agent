# -*- coding: utf-8 -*-

"""
HPO 优化结果数据模型单元测试

对应源文件：
- result.py: TrialInfo, HPOResult

测试范围：
- TrialInfo 基本属性（number/params/scores/operator）
- TrialInfo.score 属性（正常/空 scores）
- TrialInfo.score_name 属性（正常/空 scores）
- HPOResult 基本属性（best_trials/all_trials/top_k/directions）
- HPOResult.best_params / best_score / best_score_value / best_operator
- HPOResult 空结果边界情况
- HPOResult.__repr__ 输出格式
"""

import numpy as np
import pytest

from tsas.engine.hpo.result import HPOResult, TrialInfo


class TestTrialInfo:
    """测试 TrialInfo 数据类"""

    def test_basic_attributes(self):
        """
        目的：验证 TrialInfo 基本属性存储
        输入：number=0, params={"a":1}, scores={"f1":0.85}
        预期：各属性值正确存储
        """
        trial = TrialInfo(
            number=0,
            params={"a": 1, "b": 2.5},
            scores={"f1": 0.85},
        )
        assert trial.number == 0
        assert trial.params == {"a": 1, "b": 2.5}
        assert trial.scores == {"f1": 0.85}
        assert trial.operator is None

    def test_with_operator(self):
        """
        目的：验证带算子实例的 TrialInfo
        输入：operator=某算子实例
        预期：operator 正确存储
        """
        from tsas.engine.operator.detection.zscore import ZScoreDetector
        detector = ZScoreDetector()
        trial = TrialInfo(
            number=1,
            params={"threshold": 3.0},
            scores={"f1": 0.90},
            operator=detector,
        )
        assert trial.operator is detector

    def test_score_single(self):
        """
        目的：验证单目标分数的 score 属性
        输入：scores={"f1": 0.85}
        预期：score=0.85
        """
        trial = TrialInfo(number=0, params={}, scores={"f1": 0.85})
        assert trial.score == 0.85

    def test_score_multi(self):
        """
        目的：验证多目标分数的 score 属性（返回第一个）
        输入：scores={"f1": 0.8, "precision": 0.9}
        预期：score=0.8（第一个值）
        """
        trial = TrialInfo(number=0, params={}, scores={"f1": 0.8, "precision": 0.9})
        assert trial.score == 0.8

    def test_score_empty(self):
        """
        目的：验证空 scores 的 score 属性
        输入：scores={}
        预期：score=float('-inf')
        """
        trial = TrialInfo(number=0, params={}, scores={})
        assert trial.score == float('-inf')

    def test_score_name(self):
        """
        目的：验证 score_name 返回第一个键名
        输入：scores={"f1": 0.8, "precision": 0.9}
        预期：score_name="f1"
        """
        trial = TrialInfo(number=0, params={}, scores={"f1": 0.8, "precision": 0.9})
        assert trial.score_name == "f1"

    def test_score_name_empty(self):
        """
        目的：验证空 scores 的 score_name
        输入：scores={}
        预期：score_name=""
        """
        trial = TrialInfo(number=0, params={}, scores={})
        assert trial.score_name == ""


class TestHPOResult:
    """测试 HPOResult 数据类"""

    # ---- 基本属性 ----

    def test_basic_attributes(self):
        """
        目的：验证 HPOResult 基本属性
        输入：best_trials=[trial1], all_trials=[trial1,trial2]
        预期：各属性正确存储
        """
        t1 = TrialInfo(number=0, params={"a": 1}, scores={"f1": 0.9})
        t2 = TrialInfo(number=1, params={"a": 2}, scores={"f1": 0.7})
        result = HPOResult(
            best_trials=[t1],
            all_trials=[t1, t2],
            top_k=1,
            directions=["maximize"],
            search_space={"a": {"type": "int", "low": 1, "high": 10}},
            metric_names=["f1"],
        )
        assert result.best_trials == [t1]
        assert result.all_trials == [t1, t2]
        assert result.top_k == 1
        assert result.directions == ["maximize"]
        assert result.search_space == {"a": {"type": "int", "low": 1, "high": 10}}
        assert result.metric_names == ["f1"]

    def test_default_values(self):
        """
        目的：验证 HPOResult 默认值
        输入：HPOResult() 无参构造
        预期：所有列表为空，top_k=1, directions=["maximize"]
        """
        result = HPOResult()
        assert result.best_trials == []
        assert result.all_trials == []
        assert result.top_k == 1
        assert result.directions == ["maximize"]
        assert result.search_space == {}
        assert result.metric_names == []

    # ---- best_params ----

    def test_best_params(self):
        """
        目的：验证 best_params 返回首个 best_trial 的 params
        输入：best_trials=[TrialInfo(params={"a": 1})]
        预期：best_params={"a": 1}
        """
        t1 = TrialInfo(number=0, params={"a": 1, "b": 2.0}, scores={"f1": 0.9})
        result = HPOResult(best_trials=[t1])
        assert result.best_params == {"a": 1, "b": 2.0}

    def test_best_params_empty_raises(self):
        """
        目的：验证空 best_trials 访问 best_params 抛异常
        输入：best_trials=[]
        预期：抛出 IndexError
        """
        result = HPOResult()
        with pytest.raises(IndexError, match="best_trials 为空"):
            _ = result.best_params

    # ---- best_score ----

    def test_best_score(self):
        """
        目的：验证 best_score 返回首个 best_trial 的 scores
        输入：best_trials=[TrialInfo(scores={"f1":0.9})]
        预期：best_score={"f1":0.9}
        """
        t1 = TrialInfo(number=0, params={}, scores={"f1": 0.9})
        result = HPOResult(best_trials=[t1])
        assert result.best_score == {"f1": 0.9}

    def test_best_score_empty_raises(self):
        """
        目的：验证空 best_trials 访问 best_score 抛异常
        输入：best_trials=[]
        预期：抛出 IndexError
        """
        result = HPOResult()
        with pytest.raises(IndexError, match="best_trials 为空"):
            _ = result.best_score

    # ---- best_score_value ----

    def test_best_score_value(self):
        """
        目的：验证 best_score_value 返回 score 值
        输入：best_trials=[TrialInfo(scores={"f1":0.92})]
        预期：best_score_value=0.92
        """
        t1 = TrialInfo(number=0, params={}, scores={"f1": 0.92})
        result = HPOResult(best_trials=[t1])
        assert result.best_score_value == 0.92

    def test_best_score_value_empty_raises(self):
        """
        目的：验证空 best_trials 访问 best_score_value 抛异常
        输入：best_trials=[]
        预期：抛出 IndexError
        """
        result = HPOResult()
        with pytest.raises(IndexError, match="best_trials 为空"):
            _ = result.best_score_value

    # ---- best_operator ----

    def test_best_operator(self):
        """
        目的：验证 best_operator 返回算子实例
        输入：best_trials=[TrialInfo(operator=detector)]
        预期：best_operator is detector
        """
        from tsas.engine.operator.detection.zscore import ZScoreDetector
        detector = ZScoreDetector()
        t1 = TrialInfo(number=0, params={}, scores={"f1": 0.9}, operator=detector)
        result = HPOResult(best_trials=[t1])
        assert result.best_operator is detector

    def test_best_operator_none(self):
        """
        目的：验证 operator=None 的 best_operator 返回 None
        输入：best_trials=[TrialInfo(operator=None)]
        预期：best_operator=None
        """
        t1 = TrialInfo(number=0, params={}, scores={"f1": 0.9}, operator=None)
        result = HPOResult(best_trials=[t1])
        assert result.best_operator is None

    def test_best_operator_empty_raises(self):
        """
        目的：验证空 best_trials 访问 best_operator 抛异常
        输入：best_trials=[]
        预期：抛出 IndexError
        """
        result = HPOResult()
        with pytest.raises(IndexError, match="best_trials 为空"):
            _ = result.best_operator

    # ---- __repr__ ----

    def test_repr_empty(self):
        """
        目的：验证空 HPOResult 的字符串表示
        输入：HPOResult()
        预期：包含 'HPOResult' 及 trial 统计
        """
        result = HPOResult()
        rep = repr(result)
        assert "HPOResult" in rep
        assert "0/0" in rep

    def test_repr_with_trials(self):
        """
        目的：验证有 trial 的 HPOResult 字符串表示
        输入：HPOResult(best_trials=[t1], all_trials=[t1,t2])
        预期：包含分数信息和统计
        """
        t1 = TrialInfo(number=0, params={}, scores={"f1": 0.95})
        t2 = TrialInfo(number=1, params={}, scores={"f1": 0.80})
        result = HPOResult(best_trials=[t1], all_trials=[t1, t2],
                           directions=["maximize"])
        rep = repr(result)
        assert "HPOResult" in rep
        assert "1/2" in rep
        assert "maximize" in rep

    # ---- multi-objective ----

    def test_multi_objective_directions(self):
        """
        目的：验证多目标优化方向的存储
        输入：directions=["maximize", "minimize"]
        预期：directions 正确存储
        """
        result = HPOResult(directions=["maximize", "minimize"])
        assert result.directions == ["maximize", "minimize"]

    def test_top_k_greater_than_one(self):
        """
        目的：验证 top_k > 1 的情况
        输入：top_k=5
        预期：top_k=5
        """
        result = HPOResult(top_k=5)
        assert result.top_k == 5
