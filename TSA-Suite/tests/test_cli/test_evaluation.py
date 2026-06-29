# -*- coding: utf-8 -*-

"""
评价指标算子 CLI 单元测试

对应源文件：
- cli/evaluation.py

测试范围：
- help 子命令（列表模式和详情模式）
- run 子命令（单算子、多算子、alias 去重）
- 配置校验错误场景
- _resolve_output_key 去重逻辑
- _result_to_dict 序列化逻辑
"""

import json

import numpy as np
import pandas as pd
import pytest
from pydantic import BaseModel

from tsas.engine.operator.cli.evaluation import (
    main,
    create_registry,
    _resolve_output_key,
    _result_to_dict,
)


# ============================================================================
# 公共 fixture
# ============================================================================

@pytest.fixture
def binary_csv(tmp_path):
    """创建二分类评价用 CSV（含 label 和 predict 列）"""
    df = pd.DataFrame({
        'label': [0, 0, 1, 1, 0, 1, 0, 1, 1, 0],
        'predict': [0, 1, 1, 1, 0, 0, 0, 1, 1, 1],
    })
    path = tmp_path / "predictions.csv"
    df.to_csv(path, index=False)
    return path


@pytest.fixture
def score_csv(tmp_path):
    """创建单列分数 CSV（供 self_evaluation 使用）"""
    np.random.seed(42)
    df = pd.DataFrame({'score': np.random.randn(50)})
    path = tmp_path / "scores.csv"
    df.to_csv(path, index=False)
    return path


# ============================================================================
# 测试类
# ============================================================================

class TestEvaluationHelp:
    """测试 evaluation help 子命令"""

    def test_help_list(self, capsys):
        """
        目的：验证 help 无参数时列出所有评价指标算子
        输入：['help']
        预期：输出包含 binary_classification 等算子名称
        """
        main(['help'])
        captured = capsys.readouterr()
        assert "binary_classification" in captured.out
        assert "self_evaluation" in captured.out

    def test_help_detail(self, capsys):
        """
        目的：验证 help 带算子名称时输出详情
        输入：['help', 'binary_classification']
        预期：输出包含算子描述和参数
        """
        main(['help', 'binary_classification'])
        captured = capsys.readouterr()
        assert "## binary_classification" in captured.out

    def test_help_unknown_operator(self):
        """
        目的：验证 help 查找不存在的算子时报错
        输入：['help', 'nonexistent']
        预期：抛出 KeyError
        """
        with pytest.raises(KeyError, match="未找到名为"):
            main(['help', 'nonexistent'])


class TestEvaluationRun:
    """测试 evaluation run 子命令"""

    def test_run_single_operator(self, binary_csv, tmp_path, capsys):
        """
        目的：验证单个评价算子的 run 流程
        输入：binary_classification_metric 配置 + 二分类数据
        预期：输出 JSON 包含 result 和 main_scores
        """
        config = {
            "operators": [
                {
                    "name": "binary_classification",
                    "truth_columns": ["label"],
                    "predict_columns": ["predict"],
                }
            ]
        }
        config_path = tmp_path / "config.json"
        with open(config_path, 'w') as f:
            json.dump(config, f)

        output_path = tmp_path / "result.json"
        main(['run', '--input', str(binary_csv), '--output', str(output_path),
              '--config', str(config_path)])

        captured = capsys.readouterr()
        assert "评价完成" in captured.out

        with open(output_path, 'r', encoding='utf-8') as f:
            result = json.load(f)

        assert "results" in result
        assert "binary_classification" in result["results"]
        assert "result" in result["results"]["binary_classification"]

    def test_run_multiple_operators(self, binary_csv, tmp_path, capsys):
        """
        目的：验证多个评价算子的 run 流程
        输入：两个不同算子的配置
        预期：输出 JSON 包含两个算子的结果
        """
        config = {
            "operators": [
                {
                    "name": "binary_classification",
                    "truth_columns": ["label"],
                    "predict_columns": ["predict"],
                },
                {
                    "name": "point_adjust",
                    "truth_columns": ["label"],
                    "predict_columns": ["predict"],
                },
            ]
        }
        config_path = tmp_path / "config.json"
        with open(config_path, 'w') as f:
            json.dump(config, f)

        output_path = tmp_path / "result.json"
        main(['run', '--input', str(binary_csv), '--output', str(output_path),
              '--config', str(config_path)])

        with open(output_path, 'r', encoding='utf-8') as f:
            result = json.load(f)

        assert "binary_classification" in result["results"]
        assert "point_adjust" in result["results"]

    def test_run_with_alias(self, binary_csv, tmp_path, capsys):
        """
        目的：验证 alias 字段作为输出 key
        输入：配置中指定 alias
        预期：输出 JSON 使用 alias 作为 key
        """
        config = {
            "operators": [
                {
                    "name": "binary_classification",
                    "alias": "my_metric",
                    "truth_columns": ["label"],
                    "predict_columns": ["predict"],
                }
            ]
        }
        config_path = tmp_path / "config.json"
        with open(config_path, 'w') as f:
            json.dump(config, f)

        output_path = tmp_path / "result.json"
        main(['run', '--input', str(binary_csv), '--output', str(output_path),
              '--config', str(config_path)])

        with open(output_path, 'r', encoding='utf-8') as f:
            result = json.load(f)

        assert "my_metric" in result["results"]

    def test_run_single_input(self, score_csv, tmp_path, capsys):
        """
        目的：验证单输入算子（self_evaluation）的 run 流程
        输入：单列分数数据 + input_columns 配置
        预期：输出 JSON 包含 self_evaluation 结果
        """
        config = {
            "operators": [
                {
                    "name": "self_evaluation",
                    "input_columns": ["score"],
                }
            ]
        }
        config_path = tmp_path / "config.json"
        with open(config_path, 'w') as f:
            json.dump(config, f)

        output_path = tmp_path / "result.json"
        main(['run', '--input', str(score_csv), '--output', str(output_path),
              '--config', str(config_path)])

        with open(output_path, 'r', encoding='utf-8') as f:
            result = json.load(f)

        assert "self_evaluation" in result["results"]

    def test_run_empty_operators_raises(self, binary_csv, tmp_path):
        """
        目的：验证空算子列表时报错
        输入：operators 为空列表
        预期：抛出 ValueError
        """
        config = {"operators": []}
        config_path = tmp_path / "config.json"
        with open(config_path, 'w') as f:
            json.dump(config, f)

        output_path = tmp_path / "result.json"
        with pytest.raises(ValueError, match="不能为空"):
            main(['run', '--input', str(binary_csv), '--output', str(output_path),
                  '--config', str(config_path)])

    def test_run_missing_name_raises(self, binary_csv, tmp_path):
        """
        目的：验证算子缺少 name 字段时报错
        输入：无 name 的算子配置
        预期：抛出 ValueError
        """
        config = {"operators": [{"truth_columns": ["label"]}]}
        config_path = tmp_path / "config.json"
        with open(config_path, 'w') as f:
            json.dump(config, f)

        output_path = tmp_path / "result.json"
        with pytest.raises(ValueError, match="缺少 'name'"):
            main(['run', '--input', str(binary_csv), '--output', str(output_path),
                  '--config', str(config_path)])


class TestResolveOutputKey:
    """测试 _resolve_output_key 去重逻辑"""

    def test_no_alias_uses_name(self):
        """
        目的：验证无 alias 时使用算子名称
        输入：无 alias 的 spec
        预期：返回算子名称
        """
        used = set()
        key = _resolve_output_key({}, "my_op", used)
        assert key == "my_op"

    def test_alias_overrides_name(self):
        """
        目的：验证 alias 覆盖算子名称
        输入：spec 中有 alias
        预期：返回 alias
        """
        used = set()
        key = _resolve_output_key({"alias": "custom"}, "my_op", used)
        assert key == "custom"

    def test_auto_dedup(self):
        """
        目的：验证重复 key 自动追加后缀
        输入：已使用的 key 集合中已有 "my_op"
        预期：返回 "my_op_1"
        """
        used = {"my_op"}
        key = _resolve_output_key({}, "my_op", used)
        assert key == "my_op_1"

    def test_auto_dedup_multiple(self):
        """
        目的：验证多次重复时后缀递增
        输入：已使用 "my_op" 和 "my_op_1"
        预期：返回 "my_op_2"
        """
        used = {"my_op", "my_op_1"}
        key = _resolve_output_key({}, "my_op", used)
        assert key == "my_op_2"


class TestResultToDict:
    """测试 _result_to_dict 序列化逻辑"""

    def test_float(self):
        """
        目的：验证 float 直接返回
        输入：0.85
        预期：返回 0.85
        """
        assert _result_to_dict(0.85) == 0.85

    def test_int(self):
        """
        目的：验证 int 直接返回
        输入：42
        预期：返回 42
        """
        assert _result_to_dict(42) == 42

    def test_basemodel(self):
        """
        目的：验证 BaseModel 转为字典
        输入：Pydantic 模型实例
        预期：返回 model_dump() 结果
        """
        class _M(BaseModel):
            f1: float = 0.85

        result = _result_to_dict(_M())
        assert result == {"f1": 0.85}

    def test_ndarray(self):
        """
        目的：验证 ndarray 转为列表
        输入：numpy 数组
        预期：返回 tolist() 结果
        """
        arr = np.array([1.0, 2.0, 3.0])
        result = _result_to_dict(arr)
        assert result == [1.0, 2.0, 3.0]

    def test_dict(self):
        """
        目的：验证嵌套字典递归转换
        输入：包含 ndarray 的字典
        预期：内部 ndarray 被转为列表
        """
        data = {"a": np.array([1, 2])}
        result = _result_to_dict(data)
        assert result == {"a": [1, 2]}

    def test_other_type(self):
        """
        目的：验证其他类型转为字符串
        输入：非标准类型对象
        预期：返回 str() 结果
        """
        result = _result_to_dict(object())
        assert isinstance(result, str)


class TestEvaluationNoCommand:
    """测试无子命令场景"""

    def test_no_command_exits(self):
        """
        目的：验证不提供子命令时 sys.exit(1)
        输入：空参数
        预期：SystemExit
        """
        with pytest.raises(SystemExit):
            main([])


class TestCreateEvaluationRegistry:
    """测试 create_registry 工厂函数"""

    def test_create_registry(self):
        """
        目的：验证工厂函数返回已 discover 的注册中心
        输入：无
        预期：返回包含评价指标算子的 OperatorRegistry
        """
        registry = create_registry()
        assert registry.discovered is True
        assert 'binary_classification' in registry.list_all()
        assert 'self_evaluation' in registry.list_all()
