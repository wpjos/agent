# -*- coding: utf-8 -*-

"""
特征构造算子 CLI 单元测试

对应源文件：
- cli/feature_construction.py

测试范围：
- help 子命令（列表模式和详情模式）
- run 子命令（含配置文件加载、算子执行、输出保存）
- fit 子命令（可训练算子训练和保存）
- 配置校验错误场景
- 无子命令退出
"""

import json

import numpy as np
import pandas as pd
import pytest

from tsas.engine.operator.cli.feature_construction import main, create_registry


# ============================================================================
# 公共 fixture
# ============================================================================

@pytest.fixture
def sample_csv(tmp_path):
    """创建测试 CSV 数据文件（4行3列）"""
    df = pd.DataFrame({
        'col_a': [1.0, 2.0, 3.0, 4.0],
        'col_b': [5.0, 6.0, 7.0, 8.0],
        'col_c': [9.0, 10.0, 11.0, 12.0],
    })
    path = tmp_path / "input.csv"
    df.to_csv(path, index=False)
    return path


@pytest.fixture
def run_config(tmp_path):
    """创建 run 子命令的配置文件"""
    config = {
        "operators": [
            {"name": "square_feature", "config": {"input_columns": ["col_a", "col_b"]}}
        ],
        "keep_original": True,
    }
    path = tmp_path / "config.json"
    with open(path, 'w') as f:
        json.dump(config, f)
    return path


# ============================================================================
# 测试类
# ============================================================================

class TestFeatureConstructionHelp:
    """测试 feature_construction help 子命令"""

    def test_help_list(self, capsys):
        """
        目的：验证 help 无参数时列出所有特征算子
        输入：['help']
        预期：输出包含 square_feature 等算子名称
        """
        main(['help'])
        captured = capsys.readouterr()
        assert "square_feature" in captured.out
        assert "polynomial_feature" in captured.out

    def test_help_detail(self, capsys):
        """
        目的：验证 help 带算子名称时输出详情
        输入：['help', 'square_feature']
        预期：输出包含 square_feature 的描述和参数
        """
        main(['help', 'square_feature'])
        captured = capsys.readouterr()
        assert "## square_feature" in captured.out

    def test_help_unknown_operator(self):
        """
        目的：验证 help 查找不存在的算子时报错
        输入：['help', 'nonexistent']
        预期：抛出 KeyError
        """
        with pytest.raises(KeyError, match="未找到名为"):
            main(['help', 'nonexistent'])


class TestFeatureConstructionRun:
    """测试 feature_construction run 子命令"""

    def test_run_basic(self, sample_csv, run_config, tmp_path, capsys):
        """
        目的：验证 run 基本流程（加载数据→执行算子→保存结果）
        输入：输入 CSV + square_feature 配置
        预期：输出文件包含原始列和平方后的新列
        """
        output_path = tmp_path / "output.csv"
        main(['run', '--input', str(sample_csv), '--output', str(output_path),
              '--config', str(run_config)])

        captured = capsys.readouterr()
        assert "特征构造完成" in captured.out

        result = pd.read_csv(output_path)
        # keep_original=True，应保留原始 3 列 + 新增列
        assert result.shape[0] == 4
        assert result.shape[1] > 3

    def test_run_no_keep_original(self, sample_csv, tmp_path, capsys):
        """
        目的：验证 keep_original=False 时不保留原始列
        输入：keep_original 设为 false 的配置
        预期：输出只包含算子产出的列
        """
        config = {
            "operators": [
                {"name": "square_feature", "config": {"input_columns": ["col_a"]}}
            ],
            "keep_original": False,
        }
        config_path = tmp_path / "config.json"
        with open(config_path, 'w') as f:
            json.dump(config, f)

        output_path = tmp_path / "output.csv"
        main(['run', '--input', str(sample_csv), '--output', str(output_path),
              '--config', str(config_path)])

        result = pd.read_csv(output_path)
        # 不保留原始列，只有算子输出
        assert result.shape[0] == 4

    def test_run_empty_operators_raises(self, sample_csv, tmp_path):
        """
        目的：验证空算子列表时报错
        输入：operators 为空列表的配置
        预期：抛出 ValueError
        """
        config = {"operators": []}
        config_path = tmp_path / "config.json"
        with open(config_path, 'w') as f:
            json.dump(config, f)

        output_path = tmp_path / "output.csv"
        with pytest.raises(ValueError, match="不能为空"):
            main(['run', '--input', str(sample_csv), '--output', str(output_path),
                  '--config', str(config_path)])

    def test_run_missing_operator_name_raises(self, sample_csv, tmp_path):
        """
        目的：验证算子缺少 name 字段时报错
        输入：算子配置中没有 name 字段
        预期：抛出 ValueError
        """
        config = {"operators": [{"config": {}}]}
        config_path = tmp_path / "config.json"
        with open(config_path, 'w') as f:
            json.dump(config, f)

        output_path = tmp_path / "output.csv"
        with pytest.raises(ValueError, match="缺少 'name'"):
            main(['run', '--input', str(sample_csv), '--output', str(output_path),
                  '--config', str(config_path)])


class TestFeatureConstructionFit:
    """测试 feature_construction fit 子命令"""

    def test_fit_non_learnable_skips(self, sample_csv, tmp_path, capsys):
        """
        目的：验证非可训练算子被跳过
        输入：square_feature（不需要训练）
        预期：输出提示不需要训练
        """
        config = {
            "operators": [
                {"name": "square_feature", "config": {"input_columns": ["col_a"]}}
            ],
        }
        config_path = tmp_path / "config.json"
        with open(config_path, 'w') as f:
            json.dump(config, f)

        main(['fit', '--input', str(sample_csv), '--config', str(config_path)])
        captured = capsys.readouterr()
        assert "不需要训练" in captured.out

    def test_fit_learnable_and_save(self, sample_csv, tmp_path, capsys):
        """
        目的：验证可训练算子的 fit + save 流程
        输入：pca_feature 配置 + --save 参数
        预期：训练完成并保存模型到指定目录
        """
        config = {
            "operators": [
                {"name": "pca_feature", "config": {"input_columns": ["col_a", "col_b", "col_c"], "n_components": 2}}
            ],
        }
        config_path = tmp_path / "config.json"
        with open(config_path, 'w') as f:
            json.dump(config, f)

        save_dir = tmp_path / "model"
        main(['fit', '--input', str(sample_csv), '--config', str(config_path),
              '--save', str(save_dir)])

        captured = capsys.readouterr()
        assert "训练完成" in captured.out
        assert "模型已保存" in captured.out
        assert save_dir.exists()


class TestFeatureConstructionNoCommand:
    """测试无子命令场景"""

    def test_no_command_exits(self):
        """
        目的：验证不提供子命令时 sys.exit(1)
        输入：空参数
        预期：SystemExit(1)
        """
        with pytest.raises(SystemExit):
            main([])


class TestCreateRegistry:
    """测试 create_registry 工厂函数"""

    def test_create_registry(self):
        """
        目的：验证工厂函数返回已 discover 的注册中心
        输入：无
        预期：返回包含特征算子的 OperatorRegistry
        """
        registry = create_registry()
        assert registry.discovered is True
        assert 'square_feature' in registry.list_all()
