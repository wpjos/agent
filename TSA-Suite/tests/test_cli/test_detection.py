# -*- coding: utf-8 -*-

"""
异常检测算子 CLI 单元测试

对应源文件：
- cli/detection.py

测试范围：
- help 子命令（列表模式和详情模式）
- run 子命令（含 fit→save→load→run 完整流程）
- fit 子命令（训练和保存）
- 配置校验错误场景
- 非可训练算子跳过训练
"""

import json

import numpy as np
import pandas as pd
import pytest

from tsas.engine.operator.cli.detection import main, create_registry


# ============================================================================
# 公共 fixture
# ============================================================================

@pytest.fixture
def train_csv(tmp_path):
    """创建训练 CSV 数据文件（100行3列，标准正态分布）"""
    np.random.seed(42)
    df = pd.DataFrame(
        np.random.randn(100, 3),
        columns=['sensor_1', 'sensor_2', 'sensor_3'],
    )
    path = tmp_path / "train.csv"
    df.to_csv(path, index=False)
    return path


@pytest.fixture
def test_csv(tmp_path):
    """创建测试 CSV 数据文件（20行3列，含异常点）"""
    np.random.seed(123)
    normal = np.random.randn(18, 3)
    abnormal = np.random.randn(2, 3) + 10
    data = np.vstack([normal, abnormal])
    df = pd.DataFrame(data, columns=['sensor_1', 'sensor_2', 'sensor_3'])
    path = tmp_path / "test.csv"
    df.to_csv(path, index=False)
    return path


# ============================================================================
# 测试类
# ============================================================================

class TestDetectionHelp:
    """测试 detection help 子命令"""

    def test_help_list(self, capsys):
        """
        目的：验证 help 无参数时列出所有检测算子
        输入：['help']
        预期：输出包含 knn_scorer, knn_detector 等算子名称
        """
        main(['help'])
        captured = capsys.readouterr()
        assert "knn_scorer" in captured.out
        assert "knn_detector" in captured.out

    def test_help_detail(self, capsys):
        """
        目的：验证 help 带算子名称时输出详情
        输入：['help', 'knn_scorer']
        预期：输出包含 knn_scorer 的参数表格
        """
        main(['help', 'knn_scorer'])
        captured = capsys.readouterr()
        assert "## knn_scorer" in captured.out
        assert "n_neighbors" in captured.out

    def test_help_unknown_operator(self):
        """
        目的：验证 help 查找不存在的算子时报错
        输入：['help', 'nonexistent']
        预期：抛出 KeyError
        """
        with pytest.raises(KeyError, match="未找到名为"):
            main(['help', 'nonexistent'])


class TestDetectionFitAndRun:
    """测试 detection fit 和 run 完整流程"""

    def test_fit_and_save(self, train_csv, tmp_path, capsys):
        """
        目的：验证 KNN 检测器的 fit + save 流程
        输入：训练数据 + KNN 配置 + --save 参数
        预期：fit 成功，模型保存到指定目录
        """
        config = {
            "operator": {
                "name": "knn_detector",
                "input_columns": ["sensor_1", "sensor_2", "sensor_3"],
                "config": {"n_neighbors": 5, "percentile": 95.0},
            }
        }
        config_path = tmp_path / "config.json"
        with open(config_path, 'w') as f:
            json.dump(config, f)

        # fit + save
        save_dir = tmp_path / "model"
        main(['fit', '--input', str(train_csv), '--config', str(config_path),
              '--save', str(save_dir)])

        captured = capsys.readouterr()
        assert "训练完成" in captured.out
        assert "模型已保存" in captured.out
        assert save_dir.exists()

    def test_run_threshold_decider(self, test_csv, tmp_path, capsys):
        """
        目的：验证无需训练的算子（threshold_decider）直接 run
        输入：测试数据 + threshold_decider 配置
        预期：检测结果 CSV 包含原始列和检测结果列
        """
        config = {
            "operator": {
                "name": "threshold_decider",
                "input_columns": ["sensor_1"],
                "config": {"threshold": 3.0},
            }
        }
        config_path = tmp_path / "config.json"
        with open(config_path, 'w') as f:
            json.dump(config, f)

        output_path = tmp_path / "result.csv"
        main(['run', '--input', str(test_csv), '--output', str(output_path),
              '--config', str(config_path)])

        captured = capsys.readouterr()
        assert "异常检测完成" in captured.out

        result = pd.read_csv(output_path)
        assert result.shape[0] == 20
        assert result.shape[1] > 3

    def test_fit_non_learnable(self, test_csv, tmp_path, capsys):
        """
        目的：验证非可训练算子（如 threshold_decider）的 fit 提示不需要训练
        输入：threshold_decider 配置
        预期：输出提示不需要训练
        """
        config = {
            "operator": {
                "name": "threshold_decider",
                "config": {"threshold": 3.0},
            }
        }
        config_path = tmp_path / "config.json"
        with open(config_path, 'w') as f:
            json.dump(config, f)

        main(['fit', '--input', str(test_csv), '--config', str(config_path)])
        captured = capsys.readouterr()
        assert "不需要训练" in captured.out


class TestDetectionConfigErrors:
    """测试检测算子配置错误场景"""

    def test_missing_operator_field(self, train_csv, tmp_path):
        """
        目的：验证配置中缺少 operator 字段时报错
        输入：空配置
        预期：抛出 ValueError
        """
        config = {}
        config_path = tmp_path / "config.json"
        with open(config_path, 'w') as f:
            json.dump(config, f)

        with pytest.raises(ValueError, match="缺少 'operator'"):
            main(['fit', '--input', str(train_csv), '--config', str(config_path)])

    def test_missing_operator_name(self, train_csv, tmp_path):
        """
        目的：验证算子配置中缺少 name 字段时报错
        输入：无 name 的算子配置
        预期：抛出 ValueError
        """
        config = {"operator": {"config": {}}}
        config_path = tmp_path / "config.json"
        with open(config_path, 'w') as f:
            json.dump(config, f)

        with pytest.raises(ValueError, match="缺少 'name'"):
            main(['fit', '--input', str(train_csv), '--config', str(config_path)])


class TestDetectionNoCommand:
    """测试无子命令场景"""

    def test_no_command_exits(self):
        """
        目的：验证不提供子命令时 sys.exit(1)
        输入：空参数
        预期：SystemExit
        """
        with pytest.raises(SystemExit):
            main([])


class TestCreateDetectionRegistry:
    """测试 create_registry 工厂函数"""

    def test_create_registry(self):
        """
        目的：验证工厂函数返回已 discover 的注册中心
        输入：无
        预期：返回包含检测算子的 OperatorRegistry
        """
        registry = create_registry()
        assert registry.discovered is True
        assert 'knn_scorer' in registry.list_all()
        assert 'knn_detector' in registry.list_all()

    def test_registry_includes_predictors(self):
        """
        目的：验证注册表现在包含 Predictor 类型的算子
        输入：无
        预期：mean_predictor、pca_predictor 等 Predictor 算子可被发现
        """
        registry = create_registry()
        operators = registry.list_all()
        # Predictor 现在应该包含在注册表中
        assert 'mean_predictor' in operators
        assert 'pca_predictor' in operators
