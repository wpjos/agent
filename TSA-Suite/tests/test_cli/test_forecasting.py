# -*- coding: utf-8 -*-

"""
iTransformer 工业时序预测算子 CLI 单元测试

对应源文件：
- cli/forecasting.py
- forecasting/itransformer.py

测试范围：
- help 子命令（列表模式和详情模式）
- fit 子命令（训练并保存模型）
- run 子命令（加载已训练模型并预测）
- fit → save → load → run 完整流程
- 配置校验错误场景
"""

import json

import numpy as np
import pandas as pd
import pytest

# 若当前环境缺少 torch，则跳过需要训练/推理的测试
torch = pytest.importorskip("torch", reason="需要安装 torch 才能运行 iTransformer 测试")

from tsas.engine.operator.cli.forecasting import main, create_registry


# ============================================================================
# 公共 fixture
# ============================================================================

@pytest.fixture
def train_csv(tmp_path):
    """创建训练 CSV 数据文件（500行4列，满足 seq_len=100, pred_len=20 的窗口需求）"""
    np.random.seed(42)
    timesteps = 500
    data = np.cumsum(np.random.randn(timesteps, 4), axis=0)
    df = pd.DataFrame(data, columns=['feat_0', 'feat_1', 'feat_2', 'target'])
    path = tmp_path / "train.csv"
    df.to_csv(path, index=False)
    return path


@pytest.fixture
def test_window_csv(tmp_path):
    """创建推理窗口 CSV 数据文件（100行4列，与 seq_len 对齐）"""
    np.random.seed(123)
    data = np.cumsum(np.random.randn(100, 4), axis=0)
    df = pd.DataFrame(data, columns=['feat_0', 'feat_1', 'feat_2', 'target'])
    path = tmp_path / "test_window.csv"
    df.to_csv(path, index=False)
    return path


@pytest.fixture
def minimal_forecaster_config(tmp_path):
    """创建最小化的 iTransformer 配置文件（小模型、少 epoch，仅用于测试）"""
    config = {
        "operator": {
            "name": "itransformer_forecaster",
            "input_columns": ["feat_0", "feat_1", "feat_2", "target"],
            "config": {
                "seq_len": 100,
                "pred_len": 20,
                "d_model": 32,
                "nhead": 2,
                "num_layers": 1,
                "dim_feedforward": 64,
                "dropout": 0.1,
                "lag_aware": True,
                "lag_max": 8,
                "kan_grid_size": 3,
                "target_idx": -1,
                "epochs": 2,
                "batch_size": 32,
                "lr": 0.001,
                "early_stop_patience": 5,
                "train_ratio": 0.7,
                "val_ratio": 0.15,
                "device": "cpu",
            },
        }
    }
    config_path = tmp_path / "forecaster_config.json"
    with open(config_path, 'w') as f:
        json.dump(config, f)
    return config_path


# ============================================================================
# 测试类
# ============================================================================

class TestForecastingHelp:
    """测试 forecasting help 子命令"""

    def test_help_list(self, capsys):
        """
        目的：验证 help 无参数时列出所有预测算子
        输入：['help']
        预期：输出包含 itransformer_forecaster
        """
        main(['help'])
        captured = capsys.readouterr()
        assert "itransformer_forecaster" in captured.out

    def test_help_detail(self, capsys):
        """
        目的：验证 help 带算子名称时输出详情
        输入：['help', 'itransformer_forecaster']
        预期：输出包含关键参数名
        """
        main(['help', 'itransformer_forecaster'])
        captured = capsys.readouterr()
        assert "## itransformer_forecaster" in captured.out
        assert "seq_len" in captured.out
        assert "pred_len" in captured.out

    def test_help_unknown_operator(self):
        """
        目的：验证 help 查找不存在的算子时报错
        输入：['help', 'nonexistent']
        预期：抛出 KeyError
        """
        with pytest.raises(KeyError, match="未找到名为"):
            main(['help', 'nonexistent'])


class TestForecastingFitAndRun:
    """测试 forecasting fit 和 run 完整流程"""

    def test_fit_and_save(self, train_csv, minimal_forecaster_config, tmp_path, capsys):
        """
        目的：验证 itransformer_forecaster 的 fit + save 流程
        输入：训练数据 + 小模型配置 + --save 参数
        预期：fit 成功，模型保存到指定目录
        """
        save_dir = tmp_path / "model"
        main([
            'fit',
            '--input', str(train_csv),
            '--target', 'target',
            '--config', str(minimal_forecaster_config),
            '--save', str(save_dir),
        ])

        captured = capsys.readouterr()
        assert "训练完成" in captured.out
        assert "模型已保存" in captured.out
        assert save_dir.exists()
        # 检查保存的关键文件
        assert (save_dir / "config.json").exists()
        assert (save_dir / "_model_weights.pt").exists()
        assert (save_dir / "_scaler.npz").exists()
        assert (save_dir / "_forecaster_state.npz").exists()

    def test_fit_save_load_run(self, train_csv, test_window_csv,
                               minimal_forecaster_config, tmp_path, capsys):
        """
        目的：验证 fit → save → load → run 完整流程
        输入：训练数据、推理窗口、模型配置
        预期：预测结果文件存在且形状为 (pred_len, 1)
        """
        save_dir = tmp_path / "model"
        output_path = tmp_path / "pred.csv"

        # fit + save
        main([
            'fit',
            '--input', str(train_csv),
            '--target', 'target',
            '--config', str(minimal_forecaster_config),
            '--save', str(save_dir),
        ])
        assert "训练完成" in capsys.readouterr().out

        # load + run
        main([
            'run',
            '--input', str(test_window_csv),
            '--config', str(minimal_forecaster_config),
            '--load', str(save_dir),
            '--output', str(output_path),
        ])
        captured = capsys.readouterr()
        assert "预测完成" in captured.out
        assert output_path.exists()

        pred = pd.read_csv(output_path)
        # 预测输出应为 (pred_len, num_targets) = (20, 1)
        assert pred.shape[0] == 20
        assert pred.shape[1] == 1


class TestForecastingConfigErrors:
    """测试预测算子配置错误场景"""

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
            main(['fit', '--input', str(train_csv), '--target', 'target',
                  '--config', str(config_path)])

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
            main(['fit', '--input', str(train_csv), '--target', 'target',
                  '--config', str(config_path)])


class TestForecastingNoCommand:
    """测试无子命令场景"""

    def test_no_command_exits(self):
        """
        目的：验证不提供子命令时 sys.exit(1)
        输入：空参数
        预期：SystemExit
        """
        with pytest.raises(SystemExit):
            main([])


class TestCreateForecastingRegistry:
    """测试 create_registry 工厂函数"""

    def test_create_registry(self):
        """
        目的：验证工厂函数返回已 discover 的注册中心
        输入：无
        预期：返回包含 itransformer_forecaster 的 OperatorRegistry
        """
        registry = create_registry()
        assert registry.discovered is True
        operators = registry.list_all()
        assert 'itransformer_forecaster' in operators
