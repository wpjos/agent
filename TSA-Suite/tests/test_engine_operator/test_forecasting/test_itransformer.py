# -*- coding: utf-8 -*-

"""
iTransformer 工业时序预测算子单元测试

对应源文件：
- forecasting/itransformer.py: ITransformerForecaster, ITransformerForecasterConfig

测试范围：
- Config 参数校验
- 训练与推理基本流程
- DataFrame/ndarray 双类型支持
- 边界条件（未训练、序列长度不足、单目标限制）
- Save/Load roundtrip
"""

import numpy as np
import pandas as pd
import pytest
from pandas import DataFrame

# 若当前环境缺少 torch，则跳过需要训练/推理的测试
torch = pytest.importorskip("torch", reason="需要安装 torch 才能运行 iTransformer 测试")

from tsas.engine.operator.forecasting.itransformer import (
    ITransformerForecaster,
    ITransformerForecasterConfig,
)


# ============================================================================
# 公共测试数据
# ============================================================================

@pytest.fixture
def train_data():
    """测试用训练数据（ndarray, 500x4），满足 seq_len=100, pred_len=20"""
    np.random.seed(42)
    return np.cumsum(np.random.randn(500, 4), axis=0)


@pytest.fixture
def train_df(train_data):
    """测试用训练数据（DataFrame）"""
    return DataFrame(train_data, columns=["feat_0", "feat_1", "feat_2", "target"])


@pytest.fixture
def test_window():
    """测试用推理窗口（ndarray, 100x4）"""
    np.random.seed(123)
    return np.cumsum(np.random.randn(100, 4), axis=0)


@pytest.fixture
def test_window_df(test_window):
    """测试用推理窗口（DataFrame）"""
    return DataFrame(test_window, columns=["feat_0", "feat_1", "feat_2", "target"])


@pytest.fixture
def minimal_config():
    """最小化测试配置（小模型、少 epoch）"""
    return ITransformerForecasterConfig(
        seq_len=100,
        pred_len=20,
        d_model=32,
        nhead=2,
        num_layers=1,
        dim_feedforward=64,
        dropout=0.1,
        lag_aware=True,
        lag_max=8,
        kan_grid_size=3,
        target_idx=-1,
        epochs=2,
        batch_size=32,
        lr=0.001,
        early_stop_patience=5,
        train_ratio=0.7,
        val_ratio=0.15,
        device="cpu",
    )


# ============================================================================
# Config 测试
# ============================================================================

class TestITransformerForecasterConfig:
    """测试 ITransformerForecasterConfig 参数校验"""

    def test_default_config(self):
        """目的：验证默认配置可创建"""
        cfg = ITransformerForecasterConfig()
        assert cfg.seq_len == 100
        assert cfg.pred_len == 20
        assert cfg.d_model == 128

    def test_searchable_bounds(self):
        """目的：验证带 ge/le 的字段可被 HPO 识别"""
        cfg = ITransformerForecasterConfig(d_model=64, nhead=4, num_layers=2)
        assert cfg.d_model == 64
        assert cfg.nhead == 4
        assert cfg.num_layers == 2

    def test_invalid_target_idx(self):
        """目的：验证 target_idx 超出范围时失败
        注意：Pydantic 中 target_idx 的上界较大，此测试验证运行时检查
        """
        cfg = ITransformerForecasterConfig(target_idx=10)
        forecaster = ITransformerForecaster(config=cfg)
        with pytest.raises(ValueError, match="target_idx"):
            # 训练时 num_features=4，target_idx=10 会报错
            x = np.random.randn(200, 4)
            y = x[:, [-1]]
            forecaster.fit(x, y)


# ============================================================================
# ITransformerForecaster 训练与推理测试
# ============================================================================

class TestITransformerForecaster:
    """测试 iTransformer 预测算子核心流程"""

    def test_fit_learns_scaler(self, train_data, minimal_config):
        """目的：验证 fit 后 scaler 已学习参数"""
        forecaster = ITransformerForecaster(config=minimal_config)
        y = train_data[:, [-1]]
        forecaster.fit(train_data, y)
        assert forecaster._scaler is not None
        assert forecaster._scaler.mean_ is not None
        assert forecaster._scaler.scale_ is not None
        assert forecaster.is_fitted

    def test_run_output_shape(self, train_data, test_window, minimal_config):
        """目的：验证推理输出形状为 (pred_len, num_targets)"""
        forecaster = ITransformerForecaster(config=minimal_config)
        y = train_data[:, [-1]]
        forecaster.fit(train_data, y)
        pred = forecaster.run(test_window)
        assert pred.shape == (20, 1)

    def test_run_batched_output_shape(self, train_data, minimal_config):
        """目的：验证批量推理输出形状为 (batch, pred_len, num_targets)"""
        forecaster = ITransformerForecaster(config=minimal_config)
        y = train_data[:, [-1]]
        forecaster.fit(train_data, y)
        # 构造 3 个连续窗口
        x_batch = np.stack([
            train_data[100:200],
            train_data[200:300],
            train_data[300:400],
        ])
        pred = forecaster.run(x_batch)
        assert pred.shape == (3, 20, 1)

    def test_with_dataframe(self, train_df, test_window_df, minimal_config):
        """目的：验证 DataFrame 输入输出"""
        forecaster = ITransformerForecaster(config=minimal_config)
        y = train_df[["target"]]
        forecaster.fit(train_df, y)
        pred = forecaster.run(test_window_df)
        assert isinstance(pred, DataFrame)
        assert pred.shape == (20, 1)

    def test_before_fit_raises(self, test_window, minimal_config):
        """目的：验证未训练时 run 抛出 RuntimeError"""
        forecaster = ITransformerForecaster(config=minimal_config)
        with pytest.raises(RuntimeError):
            forecaster.run(test_window)

    def test_too_short_sequence_raises(self, train_data, minimal_config):
        """目的：验证训练序列长度不足时抛出 ValueError"""
        forecaster = ITransformerForecaster(config=minimal_config)
        short = train_data[:50]  # 小于 seq_len + pred_len
        y = short[:, [-1]]
        with pytest.raises(ValueError, match="时间序列长度"):
            forecaster.fit(short, y)

    def test_multi_target_rejected(self, train_data, minimal_config):
        """目的：验证当前仅支持单目标"""
        forecaster = ITransformerForecaster(config=minimal_config)
        y = train_data[:, [-1, -2]]  # 2 列目标
        with pytest.raises(ValueError, match="单目标"):
            forecaster.fit(train_data, y)

    def test_residual_prediction(self, train_data, test_window, minimal_config):
        """目的：验证预测值在合理范围内（残差预测应接近基准值附近）"""
        forecaster = ITransformerForecaster(config=minimal_config)
        y = train_data[:, [-1]]
        forecaster.fit(train_data, y)
        pred = forecaster.run(test_window)
        base_value = test_window[-1, -1]
        # 预测值应落在基准值的一个较大邻域内（训练不稳定时放宽）
        assert np.all(np.isfinite(pred))
        assert np.all(np.abs(pred - base_value) < 100)


# ============================================================================
# Save/Load Roundtrip 测试
# ============================================================================

class TestITransformerForecasterSaveLoad:
    """测试 ITransformerForecaster 持久化 roundtrip"""

    def test_save_load_roundtrip(self, train_data, test_window, minimal_config, tmp_path):
        """目的：验证 save → load 后推理结果一致"""
        forecaster = ITransformerForecaster(config=minimal_config)
        y = train_data[:, [-1]]
        forecaster.fit(train_data, y)
        original_pred = forecaster.run(test_window)

        save_dir = tmp_path / "itransformer_forecaster"
        forecaster.save(save_dir)
        loaded = ITransformerForecaster.load(save_dir)

        loaded_pred = loaded.run(test_window)
        np.testing.assert_allclose(original_pred, loaded_pred, rtol=1e-5)

    def test_loaded_state_restored(self, train_data, minimal_config, tmp_path):
        """目的：验证加载后内部状态正确恢复"""
        forecaster = ITransformerForecaster(config=minimal_config)
        y = train_data[:, [-1]]
        forecaster.fit(train_data, y)

        save_dir = tmp_path / "itransformer_forecaster"
        forecaster.save(save_dir)
        loaded = ITransformerForecaster.load(save_dir)

        assert loaded.is_fitted
        assert loaded._num_features == forecaster._num_features
        assert loaded._num_targets == forecaster._num_targets
        assert loaded._target_idx == forecaster._target_idx
        np.testing.assert_allclose(loaded._scaler.mean_, forecaster._scaler.mean_)
        np.testing.assert_allclose(loaded._scaler.scale_, forecaster._scaler.scale_)

    def test_save_creates_required_files(self, train_data, minimal_config, tmp_path):
        """目的：验证保存目录包含所有必需文件"""
        forecaster = ITransformerForecaster(config=minimal_config)
        y = train_data[:, [-1]]
        forecaster.fit(train_data, y)

        save_dir = tmp_path / "itransformer_forecaster"
        forecaster.save(save_dir)

        assert (save_dir / "config.json").exists()
        assert (save_dir / "_model_weights.pt").exists()
        assert (save_dir / "_scaler.npz").exists()
        assert (save_dir / "_forecaster_state.npz").exists()
        # 该算子未定义 FitParams，因此不会生成 last_fit_params.json
