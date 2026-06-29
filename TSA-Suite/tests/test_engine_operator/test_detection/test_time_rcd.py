# -*- coding: utf-8 -*-

"""
Time-RCD 评分器单元测试

对应源文件：
- time_rcd.py: TimeRCDScorer

测试范围：
- Config 参数验证
- fit/run 基本流程（fit 仅加载 checkpoint，不学统计量）
- DataFrame/ndarray 双类型支持
- save/load 持久化（仅 meta，无 .pt 文件）
- 边界条件（数据过短、未训练先推理、1D 输入等）

注意：
- HF 下载在 session-scoped fixture 中只触发一次
- 默认 win_size=200, batch_size=2 以加速 CI
"""

import os

import numpy as np
import pytest
from pandas import DataFrame
from pydantic import ValidationError

from tsas.engine.operator.detection.time_rcd import (
    TimeRCDScorer,
    TimeRCDScorerConfig,
    TimeRCDScorerRunParams,
)


# ============================================================================
# 公共测试数据 / fixture
# ============================================================================

@pytest.fixture(scope="session", autouse=True)
def _hf_mirror():
    """切换到清华镜像，避免 HF Hub 在国内拉取超时。"""
    from bq_rcd.utils.checkpoint_utils import use_hf_mirror

    use_hf_mirror(True)


@pytest.fixture
def train_data():
    """训练/校准数据（500x1, float32）。"""
    np.random.seed(42)
    return np.random.randn(500, 1).astype(np.float32)


@pytest.fixture
def test_data():
    """测试数据（500x1, float32，含异常段）。"""
    np.random.seed(123)
    normal = np.random.randn(400, 1).astype(np.float32)
    abnormal = (np.random.randn(100, 1) * 5 + 10).astype(np.float32)
    return np.vstack([normal, abnormal])


@pytest.fixture
def train_df(train_data):
    """训练数据 DataFrame 版本。"""
    return DataFrame(train_data, columns=["a"])


@pytest.fixture
def test_df(test_data):
    """测试数据 DataFrame 版本。"""
    return DataFrame(test_data, columns=["a"])


def _make_scorer(**overrides):
    """以测试友好的最小配置构造 TimeRCDScorer。"""
    defaults = dict(
        win_size=200,
        batch_size=2,
        patch_size=16,
    )
    defaults.update(overrides)
    return TimeRCDScorer(**defaults)


# ============================================================================
# Config 测试
# ============================================================================

class TestTimeRCDScorerConfig:
    """测试 TimeRCDScorerConfig 参数验证。"""

    def test_config_defaults(self):
        """默认值与 Time-RCD 推荐配置一致。"""
        cfg = TimeRCDScorerConfig()
        assert cfg.win_size == 5000
        assert cfg.batch_size == 64
        assert cfg.patch_size == 16
        assert cfg.num_features is None
        assert cfg.checkpoint is None

    def test_run_params_defaults(self):
        """score_form 默认为 'prob'。"""
        rp = TimeRCDScorerRunParams()
        assert rp.score_form == "prob"

    def test_config_frozen(self):
        """Config 不可变。"""
        cfg = TimeRCDScorerConfig()
        with pytest.raises(ValidationError):
            cfg.win_size = 100

    def test_config_validation_win_size(self):
        """win_size 必须 > 0。"""
        with pytest.raises(ValidationError):
            TimeRCDScorerConfig(win_size=0)

    def test_config_validation_batch_size(self):
        """batch_size 必须 > 0。"""
        with pytest.raises(ValidationError):
            TimeRCDScorerConfig(batch_size=-1)

    def test_run_params_score_form_validation(self):
        """score_form 仅接受 'prob' / 'logit'。"""
        with pytest.raises(ValidationError):
            TimeRCDScorerRunParams(score_form="invalid")

    def test_config_custom_values(self):
        """自定义参数正确传递。"""
        cfg = TimeRCDScorerConfig(win_size=200, batch_size=4)
        assert cfg.win_size == 200
        assert cfg.batch_size == 4

    def test_run_params_custom_score_form(self):
        """RunParams 可在 'prob' / 'logit' 间切换。"""
        rp = TimeRCDScorerRunParams(score_form="logit")
        assert rp.score_form == "logit"


# ============================================================================
# Fit 测试
# ============================================================================

class TestTimeRCDScorerFit:
    """测试 TimeRCDScorer 训练流程（实为加载 checkpoint）。"""

    def test_fit_creates_tester(self, train_data):
        """fit 后 tester 已构造。"""
        scorer = _make_scorer()
        scorer.fit(train_data)
        assert scorer._tester is not None
        assert scorer.is_fitted is True

    def test_fit_auto_detect_features(self, train_data):
        """num_features 自动从数据推断。"""
        scorer = _make_scorer()
        scorer.fit(train_data)
        assert scorer._num_features_detected == 1

    def test_fit_with_explicit_features(self, train_data):
        """显式指定 num_features 时使用指定值。"""
        scorer = _make_scorer(num_features=1)
        scorer.fit(train_data)
        assert scorer._num_features_detected == 1

    def test_fit_records_checkpoint_path(self, train_data):
        """fit 后记录已解析的 checkpoint 路径。"""
        scorer = _make_scorer()
        scorer.fit(train_data)
        assert scorer._checkpoint_path_resolved is not None
        assert os.path.exists(scorer._checkpoint_path_resolved)

    def test_fit_data_too_short_raises(self):
        """数据行数 < win_size 时报错。"""
        scorer = _make_scorer()
        short_data = np.random.randn(50, 1).astype(np.float32)
        with pytest.raises(ValueError, match="win_size"):
            scorer.fit(short_data)

    def test_fit_1d_input_raises(self):
        """1D 输入报错。"""
        scorer = _make_scorer()
        data_1d = np.random.randn(500).astype(np.float32)
        with pytest.raises(ValueError, match="2D"):
            scorer.fit(data_1d)

    def test_fit_with_dataframe(self, train_df):
        """DataFrame 输入可以训练。"""
        scorer = _make_scorer()
        scorer.fit(train_df)
        assert scorer.is_fitted is True


# ============================================================================
# Run 测试
# ============================================================================

class TestTimeRCDScorerRun:
    """测试 TimeRCDScorer 推理流程。"""

    def test_run_output_shape(self, train_data, test_data):
        """输出长度与输入行数一致，单输出 (N,)。"""
        scorer = _make_scorer()
        scorer.fit(train_data)
        scores = scorer.run(test_data)
        assert scores.shape == (test_data.shape[0],)

    def test_run_values_finite(self, train_data, test_data):
        """异常分数不含 NaN/Inf。"""
        scorer = _make_scorer()
        scorer.fit(train_data)
        scores = scorer.run(test_data)
        assert np.all(np.isfinite(scores))

    def test_run_prob_in_range(self, train_data, test_data):
        """score_form='prob' 时输出在 [0, 1]。"""
        scorer = _make_scorer()
        scorer.fit(train_data)
        scores = scorer.run(test_data, score_form="prob")
        assert (scores >= 0).all() and (scores <= 1).all()

    def test_run_logit_unbounded(self, train_data, test_data):
        """score_form='logit' 输出可超出 [0, 1]（softmax 前 logit 差）。"""
        scorer = _make_scorer()
        scorer.fit(train_data)
        scores = scorer.run(test_data, score_form="logit")
        assert np.all(np.isfinite(scores))
        # logit 值域不应被夹紧在 [0,1]
        assert scores.min() < 0 or scores.max() > 1

    def test_run_before_fit_raises(self, test_data):
        """未训练时 run 报 RuntimeError。"""
        scorer = _make_scorer()
        with pytest.raises(RuntimeError, match="训练尚未完成"):
            scorer.run(test_data)

    def test_run_with_dataframe(self, train_df, test_df):
        """DataFrame 输入输出，列名为 ['score']。"""
        scorer = _make_scorer()
        scorer.fit(train_df)
        scores = scorer.run(test_df)
        assert isinstance(scores, DataFrame)
        assert list(scores.columns) == ["score"]
        assert len(scores) == len(test_df)

    def test_run_with_ndarray(self, train_data, test_data):
        """ndarray 输入返回 ndarray。"""
        scorer = _make_scorer()
        scorer.fit(train_data)
        scores = scorer.run(test_data)
        assert isinstance(scores, np.ndarray)


# ============================================================================
# Save / Load 测试
# ============================================================================

class TestTimeRCDScorerSaveLoad:
    """测试 TimeRCDScorer 持久化（仅 meta，无 .pt 文件）。"""

    def test_save_creates_only_meta(self, train_data, tmp_path):
        """save 仅生成 config.json + time_rcd_meta.json，不存 .pt 文件。"""
        scorer = _make_scorer()
        scorer.fit(train_data)

        save_dir = tmp_path / "rcd_model"
        scorer.save(save_dir)

        assert (save_dir / "config.json").exists()
        assert (save_dir / "time_rcd_meta.json").exists()
        # 确认没有 .pt 文件落盘（与 cicada 显著区别）
        pt_files = list(save_dir.glob("*.pt"))
        assert len(pt_files) == 0

    def test_load_restores_fitted_state(self, train_data, tmp_path):
        """load 后 is_fitted == True。"""
        scorer = _make_scorer()
        scorer.fit(train_data)

        save_dir = tmp_path / "rcd_model"
        scorer.save(save_dir)

        loaded = TimeRCDScorer.load(save_dir)
        assert loaded.is_fitted is True
        assert loaded._tester is not None

    def test_load_num_features_preserved(self, train_data, tmp_path):
        """num_features_detected 持久化恢复。"""
        scorer = _make_scorer(num_features=None)
        scorer.fit(train_data)
        assert scorer._num_features_detected == 1

        save_dir = tmp_path / "rcd_model"
        scorer.save(save_dir)

        loaded = TimeRCDScorer.load(save_dir)
        assert loaded._num_features_detected == 1

    def test_save_load_roundtrip(self, train_data, test_data, tmp_path):
        """save + load 后推理输出一致（确定性）。"""
        scorer = _make_scorer()
        scorer.fit(train_data)
        scores_before = scorer.run(test_data)

        save_dir = tmp_path / "rcd_model"
        scorer.save(save_dir)

        loaded = TimeRCDScorer.load(save_dir)
        scores_after = loaded.run(test_data)

        np.testing.assert_allclose(scores_after, scores_before, atol=1e-5)
