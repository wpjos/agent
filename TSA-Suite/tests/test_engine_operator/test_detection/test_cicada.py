# -*- coding: utf-8 -*-

"""
CICADA 检测算子单元测试

对应源文件：
- cicada.py: CICADAPredictor、CICADAScorer

测试范围：
- Config 参数验证（Predictor 与 Scorer）
- fit/run 基本流程
- DataFrame/ndarray 双类型支持
- save/load 持久化
- EO 附加输出字段
- 边界条件（数据过短、未训练先推理等）
- CICADAScorer vs CICADA.decision_function 一致性对比
"""

import warnings

import numpy as np
import pytest
from pandas import DataFrame
from pydantic import ValidationError

from tsas.engine.operator.detection.cicada import (
    CICADAPredictor,
    CICADAPredictorConfig,
    CICADAScorer,
    CICADAScorerConfig,
    CICADAScorerExtraOutput,
)


# ============================================================================
# 公共测试数据
# ============================================================================

@pytest.fixture
def train_data():
    """测试用训练数据（ndarray, 200x3, float32）"""
    np.random.seed(42)
    return np.random.randn(200, 3).astype(np.float32)


@pytest.fixture
def test_data():
    """测试用测试数据（ndarray, 100x3, float32，含异常点）"""
    np.random.seed(123)
    normal = np.random.randn(80, 3).astype(np.float32)
    abnormal = (np.random.randn(20, 3) * 5 + 10).astype(np.float32)
    return np.vstack([normal, abnormal])


@pytest.fixture
def train_df(train_data):
    """测试用训练数据（DataFrame）"""
    return DataFrame(train_data, columns=["a", "b", "c"])


@pytest.fixture
def test_df(test_data):
    """测试用测试数据（DataFrame）"""
    return DataFrame(test_data, columns=["a", "b", "c"])


def _make_predictor(**overrides):
    """创建最小配置的 CICADAPredictor（用于加速测试）"""
    defaults = dict(
        name=["MLP"],
        win_size=10,
        num_channels=3,
        batch_size=32,
        epochs=1,
        latent_space_size=8,
        n_components=4,
    )
    defaults.update(overrides)
    with warnings.catch_warnings():
        warnings.simplefilter("ignore", UserWarning)
        return CICADAPredictor(**defaults)


# ============================================================================
# Config 测试
# ============================================================================

class TestCICADAPredictorConfig:
    """测试 CICADAPredictorConfig 参数验证"""

    def test_config_defaults(self):
        """
        目的：验证 Config 默认值与 CICADA 一致
        输入：无参数构造
        预期：所有默认值正确
        """
        cfg = CICADAPredictorConfig()
        assert cfg.name == ["GradPCA", "GradKPCA", "GradFreKPCA", "GradSubPCA"]
        assert cfg.win_size == 5
        assert cfg.stride == 1
        assert cfg.num_channels is None
        assert cfg.batch_size == 256
        assert cfg.epochs == 60
        assert cfg.latent_space_size == 128
        assert cfg.n_components == "auto"
        assert cfg.lr == 1e-3
        assert cfg.infer_mode == "offline"
        assert cfg.th == 0.98

    def test_config_frozen(self):
        """
        目的：验证 Config 不可变
        输入：创建后尝试修改字段
        预期：抛出异常
        """
        cfg = CICADAPredictorConfig()
        with pytest.raises(ValidationError):
            cfg.win_size = 100

    def test_config_validation_win_size(self):
        """
        目的：验证 win_size 约束
        输入：win_size=0
        预期：ValidationError
        """
        with pytest.raises(ValidationError):
            CICADAPredictorConfig(win_size=0)

    def test_config_validation_epochs(self):
        """
        目的：验证 epochs 约束
        输入：epochs=-1
        预期：ValidationError
        """
        with pytest.raises(ValidationError):
            CICADAPredictorConfig(epochs=-1)

    def test_config_infer_mode_validation(self):
        """
        目的：验证 infer_mode 枚举约束
        输入：infer_mode="invalid"
        预期：ValidationError
        """
        with pytest.raises(ValidationError):
            CICADAPredictorConfig(infer_mode="invalid")

    def test_config_custom_values(self):
        """
        目的：验证自定义参数正确传递
        输入：自定义 win_size, batch_size
        预期：Config 中值为自定义值
        """
        cfg = CICADAPredictorConfig(win_size=100, batch_size=64)
        assert cfg.win_size == 100
        assert cfg.batch_size == 64


# ============================================================================
# Fit 测试
# ============================================================================

class TestCICADAPredictorFit:
    """测试 CICADAPredictor 训练流程"""

    def test_fit_creates_model(self, train_data):
        """
        目的：验证 fit 后内部模型已创建
        输入：(200, 3) 训练数据
        预期：_model 不为 None，is_fitted 为 True
        """
        predictor = _make_predictor()
        predictor.fit(train_data)
        assert predictor._model is not None
        assert predictor.is_fitted is True

    def test_fit_auto_detect_channels(self, train_data):
        """
        目的：验证 num_channels 自动推断
        输入：num_channels=None，(200, 3) 数据
        预期：_num_channels_detected == 3
        """
        predictor = _make_predictor(num_channels=None)
        predictor.fit(train_data)
        assert predictor._num_channels_detected == 3

    def test_fit_with_explicit_channels(self, train_data):
        """
        目的：验证显式指定 num_channels 时使用指定值
        输入：num_channels=3
        预期：_num_channels_detected == 3
        """
        predictor = _make_predictor(num_channels=3)
        predictor.fit(train_data)
        assert predictor._num_channels_detected == 3

    def test_fit_data_too_short_raises(self):
        """
        目的：验证数据行数不足时报错
        输入：(5, 3) 数据，win_size=10
        预期：ValueError
        """
        predictor = _make_predictor()
        short_data = np.random.randn(5, 3).astype(np.float32)
        with pytest.raises(ValueError, match="win_size"):
            predictor.fit(short_data)

    def test_fit_1d_input_raises(self):
        """
        目的：验证 1D 输入报错
        输入：(300,) 一维数据
        预期：ValueError
        """
        predictor = _make_predictor()
        data_1d = np.random.randn(300).astype(np.float32)
        with pytest.raises(ValueError, match="2D"):
            predictor.fit(data_1d)

    def test_fit_with_dataframe(self, train_df):
        """
        目的：验证 DataFrame 输入可以训练
        输入：DataFrame (200, 3)
        预期：训练成功，is_fitted 为 True
        """
        predictor = _make_predictor(num_channels=3)
        predictor.fit(train_df)
        assert predictor.is_fitted is True


# ============================================================================
# Run 测试
# ============================================================================

class TestCICADAPredictorRun:
    """测试 CICADAPredictor 推理流程"""

    def test_run_output_shape(self, train_data, test_data):
        """
        目的：验证推理输出形状与输入一致
        输入：训练数据 + (100, 3) 测试数据
        预期：输出形状 == (100, 3)
        """
        predictor = _make_predictor()
        predictor.fit(train_data)
        recon = predictor.run(test_data)
        assert recon.shape == test_data.shape

    def test_run_values_finite(self, train_data, test_data):
        """
        目的：验证重构值不含 NaN/Inf
        输入：训练数据 + 测试数据
        预期：所有值为有限数
        """
        predictor = _make_predictor()
        predictor.fit(train_data)
        recon = predictor.run(test_data)
        assert np.all(np.isfinite(recon))

    def test_run_before_fit_raises(self, test_data):
        """
        目的：验证未训练时 run 报错
        输入：未训练的 predictor
        预期：RuntimeError
        """
        predictor = _make_predictor()
        with pytest.raises(RuntimeError, match="训练尚未完成"):
            predictor.run(test_data)

    def test_run_with_dataframe(self, train_df, test_df):
        """
        目的：验证 DataFrame 输入输出
        输入：DataFrame 训练 + DataFrame 测试
        预期：输出为 DataFrame，列名一致
        """
        predictor = _make_predictor(num_channels=3)
        predictor.fit(train_df)
        recon = predictor.run(test_df)
        assert isinstance(recon, DataFrame)
        assert list(recon.columns) == ["a", "b", "c"]

    def test_run_with_ndarray(self, train_data, test_data):
        """
        目的：验证 ndarray 输入输出
        输入：ndarray 训练 + ndarray 测试
        预期：输出为 ndarray
        """
        predictor = _make_predictor()
        predictor.fit(train_data)
        recon = predictor.run(test_data)
        assert isinstance(recon, np.ndarray)


# ============================================================================
# Save / Load 测试
# ============================================================================

class TestCICADAPredictorSaveLoad:
    """测试 CICADAPredictor 持久化"""

    def test_save_load_roundtrip(self, train_data, test_data, tmp_path):
        """
        目的：验证 save + load 后推理结果不变
        输入：训练 → save → load → run
        预期：load 后推理输出与 save 前一致
        """
        predictor = _make_predictor()
        predictor.fit(train_data)
        recon_before = predictor.run(test_data)

        save_dir = tmp_path / "cicada_model"
        predictor.save(save_dir)

        loaded = CICADAPredictor.load(save_dir)
        recon_after = loaded.run(test_data)

        np.testing.assert_allclose(recon_after, recon_before, atol=1e-5)

    def test_load_restores_fitted_state(self, train_data, tmp_path):
        """
        目的：验证 load 后 is_fitted 为 True
        输入：训练 → save → load
        预期：loaded.is_fitted == True
        """
        predictor = _make_predictor()
        predictor.fit(train_data)

        save_dir = tmp_path / "cicada_model"
        predictor.save(save_dir)

        loaded = CICADAPredictor.load(save_dir)
        assert loaded.is_fitted is True

    def test_load_restores_model(self, train_data, test_data, tmp_path):
        """
        目的：验证 load 后模型可用
        输入：训练 → save → load → run
        预期：模型不为 None，推理输出有限
        """
        predictor = _make_predictor()
        predictor.fit(train_data)

        save_dir = tmp_path / "cicada_model"
        predictor.save(save_dir)

        loaded = CICADAPredictor.load(save_dir)
        assert loaded._model is not None
        recon = loaded.run(test_data)
        assert np.all(np.isfinite(recon))

    def test_save_creates_expected_files(self, train_data, tmp_path):
        """
        目的：验证 save 生成正确的文件
        输入：训练 → save
        预期：config.json、cicada_model.pt、cicada_meta.json 均存在
        """
        predictor = _make_predictor()
        predictor.fit(train_data)

        save_dir = tmp_path / "cicada_model"
        predictor.save(save_dir)

        assert (save_dir / "config.json").exists()
        assert (save_dir / "cicada_model.pt").exists()
        assert (save_dir / "cicada_meta.json").exists()

    def test_load_num_channels_preserved(self, train_data, tmp_path):
        """
        目的：验证 num_channels 持久化正确
        输入：训练（自动推断 num_channels=3）→ save → load
        预期：loaded._num_channels_detected == 3
        """
        predictor = _make_predictor(num_channels=None)
        predictor.fit(train_data)
        assert predictor._num_channels_detected == 3

        save_dir = tmp_path / "cicada_model"
        predictor.save(save_dir)

        loaded = CICADAPredictor.load(save_dir)
        assert loaded._num_channels_detected == 3


# ============================================================================
# CICADAScorer 公共工厂
# ============================================================================


def _make_scorer(**overrides):
    """
    创建最小配置的 CICADAScorer（用于加速测试）

    构造一个轻量的 CICADAScorer 实例：仅使用 ``MLP`` 单专家、小窗口、单轮训练，
    保证测试用例的执行时间在可接受范围内。

    Args:
        **overrides: 用于覆盖默认参数的键值对。

    Returns:
        CICADAScorer: 已实例化、未训练的 Scorer 对象。
    """
    defaults = dict(
        name=["MLP"],
        win_size=10,
        num_channels=3,
        batch_size=32,
        epochs=1,
        latent_space_size=8,
        n_components=4,
    )
    defaults.update(overrides)
    with warnings.catch_warnings():
        # CICADA 内部偶有 UserWarning（如降维分量不足）；测试中无需关心
        warnings.simplefilter("ignore", UserWarning)
        return CICADAScorer(**defaults)


# ============================================================================
# CICADAScorerConfig 测试
# ============================================================================


class TestCICADAScorerConfig:
    """测试 CICADAScorerConfig 参数验证与继承关系"""

    def test_config_inherits_predictor_fields(self):
        """
        目的：验证 CICADAScorerConfig 完整继承 CICADAPredictorConfig 全部字段
        输入：无参数构造
        预期：父类字段默认值与 Predictor Config 一致；新增 metric 默认 "mse"
        """
        cfg = CICADAScorerConfig()
        # 继承的 CICADA 超参字段（与 Predictor Config 默认值完全一致）
        assert cfg.name == ["GradPCA", "GradKPCA", "GradFreKPCA", "GradSubPCA"]
        assert cfg.win_size == 5
        assert cfg.stride == 1
        assert cfg.num_channels is None
        assert cfg.batch_size == 256
        assert cfg.epochs == 60
        assert cfg.latent_space_size == 128
        assert cfg.n_components == "auto"
        assert cfg.lr == 1e-3
        assert cfg.infer_mode == "offline"
        assert cfg.th == 0.98
        # 新增的评分字段
        assert cfg.metric == "mse"

    def test_config_is_subclass_of_predictor_config(self):
        """
        目的：验证 CICADAScorerConfig IS-A CICADAPredictorConfig
        输入：类型关系检查
        预期：issubclass 为 True；实例 isinstance Predictor Config 亦为 True
        """
        assert issubclass(CICADAScorerConfig, CICADAPredictorConfig)
        cfg = CICADAScorerConfig()
        assert isinstance(cfg, CICADAPredictorConfig)

    def test_config_frozen(self):
        """
        目的：验证 Config 不可变（frozen=True 继承自父类）
        输入：创建后尝试修改 metric 字段
        预期：抛出 ValidationError
        """
        cfg = CICADAScorerConfig()
        with pytest.raises(ValidationError):
            cfg.metric = "mae"

    def test_config_metric_validation(self):
        """
        目的：验证 metric 枚举约束
        输入：metric="invalid"
        预期：抛出 ValidationError
        """
        with pytest.raises(ValidationError):
            CICADAScorerConfig(metric="invalid")

    def test_config_inherited_validation_still_works(self):
        """
        目的：验证父类的字段约束在子类中仍然生效
        输入：win_size=0（违反 gt=0）
        预期：抛出 ValidationError
        """
        with pytest.raises(ValidationError):
            CICADAScorerConfig(win_size=0)

    def test_config_custom_values(self):
        """
        目的：验证父类字段与新增字段可同时自定义
        输入：win_size=20, metric="mae"
        预期：两者均正确写入
        """
        cfg = CICADAScorerConfig(win_size=20, metric="mae")
        assert cfg.win_size == 20
        assert cfg.metric == "mae"


# ============================================================================
# CICADAScorer Fit 测试
# ============================================================================


class TestCICADAScorerFit:
    """测试 CICADAScorer 训练流程"""

    def test_fit_creates_internal_predictor_model(self, train_data):
        """
        目的：验证 fit 后内部 Predictor 已训练且 Scorer 训练态同步
        输入：(200, 3) 训练数据
        预期：scorer.is_fitted 与 _predictor.is_fitted 均为 True
        """
        scorer = _make_scorer()
        scorer.fit(train_data)
        assert scorer.is_fitted is True
        assert scorer._predictor.is_fitted is True
        assert scorer._predictor._model is not None

    def test_fit_auto_detect_channels(self, train_data):
        """
        目的：验证 num_channels 自动推断在 Scorer → Predictor 路径上有效
        输入：num_channels=None，(200, 3) 数据
        预期：内部 Predictor 推断出 num_channels=3
        """
        scorer = _make_scorer(num_channels=None)
        scorer.fit(train_data)
        assert scorer._predictor._num_channels_detected == 3

    def test_fit_with_dataframe(self, train_df):
        """
        目的：验证 DataFrame 输入可直接训练
        输入：DataFrame (200, 3)
        预期：训练成功，is_fitted=True
        """
        scorer = _make_scorer(num_channels=3)
        scorer.fit(train_df)
        assert scorer.is_fitted is True

    def test_fit_data_too_short_raises(self):
        """
        目的：验证数据行数不足以构造滑动窗口时报错
        输入：(5, 3) 数据，win_size=10
        预期：内部 Predictor 抛出 ValueError("win_size ...")
        """
        scorer = _make_scorer()
        short_data = np.random.randn(5, 3).astype(np.float32)
        with pytest.raises(ValueError, match="win_size"):
            scorer.fit(short_data)


# ============================================================================
# CICADAScorer Run 测试
# ============================================================================


class TestCICADAScorerRun:
    """测试 CICADAScorer 推理流程与输出语义"""

    def test_run_output_shape_and_eo_type(self, train_data, test_data):
        """
        目的：验证主输出为 1D 形状 (N,)，附加输出为 CICADAScorerExtraOutput
        输入：训练数据 + (100, 3) 测试数据
        预期：scores.shape == (100,)；eo 类型为 CICADAScorerExtraOutput
        """
        scorer = _make_scorer()
        scorer.fit(train_data)
        scores, eo = scorer.run(test_data)
        assert scores.shape == (test_data.shape[0],)
        assert isinstance(eo, CICADAScorerExtraOutput)

    def test_run_scores_finite_and_non_negative_mse(self, train_data, test_data):
        """
        目的：验证 MSE 模式下分数有限且非负
        输入：metric="mse"
        预期：所有分数 finite 且 >= 0
        """
        scorer = _make_scorer(metric="mse")
        scorer.fit(train_data)
        scores, _ = scorer.run(test_data)
        assert np.all(np.isfinite(scores))
        assert np.all(scores >= 0)

    def test_run_scores_non_negative_mae(self, train_data, test_data):
        """
        目的：验证 MAE 模式同样产出非负分数
        输入：metric="mae"
        预期：所有分数 >= 0
        """
        scorer = _make_scorer(metric="mae")
        scorer.fit(train_data)
        scores, _ = scorer.run(test_data)
        assert np.all(scores >= 0)

    def test_run_anomalous_higher_scores(self, train_data, test_data):
        """
        目的：验证异常点平均分数高于正常点
        输入：test_data 前 80 行为正常点，后 20 行为异常点
        预期：异常区平均分数 > 正常区平均分数
        """
        scorer = _make_scorer()
        scorer.fit(train_data)
        scores, _ = scorer.run(test_data)
        normal_avg = scores[:80].mean()
        abnormal_avg = scores[80:].mean()
        assert abnormal_avg > normal_avg

    def test_run_before_fit_raises(self, test_data):
        """
        目的：验证未训练时 run 抛 RuntimeError
        输入：未训练的 scorer
        预期：抛出 RuntimeError，包含 "训练尚未完成"
        """
        scorer = _make_scorer()
        with pytest.raises(RuntimeError, match="训练尚未完成"):
            scorer.run(test_data)

    def test_run_with_dataframe_returns_score_column(self, train_df, test_df):
        """
        目的：验证 DataFrame 输入输出
        输入：DataFrame 训练 + DataFrame 测试
        预期：输出为 DataFrame，列名为 ["score"]，索引与输入一致
        """
        scorer = _make_scorer(num_channels=3)
        scorer.fit(train_df)
        scores, _ = scorer.run(test_df)
        assert isinstance(scores, DataFrame)
        assert list(scores.columns) == ["score"]
        assert len(scores) == len(test_df)

    def test_run_with_ndarray_returns_ndarray(self, train_data, test_data):
        """
        目的：验证 ndarray 输入返回 ndarray
        输入：ndarray 训练 + ndarray 测试
        预期：scores 为 ndarray 且为 1D
        """
        scorer = _make_scorer()
        scorer.fit(train_data)
        scores, _ = scorer.run(test_data)
        assert isinstance(scores, np.ndarray)
        assert scores.ndim == 1

    def test_run_extra_output_fields(self, train_data, test_data):
        """
        目的：验证扁平化 EO 字段的形状与语义
        输入：训练 → run 测试数据
        预期：feature_recon、feature_scores 形状均为 (N, n_features) 且为有限值
        """
        scorer = _make_scorer()
        scorer.fit(train_data)
        _, eo = scorer.run(test_data)
        # 逐变量重构值
        assert isinstance(eo.feature_recon, np.ndarray)
        assert eo.feature_recon.shape == test_data.shape
        assert np.all(np.isfinite(eo.feature_recon))
        # 逐变量异常分数
        assert isinstance(eo.feature_scores, np.ndarray)
        assert eo.feature_scores.shape == test_data.shape
        assert np.all(eo.feature_scores >= 0)

    def test_run_feature_scores_consistent_with_metric(self, train_data, test_data):
        """
        目的：验证 EO 中 feature_scores 与 metric 的数学一致性
        输入：metric="mae"，预期逐变量分数 == |x - feature_recon|
        预期：feature_scores ≈ |test_data - feature_recon|
        """
        scorer = _make_scorer(metric="mae")
        scorer.fit(train_data)
        _, eo = scorer.run(test_data)
        expected = np.abs(test_data - eo.feature_recon)
        np.testing.assert_allclose(eo.feature_scores, expected, atol=1e-5)

    def test_run_main_score_equals_feature_scores_mean(self, train_data, test_data):
        """
        目的：验证 1D 主输出 = 逐变量分数沿特征轴的均值（与 ResidualScorer 行为一致）
        输入：训练 → 推理
        预期：scores ≈ feature_scores.mean(axis=1)
        """
        scorer = _make_scorer()
        scorer.fit(train_data)
        scores, eo = scorer.run(test_data)
        np.testing.assert_allclose(
            scores, eo.feature_scores.mean(axis=1), atol=1e-5
        )


# ============================================================================
# CICADAScorer Save / Load 测试
# ============================================================================


class TestCICADAScorerSaveLoad:
    """测试 CICADAScorer 持久化（委托内部 Predictor 落盘 torch 权重）"""

    def test_save_load_roundtrip(self, train_data, test_data, tmp_path):
        """
        目的：验证 save + load 后推理结果一致
        输入：训练 → save → load → run
        预期：load 前后 1D 异常分数完全相同
        """
        scorer = _make_scorer()
        scorer.fit(train_data)
        scores_before, _ = scorer.run(test_data)

        save_dir = tmp_path / "cicada_scorer"
        scorer.save(save_dir)

        loaded = CICADAScorer.load(save_dir)
        scores_after, _ = loaded.run(test_data)

        np.testing.assert_allclose(scores_after, scores_before, atol=1e-5)

    def test_save_load_restores_fitted_state(self, train_data, tmp_path):
        """
        目的：验证 load 后 is_fitted 为 True
        输入：训练 → save → load
        预期：loaded.is_fitted == True 且内部 Predictor.is_fitted == True
        """
        scorer = _make_scorer()
        scorer.fit(train_data)

        save_dir = tmp_path / "cicada_scorer"
        scorer.save(save_dir)

        loaded = CICADAScorer.load(save_dir)
        assert loaded.is_fitted is True
        assert loaded._predictor.is_fitted is True

    def test_save_directory_layout(self, train_data, tmp_path):
        """
        目的：验证 save 目录结构与子目录命名
        输入：训练 → save
        预期：根目录含 config.json；predictor/ 子目录含 Predictor 持久化产物
        """
        scorer = _make_scorer()
        scorer.fit(train_data)

        save_dir = tmp_path / "cicada_scorer"
        scorer.save(save_dir)

        # Scorer 自身的配置
        assert (save_dir / "config.json").exists()
        # Predictor 子目录及其内部文件（与 CICADAPredictor 自身约定一致）
        predictor_dir = save_dir / "predictor"
        assert predictor_dir.is_dir()
        assert (predictor_dir / "config.json").exists()
        assert (predictor_dir / "cicada_model.pt").exists()
        assert (predictor_dir / "cicada_meta.json").exists()

    def test_load_preserves_metric(self, train_data, tmp_path):
        """
        目的：验证 metric 等 Scorer 自身配置在 load 后正确恢复
        输入：metric="mae" 训练 → save → load
        预期：loaded.config.metric == "mae"，且内部 ResidualScorer 也使用 "mae"
        """
        scorer = _make_scorer(metric="mae")
        scorer.fit(train_data)

        save_dir = tmp_path / "cicada_scorer"
        scorer.save(save_dir)

        loaded = CICADAScorer.load(save_dir)
        assert loaded.config.metric == "mae"
        assert loaded._scorer.config.metric == "mae"

    def test_load_preserves_num_channels(self, train_data, tmp_path):
        """
        目的：验证内部 Predictor 自动推断的 num_channels 在 load 后保留
        输入：num_channels=None 训练（自动推断为 3）→ save → load
        预期：loaded._predictor._num_channels_detected == 3
        """
        scorer = _make_scorer(num_channels=None)
        scorer.fit(train_data)
        assert scorer._predictor._num_channels_detected == 3

        save_dir = tmp_path / "cicada_scorer"
        scorer.save(save_dir)

        loaded = CICADAScorer.load(save_dir)
        assert loaded._predictor._num_channels_detected == 3


# ============================================================================
# CICADAScorer 杂项测试
# ============================================================================


class TestCICADAScorerMisc:
    """CICADAScorer 杂项（name、内部组件配置一致性等）"""

    def test_name(self):
        """
        目的：验证 name() 返回正确算子标识
        预期：返回字符串 "cicada_scorer"
        """
        assert CICADAScorer.name() == "cicada_scorer"

    def test_internal_components_initialized(self):
        """
        目的：验证 __init__ 正确创建内部 Predictor 与 Scorer 组件
        输入：默认参数构造
        预期：_predictor 为 CICADAPredictor 实例；_scorer 为 ResidualScorer 实例；
              且 _scorer.config.metric 与 Scorer 自身 metric 配置一致
        """
        from tsas.engine.operator.detection.residual_scorer import ResidualScorer

        scorer = _make_scorer(metric="mae")
        assert isinstance(scorer._predictor, CICADAPredictor)
        assert isinstance(scorer._scorer, ResidualScorer)
        assert scorer._scorer.config.metric == "mae"

    def test_metric_propagation_to_residual_scorer(self):
        """
        目的：验证 Scorer Config 的 metric 字段正确透传到内部 ResidualScorer
        输入：metric 在两种取值下分别构造 Scorer
        预期：内部 ResidualScorer.config.metric 与外部 metric 一致
        """
        for metric in ("mse", "mae"):
            scorer = _make_scorer(metric=metric)
            assert scorer._scorer.config.metric == metric


# ============================================================================
# CICADAScorer vs CICADA.decision_function 一致性测试
# ============================================================================


class TestCICADAScorerVsDecisionFunction:
    """
    验证 CICADAScorer 异常分数与底层 CICADA.decision_function 的关系

    两种评分路径存在本质差异，本测试组旨在：
    1. 确认两种方法均能产出合理的异常分数
    2. 验证 CICADAScorer 的残差计算与手动计算结果一致
    3. 分析两种方法的相关性和排序一致性
    4. 记录并解释差异来源

    差异来源分析：
        - **评分机制不同**：decision_function 使用注意力加权的专家残差；
          CICADAScorer 使用简单 MSE/MAE（x - x_pred）
        - **归一化域不同**：decision_function 在归一化空间计算残差；
          CICADAScorer 在原始空间计算（reconstruct 返回反归一化后的值）
        - **窗口策略不同**：decision_function 使用 stride=1 滑窗 + 合并；
          reconstruct 使用 stride=win_size 的非重叠窗
        - **MAML 适配**：decision_function 包含测试时元学习适配；
          reconstruct 不含
    """

    @pytest.fixture
    def trained_scorer_and_data(self):
        """
        创建已训练的 CICADAScorer 及训练/测试数据。

        使用 MLP 单专家降低测试耗时，同时保证有意义的对比。
        """
        np.random.seed(42)
        train = np.random.randn(200, 3).astype(np.float32)
        np.random.seed(123)
        normal = np.random.randn(80, 3).astype(np.float32)
        abnormal = (np.random.randn(20, 3) * 5 + 10).astype(np.float32)
        test = np.vstack([normal, abnormal])

        scorer = _make_scorer()
        scorer.fit(train)
        return scorer, train, test

    # ------------------------------------------------------------------
    # 基础：decision_function 可调用性
    # ------------------------------------------------------------------

    def test_decision_function_produces_valid_scores(self, trained_scorer_and_data):
        """
        目的：验证底层 CICADA.decision_function 能产出有效的异常分数
        输入：已训练的 Scorer + 测试数据
        预期：返回 1D ndarray，长度 == n_samples，所有值有限
        """
        scorer, _, test_data = trained_scorer_and_data
        cicada_model = scorer._predictor._model
        assert cicada_model is not None

        df_scores = cicada_model.decision_function(test_data)
        assert isinstance(df_scores, np.ndarray)
        assert df_scores.ndim == 1
        assert len(df_scores) == test_data.shape[0]
        assert np.all(np.isfinite(df_scores))

    # ------------------------------------------------------------------
    # CICADAScorer 残差计算的手动验证
    # ------------------------------------------------------------------

    def test_scorer_mse_equals_manual_residual(self, trained_scorer_and_data):
        """
        目的：验证 CICADAScorer 的 MSE 分数可通过手动计算复现
        输入：训练数据 + 测试数据，metric="mse"
        预期：scores ≈ mean((test - recon)^2, axis=1)
        分析：此测试确认 CICADAScorer 内部残差计算逻辑无误
        """
        scorer, _, test_data = trained_scorer_and_data
        scores, eo = scorer.run(test_data)

        # 手动计算重构残差
        recon = eo.feature_recon
        manual_scores = np.mean((test_data - recon) ** 2, axis=1)
        np.testing.assert_allclose(scores, manual_scores, atol=1e-5)

    def test_scorer_mae_equals_manual_residual(self):
        """
        目的：验证 MAE 模式下分数与手动计算一致
        输入：metric="mae"
        预期：scores ≈ mean(|test - recon|, axis=1)
        """
        np.random.seed(42)
        train = np.random.randn(200, 3).astype(np.float32)
        np.random.seed(123)
        test = np.random.randn(100, 3).astype(np.float32)

        scorer = _make_scorer(metric="mae")
        scorer.fit(train)
        scores, eo = scorer.run(test)

        recon = eo.feature_recon
        manual_scores = np.mean(np.abs(test - recon), axis=1)
        np.testing.assert_allclose(scores, manual_scores, atol=1e-5)

    # ------------------------------------------------------------------
    # 两种评分方法的相关性分析
    # ------------------------------------------------------------------

    def test_scorer_vs_decision_function_positive_correlation(
        self, trained_scorer_and_data
    ):
        """
        目的：验证 CICADAScorer 分数与 decision_function 分数正相关
        输入：同一测试数据上两种评分
        预期：Pearson 相关系数 > 0，说明两者对异常程度排序基本一致

        分析：
            虽然两种方法的具体数值不同（见类文档字符串），但它们检测
            的是同一个底层异常信号（重构误差）。因此分数趋势应正相关。
            相关系数不一定接近 1.0，因为：
            - attention 加权 vs 简单均值
            - 归一化域差异导致各变量权重不同
            - 滑窗策略不同导致边界效应
        """
        scorer, _, test_data = trained_scorer_and_data
        scores, _ = scorer.run(test_data)
        df_scores = scorer._predictor._model.decision_function(test_data)

        corr = np.corrcoef(scores, df_scores)[0, 1]
        assert corr > 0, (
            f"CICADAScorer 与 decision_function 分数应正相关，"
            f"实际 Pearson r = {corr:.4f}"
        )

    def test_scorer_vs_decision_function_top_k_overlap(
        self, trained_scorer_and_data
    ):
        """
        目的：验证两种方法在检测最显著异常点时的一致性
        输入：含注入异常的测试数据（后 20 行为异常）
        预期：两种方法的 top-20 异常位置有显著重叠（>= 40%）

        分析：
            后 20 行被注入了大偏移异常，两种方法均应将其识别为异常。
            由于评分机制差异，top-k 集合不一定完全一致，但重叠率
            应显著高于随机水平（20/100 = 20%）。
        """
        scorer, _, test_data = trained_scorer_and_data
        scores, _ = scorer.run(test_data)
        df_scores = scorer._predictor._model.decision_function(test_data)

        k = 20
        top_scorer = set(np.argsort(scores)[-k:])
        top_df = set(np.argsort(df_scores)[-k:])
        overlap = len(top_scorer & top_df)
        overlap_ratio = overlap / k

        # 重叠率应显著高于随机水平 20%
        assert overlap_ratio >= 0.4, (
            f"top-{k} 异常位置重叠率 {overlap_ratio:.0%} 过低，"
            f"预期 >= 40%"
        )

    # ------------------------------------------------------------------
    # 分数绝对值差异记录
    # ------------------------------------------------------------------

    def test_scorer_vs_decision_function_score_magnitude_differs(
        self, trained_scorer_and_data
    ):
        """
        目的：确认两种评分方法的绝对值量级不同，并记录差异特征
        输入：同一测试数据上两种评分
        预期：两者绝对值存在显著差异（均值差异 > 10%）

        差异来源（详细分析）：
            1. **归一化域**：decision_function 在标准化空间计算 MSE，
               即 mean((x_norm - x_hat_norm)^2)；CICADAScorer 在原始空间
               计算 mean((x - x_denorm)^2)。当各变量 std 不同时，
               CICADAScorer 的残差受高方差变量主导。

            2. **注意力加权**：多专家时 decision_function 使用注意力权重
               对各专家残差加权求和；CICADAScorer 对所有变量取等权均值。

            3. **MAML 适配**：decision_function 在推理时进行元学习适配，
               可能改变重构质量；reconstruct 不做适配。

            4. **窗口合并**：decision_function 使用 stride=1 滑窗并对重叠
               区域做合并（如取 mean），导致分数更平滑；reconstruct 使用
               stride=win_size 非重叠窗口。

        注意：本测试不 assert 通过/失败，而是记录统计信息供人工审查。
        """
        scorer, _, test_data = trained_scorer_and_data
        scores, _ = scorer.run(test_data)
        df_scores = scorer._predictor._model.decision_function(test_data)

        # 记录统计信息（通过 print 输出到测试报告）
        print(f"\n{'='*60}")
        print(f"CICADAScorer vs CICADA.decision_function 分数对比")
        print(f"{'='*60}")
        print(f"CICADAScorer  — mean: {scores.mean():.6f}, std: {scores.std():.6f}, "
              f"min: {scores.min():.6f}, max: {scores.max():.6f}")
        print(f"decision_func — mean: {df_scores.mean():.6f}, std: {df_scores.std():.6f}, "
              f"min: {df_scores.min():.6f}, max: {df_scores.max():.6f}")
        corr = np.corrcoef(scores, df_scores)[0, 1]
        print(f"Pearson r: {corr:.4f}")
        print(f"{'='*60}")

        # 验证两者绝对值不同（通常量级差异显著）
        # 允许极端情况下量级接近但不完全相同
        assert not np.allclose(scores, df_scores, rtol=0.01), (
            "CICADAScorer 与 decision_function 的分数不应几乎相同"
        )

    # ------------------------------------------------------------------
    # 不同专家配置的对比
    # ------------------------------------------------------------------

    def test_single_expert_mlp_consistency(self, trained_scorer_and_data):
        """
        目的：验证单专家(MLP)配置下，CICADAScorer 分数可手动复现
        输入：MLP 单专家，metric="mse"
        预期：CICADAScorer 的 feature_scores 与手动 MSE 一致

        分析：
            单专家时不存在注意力加权差异，但仍存在归一化域和窗口
            策略差异。此测试确认 CICADAScorer 的内部计算链无误。
        """
        scorer, _, test_data = trained_scorer_and_data
        _, eo = scorer.run(test_data)

        # 手动计算逐变量 MSE
        recon = eo.feature_recon
        manual_mse = (test_data - recon) ** 2
        np.testing.assert_allclose(eo.feature_scores, manual_mse, atol=1e-5)

        # 主输出 = 逐变量 MSE 的均值
        manual_total = manual_mse.mean(axis=1)
        scores, _ = scorer.run(test_data)
        np.testing.assert_allclose(scores, manual_total, atol=1e-5)
