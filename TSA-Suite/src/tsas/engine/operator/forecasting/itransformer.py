# -*- coding: utf-8 -*-

"""
iTransformer 工业时序预测算子

基于 HBHD_predict_v1.5 中的 NPU-compatible iTransformer 模型：
- Dense NPU Transformer Encoder（MoE 消融为 num_experts=1）
- KAN 预测头
- Lag-Aware Refiner
- 残差预测策略 + 目标历史平线掩码

算子负责：
1. 训练阶段：标准化、滑动窗口构造、训练循环（加权 MSE + 趋势损失 + EarlyStopping）
2. 推理阶段：标准化输入、预测归一化残差、反归一化并加基准值得到物理量预测
"""

from pathlib import Path
from typing import Literal, Optional

import numpy as np
import pandas as pd
import torch
import torch.nn as nn
from loguru import logger
from pydantic import BaseModel, Field
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from torch.utils.data import DataLoader, Dataset

from tsas.engine.operator.forecasting.base import BaseForecaster, ForecastExtraOutput
from tsas.engine.operator.forecasting._models.itransformer_kan_res import (
    ITransformerRegressor,
)
from tsas.engine.util.pt_helper import _try_import_npu

__all__ = [
    'ITransformerForecasterConfig',
    'ITransformerForecaster',
]


class ITransformerForecasterConfig(BaseModel):
    """iTransformer 预测算子实例参数。

    所有带 ``ge`` / ``le`` 约束的数值字段均可被 HPO 自动搜索。
    """

    # ---- 模型结构参数 ----
    seq_len: int = Field(default=100, ge=8, le=2000, description="输入历史窗口长度")
    pred_len: int = Field(default=20, ge=1, le=500, description="预测未来步长")
    d_model: int = Field(default=128, ge=16, le=1024, description="模型嵌入维度")
    nhead: int = Field(default=4, ge=1, le=16, description="注意力头数")
    num_layers: int = Field(default=2, ge=1, le=12, description="Encoder 层数")
    dim_feedforward: Optional[int] = Field(default=None, ge=16, le=4096,
                                        description="FFN 隐藏层维度，None 时取 2*d_model")
    dropout: float = Field(default=0.2, ge=0.0, le=0.8, description="Dropout 比率")
    step_cond_head: bool = Field(default=False, description="是否使用步长条件预测头")
    lag_aware: bool = Field(default=True, description="是否启用 Lag-Aware Refiner")
    lag_max: int = Field(default=16, ge=0, le=128, description="最大滞后步数")
    lag_bias_scale: float = Field(default=2.0, ge=0.0, le=10.0, description="互相关先验偏置缩放")
    lag_dropout: float = Field(default=0.1, ge=0.0, le=0.8, description="Lag Refiner Dropout")
    kan_grid_size: int = Field(default=5, ge=1, le=20, description="KAN 网格大小")
    target_idx: int = Field(default=-1, ge=-1, le=10000,
                            description="目标变量在特征中的列索引，-1 表示最后一列")

    # ---- 训练参数 ----
    epochs: int = Field(default=30, ge=1, le=1000, description="最大训练轮数")
    batch_size: int = Field(default=128, ge=1, le=2048, description="训练批次大小")
    lr: float = Field(default=0.001, ge=1e-6, le=1e-1, description="学习率")
    weight_decay: float = Field(default=1e-5, ge=0.0, le=1e-1, description="权重衰减")
    early_stop_patience: int = Field(default=12, ge=1, le=100, description="早停耐心轮数")
    train_ratio: float = Field(default=0.7, ge=0.1, le=0.9, description="训练集占比")
    val_ratio: float = Field(default=0.15, ge=0.05, le=0.45, description="验证集占剩余数据比例")
    trend_weight: float = Field(default=1.0, ge=0.0, le=10.0, description="趋势损失权重")
    time_weight_start: float = Field(default=0.1, ge=0.0, le=1.0, description="时间加权损失起始权重")
    time_weight_end: float = Field(default=1.0, ge=0.0, le=1.0, description="时间加权损失结束权重")
    max_grad_norm: float = Field(default=1.0, ge=0.1, le=10.0, description="梯度裁剪范数")
    scheduler_factor: float = Field(default=0.5, ge=0.1, le=0.9, description="学习率衰减因子")
    scheduler_patience: int = Field(default=3, ge=1, le=20, description="学习率衰减耐心轮数")

    # ---- 运行参数 ----
    device: Literal['auto', 'cpu', 'cuda', 'npu'] = Field(default='auto', description="计算设备（auto/cpu/cuda/npu）")

    class Config:
        extra = 'forbid'


class _TimeSeriesDataset(Dataset):
    """内部数据集：从完整时间序列构造 (X_seq, Y_residual, Y_base, Y_future) 样本。"""

    def __init__(self, x: np.ndarray, y: np.ndarray, indices: np.ndarray, seq_len: int, pred_len: int):
        self.x = torch.tensor(x, dtype=torch.float32)
        self.y = torch.tensor(y, dtype=torch.float32)
        self.indices = indices
        self.seq_len = seq_len
        self.pred_len = pred_len

    def __len__(self):
        return len(self.indices)

    def __getitem__(self, idx):
        start = self.indices[idx]
        end_x = start + self.seq_len
        end_y = end_x + self.pred_len

        x_seq = self.x[start:end_x]
        y_future = self.y[end_x:end_y]
        y_base = self.y[end_x - 1]
        y_residual = y_future - y_base
        return x_seq, y_residual, y_base, y_future


class ITransformerForecaster(BaseForecaster[ForecastExtraOutput,
                                           ITransformerForecasterConfig,
                                           None,
                                           None]):
    """iTransformer 工业时序预测算子。

    训练阶段学习标准化残差；推理阶段输出物理量预测。

    输入输出约定::

        fit(x, y):
            x: (timesteps, num_features)  DataFrame / ndarray
            y: (timesteps, num_targets)   DataFrame / ndarray，通常 num_targets=1

        run(x):
            x: (seq_len, num_features) 或 (batch, seq_len, num_features)
            返回: (pred_len, num_targets) 或 (batch, pred_len, num_targets)
    """

    _SCALER_FILE = '_scaler.npz'
    _MODEL_FILE = '_model_weights.pt'
    _STATE_FILE = '_forecaster_state.npz'

    @classmethod
    def name(cls) -> str:
        return "itransformer_forecaster"

    @classmethod
    def version(cls) -> tuple[int, ...]:
        """返回算子版本号。

        Returns:
            tuple[int, ...]: 版本号三元组 ``(1, 0, 0)``
        """
        return (1, 0, 0)

    def __init__(self, *, oid: str | None = None, config: ITransformerForecasterConfig | None = None, **kwargs):
        super().__init__(oid=oid, config=config, **kwargs)
        self._scaler: StandardScaler | None = None
        self._model: nn.Module | None = None
        self._target_idx: int | None = None
        self._num_features: int | None = None
        self._num_targets: int | None = None
        self._device: torch.device = self._resolve_device()

    def _resolve_device(self) -> torch.device:
        cfg = self.config
        if cfg.device == 'auto':
            # 自动检测优先级：cuda > npu > cpu
            if torch.cuda.is_available():
                return torch.device('cuda')
            if _try_import_npu():
                return torch.device('npu')
            return torch.device('cpu')
        if cfg.device == 'npu':
            if not _try_import_npu():
                logger.warning("NPU 不可用（未安装 torch_npu 或设备未就绪），降级为 CPU")
                return torch.device('cpu')
            return torch.device('npu')
        return torch.device(cfg.device)

    def _resolve_target_idx(self, num_features: int) -> int:
        cfg = self.config
        if cfg.target_idx == -1:
            return num_features - 1
        if cfg.target_idx >= num_features:
            raise ValueError(f"target_idx {cfg.target_idx} 超出特征数 {num_features}")
        return cfg.target_idx

    def _build_model(self) -> nn.Module:
        cfg = self.config
        dim_feedforward = cfg.dim_feedforward if cfg.dim_feedforward is not None else cfg.d_model * 2
        model = ITransformerRegressor(
            seq_len=cfg.seq_len,
            num_features=self._num_features,
            out_dim=cfg.pred_len * self._num_targets,
            target_idx=self._target_idx,
            d_model=cfg.d_model,
            nhead=cfg.nhead,
            num_layers=cfg.num_layers,
            dim_feedforward=dim_feedforward,
            dropout=cfg.dropout,
            step_cond_head=cfg.step_cond_head,
            lag_aware=cfg.lag_aware,
            lag_max=cfg.lag_max,
            lag_bias_scale=cfg.lag_bias_scale,
            lag_dropout=cfg.lag_dropout,
            kan_grid_size=cfg.kan_grid_size,
        )
        return model.to(self._device)

    def _make_datasets(self, x: np.ndarray, y: np.ndarray):
        """从完整时间序列构造训练、验证数据集与索引。"""
        cfg = self.config
        n_total = len(x)
        n_samples = n_total - cfg.seq_len - cfg.pred_len + 1
        if n_samples <= 0:
            raise ValueError(
                f"时间序列长度 {n_total} 不足以构造窗口 "
                f"(seq_len={cfg.seq_len}, pred_len={cfg.pred_len})"
            )
        all_indices = np.arange(n_samples)

        # 按时间顺序划分：训练 / 验证 / 测试（测试仅保留，不使用）
        idx_train, idx_temp = train_test_split(
            all_indices, test_size=1 - cfg.train_ratio, random_state=42, shuffle=False
        )
        val_size = cfg.val_ratio / (1 - cfg.train_ratio)
        idx_val, _ = train_test_split(
            idx_temp, test_size=1 - val_size, random_state=42, shuffle=False
        )

        train_ds = _TimeSeriesDataset(x, y, idx_train, cfg.seq_len, cfg.pred_len)
        val_ds = _TimeSeriesDataset(x, y, idx_val, cfg.seq_len, cfg.pred_len)
        return train_ds, val_ds

    def _fit_data(self, x: np.ndarray, y: np.ndarray, *, params: None) -> None:
        cfg = self.config
        self._num_features = x.shape[1]
        self._num_targets = y.shape[1]
        self._target_idx = self._resolve_target_idx(self._num_features)

        # 1. 标准化（仅使用训练数据拟合，避免泄漏）
        if y.shape[1] != 1:
            raise ValueError(
                f"ITransformerForecaster 当前仅支持单目标预测，y 列数应为 1，"
                f"但当前为 {y.shape[1]}"
            )
        max_train = int(len(x) * cfg.train_ratio)
        self._scaler = StandardScaler()
        self._scaler.fit(x[:max_train])
        x_scaled = self._scaler.transform(x)
        y_scaled = x_scaled[:, [self._target_idx]]

        # 2. 构造数据集
        train_ds, val_ds = self._make_datasets(x_scaled, y_scaled)
        train_loader = DataLoader(train_ds, batch_size=cfg.batch_size, shuffle=True)
        val_loader = DataLoader(val_ds, batch_size=cfg.batch_size, shuffle=False)

        # 3. 构建模型
        self._model = self._build_model()
        optimizer = torch.optim.Adam(
            self._model.parameters(), lr=cfg.lr * 0.2, weight_decay=cfg.weight_decay
        )
        scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
            optimizer, mode='min', factor=cfg.scheduler_factor, patience=cfg.scheduler_patience
        )

        # 4. 训练循环
        time_weights = torch.linspace(
            cfg.time_weight_start, cfg.time_weight_end, steps=cfg.pred_len
        ).to(self._device).view(1, cfg.pred_len, 1)

        best_val_loss = float('inf')
        epochs_no_improve = 0
        best_state = None

        for epoch in range(cfg.epochs):
            self._model.train()
            train_loss = 0.0
            for x_batch, y_res_batch, _, _ in train_loader:
                x_batch = x_batch.to(self._device)
                y_res_batch = y_res_batch.to(self._device)
                B = x_batch.size(0)

                optimizer.zero_grad()
                y_pred_res_flat = self._model(x_batch)
                y_pred_res = y_pred_res_flat.view(B, cfg.pred_len, self._num_targets)

                loss_abs = ((y_pred_res - y_res_batch) ** 2 * time_weights).mean()
                diff_pred = y_pred_res[:, 1:, :] - y_pred_res[:, :-1, :]
                diff_true = y_res_batch[:, 1:, :] - y_res_batch[:, :-1, :]
                loss_trend = nn.functional.mse_loss(diff_pred, diff_true)
                loss = loss_abs + cfg.trend_weight * loss_trend

                loss.backward()
                torch.nn.utils.clip_grad_norm_(self._model.parameters(), cfg.max_grad_norm)
                optimizer.step()
                train_loss += loss.item()

            train_loss /= len(train_loader)

            # 验证
            self._model.eval()
            val_loss = 0.0
            with torch.no_grad():
                for x_batch, y_res_batch, _, _ in val_loader:
                    x_batch = x_batch.to(self._device)
                    y_res_batch = y_res_batch.to(self._device)
                    B = x_batch.size(0)
                    y_pred_res_flat = self._model(x_batch)
                    y_pred_res = y_pred_res_flat.view(B, cfg.pred_len, self._num_targets)

                    v_loss_abs = ((y_pred_res - y_res_batch) ** 2 * time_weights).mean()
                    v_diff_pred = y_pred_res[:, 1:, :] - y_pred_res[:, :-1, :]
                    v_diff_true = y_res_batch[:, 1:, :] - y_res_batch[:, :-1, :]
                    v_loss_trend = nn.functional.mse_loss(v_diff_pred, v_diff_true)
                    val_loss += (v_loss_abs + cfg.trend_weight * v_loss_trend).item()

            val_loss /= len(val_loader)
            scheduler.step(val_loss)

            logger.info(
                f"[{self.oid}] Epoch {epoch + 1}/{cfg.epochs} "
                f"train_loss={train_loss:.4f} val_loss={val_loss:.4f}"
            )

            if val_loss < best_val_loss:
                best_val_loss = val_loss
                epochs_no_improve = 0
                best_state = {k: v.cpu().clone() for k, v in self._model.state_dict().items()}
            else:
                epochs_no_improve += 1
                if epochs_no_improve >= cfg.early_stop_patience:
                    logger.info(f"Early stop triggered at epoch {epoch + 1}")
                    break

        # 恢复最佳权重
        if best_state is not None:
            self._model.load_state_dict(best_state)
        self._model.eval()

    def _run_data(self, x: np.ndarray, *, params: None) -> np.ndarray:
        if self._model is None or self._scaler is None:
            raise RuntimeError("模型尚未训练，无法执行推理")

        cfg = self.config
        batched = x.ndim == 3
        if not batched:
            x = x[np.newaxis, ...]  # (1, seq_len, num_features)

        # 标准化
        x_scaled = self._scaler.transform(x.reshape(-1, self._num_features)).reshape(x.shape)
        x_tensor = torch.tensor(x_scaled, dtype=torch.float32, device=self._device)

        with torch.no_grad():
            pred_res_scaled = self._model(x_tensor).cpu().numpy()

        pred_res_scaled = pred_res_scaled.reshape(-1, cfg.pred_len, self._num_targets)

        # 反标准化残差：delta_physical = normalized_delta * scale[target_idx]
        target_scale = self._scaler.scale_[self._target_idx]
        pred_res = pred_res_scaled * target_scale

        # 基准值为输入窗口最后一个时间步的目标物理值
        base_value = x[:, -1:, [self._target_idx]]  # (B, 1, num_targets)
        pred = base_value + pred_res

        return pred if batched else pred[0]

    def _save_fit_state(self, path: Path) -> None:
        super()._save_fit_state(path)
        if self._scaler is None or self._model is None:
            return

        # 保存 scaler
        np.savez(
            path / self._SCALER_FILE,
            mean=self._scaler.mean_,
            scale=self._scaler.scale_,
            n_features_in_=self._scaler.n_features_in_,
        )

        # 保存模型权重
        torch.save(self._model.state_dict(), path / self._MODEL_FILE)

        # 保存推理所需的元信息
        np.savez(
            path / self._STATE_FILE,
            target_idx=self._target_idx,
            num_features=self._num_features,
            num_targets=self._num_targets,
        )

    def _load_fit_state(self, path: Path) -> None:
        super()._load_fit_state(path)
        from sklearn.preprocessing import StandardScaler

        # 恢复 scaler
        scaler_data = np.load(path / self._SCALER_FILE)
        self._scaler = StandardScaler()
        self._scaler.mean_ = scaler_data['mean']
        self._scaler.scale_ = scaler_data['scale']
        self._scaler.n_features_in_ = int(scaler_data['n_features_in_'])

        # 恢复元信息
        state = np.load(path / self._STATE_FILE)
        self._target_idx = int(state['target_idx'])
        self._num_features = int(state['num_features'])
        self._num_targets = int(state['num_targets'])
        self._device = self._resolve_device()

        # 重建模型并加载权重
        self._model = self._build_model()
        self._model.load_state_dict(torch.load(path / self._MODEL_FILE, map_location=self._device))
        self._model.eval()

        self._fitted = True
