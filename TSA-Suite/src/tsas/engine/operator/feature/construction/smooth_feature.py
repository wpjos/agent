# -*- coding: utf-8 -*-

"""
低频特征增强算子实现模块

提供 9 个面向低频振动/电流信号的特征增强算子，均基于 IndependentMapFeature 模式。
数据模型：每个 DataFrame 格子存放一个信号段（list/ndarray），compute 逐格计算产生等长序列。

特征分组：
- Group H: 滑窗基础统计 (5) — smooth_mean, smooth_min, smooth_max, smooth_std, smooth_slope
- Group I: 滑窗梯度特征 (3) — smooth_grad_mean, smooth_grad_std, smooth_grad2_mean
- Group J: 滑窗频域特征 (1) — smooth_fft_energy

参考 bq_ops 项目的 smooth_* 算子实现，使用纯 NumPy 后端。

所有算子共享 SmoothFeatureConfig，包含 win/stride/padding_mode 三个超参数：
    - win: 滑窗长度（默认 30，建议覆盖 3-5 个基频周期）
    - stride: 步长（默认 1）
    - padding_mode: 填充模式（默认 'backward' 因果填充）
"""

from typing import Literal

import numpy as np
from pydantic import Field

from tsas.engine.operator.feature.construction.base import (
    BaseFeatureConfig,
    IndependentMapFeature,
)

__all__ = [
    # Config
    'SmoothFeatureConfig',
    # Group H: 滑窗基础统计
    'SmoothMeanFeature',
    'SmoothMinFeature',
    'SmoothMaxFeature',
    'SmoothStdFeature',
    'SmoothSlopeFeature',
    # Group I: 滑窗梯度特征
    'SmoothGradMeanFeature',
    'SmoothGradStdFeature',
    'SmoothGrad2MeanFeature',
    # Group J: 滑窗频域特征
    'SmoothFftEnergyFeature',
]


# ============================================================================
# 公共辅助函数
# ============================================================================

def _apply_per_cell_1d(x: np.ndarray, func) -> np.ndarray:
    """对 object ndarray 的每个格子应用 1D→1D 函数，返回 object ndarray。

    Args:
        x: 输入 ndarray，每个格子是一个信号段（list 或 ndarray）
        func: 变换函数，接收 1D float ndarray，返回 1D float ndarray

    Returns:
        object ndarray，形状与 x 相同，每个格子是变换后的序列
    """
    result = np.empty(x.shape, dtype=object)
    for idx in np.ndindex(x.shape):
        sig = np.asarray(x[idx], dtype=float)
        result[idx] = func(sig)
    return result


# ============================================================================
# 滑窗视图辅助（从 bq_ops/_pipeline.py 搬运）
# ============================================================================

PadMode = Literal['forward', 'backward', 'bidirectional']


def _compute_extension(n: int, win: int, stride: int) -> int:
    """计算需要填充的总长度。"""
    total_needed = (n - 1) * stride + win
    return max(0, total_needed - n)


def _split_extension(extension: int, n: int, padding_mode: PadMode) -> tuple[int, int]:
    """根据填充模式分配前/后填充长度。"""
    if extension == 0:
        return 0, 0

    if padding_mode == 'forward':
        front, back = 0, extension
    elif padding_mode == 'backward':
        front, back = extension, 0
    elif padding_mode == 'bidirectional':
        front = extension // 2
        back = extension - front
    else:
        raise ValueError(f"Unknown padding_mode: {padding_mode!r}")

    return min(front, n), min(back, n)


def pmt_numpy(sig: np.ndarray, win: int, stride: int = 1, padding_mode: PadMode = 'backward') -> np.ndarray:
    """循环填充的滑窗视图（1D 信号）。

    Args:
        sig: 1D 时域信号
        win: 滑窗长度
        stride: 步长
        padding_mode: 填充模式

    Returns:
        2D ndarray，形状为 (n, win)，每行是一个滑窗切片
    """
    sig = np.asarray(sig, dtype=float)
    n = len(sig)

    if win <= 0:
        raise ValueError(f"win must be positive, got {win}")
    if stride <= 0:
        raise ValueError(f"stride must be positive, got {stride}")

    if n == 0:
        return np.empty((0, win))

    extension = _compute_extension(n, win, stride)
    front, back = _split_extension(extension, n, padding_mode)

    if front > 0 or back > 0:
        padded = np.concatenate([sig[-front:] if front > 0 else [], sig, sig[:back] if back > 0 else []])
    else:
        padded = sig

    windows = np.lib.stride_tricks.sliding_window_view(padded, win)

    if stride > 1:
        windows = windows[::stride]

    raw = windows[:min(n, len(windows))]

    if stride == 1:
        return raw

    repeated = np.repeat(raw, stride, axis=0)
    total = len(repeated)

    if padding_mode == 'forward':
        sl = slice(0, n)
    elif padding_mode == 'backward':
        sl = slice(total - n, total)
    elif padding_mode == 'bidirectional':
        sl = slice((total - n) // 2, (total - n) // 2 + n)

    return repeated[sl]


def diff2_numpy(sig: np.ndarray) -> np.ndarray:
    """三点二阶中心差分（离散拉普拉斯）。

    Formula: y[i] = x[i+1] - 2*x[i] + x[i-1]
    边界用镜像填充处理，输出长度与输入一致。

    Args:
        sig: 1D 信号

    Returns:
        二阶差分序列，长度与输入一致
    """
    sig = np.asarray(sig, dtype=float)
    n = len(sig)

    if n <= 1:
        return np.zeros_like(sig)

    # 镜像填充 1 个元素
    padded = np.concatenate([sig[1:2], sig, sig[-2:-1]])

    # 三点模板: x[i+1] - 2*x[i] + x[i-1]
    return padded[2:] - 2 * padded[1:-1] + padded[:-2]


# ============================================================================
# Config 定义
# ============================================================================

class SmoothFeatureConfig(BaseFeatureConfig):
    """低频特征增强算子 Config。

    包含三个滑窗超参数：
        - win: 滑窗长度（建议覆盖 3-5 个基频周期）
        - stride: 步长
        - padding_mode: 填充模式（forward/backward/bidirectional）
    """
    win: int = Field(default=30, ge=1, description="滑窗长度，建议覆盖 3-5 个基频周期")
    stride: int = Field(default=1, ge=1, description="步长")
    padding_mode: PadMode = Field(default='backward', description="填充模式: forward/backward/bidirectional")


# ============================================================================
# Group H: 滑窗基础统计 (5)
# ============================================================================

class SmoothMeanFeature(IndependentMapFeature[SmoothFeatureConfig]):
    """滑动窗口均值（移动平均）特征。

    平滑信号同时保留时间分辨率，提取低频趋势。
    

    Input:
        x: 特征矩阵，形状 (n_samples, n_features)，由 Config 的 ``input_columns`` 选取

    Output:
        平滑变换后的特征矩阵，列数与输入相同，行数与输入相同
    """

    @classmethod
    def name(cls) -> str:
        return "smooth_mean_feature"

    @classmethod
    def version(cls) -> tuple[int, ...]:
        """返回算子版本号

        Returns:
            tuple[int, ...]: 版本号三元组 ``(1, 0, 0)``
        """
        return (1, 0, 0)

    @staticmethod
    def compute(x, *, state=None, **params):
        win = params.get('win', 30)
        stride = params.get('stride', 1)
        padding_mode = params.get('padding_mode', 'backward')

        def _smooth_mean_1d(sig: np.ndarray) -> np.ndarray:
            if sig.size == 0:
                return np.array([])
            windows = pmt_numpy(sig, win, stride, padding_mode)
            return np.mean(windows, axis=-1)

        return _apply_per_cell_1d(x, _smooth_mean_1d)

    def _get_compute_params(self):
        return {
            'win': self.config.win,
            'stride': self.config.stride,
            'padding_mode': self.config.padding_mode,
        }

    def _name_output_column(self, input_col, output_val):
        return self._make_output_column_name(input_col, f"smooth_mean_win{self.config.win}")


class SmoothMinFeature(IndependentMapFeature[SmoothFeatureConfig]):
    """滑动窗口最小值特征。

    形态学腐蚀滤波，提取信号下包络。
    

    Input:
        x: 特征矩阵，形状 (n_samples, n_features)，由 Config 的 ``input_columns`` 选取

    Output:
        平滑变换后的特征矩阵，列数与输入相同，行数与输入相同
    """

    @classmethod
    def name(cls) -> str:
        return "smooth_min_feature"

    @classmethod
    def version(cls) -> tuple[int, ...]:
        """返回算子版本号

        Returns:
            tuple[int, ...]: 版本号三元组 ``(1, 0, 0)``
        """
        return (1, 0, 0)

    @staticmethod
    def compute(x, *, state=None, **params):
        win = params.get('win', 30)
        stride = params.get('stride', 1)
        padding_mode = params.get('padding_mode', 'backward')

        def _smooth_min_1d(sig: np.ndarray) -> np.ndarray:
            if sig.size == 0:
                return np.array([])
            windows = pmt_numpy(sig, win, stride, padding_mode)
            return np.min(windows, axis=-1)

        return _apply_per_cell_1d(x, _smooth_min_1d)

    def _get_compute_params(self):
        return {
            'win': self.config.win,
            'stride': self.config.stride,
            'padding_mode': self.config.padding_mode,
        }

    def _name_output_column(self, input_col, output_val):
        return self._make_output_column_name(input_col, f"smooth_min_win{self.config.win}")


class SmoothMaxFeature(IndependentMapFeature[SmoothFeatureConfig]):
    """滑动窗口最大值特征。

    形态学膨胀滤波，提取信号上包络。
    

    Input:
        x: 特征矩阵，形状 (n_samples, n_features)，由 Config 的 ``input_columns`` 选取

    Output:
        平滑变换后的特征矩阵，列数与输入相同，行数与输入相同
    """

    @classmethod
    def name(cls) -> str:
        return "smooth_max_feature"

    @classmethod
    def version(cls) -> tuple[int, ...]:
        """返回算子版本号

        Returns:
            tuple[int, ...]: 版本号三元组 ``(1, 0, 0)``
        """
        return (1, 0, 0)

    @staticmethod
    def compute(x, *, state=None, **params):
        win = params.get('win', 30)
        stride = params.get('stride', 1)
        padding_mode = params.get('padding_mode', 'backward')

        def _smooth_max_1d(sig: np.ndarray) -> np.ndarray:
            if sig.size == 0:
                return np.array([])
            windows = pmt_numpy(sig, win, stride, padding_mode)
            return np.max(windows, axis=-1)

        return _apply_per_cell_1d(x, _smooth_max_1d)

    def _get_compute_params(self):
        return {
            'win': self.config.win,
            'stride': self.config.stride,
            'padding_mode': self.config.padding_mode,
        }

    def _name_output_column(self, input_col, output_val):
        return self._make_output_column_name(input_col, f"smooth_max_win{self.config.win}")


class SmoothStdFeature(IndependentMapFeature[SmoothFeatureConfig]):
    """滑动窗口标准差特征。

    量化信号局部波动程度，监测不稳定性的变化趋势。
    

    Input:
        x: 特征矩阵，形状 (n_samples, n_features)，由 Config 的 ``input_columns`` 选取

    Output:
        平滑变换后的特征矩阵，列数与输入相同，行数与输入相同
    """

    @classmethod
    def name(cls) -> str:
        return "smooth_std_feature"

    @classmethod
    def version(cls) -> tuple[int, ...]:
        """返回算子版本号

        Returns:
            tuple[int, ...]: 版本号三元组 ``(1, 0, 0)``
        """
        return (1, 0, 0)

    @staticmethod
    def compute(x, *, state=None, **params):
        win = params.get('win', 30)
        stride = params.get('stride', 1)
        padding_mode = params.get('padding_mode', 'backward')

        def _smooth_std_1d(sig: np.ndarray) -> np.ndarray:
            if sig.size == 0:
                return np.array([])
            windows = pmt_numpy(sig, win, stride, padding_mode)
            return np.std(windows, axis=-1, ddof=1)

        return _apply_per_cell_1d(x, _smooth_std_1d)

    def _get_compute_params(self):
        return {
            'win': self.config.win,
            'stride': self.config.stride,
            'padding_mode': self.config.padding_mode,
        }

    def _name_output_column(self, input_col, output_val):
        return self._make_output_column_name(input_col, f"smooth_std_win{self.config.win}")


class SmoothSlopeFeature(IndependentMapFeature[SmoothFeatureConfig]):
    """滑动窗口 OLS 回归斜率特征。

    量化信号局部趋势变化，监测漂移/渐变过程。

    使用闭式解: β = (Σ i·y - n·x̄·ȳ) / (n(n²-1)/12)
    其中 x = [0, 1, ..., win-1]
    

    Input:
        x: 特征矩阵，形状 (n_samples, n_features)，由 Config 的 ``input_columns`` 选取

    Output:
        平滑变换后的特征矩阵，列数与输入相同，行数与输入相同
    """

    @classmethod
    def name(cls) -> str:
        return "smooth_slope_feature"

    @classmethod
    def version(cls) -> tuple[int, ...]:
        """返回算子版本号

        Returns:
            tuple[int, ...]: 版本号三元组 ``(1, 0, 0)``
        """
        return (1, 0, 0)

    @staticmethod
    def compute(x, *, state=None, **params):
        win = params.get('win', 30)
        stride = params.get('stride', 1)
        padding_mode = params.get('padding_mode', 'backward')

        def _smooth_slope_1d(sig: np.ndarray) -> np.ndarray:
            if sig.size == 0:
                return np.array([])
            if win < 2:
                return np.zeros_like(sig)

            windows = pmt_numpy(sig, win, stride, padding_mode)

            # 闭式 OLS
            n = win
            x_mean = (n - 1) / 2.0
            denominator = n * (n * n - 1) / 12.0
            x_arr = np.arange(n, dtype=float)

            # dot(x, y) along last axis
            dot_xy = np.einsum('...i,i->...', windows, x_arr)
            y_mean = np.mean(windows, axis=-1)

            return (dot_xy - n * x_mean * y_mean) / denominator

        return _apply_per_cell_1d(x, _smooth_slope_1d)

    def _get_compute_params(self):
        return {
            'win': self.config.win,
            'stride': self.config.stride,
            'padding_mode': self.config.padding_mode,
        }

    def _name_output_column(self, input_col, output_val):
        return self._make_output_column_name(input_col, f"smooth_slope_win{self.config.win}")


# ============================================================================
# Group I: 滑窗梯度特征 (3)
# ============================================================================

class SmoothGradMeanFeature(IndependentMapFeature[SmoothFeatureConfig]):
    """滑动窗口均值的一阶梯度特征。

    量化平滑后信号的变化率，监测趋势转折点。
    先 smooth_mean 再 gradient。
    

    Input:
        x: 特征矩阵，形状 (n_samples, n_features)，由 Config 的 ``input_columns`` 选取

    Output:
        平滑变换后的特征矩阵，列数与输入相同，行数与输入相同
    """

    @classmethod
    def name(cls) -> str:
        return "smooth_grad_mean_feature"

    @classmethod
    def version(cls) -> tuple[int, ...]:
        """返回算子版本号

        Returns:
            tuple[int, ...]: 版本号三元组 ``(1, 0, 0)``
        """
        return (1, 0, 0)

    @staticmethod
    def compute(x, *, state=None, **params):
        win = params.get('win', 30)
        stride = params.get('stride', 1)
        padding_mode = params.get('padding_mode', 'backward')

        def _smooth_grad_mean_1d(sig: np.ndarray) -> np.ndarray:
            if sig.size == 0:
                return np.array([])
            windows = pmt_numpy(sig, win, stride, padding_mode)
            smoothed = np.mean(windows, axis=-1)
            return np.gradient(smoothed)

        return _apply_per_cell_1d(x, _smooth_grad_mean_1d)

    def _get_compute_params(self):
        return {
            'win': self.config.win,
            'stride': self.config.stride,
            'padding_mode': self.config.padding_mode,
        }

    def _name_output_column(self, input_col, output_val):
        return self._make_output_column_name(input_col, f"smooth_grad_mean_win{self.config.win}")


class SmoothGradStdFeature(IndependentMapFeature[SmoothFeatureConfig]):
    """滑动窗口标准差的一阶梯度特征。

    量化波动程度的变化率，监测波动突变点。
    先 smooth_std 再 gradient。
    

    Input:
        x: 特征矩阵，形状 (n_samples, n_features)，由 Config 的 ``input_columns`` 选取

    Output:
        平滑变换后的特征矩阵，列数与输入相同，行数与输入相同
    """

    @classmethod
    def name(cls) -> str:
        return "smooth_grad_std_feature"

    @classmethod
    def version(cls) -> tuple[int, ...]:
        """返回算子版本号

        Returns:
            tuple[int, ...]: 版本号三元组 ``(1, 0, 0)``
        """
        return (1, 0, 0)

    @staticmethod
    def compute(x, *, state=None, **params):
        win = params.get('win', 30)
        stride = params.get('stride', 1)
        padding_mode = params.get('padding_mode', 'backward')

        def _smooth_grad_std_1d(sig: np.ndarray) -> np.ndarray:
            if sig.size == 0:
                return np.array([])
            windows = pmt_numpy(sig, win, stride, padding_mode)
            smoothed = np.std(windows, axis=-1, ddof=1)
            return np.gradient(smoothed)

        return _apply_per_cell_1d(x, _smooth_grad_std_1d)

    def _get_compute_params(self):
        return {
            'win': self.config.win,
            'stride': self.config.stride,
            'padding_mode': self.config.padding_mode,
        }

    def _name_output_column(self, input_col, output_val):
        return self._make_output_column_name(input_col, f"smooth_grad_std_win{self.config.win}")


class SmoothGrad2MeanFeature(IndependentMapFeature[SmoothFeatureConfig]):
    """滑动窗口均值的二阶差分特征。

    量化平滑信号的曲率/加速度，检测趋势拐点。
    先 smooth_mean 再 diff2。
    

    Input:
        x: 特征矩阵，形状 (n_samples, n_features)，由 Config 的 ``input_columns`` 选取

    Output:
        平滑变换后的特征矩阵，列数与输入相同，行数与输入相同
    """

    @classmethod
    def name(cls) -> str:
        return "smooth_grad2_mean_feature"

    @classmethod
    def version(cls) -> tuple[int, ...]:
        """返回算子版本号

        Returns:
            tuple[int, ...]: 版本号三元组 ``(1, 0, 0)``
        """
        return (1, 0, 0)

    @staticmethod
    def compute(x, *, state=None, **params):
        win = params.get('win', 30)
        stride = params.get('stride', 1)
        padding_mode = params.get('padding_mode', 'backward')

        def _smooth_grad2_mean_1d(sig: np.ndarray) -> np.ndarray:
            if sig.size == 0:
                return np.array([])
            windows = pmt_numpy(sig, win, stride, padding_mode)
            smoothed = np.mean(windows, axis=-1)
            return diff2_numpy(smoothed)

        return _apply_per_cell_1d(x, _smooth_grad2_mean_1d)

    def _get_compute_params(self):
        return {
            'win': self.config.win,
            'stride': self.config.stride,
            'padding_mode': self.config.padding_mode,
        }

    def _name_output_column(self, input_col, output_val):
        return self._make_output_column_name(input_col, f"smooth_grad2_mean_win{self.config.win}")


# ============================================================================
# Group J: 滑窗频域特征 (1)
# ============================================================================

class SmoothFftEnergyFeature(IndependentMapFeature[SmoothFeatureConfig]):
    """滑动窗口 FFT 能量特征。

    量化信号局部频谱总能量，监测能量突变点。
    

    Input:
        x: 特征矩阵，形状 (n_samples, n_features)，由 Config 的 ``input_columns`` 选取

    Output:
        平滑变换后的特征矩阵，列数与输入相同，行数与输入相同
    """

    @classmethod
    def name(cls) -> str:
        return "smooth_fft_energy_feature"

    @classmethod
    def version(cls) -> tuple[int, ...]:
        """返回算子版本号

        Returns:
            tuple[int, ...]: 版本号三元组 ``(1, 0, 0)``
        """
        return (1, 0, 0)

    @staticmethod
    def compute(x, *, state=None, **params):
        win = params.get('win', 30)
        stride = params.get('stride', 1)
        padding_mode = params.get('padding_mode', 'backward')

        def _smooth_fft_energy_1d(sig: np.ndarray) -> np.ndarray:
            if sig.size == 0:
                return np.array([])
            windows = pmt_numpy(sig, win, stride, padding_mode)
            spectrum = np.fft.rfft(windows, axis=-1)
            energy = spectrum.real ** 2 + spectrum.imag ** 2
            return np.sum(energy, axis=-1)

        return _apply_per_cell_1d(x, _smooth_fft_energy_1d)

    def _get_compute_params(self):
        return {
            'win': self.config.win,
            'stride': self.config.stride,
            'padding_mode': self.config.padding_mode,
        }

    def _name_output_column(self, input_col, output_val):
        return self._make_output_column_name(input_col, f"smooth_fft_energy_win{self.config.win}")
