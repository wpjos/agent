# -*- coding: utf-8 -*-

"""
信号特征构造算子实现模块

提供 32 个面向预测性维护场景的信号特征算子，均基于 IndependentMapFeature + Base 模式。
数据模型：每个 DataFrame 格子存放一个信号段（list/ndarray），compute 逐格计算产生标量特征值。

特征分组：
- Group A: 简单统计特征 (11) — 无额外参数
- Group B: 需要采样率的特征 (3) — sample_rate
- Group C: 频域特征 (5) — sample_rate
- Group D: 复合特征 (3) — envelope_rms, average_kurtosis, hnr
- Group E: 频带特征 (3 类，可参数化为 9 个实例) — sample_rate + band boundaries
- Group F: 转速特征 (1) — HPS+ speed_rpm

参考 ops 项目的算子实现，使用纯 NumPy 后端。
"""

import math

import numpy as np
from pydantic import Field

from tsas.engine.operator.feature.construction.base import (
    BaseFeatureConfig,
    IndependentMapFeature,
)

__all__ = [
    # Config
    'SampleRateFeatureConfig',
    'BandFeatureConfig',
    'AverageKurtosisConfig',
    'SpeedRpmConfig',
    # Group A: 简单统计特征
    'MeanSquareFeature',
    'VarianceFeature',
    'RmsFeature',
    'PeakPeakFeature',
    'ShapeFactorFeature',
    'CrestFeature',
    'ImpulseFeature',
    'ClearanceFeature',
    'SkewnessFeature',
    'KurtosisFeature',
    'GiniIndexFeature',
    # Group B: 需要采样率的特征
    'SpectralEntropyFeature',
    'RoughnessFeature',
    'SharpnessFeature',
    # Group C: 频域特征
    'SpectralCentroidFeature',
    'MeanSquareFrequencyFeature',
    'RmsFrequencyFeature',
    'FrequencyVarianceFeature',
    'FrequencyStdFeature',
    # Group D: 复合特征
    'EnvelopeRmsFeature',
    'AverageKurtosisFeature',
    'HnrFeature',
    # Group E: 频带特征
    'BandKurtosisFeature',
    'BandRmsFeature',
    'BandHnrFeature',
    # Group F: 转速特征
    'SpeedRpmFeature',
]


# ============================================================================
# 公共辅助函数
# ============================================================================

def _apply_per_cell(x: np.ndarray, func) -> np.ndarray:
    """对 object ndarray 的每个格子应用标量函数，返回 float ndarray。

    Args:
        x: 输入 ndarray，每个格子是一个信号段（list 或 ndarray）
        func: 标量函数，接收 1D float ndarray，返回 float

    Returns:
        float ndarray，形状与 x 相同
    """
    result = np.empty(x.shape, dtype=float)
    for idx in np.ndindex(x.shape):
        sig = np.asarray(x[idx], dtype=float)
        result[idx] = func(sig)
    return result


def _safe_div(num: float, den: float) -> float:
    """安全除法，分母为 0 时返回 0.0。"""
    return 0.0 if den == 0.0 else float(num / den)


# ============================================================================
# 内部计算函数（1D 信号处理）
# ============================================================================

def _rms_1d(sig: np.ndarray) -> float:
    """RMS: sqrt(mean(x^2))。"""
    if sig.size == 0:
        return 0.0
    return float(np.sqrt(np.mean(sig ** 2)))


def _mean_square_1d(sig: np.ndarray) -> float:
    """均方值: mean(x^2)。"""
    if sig.size == 0:
        return 0.0
    return float(np.mean(sig ** 2))


def _variance_1d(sig: np.ndarray) -> float:
    """方差: var(x, ddof=1)。"""
    if sig.size <= 1:
        return 0.0
    return float(np.var(sig, ddof=1))


def _peak_peak_1d(sig: np.ndarray) -> float:
    """峰峰值: max(x) - min(x)。"""
    if sig.size == 0:
        return 0.0
    return float(np.ptp(sig))


def _shape_factor_1d(sig: np.ndarray) -> float:
    """波形因子: RMS / mean(|x|)。"""
    if sig.size == 0:
        return 0.0
    rms_val = _rms_1d(sig)
    abs_mean = float(np.mean(np.abs(sig)))
    return _safe_div(rms_val, abs_mean)


def _crest_1d(sig: np.ndarray) -> float:
    """峰值因子: max(|x|) / RMS。"""
    if sig.size == 0:
        return 0.0
    max_abs = float(np.max(np.abs(sig)))
    rms_val = _rms_1d(sig)
    return _safe_div(max_abs, rms_val)


def _impulse_1d(sig: np.ndarray) -> float:
    """脉冲因子: max(|x|) / mean(|x|)。"""
    if sig.size == 0:
        return 0.0
    max_abs = float(np.max(np.abs(sig)))
    abs_mean = float(np.mean(np.abs(sig)))
    return _safe_div(max_abs, abs_mean)


def _clearance_1d(sig: np.ndarray) -> float:
    """裕度因子: max(|x|) / (mean(sqrt(|x|)))^2。"""
    if sig.size == 0:
        return 0.0
    max_abs = float(np.max(np.abs(sig)))
    sqrt_abs_mean = float(np.mean(np.sqrt(np.abs(sig))))
    denominator = sqrt_abs_mean ** 2
    return _safe_div(max_abs, denominator)


def _skewness_1d(sig: np.ndarray) -> float:
    """偏斜度: mean((x - mu)^3) / std^3。"""
    if sig.size < 2:
        return 0.0
    mean_val = np.mean(sig)
    std_val = np.std(sig, ddof=1)
    if std_val == 0.0:
        return 0.0
    third_moment = np.mean((sig - mean_val) ** 3)
    return float(third_moment / (std_val ** 3))


def _kurtosis_1d(sig: np.ndarray) -> float:
    """超额峭度: mean((x - mu)^4) / std^4 - 3。"""
    if sig.size < 2:
        return 0.0
    mean_val = np.mean(sig)
    std_val = np.std(sig, ddof=1)
    if std_val == 0.0:
        return 0.0
    fourth_moment = np.mean((sig - mean_val) ** 4)
    return float(fourth_moment / (std_val ** 4) - 3)


def _gini_index_1d(sig: np.ndarray) -> float:
    """Gini 指数: 对 |x| 计算分布不均匀度。"""
    abs_sig = np.abs(sig)
    if abs_sig.size == 0:
        return 0.0
    sorted_sig = np.sort(abs_sig)
    n = len(sorted_sig)
    total = np.sum(sorted_sig)
    if total == 0.0:
        return 0.0
    indices = np.arange(1, n + 1)
    weighted_sum = np.sum(indices * sorted_sig)
    return float((2 * weighted_sum) / (n * total) - (n + 1) / n)


def _hilbert_envelope_1d(sig: np.ndarray) -> np.ndarray:
    """Hilbert 包络（纯 NumPy 实现）。"""
    n = len(sig)
    if n < 2:
        return np.abs(sig)
    X = np.fft.fft(sig)
    h = np.zeros(n)
    if n % 2 == 0:
        h[0] = 1
        h[1:n // 2] = 2
        h[n // 2] = 1
    else:
        h[0] = 1
        h[1:(n + 1) // 2] = 2
    analytic = np.fft.ifft(X * h)
    return np.abs(analytic)


def _envelope_rms_1d(sig: np.ndarray) -> float:
    """包络 RMS: envelope → rms。"""
    if sig.size < 2:
        return 0.0
    env = _hilbert_envelope_1d(sig)
    return _rms_1d(env)


def _average_kurtosis_1d(sig: np.ndarray, n_segments: int) -> float:
    """分段峭度均值。"""
    if sig.size < 2:
        return 0.0
    n = len(sig)
    seg_len = n // n_segments
    if seg_len < 2:
        return _kurtosis_1d(sig)
    segments = [sig[i * seg_len:(i + 1) * seg_len] for i in range(n_segments)]
    valid_kurtoses = [_kurtosis_1d(seg) for seg in segments if len(seg) >= 2]
    if not valid_kurtoses:
        return 0.0
    return float(np.mean(valid_kurtoses))


def _hnr_1d(sig: np.ndarray) -> float:
    """谐噪比（基于自相关峰值近似），返回 dB。"""
    if sig.size < 3:
        return 0.0
    x = sig - np.mean(sig)
    n = len(x)
    n_fft = 1 << (2 * n - 1).bit_length()
    spectrum = np.fft.rfft(x, n=n_fft)
    corr = np.fft.irfft(spectrum * np.conj(spectrum), n=n_fft)[:n]
    if corr[0] == 0:
        return 0.0
    corr = corr / corr[0]
    r = float(np.max(corr[1:]))
    r = np.clip(r, 0.0, 0.999999)
    return float(10.0 * np.log10(_safe_div(r, 1.0 - r)))


def _bandpass_fft_1d(sig: np.ndarray, sample_rate: float, low: float, high: float | None) -> np.ndarray:
    """FFT 频域带通滤波。high=None 表示到 Nyquist。"""
    n = len(sig)
    if n < 2:
        return sig
    X = np.fft.rfft(sig)
    freqs = np.fft.rfftfreq(n, d=1.0 / sample_rate)
    high_hz = sample_rate / 2 if high is None else high
    mask = (freqs >= low) & (freqs <= high_hz)
    return np.fft.irfft(X * mask, n=n)


def _frequency_features_1d(sig: np.ndarray, sample_rate: float) -> tuple:
    """频域特征: (centroid, msf, rmsf, freq_var, freq_std)。"""
    if sig.size < 2:
        return (0.0, 0.0, 0.0, 0.0, 0.0)
    freqs = np.fft.rfftfreq(len(sig), d=1.0 / sample_rate)
    power = np.abs(np.fft.rfft(sig)) ** 2
    total = np.sum(power)
    if total == 0.0:
        return (0.0, 0.0, 0.0, 0.0, 0.0)
    centroid = float(np.sum(freqs * power) / total)
    msf = float(np.sum((freqs ** 2) * power) / total)
    rmsf = float(np.sqrt(msf))
    freq_var = float(np.sum(((freqs - centroid) ** 2) * power) / total)
    freq_std = float(np.sqrt(freq_var))
    return (centroid, msf, rmsf, freq_var, freq_std)


def _welch_psd(sig: np.ndarray, sample_rate: float, nperseg: int = 256) -> np.ndarray:
    """NumPy Welch PSD 估计（与 scipy.signal.welch 结果对齐）。"""
    n = len(sig)
    nperseg = min(nperseg, n)
    noverlap = nperseg // 2
    step = nperseg - noverlap
    window = 0.5 * (1 - np.cos(2 * np.pi * np.arange(nperseg) / nperseg))
    win_power = np.sum(window ** 2)
    n_fft = nperseg
    n_freqs = n_fft // 2 + 1
    n_segments = max(1, (n - noverlap) // step)
    psd_acc = np.zeros(n_freqs)
    for i in range(n_segments):
        start = i * step
        segment = sig[start:start + nperseg].copy()
        segment -= np.mean(segment)
        segment *= window
        X = np.fft.rfft(segment, n=n_fft)
        psd_acc += np.abs(X) ** 2
    psd = psd_acc / (n_segments * sample_rate * win_power)
    if n_fft % 2 == 0:
        if n_freqs > 2:
            psd[1:-1] *= 2.0
    else:
        if n_freqs > 1:
            psd[1:] *= 2.0
    return psd


def _spectral_entropy_1d(sig: np.ndarray) -> float:
    """功率谱熵（Welch PSD → 归一化 Shannon 熵）。"""
    if sig.size < 2:
        return 0.0
    nperseg = min(256, len(sig))
    psd = _welch_psd(sig, 1.0, nperseg)
    total = np.sum(psd)
    if total <= 0.0:
        return 0.0
    p = psd / total
    p_pos = p[p > 0]
    entropy = -np.sum(p_pos * np.log2(p_pos))
    max_entropy = np.log2(len(p_pos))
    if max_entropy <= 0:
        return 0.0
    return float(entropy / max_entropy)


# ============================================================================
# 心理声学特征辅助（Bark 频谱 + 特定响度）
# ============================================================================

# 阈值静音曲线（dB SPL per Bark band）
_THRESHOLD_OF_QUIET = np.array([
    44, 32, 25, 19, 14, 11, 8, 6, 5, 5,
    5, 5, 5, 5, 5, 5, 5, 5, 5, 5,
    5, 5, 5, 5,
])


def _hz_to_bark(f: float | np.ndarray) -> float | np.ndarray:
    """Hz → Bark (Zwicker 1961)。"""
    return 13.0 * np.arctan(0.00076 * f) + 3.5 * np.arctan((f / 7500.0) ** 2)


def _bark_band_edges(n_barks: int = 24) -> np.ndarray:
    """Bark 频带边界（Hz）。"""
    bark_edges = np.arange(n_barks + 1)
    hz_edges = np.zeros(n_barks + 1)
    for i, z in enumerate(bark_edges):
        hz_edges[i] = _bark_to_hz_approx(z)
    return hz_edges


def _bark_to_hz_approx(z: float) -> float:
    """Bark → Hz 近似逆变换（二分法）。"""
    if z <= 0:
        return 0.0
    if z >= 24:
        return 15500.0
    lo, hi = 0.0, 20000.0
    for _ in range(50):
        mid = (lo + hi) / 2
        if _hz_to_bark(mid) < z:
            lo = mid
        else:
            hi = mid
    return (lo + hi) / 2


def _bark_spectrum_from_signal(sig: np.ndarray, sample_rate: float, n_barks: int = 24) -> np.ndarray:
    """从时域信号计算 Bark 功率谱。"""
    n = len(sig)
    if n == 0:
        return np.zeros(n_barks)
    X = np.fft.rfft(sig)
    power = np.abs(X) ** 2 / n
    freqs = np.fft.rfftfreq(n, d=1.0 / sample_rate)
    edges = _bark_band_edges(n_barks)
    bark_power = np.zeros(n_barks)
    for i in range(n_barks):
        mask = (freqs >= edges[i]) & (freqs < edges[i + 1])
        bark_power[i] = np.sum(power[mask])
    return bark_power


def _power_to_specific_loudness(bark_power: np.ndarray, ref_power: float = 1e-12) -> np.ndarray:
    """Bark 功率 → 特定响度 N'(z)（简化 Zwicker 模型）。"""
    bark_power = np.asarray(bark_power, dtype=float)
    if bark_power.size == 0:
        return np.array([])
    eps = 1e-20
    level_db = 10 * np.log10(np.maximum(bark_power, eps) / ref_power)
    effective_level = level_db - _THRESHOLD_OF_QUIET
    loudness = np.zeros_like(effective_level)
    above_threshold = effective_level > 0
    if above_threshold.any():
        loudness[above_threshold] = 0.08 * (10 ** (effective_level[above_threshold] / 10)) ** 0.23
    return loudness


def _sharpness_from_specific_loudness(n_prime: np.ndarray) -> float:
    """特定响度 → 锐度（acum）。"""
    n_prime = np.asarray(n_prime, dtype=float)
    total_loudness = np.sum(n_prime)
    if total_loudness <= 0:
        return 0.0
    z = np.arange(len(n_prime)) + 0.5
    g = np.ones_like(z, dtype=float)
    high = z > 15
    g[high] = 0.066 * np.exp(0.171 * z[high])
    weighted_sum = np.sum(n_prime * g * z)
    return float(0.11 * weighted_sum / total_loudness)


def _sharpness_1d(sig: np.ndarray, sample_rate: float) -> float:
    """锐度计算（从时域信号）。"""
    if sig.size < 2:
        return 0.0
    bark_power = _bark_spectrum_from_signal(sig, sample_rate)
    n_prime = _power_to_specific_loudness(bark_power)
    return _sharpness_from_specific_loudness(n_prime)


def _roughness_1d(sig: np.ndarray, sample_rate: float) -> float:
    """粗糙度计算（Zwicker & Fastl 模型，从时域信号）。"""
    if sig.size < 8:
        return 0.0
    n_barks = 24
    edges = _bark_band_edges(n_barks)
    total_roughness = 0.0
    n = len(sig)
    X_full = np.fft.fft(sig)
    freqs_full = np.fft.fftfreq(n, d=1.0 / sample_rate)

    for band_idx in range(n_barks):
        lo_hz = edges[band_idx]
        hi_hz = edges[band_idx + 1]
        mask = np.zeros(n, dtype=bool)
        mask[(np.abs(freqs_full) >= lo_hz) & (np.abs(freqs_full) < hi_hz)] = True
        mask[0] = False
        band_signal = np.real(np.fft.ifft(X_full * mask))
        band_power = np.mean(band_signal ** 2)
        if band_power < 1e-20:
            continue
        env = _hilbert_envelope_1d(band_signal)
        mean_env = np.mean(env)
        if mean_env <= 0:
            continue
        mod_depth = min(np.std(env - mean_env) / mean_env, 1.0)
        if mod_depth <= 0:
            continue
        ac = np.correlate(env - mean_env, env - mean_env, mode='full')
        ac = ac[len(ac) // 2:]
        if ac[0] <= 0:
            continue
        ac = ac / ac[0]
        min_lag = max(1, int(sample_rate / 500))
        if min_lag >= len(ac):
            continue
        ac_trimmed = ac[min_lag:]
        if len(ac_trimmed) == 0:
            continue
        peak_idx = np.argmax(ac_trimmed)
        peak_lag = peak_idx + min_lag
        if peak_lag <= 0 or ac_trimmed[peak_idx] <= 0:
            continue
        mod_freq = sample_rate / peak_lag
        if mod_freq <= 0:
            continue
        weight = mod_freq / 70.0 * np.exp(1 - mod_freq / 70.0)
        total_roughness += 0.3 * mod_depth * weight * band_power
    return float(total_roughness)


# ============================================================================
# HPS+ 转速估计辅助函数
# ============================================================================

def _hps_spectral_normalize(spectrum: np.ndarray, win_bins: int) -> np.ndarray:
    """频谱归一化（平滑包络拉平）。

    将频谱除以平滑后的自身，让各频段峰值公平竞争。
    """
    win = max(2 * win_bins + 1, 3)
    if spectrum.size < win:
        return spectrum
    kernel = np.ones(win) / win
    smoothed = np.convolve(spectrum, kernel, mode="valid")
    shift = 2 * win_bins - 1
    baseline = np.full(spectrum.size, np.nan)
    fill1 = min(baseline.size, smoothed.size - shift)
    if fill1 > 0:
        baseline[:fill1] = smoothed[shift: shift + fill1]
    start2 = 3 * win_bins
    fill2 = min(baseline.size - start2, smoothed.size)
    if fill2 > 0:
        region = baseline[start2: start2 + fill2]
        mask = ~np.isnan(region)
        region[mask] = (region[mask] + smoothed[: np.sum(mask)]) / 2.0
        region[~mask] = smoothed[: np.sum(~mask)]
        baseline[start2: start2 + fill2] = region
    bl = np.nan_to_num(baseline, nan=1.0)
    bl = np.maximum(bl, 1e-10)
    return spectrum / bl


def _hps_core(spectrum: np.ndarray, n_harmonics: int = 5, normalize_win: int | None = None) -> np.ndarray:
    """Harmonic Product Spectrum（对数域）。

    HPS(f) = Π(h=1..H) |S(h·f)|，在对数域计算避免数值溢出。
    """
    spectrum = np.asarray(spectrum, dtype=float)
    if spectrum.size == 0:
        return np.array([], dtype=float)
    if spectrum.size == 1:
        return np.array([0.0])
    if normalize_win is not None and normalize_win > 0:
        spectrum = _hps_spectral_normalize(spectrum, normalize_win)
    out_len = spectrum.size // n_harmonics
    if out_len == 0:
        return np.array([0.0])
    log_spec = np.log(np.maximum(spectrum[:out_len], 1e-30))
    result = log_spec.copy()
    for h in range(2, n_harmonics + 1):
        downsampled = spectrum[::h][:out_len]
        if downsampled.size < out_len:
            break
        result += np.log(np.maximum(downsampled, 1e-30))
    return result


def _hps_pick_peak_validated(product: np.ndarray, spectrum: np.ndarray,
                             bin_min: int, bin_max: int, n_harmonics: int = 5) -> int | None:
    """谐波验证选峰（HPS+ 核心策略）。

    在 HPS 产品中找峰值候选，对每个候选统计其谐波族在原始频谱中的峰个数和突出度，
    选择得分最高的候选。这是 HPS+ 抗噪的关键创新。
    """
    bin_min = max(bin_min, 0)
    bin_max = min(bin_max, product.size - 1)
    if bin_min >= bin_max:
        return None
    segment = product[bin_min: bin_max + 1]
    padded = np.pad(segment, 1, constant_values=-np.inf)
    is_peak = (segment > padded[:-2]) & (segment > padded[2:])
    if not np.any(is_peak):
        return bin_min + int(np.argmax(segment))
    candidates = np.where(is_peak)[0] + bin_min
    if candidates.size == 1:
        return int(candidates[0])
    spec_padded = np.pad(spectrum, 1, constant_values=-np.inf)
    spec_is_peak = (spectrum > spec_padded[:-2]) & (spectrum > spec_padded[2:])
    spec_peaks = set(np.where(spec_is_peak)[0])

    def compute_prominence(hb):
        half_w = max(2, hb // 4) if hb > 0 else 2
        lo = max(0, hb - half_w)
        hi = min(spectrum.size, hb + half_w + 1)
        neighborhood = np.concatenate([spectrum[lo:hb], spectrum[hb + 1:hi]])
        if neighborhood.size == 0:
            return spectrum[hb]
        return spectrum[hb] - np.mean(neighborhood)

    best_bin = None
    best_peak_count = -1
    best_prominence = -np.inf
    for c in candidates:
        c = int(c)
        peak_count = 0
        prominence_sum = 0.0
        for h in range(1, n_harmonics + 1):
            hb = c * h
            if hb >= spectrum.size:
                break
            if hb in spec_peaks:
                peak_count += 1
            prominence_sum += compute_prominence(hb)
        if peak_count > best_peak_count:
            best_peak_count = peak_count
            best_prominence = prominence_sum
            best_bin = c
        elif peak_count == best_peak_count and prominence_sum > best_prominence:
            best_prominence = prominence_sum
            best_bin = c
    return best_bin


def _speed_rpm_1d(sig: np.ndarray, sample_rate: float, speed_min: float, speed_max: float,
                  n_harmonics: int = 5, speed_delta: float = 10.0, std_min: float = 0.01) -> float:
    """转速估计（HPS+ 方法）。

    Args:
        sig: 时域振动信号
        sample_rate: 采样率 (Hz)
        speed_min, speed_max: 转速搜索范围 (RPM)
        n_harmonics: 谐波数（默认 5）
        speed_delta: 转速分辨率 (RPM)，用于频谱归一化窗口
        std_min: 信号最小标准差（低于此返回 0）

    Returns:
        float: 估计转速 (RPM)，失败返回 0.0
    """
    x = np.asarray(sig, dtype=float)
    if x.size < 2 or np.std(x) < std_min:
        return 0.0
    N = x.size
    spectrum = np.abs(np.fft.rfft(x))
    # 频谱归一化窗口
    ndeta = max(1, math.ceil(speed_delta / 60 * N / sample_rate))
    normed = _hps_spectral_normalize(spectrum, ndeta)
    # HPS 核心
    product = _hps_core(normed, n_harmonics=n_harmonics)
    # 搜索范围映射到 bin
    bin_min = max(1, int(speed_min / 60 * N / sample_rate))
    bin_max = min(math.ceil(speed_max / 60 * N / sample_rate), product.size - 1)
    # 谐波验证选峰
    best_bin = _hps_pick_peak_validated(product, normed, bin_min, bin_max, n_harmonics)
    if best_bin is None:
        return 0.0
    return float(best_bin / N * sample_rate * 60)


# ============================================================================
# Config 定义
# ============================================================================

class SampleRateFeatureConfig(BaseFeatureConfig):
    """需要采样率的特征 Config。"""
    sample_rate: float = Field(gt=0, description="采样率 (Hz)")


class BandFeatureConfig(SampleRateFeatureConfig):
    """频带特征 Config。"""
    low: float = Field(default=0.0, ge=0, description="频带下界 (Hz)")
    high: float | None = Field(default=None, description="频带上界 (Hz)，None 表示 Nyquist")


class AverageKurtosisConfig(BaseFeatureConfig):
    """分段峭度均值特征 Config。"""
    n_segments: int = Field(default=10, ge=1, description="分段数")


class SpeedRpmConfig(SampleRateFeatureConfig):
    """转速估计（HPS+）特征 Config。"""
    speed_min: float = Field(gt=0, description="最小转速搜索范围 (RPM)")
    speed_max: float = Field(gt=0, description="最大转速搜索范围 (RPM)")
    n_harmonics: int = Field(default=5, ge=1, description="谐波数")
    speed_delta: float = Field(default=10.0, gt=0, description="转速分辨率 (RPM)")
    std_min: float = Field(default=0.01, ge=0, description="信号最小标准差阈值")


# ============================================================================
# Group A: 简单统计特征 (11)
# ============================================================================

class MeanSquareFeature(IndependentMapFeature[BaseFeatureConfig]):
    """均方值特征: mean(x^2)。

    Input:
        x: 特征矩阵，形状 (n_samples, n_features)，由 Config 的 ``input_columns`` 选取

    Output:
        信号特征变换后的矩阵，列数与输入相同，行数与输入相同
    """

    @classmethod
    def name(cls) -> str:
        return "mean_square_feature"

    @classmethod
    def version(cls) -> tuple[int, ...]:
        """返回算子版本号

        Returns:
            tuple[int, ...]: 版本号三元组 ``(1, 0, 0)``
        """
        return (1, 0, 0)

    @staticmethod
    def compute(x, *, state=None, **params):
        return _apply_per_cell(x, _mean_square_1d)

    def _name_output_column(self, input_col, output_val):
        return self._make_output_column_name(input_col, "mean_square")


class VarianceFeature(IndependentMapFeature[BaseFeatureConfig]):
    """方差特征: var(x, ddof=1)。

    Input:
        x: 特征矩阵，形状 (n_samples, n_features)，由 Config 的 ``input_columns`` 选取

    Output:
        信号特征变换后的矩阵，列数与输入相同，行数与输入相同
    """

    @classmethod
    def name(cls) -> str:
        return "variance_feature"

    @classmethod
    def version(cls) -> tuple[int, ...]:
        """返回算子版本号

        Returns:
            tuple[int, ...]: 版本号三元组 ``(1, 0, 0)``
        """
        return (1, 0, 0)

    @staticmethod
    def compute(x, *, state=None, **params):
        return _apply_per_cell(x, _variance_1d)

    def _name_output_column(self, input_col, output_val):
        return self._make_output_column_name(input_col, "variance")


class RmsFeature(IndependentMapFeature[BaseFeatureConfig]):
    """有效值特征: sqrt(mean(x^2))。

    Input:
        x: 特征矩阵，形状 (n_samples, n_features)，由 Config 的 ``input_columns`` 选取

    Output:
        信号特征变换后的矩阵，列数与输入相同，行数与输入相同
    """

    @classmethod
    def name(cls) -> str:
        return "rms_feature"

    @classmethod
    def version(cls) -> tuple[int, ...]:
        """返回算子版本号

        Returns:
            tuple[int, ...]: 版本号三元组 ``(1, 0, 0)``
        """
        return (1, 0, 0)

    @staticmethod
    def compute(x, *, state=None, **params):
        return _apply_per_cell(x, _rms_1d)

    def _name_output_column(self, input_col, output_val):
        return self._make_output_column_name(input_col, "rms")


class PeakPeakFeature(IndependentMapFeature[BaseFeatureConfig]):
    """峰峰值特征: max(x) - min(x)。

    Input:
        x: 特征矩阵，形状 (n_samples, n_features)，由 Config 的 ``input_columns`` 选取

    Output:
        信号特征变换后的矩阵，列数与输入相同，行数与输入相同
    """

    @classmethod
    def name(cls) -> str:
        return "peak_peak_feature"

    @classmethod
    def version(cls) -> tuple[int, ...]:
        """返回算子版本号

        Returns:
            tuple[int, ...]: 版本号三元组 ``(1, 0, 0)``
        """
        return (1, 0, 0)

    @staticmethod
    def compute(x, *, state=None, **params):
        return _apply_per_cell(x, _peak_peak_1d)

    def _name_output_column(self, input_col, output_val):
        return self._make_output_column_name(input_col, "peak_peak")


class ShapeFactorFeature(IndependentMapFeature[BaseFeatureConfig]):
    """波形因子: RMS / mean(|x|)。

    Input:
        x: 特征矩阵，形状 (n_samples, n_features)，由 Config 的 ``input_columns`` 选取

    Output:
        信号特征变换后的矩阵，列数与输入相同，行数与输入相同
    """

    @classmethod
    def name(cls) -> str:
        return "shape_factor_feature"

    @classmethod
    def version(cls) -> tuple[int, ...]:
        """返回算子版本号

        Returns:
            tuple[int, ...]: 版本号三元组 ``(1, 0, 0)``
        """
        return (1, 0, 0)

    @staticmethod
    def compute(x, *, state=None, **params):
        return _apply_per_cell(x, _shape_factor_1d)

    def _name_output_column(self, input_col, output_val):
        return self._make_output_column_name(input_col, "shape_factor")


class CrestFeature(IndependentMapFeature[BaseFeatureConfig]):
    """峰值因子: max(|x|) / RMS。

    Input:
        x: 特征矩阵，形状 (n_samples, n_features)，由 Config 的 ``input_columns`` 选取

    Output:
        信号特征变换后的矩阵，列数与输入相同，行数与输入相同
    """

    @classmethod
    def name(cls) -> str:
        return "crest_feature"

    @classmethod
    def version(cls) -> tuple[int, ...]:
        """返回算子版本号

        Returns:
            tuple[int, ...]: 版本号三元组 ``(1, 0, 0)``
        """
        return (1, 0, 0)

    @staticmethod
    def compute(x, *, state=None, **params):
        return _apply_per_cell(x, _crest_1d)

    def _name_output_column(self, input_col, output_val):
        return self._make_output_column_name(input_col, "crest")


class ImpulseFeature(IndependentMapFeature[BaseFeatureConfig]):
    """脉冲因子: max(|x|) / mean(|x|)。

    Input:
        x: 特征矩阵，形状 (n_samples, n_features)，由 Config 的 ``input_columns`` 选取

    Output:
        信号特征变换后的矩阵，列数与输入相同，行数与输入相同
    """

    @classmethod
    def name(cls) -> str:
        return "impulse_feature"

    @classmethod
    def version(cls) -> tuple[int, ...]:
        """返回算子版本号

        Returns:
            tuple[int, ...]: 版本号三元组 ``(1, 0, 0)``
        """
        return (1, 0, 0)

    @staticmethod
    def compute(x, *, state=None, **params):
        return _apply_per_cell(x, _impulse_1d)

    def _name_output_column(self, input_col, output_val):
        return self._make_output_column_name(input_col, "impulse")


class ClearanceFeature(IndependentMapFeature[BaseFeatureConfig]):
    """裕度因子: max(|x|) / (mean(sqrt(|x|)))^2。

    Input:
        x: 特征矩阵，形状 (n_samples, n_features)，由 Config 的 ``input_columns`` 选取

    Output:
        信号特征变换后的矩阵，列数与输入相同，行数与输入相同
    """

    @classmethod
    def name(cls) -> str:
        return "clearance_feature"

    @classmethod
    def version(cls) -> tuple[int, ...]:
        """返回算子版本号

        Returns:
            tuple[int, ...]: 版本号三元组 ``(1, 0, 0)``
        """
        return (1, 0, 0)

    @staticmethod
    def compute(x, *, state=None, **params):
        return _apply_per_cell(x, _clearance_1d)

    def _name_output_column(self, input_col, output_val):
        return self._make_output_column_name(input_col, "clearance")


class SkewnessFeature(IndependentMapFeature[BaseFeatureConfig]):
    """偏斜度特征。

    Input:
        x: 特征矩阵，形状 (n_samples, n_features)，由 Config 的 ``input_columns`` 选取

    Output:
        信号特征变换后的矩阵，列数与输入相同，行数与输入相同
    """

    @classmethod
    def name(cls) -> str:
        return "skewness_feature"

    @classmethod
    def version(cls) -> tuple[int, ...]:
        """返回算子版本号

        Returns:
            tuple[int, ...]: 版本号三元组 ``(1, 0, 0)``
        """
        return (1, 0, 0)

    @staticmethod
    def compute(x, *, state=None, **params):
        return _apply_per_cell(x, _skewness_1d)

    def _name_output_column(self, input_col, output_val):
        return self._make_output_column_name(input_col, "skewness")


class KurtosisFeature(IndependentMapFeature[BaseFeatureConfig]):
    """超额峭度特征。

    Input:
        x: 特征矩阵，形状 (n_samples, n_features)，由 Config 的 ``input_columns`` 选取

    Output:
        信号特征变换后的矩阵，列数与输入相同，行数与输入相同
    """

    @classmethod
    def name(cls) -> str:
        return "kurtosis_feature"

    @classmethod
    def version(cls) -> tuple[int, ...]:
        """返回算子版本号

        Returns:
            tuple[int, ...]: 版本号三元组 ``(1, 0, 0)``
        """
        return (1, 0, 0)

    @staticmethod
    def compute(x, *, state=None, **params):
        return _apply_per_cell(x, _kurtosis_1d)

    def _name_output_column(self, input_col, output_val):
        return self._make_output_column_name(input_col, "kurtosis")


class GiniIndexFeature(IndependentMapFeature[BaseFeatureConfig]):
    """Gini 指数特征（假设非负值）。

    Input:
        x: 特征矩阵，形状 (n_samples, n_features)，由 Config 的 ``input_columns`` 选取

    Output:
        信号特征变换后的矩阵，列数与输入相同，行数与输入相同
    """

    @classmethod
    def name(cls) -> str:
        return "gini_index_feature"

    @classmethod
    def version(cls) -> tuple[int, ...]:
        """返回算子版本号

        Returns:
            tuple[int, ...]: 版本号三元组 ``(1, 0, 0)``
        """
        return (1, 0, 0)

    @staticmethod
    def compute(x, *, state=None, **params):
        return _apply_per_cell(x, _gini_index_1d)

    def _name_output_column(self, input_col, output_val):
        return self._make_output_column_name(input_col, "gini_index")


# ============================================================================
# Group B: 需要采样率的特征 (3)
# ============================================================================

class SpectralEntropyFeature(IndependentMapFeature[SampleRateFeatureConfig]):
    """功率谱熵特征（归一化 Shannon 熵）。

    Input:
        x: 特征矩阵，形状 (n_samples, n_features)，由 Config 的 ``input_columns`` 选取

    Output:
        信号特征变换后的矩阵，列数与输入相同，行数与输入相同
    """

    @classmethod
    def name(cls) -> str:
        return "spectral_entropy_feature"

    @classmethod
    def version(cls) -> tuple[int, ...]:
        """返回算子版本号

        Returns:
            tuple[int, ...]: 版本号三元组 ``(1, 0, 0)``
        """
        return (1, 0, 0)

    @staticmethod
    def compute(x, *, state=None, **params):
        return _apply_per_cell(x, _spectral_entropy_1d)

    def _name_output_column(self, input_col, output_val):
        return self._make_output_column_name(input_col, "spectral_entropy")


class RoughnessFeature(IndependentMapFeature[SampleRateFeatureConfig]):
    """粗糙度特征（心理声学，Zwicker & Fastl 模型）。

    Input:
        x: 特征矩阵，形状 (n_samples, n_features)，由 Config 的 ``input_columns`` 选取

    Output:
        信号特征变换后的矩阵，列数与输入相同，行数与输入相同
    """

    @classmethod
    def name(cls) -> str:
        return "roughness_feature"

    @classmethod
    def version(cls) -> tuple[int, ...]:
        """返回算子版本号

        Returns:
            tuple[int, ...]: 版本号三元组 ``(1, 0, 0)``
        """
        return (1, 0, 0)

    @staticmethod
    def compute(x, *, state=None, **params):
        sample_rate = params.get('sample_rate', 44100)
        return _apply_per_cell(x, lambda sig: _roughness_1d(sig, sample_rate))

    def _get_compute_params(self):
        return {'sample_rate': self.config.sample_rate}

    def _name_output_column(self, input_col, output_val):
        return self._make_output_column_name(input_col, "roughness")


class SharpnessFeature(IndependentMapFeature[SampleRateFeatureConfig]):
    """锐度特征（心理声学，Zwicker DIN 45692）。

    Input:
        x: 特征矩阵，形状 (n_samples, n_features)，由 Config 的 ``input_columns`` 选取

    Output:
        信号特征变换后的矩阵，列数与输入相同，行数与输入相同
    """

    @classmethod
    def name(cls) -> str:
        return "sharpness_feature"

    @classmethod
    def version(cls) -> tuple[int, ...]:
        """返回算子版本号

        Returns:
            tuple[int, ...]: 版本号三元组 ``(1, 0, 0)``
        """
        return (1, 0, 0)

    @staticmethod
    def compute(x, *, state=None, **params):
        sample_rate = params.get('sample_rate', 44100)
        return _apply_per_cell(x, lambda sig: _sharpness_1d(sig, sample_rate))

    def _get_compute_params(self):
        return {'sample_rate': self.config.sample_rate}

    def _name_output_column(self, input_col, output_val):
        return self._make_output_column_name(input_col, "sharpness")


# ============================================================================
# Group C: 频域特征 (5)
# ============================================================================

class SpectralCentroidFeature(IndependentMapFeature[SampleRateFeatureConfig]):
    """频谱重心特征。

    Input:
        x: 特征矩阵，形状 (n_samples, n_features)，由 Config 的 ``input_columns`` 选取

    Output:
        信号特征变换后的矩阵，列数与输入相同，行数与输入相同
    """

    @classmethod
    def name(cls) -> str:
        return "spectral_centroid_feature"

    @classmethod
    def version(cls) -> tuple[int, ...]:
        """返回算子版本号

        Returns:
            tuple[int, ...]: 版本号三元组 ``(1, 0, 0)``
        """
        return (1, 0, 0)

    @staticmethod
    def compute(x, *, state=None, **params):
        sample_rate = params.get('sample_rate', 44100)
        return _apply_per_cell(x, lambda sig: _frequency_features_1d(sig, sample_rate)[0])

    def _get_compute_params(self):
        return {'sample_rate': self.config.sample_rate}

    def _name_output_column(self, input_col, output_val):
        return self._make_output_column_name(input_col, "spectral_centroid")


class MeanSquareFrequencyFeature(IndependentMapFeature[SampleRateFeatureConfig]):
    """均方频率特征。

    Input:
        x: 特征矩阵，形状 (n_samples, n_features)，由 Config 的 ``input_columns`` 选取

    Output:
        信号特征变换后的矩阵，列数与输入相同，行数与输入相同
    """

    @classmethod
    def name(cls) -> str:
        return "mean_square_frequency_feature"

    @classmethod
    def version(cls) -> tuple[int, ...]:
        """返回算子版本号

        Returns:
            tuple[int, ...]: 版本号三元组 ``(1, 0, 0)``
        """
        return (1, 0, 0)

    @staticmethod
    def compute(x, *, state=None, **params):
        sample_rate = params.get('sample_rate', 44100)
        return _apply_per_cell(x, lambda sig: _frequency_features_1d(sig, sample_rate)[1])

    def _get_compute_params(self):
        return {'sample_rate': self.config.sample_rate}

    def _name_output_column(self, input_col, output_val):
        return self._make_output_column_name(input_col, "mean_square_frequency")


class RmsFrequencyFeature(IndependentMapFeature[SampleRateFeatureConfig]):
    """均方根频率特征。

    Input:
        x: 特征矩阵，形状 (n_samples, n_features)，由 Config 的 ``input_columns`` 选取

    Output:
        信号特征变换后的矩阵，列数与输入相同，行数与输入相同
    """

    @classmethod
    def name(cls) -> str:
        return "rms_frequency_feature"

    @classmethod
    def version(cls) -> tuple[int, ...]:
        """返回算子版本号

        Returns:
            tuple[int, ...]: 版本号三元组 ``(1, 0, 0)``
        """
        return (1, 0, 0)

    @staticmethod
    def compute(x, *, state=None, **params):
        sample_rate = params.get('sample_rate', 44100)
        return _apply_per_cell(x, lambda sig: _frequency_features_1d(sig, sample_rate)[2])

    def _get_compute_params(self):
        return {'sample_rate': self.config.sample_rate}

    def _name_output_column(self, input_col, output_val):
        return self._make_output_column_name(input_col, "rms_frequency")


class FrequencyVarianceFeature(IndependentMapFeature[SampleRateFeatureConfig]):
    """频率方差特征。

    Input:
        x: 特征矩阵，形状 (n_samples, n_features)，由 Config 的 ``input_columns`` 选取

    Output:
        信号特征变换后的矩阵，列数与输入相同，行数与输入相同
    """

    @classmethod
    def name(cls) -> str:
        return "frequency_variance_feature"

    @classmethod
    def version(cls) -> tuple[int, ...]:
        """返回算子版本号

        Returns:
            tuple[int, ...]: 版本号三元组 ``(1, 0, 0)``
        """
        return (1, 0, 0)

    @staticmethod
    def compute(x, *, state=None, **params):
        sample_rate = params.get('sample_rate', 44100)
        return _apply_per_cell(x, lambda sig: _frequency_features_1d(sig, sample_rate)[3])

    def _get_compute_params(self):
        return {'sample_rate': self.config.sample_rate}

    def _name_output_column(self, input_col, output_val):
        return self._make_output_column_name(input_col, "frequency_variance")


class FrequencyStdFeature(IndependentMapFeature[SampleRateFeatureConfig]):
    """频率标准差特征。

    Input:
        x: 特征矩阵，形状 (n_samples, n_features)，由 Config 的 ``input_columns`` 选取

    Output:
        信号特征变换后的矩阵，列数与输入相同，行数与输入相同
    """

    @classmethod
    def name(cls) -> str:
        return "frequency_std_feature"

    @classmethod
    def version(cls) -> tuple[int, ...]:
        """返回算子版本号

        Returns:
            tuple[int, ...]: 版本号三元组 ``(1, 0, 0)``
        """
        return (1, 0, 0)

    @staticmethod
    def compute(x, *, state=None, **params):
        sample_rate = params.get('sample_rate', 44100)
        return _apply_per_cell(x, lambda sig: _frequency_features_1d(sig, sample_rate)[4])

    def _get_compute_params(self):
        return {'sample_rate': self.config.sample_rate}

    def _name_output_column(self, input_col, output_val):
        return self._make_output_column_name(input_col, "frequency_std")


# ============================================================================
# Group D: 复合特征 (3)
# ============================================================================

class EnvelopeRmsFeature(IndependentMapFeature[BaseFeatureConfig]):
    """包络 RMS 特征: Hilbert 包络 → RMS。

    Input:
        x: 特征矩阵，形状 (n_samples, n_features)，由 Config 的 ``input_columns`` 选取

    Output:
        信号特征变换后的矩阵，列数与输入相同，行数与输入相同
    """

    @classmethod
    def name(cls) -> str:
        return "envelope_rms_feature"

    @classmethod
    def version(cls) -> tuple[int, ...]:
        """返回算子版本号

        Returns:
            tuple[int, ...]: 版本号三元组 ``(1, 0, 0)``
        """
        return (1, 0, 0)

    @staticmethod
    def compute(x, *, state=None, **params):
        return _apply_per_cell(x, _envelope_rms_1d)

    def _name_output_column(self, input_col, output_val):
        return self._make_output_column_name(input_col, "envelope_rms")


class AverageKurtosisFeature(IndependentMapFeature[AverageKurtosisConfig]):
    """分段峭度均值特征。

    Input:
        x: 特征矩阵，形状 (n_samples, n_features)，由 Config 的 ``input_columns`` 选取

    Output:
        信号特征变换后的矩阵，列数与输入相同，行数与输入相同
    """

    @classmethod
    def name(cls) -> str:
        return "average_kurtosis_feature"

    @classmethod
    def version(cls) -> tuple[int, ...]:
        """返回算子版本号

        Returns:
            tuple[int, ...]: 版本号三元组 ``(1, 0, 0)``
        """
        return (1, 0, 0)

    @staticmethod
    def compute(x, *, state=None, **params):
        n_segments = params.get('n_segments', 10)
        return _apply_per_cell(x, lambda sig: _average_kurtosis_1d(sig, n_segments))

    def _get_compute_params(self):
        return {'n_segments': self.config.n_segments}

    def _name_output_column(self, input_col, output_val):
        return self._make_output_column_name(input_col, "average_kurtosis")


class HnrFeature(IndependentMapFeature[BaseFeatureConfig]):
    """谐噪比特征（基于自相关峰值近似，dB）。

    Input:
        x: 特征矩阵，形状 (n_samples, n_features)，由 Config 的 ``input_columns`` 选取

    Output:
        信号特征变换后的矩阵，列数与输入相同，行数与输入相同
    """

    @classmethod
    def name(cls) -> str:
        return "hnr_feature"

    @classmethod
    def version(cls) -> tuple[int, ...]:
        """返回算子版本号

        Returns:
            tuple[int, ...]: 版本号三元组 ``(1, 0, 0)``
        """
        return (1, 0, 0)

    @staticmethod
    def compute(x, *, state=None, **params):
        return _apply_per_cell(x, _hnr_1d)

    def _name_output_column(self, input_col, output_val):
        return self._make_output_column_name(input_col, "hnr")


# ============================================================================
# Group E: 频带特征 (3 类，可参数化为 9 个实例)
# ============================================================================

class BandKurtosisFeature(IndependentMapFeature[BandFeatureConfig]):
    """频带峭度特征: 带通滤波 → 峭度。

    Input:
        x: 特征矩阵，形状 (n_samples, n_features)，由 Config 的 ``input_columns`` 选取

    Output:
        信号特征变换后的矩阵，列数与输入相同，行数与输入相同
    """

    @classmethod
    def name(cls) -> str:
        return "band_kurtosis_feature"

    @classmethod
    def version(cls) -> tuple[int, ...]:
        """返回算子版本号

        Returns:
            tuple[int, ...]: 版本号三元组 ``(1, 0, 0)``
        """
        return (1, 0, 0)

    @staticmethod
    def compute(x, *, state=None, **params):
        sample_rate = params.get('sample_rate', 44100)
        low = params.get('low', 0.0)
        high = params.get('high', None)

        def _band_kurt(sig):
            band_sig = _bandpass_fft_1d(sig, sample_rate, low, high)
            return _kurtosis_1d(band_sig)

        return _apply_per_cell(x, _band_kurt)

    def _get_compute_params(self):
        return {
            'sample_rate': self.config.sample_rate,
            'low': self.config.low,
            'high': self.config.high,
        }

    def _name_output_column(self, input_col, output_val):
        high_str = str(int(self.config.high)) if self.config.high is not None else "nyquist"
        low_str = str(int(self.config.low))
        return self._make_output_column_name(input_col, f"band_kurtosis_{low_str}_{high_str}")


class BandRmsFeature(IndependentMapFeature[BandFeatureConfig]):
    """频带 RMS 特征: 带通滤波 → RMS。

    Input:
        x: 特征矩阵，形状 (n_samples, n_features)，由 Config 的 ``input_columns`` 选取

    Output:
        信号特征变换后的矩阵，列数与输入相同，行数与输入相同
    """

    @classmethod
    def name(cls) -> str:
        return "band_rms_feature"

    @classmethod
    def version(cls) -> tuple[int, ...]:
        """返回算子版本号

        Returns:
            tuple[int, ...]: 版本号三元组 ``(1, 0, 0)``
        """
        return (1, 0, 0)

    @staticmethod
    def compute(x, *, state=None, **params):
        sample_rate = params.get('sample_rate', 44100)
        low = params.get('low', 0.0)
        high = params.get('high', None)

        def _band_rms(sig):
            band_sig = _bandpass_fft_1d(sig, sample_rate, low, high)
            return _rms_1d(band_sig)

        return _apply_per_cell(x, _band_rms)

    def _get_compute_params(self):
        return {
            'sample_rate': self.config.sample_rate,
            'low': self.config.low,
            'high': self.config.high,
        }

    def _name_output_column(self, input_col, output_val):
        high_str = str(int(self.config.high)) if self.config.high is not None else "nyquist"
        low_str = str(int(self.config.low))
        return self._make_output_column_name(input_col, f"band_rms_{low_str}_{high_str}")


class BandHnrFeature(IndependentMapFeature[BandFeatureConfig]):
    """频带 HNR 特征: 带通滤波 → HNR。

    Input:
        x: 特征矩阵，形状 (n_samples, n_features)，由 Config 的 ``input_columns`` 选取

    Output:
        信号特征变换后的矩阵，列数与输入相同，行数与输入相同
    """

    @classmethod
    def name(cls) -> str:
        return "band_hnr_feature"

    @classmethod
    def version(cls) -> tuple[int, ...]:
        """返回算子版本号

        Returns:
            tuple[int, ...]: 版本号三元组 ``(1, 0, 0)``
        """
        return (1, 0, 0)

    @staticmethod
    def compute(x, *, state=None, **params):
        sample_rate = params.get('sample_rate', 44100)
        low = params.get('low', 0.0)
        high = params.get('high', None)

        def _band_hnr(sig):
            band_sig = _bandpass_fft_1d(sig, sample_rate, low, high)
            return _hnr_1d(band_sig)

        return _apply_per_cell(x, _band_hnr)

    def _get_compute_params(self):
        return {
            'sample_rate': self.config.sample_rate,
            'low': self.config.low,
            'high': self.config.high,
        }

    def _name_output_column(self, input_col, output_val):
        high_str = str(int(self.config.high)) if self.config.high is not None else "nyquist"
        low_str = str(int(self.config.low))
        return self._make_output_column_name(input_col, f"band_hnr_{low_str}_{high_str}")


# ============================================================================
# Group F: 转速特征 (1)
# ============================================================================

class SpeedRpmFeature(IndependentMapFeature[SpeedRpmConfig]):
    """转速估计特征（HPS+ 方法）。

    基于谐波乘积谱 + 谐波验证选峰，适合旋转设备振动/声音信号的转速估计。

    Input:
        x: 特征矩阵，形状 (n_samples, n_features)，由 Config 的 ``input_columns`` 选取

    Output:
        信号特征变换后的矩阵，列数与输入相同，行数与输入相同
    """

    @classmethod
    def name(cls) -> str:
        return "speed_rpm_feature"

    @classmethod
    def version(cls) -> tuple[int, ...]:
        """返回算子版本号

        Returns:
            tuple[int, ...]: 版本号三元组 ``(1, 0, 0)``
        """
        return (1, 0, 0)

    @staticmethod
    def compute(x, *, state=None, **params):
        sample_rate = params.get('sample_rate', 44100)
        speed_min = params.get('speed_min', 600)
        speed_max = params.get('speed_max', 3600)
        n_harmonics = params.get('n_harmonics', 5)
        speed_delta = params.get('speed_delta', 10.0)
        std_min = params.get('std_min', 0.01)

        def _speed(sig):
            return _speed_rpm_1d(sig, sample_rate, speed_min, speed_max, n_harmonics, speed_delta, std_min)

        return _apply_per_cell(x, _speed)

    def _get_compute_params(self):
        return {
            'sample_rate': self.config.sample_rate,
            'speed_min': self.config.speed_min,
            'speed_max': self.config.speed_max,
            'n_harmonics': self.config.n_harmonics,
            'speed_delta': self.config.speed_delta,
            'std_min': self.config.std_min,
        }

    def _name_output_column(self, input_col, output_val):
        return self._make_output_column_name(input_col, "speed_rpm")
