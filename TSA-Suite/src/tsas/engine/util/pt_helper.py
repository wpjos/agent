# -*- coding: utf-8 -*-

"""
PyTorch 辅助工具模块

提供算子层常用的 PyTorch 设备检测与 NPU 导入辅助函数。
"""

import torch

__all__ = [
    'get_torch_device',
    '_try_import_npu',
]


def _try_import_npu() -> bool:
    """尝试导入 torch_npu 并返回是否成功。

    Returns:
        bool: torch_npu 是否可用
    """
    try:
        import torch_npu  # noqa: F401
        return True
    except ImportError:
        return False


def get_torch_device(device: str | None = None) -> torch.device:
    """根据用户指定或自动推断获取 torch 设备。

    Args:
        device (str | None): 设备标识，如 ``"cpu"``、``"cuda:0"``、``"npu:0"``。
            为 ``None`` 或 ``"auto"`` 时自动选择可用设备（cuda > npu > cpu）。

    Returns:
        torch.device: 解析后的设备对象
    """
    if device is None or device == 'auto':
        if torch.cuda.is_available():
            return torch.device('cuda')
        if _try_import_npu():
            return torch.device('npu')
        return torch.device('cpu')

    if device == 'npu':
        if not _try_import_npu():
            return torch.device('cpu')
        return torch.device('npu')

    return torch.device(device)
