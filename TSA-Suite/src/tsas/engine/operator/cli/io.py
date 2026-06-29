# -*- coding: utf-8 -*-

"""
数据输入输出模块

提供统一的数据加载和保存接口，根据文件后缀自动选择格式。
当前支持 CSV 格式，预留 TSV、MAT、HDF5 等格式的扩展能力。

支持格式:
    - ``.csv``: CSV 逗号分隔值
    - ``.tsv``: TSV 制表符分隔值（预留）
    - ``.mat``: MATLAB MAT 文件（预留）
    - ``.h5`` / ``.hdf5``: HDF5 文件（预留）

使用示例::

    from tsas.engine.operator.cli.io import load_data, save_data, save_json

    df = load_data("input.csv")
    save_data(df, "output.csv")
    save_json({"f1": 0.85}, "result.json")
"""

import json
import os
import sys
from pathlib import Path

import pandas as pd

__all__ = [
    'load_data',
    'save_data',
    'save_json',
    'ensure_encoding',
]

# 支持的数据文件后缀 → 加载/保存函数的映射键
_SUPPORTED_EXTENSIONS = {'.csv', '.tsv'}
# 预留但尚未实现的格式
_RESERVED_EXTENSIONS = {'.mat', '.h5', '.hdf5'}


def load_data(path: str | Path) -> pd.DataFrame:
    """
    根据文件后缀自动加载数据为 DataFrame

    当前支持 CSV 格式。后缀名自动判断，大小写不敏感。

    Args:
        path (str | Path): 输入文件路径

    Returns:
        pd.DataFrame: 加载的数据，第一行作为表头

    Raises:
        FileNotFoundError: 文件不存在时
        ValueError: 文件后缀不支持或尚未实现时
    """
    path = Path(path)

    if not path.exists():
        raise FileNotFoundError(f"输入文件不存在: {path}")

    suffix = path.suffix.lower()

    if suffix == '.csv':
        return pd.read_csv(path)
    elif suffix == '.tsv':
        return pd.read_csv(path, sep='\t')
    elif suffix in _RESERVED_EXTENSIONS:
        raise ValueError(
            f"文件格式 '{suffix}' 尚未实现，预留扩展。"
            f"当前支持的格式: {sorted(_SUPPORTED_EXTENSIONS)}"
        )
    else:
        raise ValueError(
            f"不支持的文件格式 '{suffix}'。"
            f"当前支持的格式: {sorted(_SUPPORTED_EXTENSIONS)}"
        )


def save_data(df: pd.DataFrame, path: str | Path) -> None:
    """
    根据文件后缀自动保存 DataFrame 到文件

    当前支持 CSV 格式。自动创建目标目录（如不存在）。

    Args:
        df (pd.DataFrame): 要保存的数据
        path (str | Path): 输出文件路径

    Raises:
        ValueError: 文件后缀不支持或尚未实现时
    """
    path = Path(path)

    # 自动创建目标目录
    path.parent.mkdir(parents=True, exist_ok=True)

    suffix = path.suffix.lower()

    if suffix == '.csv':
        df.to_csv(path, index=False)
    elif suffix == '.tsv':
        df.to_csv(path, sep='\t', index=False)
    elif suffix in _RESERVED_EXTENSIONS:
        raise ValueError(
            f"文件格式 '{suffix}' 尚未实现，预留扩展。"
            f"当前支持的格式: {sorted(_SUPPORTED_EXTENSIONS)}"
        )
    else:
        raise ValueError(
            f"不支持的文件格式 '{suffix}'。"
            f"当前支持的格式: {sorted(_SUPPORTED_EXTENSIONS)}"
        )


def save_json(data: dict, path: str | Path) -> None:
    """
    保存字典数据为 JSON 文件

    使用 UTF-8 编码，2 空格缩进，保留非 ASCII 字符。
    自动创建目标目录（如不存在）。

    Args:
        data (dict): 要保存的字典数据
        path (str | Path): 输出文件路径
    """
    path = Path(path)

    # 自动创建目标目录
    path.parent.mkdir(parents=True, exist_ok=True)

    with open(path, 'w', encoding='utf-8') as f:
        json.dump(data, f, indent=2, ensure_ascii=False, default=str)


def ensure_encoding(encoding: str | None = None) -> None:
    """
    确保终端输出使用正确的编码

    当用户通过 ``--encoding`` 参数指定编码时，使用用户指定的编码。
    当未指定时，自动探测终端编码：若当前已为 UTF-8 则不做处理，
    否则尝试将 stdout/stderr 重配置为 UTF-8，并在 Windows 上设置控制台代码页。

    Args:
        encoding (str | None): 用户指定的编码名称，如 ``'utf-8'``、``'gbk'``。
            传入 ``None`` 时启用自动探测模式，默认目标编码为 UTF-8。
    """
    target = encoding

    if target is None:
        # 自动探测模式：检查当前 stdout 编码是否已兼容 UTF-8
        enc = getattr(sys.stdout, 'encoding', '') or ''
        if 'utf' in enc.lower():
            return  # 已是 UTF-8，无需处理

        # 非 UTF-8 环境，目标设为 UTF-8
        target = 'utf-8'

        # Windows: 设置控制台代码页为 UTF-8 (65001)
        if sys.platform == 'win32':
            os.system('chcp 65001 >nul 2>&1')

    # 重配置 stdout/stderr 的编码
    for stream in [sys.stdout, sys.stderr]:
        if hasattr(stream, 'reconfigure'):
            try:
                stream.reconfigure(encoding=target, errors='replace')
            except (LookupError, UnicodeError, AttributeError):
                pass
