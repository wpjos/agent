# -*- coding: utf-8 -*-

"""
配置文件加载模块

提供统一的配置文件加载接口，根据文件后缀自动选择解析器。
支持 JSON、YAML、JSON5 三种格式。

支持格式:
    - ``.json``: 标准 JSON
    - ``.yaml`` / ``.yml``: YAML 格式
    - ``.json5``: JSON5 格式（支持注释、尾逗号等）

使用示例::

    from tsas.engine.operator.cli.config_loader import load_config

    config = load_config("pipeline.yaml")
    config = load_config("pipeline.json5")
"""

import json
from pathlib import Path

__all__ = [
    'load_config',
]

# 支持的配置文件后缀
_JSON_EXTENSIONS = {'.json'}
_YAML_EXTENSIONS = {'.yaml', '.yml'}
_JSON5_EXTENSIONS = {'.json5'}
_ALL_EXTENSIONS = _JSON_EXTENSIONS | _YAML_EXTENSIONS | _JSON5_EXTENSIONS


def load_config(path: str | Path) -> dict:
    """
    根据文件后缀自动加载配置文件为字典

    后缀名大小写不敏感。

    Args:
        path (str | Path): 配置文件路径

    Returns:
        dict: 解析后的配置字典

    Raises:
        FileNotFoundError: 文件不存在时
        ValueError: 文件后缀不支持时
        ImportError: 缺少 pyyaml 或 json5 依赖时
    """
    path = Path(path)

    if not path.exists():
        raise FileNotFoundError(f"配置文件不存在: {path}")

    suffix = path.suffix.lower()

    if suffix in _JSON_EXTENSIONS:
        return _load_json(path)
    elif suffix in _YAML_EXTENSIONS:
        return _load_yaml(path)
    elif suffix in _JSON5_EXTENSIONS:
        return _load_json5(path)
    else:
        raise ValueError(
            f"不支持的配置文件格式 '{suffix}'。"
            f"支持的格式: {sorted(_ALL_EXTENSIONS)}"
        )


def _load_json(path: Path) -> dict:
    """
    加载 JSON 配置文件

    Args:
        path (Path): JSON 文件路径

    Returns:
        dict: 解析后的字典
    """
    with open(path, 'r', encoding='utf-8') as f:
        return json.load(f)


def _load_yaml(path: Path) -> dict:
    """
    加载 YAML 配置文件

    Args:
        path (Path): YAML 文件路径

    Returns:
        dict: 解析后的字典

    Raises:
        ImportError: 未安装 pyyaml 时
    """
    try:
        import yaml
    except ImportError as e:
        raise ImportError(
            "加载 YAML 配置需要 pyyaml 库，请执行: pip install pyyaml"
        ) from e

    with open(path, 'r', encoding='utf-8') as f:
        return yaml.safe_load(f)


def _load_json5(path: Path) -> dict:
    """
    加载 JSON5 配置文件

    Args:
        path (Path): JSON5 文件路径

    Returns:
        dict: 解析后的字典

    Raises:
        ImportError: 未安装 json5 时
    """
    try:
        import json5
    except ImportError as e:
        raise ImportError(
            "加载 JSON5 配置需要 json5 库，请执行: pip install json5"
        ) from e

    with open(path, 'r', encoding='utf-8') as f:
        return json5.load(f)
