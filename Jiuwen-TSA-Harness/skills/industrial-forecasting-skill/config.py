#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
路径配置模块。

自动推断 TSA-Suite 项目根目录，并提供流程中使用的各目录路径。
"""

import os
import sys
from pathlib import Path


def get_tsa_suite_dir() -> Path:
    """获取 TSA-Suite 项目根目录路径。

    优先级：
    1. 环境变量 TSA_SUITE_DIR
    2. 向上遍历文件系统，找到包含 ``src/tsas`` 的目录
    """
    tsa_dir = os.environ.get("TSA_SUITE_DIR")
    if tsa_dir:
        return Path(tsa_dir).resolve()

    # 从本脚本所在位置向上查找项目根目录（以 src/tsas 为标记）
    current = Path(__file__).resolve().parent
    for parent in [current, *current.parents]:
        if (parent / "src" / "tsas").is_dir():
            return parent

    # 兜底：保持旧行为，向上两级
    return Path(__file__).resolve().parents[2]


def setup_paths():
    """设置所有必要的路径变量，并确保 src 在 sys.path 中。"""
    repo_root = get_tsa_suite_dir()
    src_dir = repo_root / "src"

    if str(src_dir) not in sys.path:
        sys.path.insert(0, str(src_dir))

    return {
        "REPO_ROOT": repo_root,
        "SRC_DIR": src_dir,
        "DATA_DIR": repo_root / "data" / "synthetic",
        "RESULTS_DIR": repo_root / "results",
        "MODEL_DIR_FORECAST": repo_root / "results" / "models" / "forecast_demo",
        "FORECAST_DIR": repo_root / "results" / "forecasts",
        "METRICS_DIR": repo_root / "results" / "metrics",
    }


if __name__ == "__main__":
    paths = setup_paths()
    print("TSA-Suite 项目路径配置:")
    for name, path in paths.items():
        exists = "✓ 存在" if path.exists() else "✗ 不存在"
        print(f"  {name:20s}: {path} ({exists})")
