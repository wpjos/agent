#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
低频特征增强算子一键调用 Demo

展示如何通过 TSA-Suite CLI 和 Python API 两种方式
调用套餐八的 9 个 smooth 特征。

用法:
    # 1. CLI 方式（需要先 pip install tsa_suite）
    python scripts/smooth_feature_demo.py --mode cli

    # 2. Python API 方式
    python scripts/smooth_feature_demo.py --mode api

    # 3. 两种都跑
    python scripts/smooth_feature_demo.py
"""

import argparse
import json
import sys
import tempfile
from pathlib import Path

import numpy as np
import pandas as pd


def generate_demo_data(n_samples: int = 200, seed: int = 42) -> pd.DataFrame:
    """生成模拟低频振动信号。

    包含: 趋势漂移 + 慢变正弦 + 噪声 + 一个突变点
    """
    np.random.seed(seed)
    t = np.linspace(0, 10, n_samples)

    # 基础趋势
    trend = 0.3 * t
    # 低频正弦
    low_freq = 2.0 * np.sin(2 * np.pi * 0.3 * t)
    # 噪声
    noise = 0.5 * np.random.randn(n_samples)
    # 突变点
    glitch = np.zeros(n_samples)
    glitch[n_samples // 2:n_samples // 2 + 10] = 5.0

    signal = trend + low_freq + noise + glitch

    # DataFrame 每行一个信号段（模拟预测性维护场景的数据格式）
    # 这里把整个信号作为一个段
    return pd.DataFrame({"vibration": [signal.tolist()]})


def run_api_mode():
    """Python API 方式调用 9 个 smooth 特征。"""
    from tsas.engine.operator.feature.construction.smooth_feature import (
        SmoothFeatureConfig,
        SmoothMeanFeature,
        SmoothMinFeature,
        SmoothMaxFeature,
        SmoothStdFeature,
        SmoothSlopeFeature,
        SmoothGradMeanFeature,
        SmoothGradStdFeature,
        SmoothGrad2MeanFeature,
        SmoothFftEnergyFeature,
    )

    print("=" * 60)
    print("Python API 方式: 9 个 smooth 特征")
    print("=" * 60)

    df = generate_demo_data()
    raw_signal = np.asarray(df["vibration"].iloc[0])
    print(f"\n输入信号长度: {len(raw_signal)}")

    features = [
        ("smooth_mean", SmoothMeanFeature),
        ("smooth_min", SmoothMinFeature),
        ("smooth_max", SmoothMaxFeature),
        ("smooth_std", SmoothStdFeature),
        ("smooth_slope", SmoothSlopeFeature),
        ("smooth_grad_mean", SmoothGradMeanFeature),
        ("smooth_grad_std", SmoothGradStdFeature),
        ("smooth_grad2_mean", SmoothGrad2MeanFeature),
        ("smooth_fft_energy", SmoothFftEnergyFeature),
    ]

    config = SmoothFeatureConfig(input_columns=["vibration"], win=30)
    print(f"Config: win={config.win}, stride={config.stride}, padding_mode={config.padding_mode}")
    print()

    results = {}
    for name, cls in features:
        feat = cls(config=config)
        result_df = feat.run(df)
        col_name = [c for c in result_df.columns if "smooth" in c][0]
        out = np.asarray(result_df[col_name].iloc[0])
        scalar_val = float(np.mean(out))
        results[name] = {
            "sequence_length": len(out),
            "scalar_mean": round(scalar_val, 6),
            "sequence_preview": [round(v, 4) for v in out[:5].tolist()],
        }
        print(f"  {name:25s}  len={len(out)}  mean={scalar_val:+.6f}  preview={out[:3].tolist()}")

    print(f"\n共提取 {len(results)} 个特征序列")

    # 保存结果
    out_path = Path(tempfile.mktemp(suffix=".json"))
    with open(out_path, "w") as f:
        json.dump(results, f, indent=2, ensure_ascii=False)
    print(f"结果已保存: {out_path}")

    return results


def run_cli_mode():
    """展示 CLI 调用方式。"""
    import subprocess

    print("\n" + "=" * 60)
    print("CLI 方式: smooth 特征调用示例")
    print("=" * 60)

    # CLI config 示例
    config_example = {
        "operators": [
            {"name": "smooth_mean_feature", "config": {"input_columns": ["vibration"], "win": 30}},
            {"name": "smooth_std_feature", "config": {"input_columns": ["vibration"], "win": 30}},
            {"name": "smooth_slope_feature", "config": {"input_columns": ["vibration"], "win": 30}},
            {"name": "smooth_fft_energy_feature", "config": {"input_columns": ["vibration"], "win": 30}},
        ]
    }

    print("\n1. 查看所有可用算子:")
    print("   python -m tsas.engine.operator.cli feature_construction help")

    print("\n2. 查看单个算子详情:")
    print("   python -m tsas.engine.operator.cli feature_construction help smooth_mean_feature")

    print("\n3. 运行算子 (需要 input_columns 中每格为 list/ndarray 格式):")
    print("   python -m tsas.engine.operator.cli feature_construction run \\")
    print("       --input data.csv \\")
    print("       --output result.csv \\")
    print("       --config config.json")

    print("\n4. config.json 示例:")
    print(json.dumps(config_example, indent=4, ensure_ascii=False))

    # 实际执行 help 命令验证 CLI 可用
    print("\n5. 验证 CLI 可用性:")
    result = subprocess.run(
        [sys.executable, "-m", "tsas.engine.operator.cli", "feature_construction", "help"],
        capture_output=True, text=True,
    )
    if result.returncode == 0:
        # 提取 smooth 相关行
        lines = result.stdout.strip().split("\n")
        smooth_lines = [l for l in lines if "smooth" in l]
        print(f"   已注册 {len(smooth_lines)} 个 smooth 算子:")
        for l in smooth_lines:
            print(f"   {l.strip()}")
    else:
        print(f"   CLI 执行失败: {result.stderr}")


def main():
    parser = argparse.ArgumentParser(description="TSA-Suite 低频特征增强 Demo")
    parser.add_argument(
        "--mode", choices=["api", "cli", "both"], default="both",
        help="运行模式: api / cli / both (默认 both)",
    )
    args = parser.parse_args()

    if args.mode in ("api", "both"):
        run_api_mode()

    if args.mode in ("cli", "both"):
        run_cli_mode()

    print("\n" + "=" * 60)
    print("Demo 完成")
    print("=" * 60)


if __name__ == "__main__":
    main()
