#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
生成 TSA-Suite 时序预测端到端流程的工业示例数据。

输出文件（位于 data/synthetic/）：
- forecast_demo.csv : 单一输入文件，包含 30 个工业特征列、目标列与 split 列。
  按时间顺序划分为训练集/验证集/测试集，比例为 70:15:15。
  * split=train : 用于模型训练
  * split=val   : 验证集（可在流程中用于调参/早停，当前由模型内部划分）
  * split=test  : 用于最终推理与评估

数据覆盖低压汽包水位、凝结水/给水流量、除氧器、汽机/气机负荷、压力、
温度等典型工业测点。
"""

import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd

# ---------------------------------------------------------------------------
# 0. 环境准备
# ---------------------------------------------------------------------------
SCRIPT_DIR = Path(__file__).resolve().parent
# 使用 config 模块获取 TSA-Suite 项目路径
from config import setup_paths

PATHS = setup_paths()
REPO_ROOT = PATHS["REPO_ROOT"]
DATA_DIR = PATHS["DATA_DIR"]
DATA_DIR.mkdir(parents=True, exist_ok=True)

np.random.seed(42)

# ---------------------------------------------------------------------------
# 1. 特征名定义
# ---------------------------------------------------------------------------
DATETIME_COL = "datatime"
TARGET_COL = "target_shuiwei_chazhi"

FEATURE_COLS = [
    "diya_qibao_shuiwei",
    "diya_qibao_shuiwei_youxuanzhi",
    "diya_qibao_shuiwei_celiang_waihuan",
    "diya_qibao_shuiwei_mubiao_pid",
    "ningjie_shuiliuliang_1",
    "ningjie_shuiliuliang_2",
    "ningjie_shuiliuliang_youxuanzhi",
    "ningjie_shuiliuliang_celiang_neihuan",
    "ningjie_shuiliuliang_mubiao_pid",
    "gaoya_geishuiliuliang_1",
    "gaoya_geishuiliuliang_2",
    "gaoya_geishuiliuliang_3",
    "gaoya_geishuiliuliang_youxuanzhi",
    "chuyangqi_tiaojiefa_weizhifankui",
    "chuyangqi_tiaojiefa_pid_sheding",
    "ranji_fuhe",
    "qiji_fuhe",
    "chuyangqi_rukou_yali",
    "diya_qibao_yali_1",
    "diya_qibao_yali_2",
    "diya_qibao_yali_3",
    "diya_qibao_yali_youxuanzhi",
    "diya_zhuqiliuliang_youxuanzhi",
    "gaoya_jianwenshui_liuliang",
    "ranji_paiqi_wendu",
    "mubiao_diya_qibao_shuiwei",
    "pidzidong_shoudong",
    "gongkuang",
]

ALL_VALUE_COLS = FEATURE_COLS + [TARGET_COL]


# ---------------------------------------------------------------------------
# 2. 生成预测数据
# ---------------------------------------------------------------------------
def _add_noise(size: int, scale: float = 1.0) -> np.ndarray:
    """生成高斯噪声。"""
    return np.random.randn(size) * scale


def _generate_gongkuang(n_samples: int) -> np.ndarray:
    """生成缓慢切换的工况标签（1/2/3）。"""
    gongkuang = np.ones(n_samples, dtype=int)
    change_points = np.arange(120, n_samples, np.random.randint(100, 200))
    label = 1
    prev = 0
    for cp in change_points:
        gongkuang[prev:cp] = label
        label = (label % 3) + 1
        prev = cp
    gongkuang[prev:] = label
    return gongkuang


def generate_forecast_data(
    n_samples: int = 1000,
    train_ratio: float = 0.70,
    val_ratio: float = 0.15,
    start_time: str = "2024-01-01 00:00:00",
    freq: str = "1s",
) -> pd.DataFrame:
    """生成工业时序预测示例数据，并按时间顺序划分 train/val/test。

    目标列为 ``target_shuiwei_chazhi``（目标水位差值）。返回的 DataFrame
    包含 ``split`` 列，取值依次为 ``train``、``val``、``test``。
    """
    t = np.arange(n_samples)
    data: dict[str, np.ndarray] = {}

    # 工况与 PID 手自动
    data["gongkuang"] = _generate_gongkuang(n_samples)
    data["pidzidong_shoudong"] = (np.random.rand(n_samples) > 0.05).astype(int)

    # 燃机/汽机负荷：随时间缓慢漂移，并受工况影响
    load_base = 250.0 + 80.0 * np.sin(2 * np.pi * t / 600) + 0.05 * t
    load_base += np.where(data["gongkuang"] == 2, 30.0, 0.0)
    load_base += np.where(data["gongkuang"] == 3, -20.0, 0.0)
    data["ranji_fuhe"] = load_base + _add_noise(n_samples, 5.0)
    data["qiji_fuhe"] = 0.92 * data["ranji_fuhe"] + _add_noise(n_samples, 4.0)

    # 给水系统
    data["gaoya_geishuiliuliang_youxuanzhi"] = (
        1200.0 + 2.5 * data["ranji_fuhe"] + _add_noise(n_samples, 15.0)
    )
    data["gaoya_geishuiliuliang_1"] = (
        data["gaoya_geishuiliuliang_youxuanzhi"] + _add_noise(n_samples, 8.0)
    )
    data["gaoya_geishuiliuliang_2"] = (
        data["gaoya_geishuiliuliang_youxuanzhi"] + _add_noise(n_samples, 8.0)
    )
    data["gaoya_geishuiliuliang_3"] = (
        data["gaoya_geishuiliuliang_youxuanzhi"] + _add_noise(n_samples, 8.0)
    )

    # 凝结水系统
    data["ningjie_shuiliuliang_youxuanzhi"] = (
        800.0 + 1.8 * data["ranji_fuhe"] + _add_noise(n_samples, 12.0)
    )
    data["ningjie_shuiliuliang_1"] = (
        data["ningjie_shuiliuliang_youxuanzhi"] + _add_noise(n_samples, 6.0)
    )
    data["ningjie_shuiliuliang_2"] = (
        data["ningjie_shuiliuliang_youxuanzhi"] + _add_noise(n_samples, 6.0)
    )
    data["ningjie_shuiliuliang_celiang_neihuan"] = (
        data["ningjie_shuiliuliang_youxuanzhi"] + _add_noise(n_samples, 5.0)
    )
    data["ningjie_shuiliuliang_mubiao_pid"] = (
        data["ningjie_shuiliuliang_youxuanzhi"] + _add_noise(n_samples, 2.0)
    )

    # 除氧器相关
    data["chuyangqi_rukou_yali"] = (
        1.2 + 0.001 * data["ranji_fuhe"] + _add_noise(n_samples, 0.05)
    )
    data["chuyangqi_tiaojiefa_pid_sheding"] = 65.0 + _add_noise(n_samples, 2.0)
    data["chuyangqi_tiaojiefa_weizhifankui"] = np.clip(
        data["chuyangqi_tiaojiefa_pid_sheding"] - 5.0 + _add_noise(n_samples, 3.0),
        0.0,
        100.0,
    )

    # 低压汽包压力
    data["diya_qibao_yali_youxuanzhi"] = (
        4.5 + 0.003 * data["ranji_fuhe"] + _add_noise(n_samples, 0.05)
    )
    data["diya_qibao_yali_1"] = (
        data["diya_qibao_yali_youxuanzhi"] + _add_noise(n_samples, 0.03)
    )
    data["diya_qibao_yali_2"] = (
        data["diya_qibao_yali_youxuanzhi"] + _add_noise(n_samples, 0.03)
    )
    data["diya_qibao_yali_3"] = (
        data["diya_qibao_yali_youxuanzhi"] + _add_noise(n_samples, 0.03)
    )

    # 主汽流量 / 减温水 / 排气温度
    data["diya_zhuqiliuliang_youxuanzhi"] = (
        600.0 + 1.5 * data["ranji_fuhe"] + _add_noise(n_samples, 10.0)
    )
    data["gaoya_jianwenshui_liuliang"] = (
        50.0 + 0.08 * data["ranji_fuhe"] + _add_noise(n_samples, 2.0)
    )
    data["ranji_paiqi_wendu"] = (
        580.0 + 0.25 * data["ranji_fuhe"] + _add_noise(n_samples, 3.0)
    )

    # 低压汽包水位系统
    data["mubiao_diya_qibao_shuiwei"] = 150.0 + _add_noise(n_samples, 0.5)
    data["diya_qibao_shuiwei_youxuanzhi"] = (
        data["mubiao_diya_qibao_shuiwei"] + _add_noise(n_samples, 1.0)
    )

    # 实际水位受给水、凝结水流量影响，并带惯性
    shuiwei = np.zeros(n_samples)
    shuiwei[0] = data["mubiao_diya_qibao_shuiwei"][0]
    for i in range(1, n_samples):
        inflow = 0.002 * data["gaoya_geishuiliuliang_youxuanzhi"][i]
        outflow = 0.003 * data["ningjie_shuiliuliang_youxuanzhi"][i]
        shuiwei[i] = (
            0.98 * shuiwei[i - 1]
            + 0.02 * data["mubiao_diya_qibao_shuiwei"][i]
            + 0.05 * (inflow - outflow)
            + _add_noise(1, 1.5)[0]
        )
    data["diya_qibao_shuiwei"] = shuiwei
    data["diya_qibao_shuiwei_celiang_waihuan"] = (
        data["diya_qibao_shuiwei"] + _add_noise(n_samples, 1.0)
    )
    data["diya_qibao_shuiwei_mubiao_pid"] = (
        data["mubiao_diya_qibao_shuiwei"] + _add_noise(n_samples, 0.8)
    )

    # 目标：目标水位差值（带滞后，体现预测任务）
    target = np.zeros(n_samples)
    lag = 3
    target[lag:] = (
        data["mubiao_diya_qibao_shuiwei"][lag:]
        - data["diya_qibao_shuiwei"][:-lag]
        + 0.3 * (data["gaoya_geishuiliuliang_youxuanzhi"][lag:] - 1200.0) / 100.0
        + _add_noise(n_samples - lag, 0.5)
    )
    data[TARGET_COL] = target

    # 构造 DataFrame
    df = pd.DataFrame({k: data[k] for k in ALL_VALUE_COLS})

    # 时间戳
    df.insert(0, DATETIME_COL, pd.date_range(start=start_time, periods=n_samples, freq=freq))

    # 按时间顺序划分 train/val/test
    n_train = int(n_samples * train_ratio)
    n_val = int(n_samples * val_ratio)
    splits = ["train"] * n_train + ["val"] * n_val + ["test"] * (n_samples - n_train - n_val)
    df["split"] = splits

    return df


# ---------------------------------------------------------------------------
# 3. 入口
# ---------------------------------------------------------------------------
def main() -> int:
    print(f"工作目录: {REPO_ROOT}")
    print(f"示例数据输出目录: {DATA_DIR}")

    # 预测数据：单一文件，内含 30 个工业特征 + train/val/test 划分
    forecast_df = generate_forecast_data()
    forecast_path = DATA_DIR / "forecast_demo.csv"
    forecast_path.parent.mkdir(parents=True, exist_ok=True)
    forecast_df.to_csv(forecast_path, index=False)
    split_counts = forecast_df["split"].value_counts().to_dict()
    print(f"预测数据文件: {forecast_path} ({forecast_df.shape}), 划分统计: {split_counts}")

    # 输出元信息，方便脚本读取
    meta = {
        "forecast_data": str(forecast_path),
        "split_counts": split_counts,
        "target_column": TARGET_COL,
        "feature_columns": FEATURE_COLS,
    }
    meta_path = DATA_DIR / "synthetic_meta.json"
    with open(meta_path, "w", encoding="utf-8") as f:
        json.dump(meta, f, ensure_ascii=False, indent=2)
    print(f"元信息文件: {meta_path}")

    return 0


if __name__ == "__main__":
    sys.exit(main())
