#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
run_forecast_with_params.py — 工业时序预测参数化训练与评估入口。

接收一组超参数（JSON），在示例工业数据上训练 ITransformerForecaster，
返回多目标评估指标（MSE、RMSE、MAE、MAPE、SMAPE、MASE、DTW、R²）。

本脚本作为 `industrial-forecasting-skill` 的可参数化子入口，供 orchestrator skill
（如 `hebo-forecasting-hpo`）调用。

使用示例：
  python run_forecast_with_params.py \
    --params '{"d_model": 64, "nhead": 2, "num_layers": 1, "dropout": 0.1, \
               "lag_max": 8, "kan_grid_size": 3, "epochs": 10, \
               "batch_size": 32, "lr": 0.001, "trend_weight": 1.0}' \
    --model_tag trial_001 \
    --output -
"""

import argparse
import json
import logging
import sys
import warnings
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd

# 抑制第三方库日志
warnings.filterwarnings("ignore")
logging.getLogger("GP").setLevel(logging.ERROR)
logging.getLogger("GPy").setLevel(logging.ERROR)

# ---------------------------------------------------------------------------
# 0. 环境准备
# ---------------------------------------------------------------------------
SCRIPT_DIR = Path(__file__).resolve().parent
from config import setup_paths

PATHS = setup_paths()
REPO_ROOT = PATHS["REPO_ROOT"]
SRC_DIR = PATHS["SRC_DIR"]
DATA_DIR = PATHS["DATA_DIR"]
FORECAST_DIR = PATHS["FORECAST_DIR"]
METRICS_DIR = PATHS["METRICS_DIR"]

# 延迟导入 torch
try:
    import torch
except ImportError as e:
    print("错误：当前环境缺少 torch，请先安装 torch。")
    print(str(e))
    sys.exit(1)

from tsas.engine.operator.forecasting.itransformer import (
    ITransformerForecaster,
    ITransformerForecasterConfig,
)
from tsas.engine.operator.evaluation.forecasting_metrics import (
    ForecastingMetrics,
    ForecastingMetricConfig,
)

# ---------------------------------------------------------------------------
# 1. 默认配置与特征定义（与 run_forecast_pipeline.py 保持一致）
# ---------------------------------------------------------------------------
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
    TARGET_COL,
]
TARGET_IDX = FEATURE_COLS.index(TARGET_COL)

SEQ_LEN = 100
PRED_LEN = 20
DEVICE = "cpu"


# ---------------------------------------------------------------------------
# 2. 超参数校验
# ---------------------------------------------------------------------------
def validate_params(params: dict[str, Any]) -> dict[str, Any]:
    """校验并补全超参数，返回可用于 ITransformerForecasterConfig 的字典。"""
    defaults = {
        "d_model": 64,
        "nhead": 2,
        "num_layers": 1,
        "dropout": 0.1,
        "lag_max": 8,
        "kan_grid_size": 3,
        "epochs": 10,
        "batch_size": 32,
        "lr": 0.001,
        "trend_weight": 1.0,
    }
    merged = {**defaults, **params}

    # 类型约束
    merged["d_model"] = int(np.clip(merged["d_model"], 16, 512))
    merged["nhead"] = int(np.clip(merged["nhead"], 1, 8))
    merged["num_layers"] = int(np.clip(merged["num_layers"], 1, 6))
    merged["dropout"] = float(np.clip(merged["dropout"], 0.0, 0.8))
    merged["lag_max"] = int(np.clip(merged["lag_max"], 0, 64))
    merged["kan_grid_size"] = int(np.clip(merged["kan_grid_size"], 1, 20))
    merged["epochs"] = int(np.clip(merged["epochs"], 1, 200))
    merged["batch_size"] = int(np.clip(merged["batch_size"], 1, 512))
    merged["lr"] = float(np.clip(merged["lr"], 1e-6, 1e-1))
    merged["trend_weight"] = float(np.clip(merged["trend_weight"], 0.0, 10.0))

    # d_model 必须能被 nhead 整除
    if merged["d_model"] % merged["nhead"] != 0:
        merged["d_model"] = (merged["d_model"] // merged["nhead"]) * merged["nhead"]
        if merged["d_model"] < 16:
            merged["d_model"] = merged["nhead"] * 16

    return merged


# ---------------------------------------------------------------------------
# 3. 数据加载
# ---------------------------------------------------------------------------
def ensure_data() -> pd.DataFrame:
    """确保示例数据存在，不存在则调用生成脚本。"""
    data_path = DATA_DIR / "forecast_demo.csv"
    if not data_path.exists():
        gen_script = SCRIPT_DIR / "generate_synthetic_data.py"
        import subprocess

        result = subprocess.run(
            [sys.executable, str(gen_script)],
            cwd=REPO_ROOT,
            capture_output=True,
            text=True,
        )
        if result.returncode != 0:
            raise RuntimeError(f"生成示例数据失败:\n{result.stderr}")

    return pd.read_csv(data_path)


def load_data() -> tuple[pd.DataFrame, pd.DataFrame]:
    """加载 forecast_demo.csv 并分离 train/test split。"""
    df = ensure_data()
    assert set(FEATURE_COLS + ["split"]).issubset(df.columns)

    train_df = df[df["split"] == "train"].reset_index(drop=True)
    test_df = df[df["split"] == "test"].reset_index(drop=True)
    return train_df, test_df


# ---------------------------------------------------------------------------
# 4. 单次超参数评估
# ---------------------------------------------------------------------------
def evaluate_forecast_params(
    params: dict[str, Any],
    model_dir: Path,
    device: str = DEVICE,
    logger: logging.Logger | None = None,
) -> dict[str, float]:
    """
    使用给定超参数训练并评估一次工业时序预测模型。

    Parameters
    ----------
    params    : 超参数字典
    model_dir : 模型保存目录（每次 trial 使用不同目录避免冲突）
    device    : 训练设备
    logger    : 可选日志器

    Returns
    -------
    dict : 8 项评估指标 {mse, rmse, mae, mape, smape, mase, dtw, r2}
    """
    if logger is None:
        logger = logging.getLogger("run_forecast_with_params")
        logger.setLevel(logging.WARNING)

    params = validate_params(params)
    logger.info("评估超参数: %s", params)

    train_df, test_df = load_data()

    # 构造模型配置
    config = ITransformerForecasterConfig(
        seq_len=SEQ_LEN,
        pred_len=PRED_LEN,
        d_model=params["d_model"],
        nhead=params["nhead"],
        num_layers=params["num_layers"],
        dim_feedforward=params["d_model"] * 2,
        dropout=params["dropout"],
        lag_aware=True,
        lag_max=params["lag_max"],
        kan_grid_size=params["kan_grid_size"],
        target_idx=TARGET_IDX,
        epochs=params["epochs"],
        batch_size=params["batch_size"],
        lr=params["lr"],
        early_stop_patience=5,
        trend_weight=params["trend_weight"],
        train_ratio=0.9,
        val_ratio=0.05,
        device=device,
    )

    # 训练
    x_train = train_df[FEATURE_COLS].values.astype(np.float32)
    y_train = train_df[[TARGET_COL]].values.astype(np.float32)

    forecaster = ITransformerForecaster(config=config)
    forecaster.fit(x_train, y_train)

    model_dir.mkdir(parents=True, exist_ok=True)
    forecaster.save(model_dir)

    # 推理
    loaded = ITransformerForecaster.load(model_dir)
    df_test = test_df[FEATURE_COLS]
    total_len = len(df_test)
    if total_len < SEQ_LEN + PRED_LEN:
        raise ValueError(f"测试集长度 {total_len} 不足以构造窗口")

    start_idx = total_len - SEQ_LEN - PRED_LEN
    x_window = df_test.iloc[start_idx : start_idx + SEQ_LEN].values.astype(np.float32)
    y_true = df_test.iloc[start_idx + SEQ_LEN : start_idx + SEQ_LEN + PRED_LEN][[TARGET_COL]].values.astype(np.float32)
    y_pred = loaded.run(x_window)

    # 评估
    train_y = train_df[TARGET_COL].values.astype(np.float32)
    naive_error = float(np.mean(np.abs(train_y[1:] - train_y[:-1])))

    metric_config = ForecastingMetricConfig(naive_error=naive_error)
    metrics_op = ForecastingMetrics(config=metric_config)
    metrics = metrics_op.run((y_true, y_pred))

    return {
        "mse": float(metrics.mse),
        "rmse": float(metrics.rmse),
        "mae": float(metrics.mae),
        "mape": float(metrics.mape),
        "smape": float(metrics.smape),
        "mase": float(metrics.mase),
        "dtw": float(metrics.dtw),
        "r2": float(metrics.r2),
    }


# ---------------------------------------------------------------------------
# 5. CLI 入口
# ---------------------------------------------------------------------------
def main() -> int:
    parser = argparse.ArgumentParser(
        description="工业时序预测参数化训练与评估入口",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog=__doc__,
    )
    parser.add_argument(
        "--params",
        required=True,
        help='超参数 JSON 字符串，例如：\'{"d_model": 64, "epochs": 10}\'',
    )
    parser.add_argument(
        "--model_tag",
        default="hpo_trial",
        help="本次评估的模型目录标识，默认 hpo_trial",
    )
    parser.add_argument(
        "--output",
        default="-",
        help="指标输出路径，默认 '-' 输出到 stdout",
    )
    parser.add_argument(
        "--device",
        default=DEVICE,
        help=f"训练设备，默认 {DEVICE}",
    )

    args = parser.parse_args()

    try:
        params = json.loads(args.params)
    except json.JSONDecodeError as e:
        parser.error(f"--params 不是合法 JSON：{e}")
        return 1

    if not isinstance(params, dict):
        parser.error("--params 必须是一个 JSON 对象")
        return 1

    model_dir = METRICS_DIR.parent / "models_hpo" / args.model_tag

    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s [%(levelname)s] %(message)s",
        datefmt="%Y-%m-%d %H:%M:%S",
    )
    logger = logging.getLogger("run_forecast_with_params")

    try:
        metrics = evaluate_forecast_params(params, model_dir, args.device, logger)
    except Exception as e:
        logger.error("评估失败: %s", e)
        return 1

    result = {
        "params": params,
        "metrics": metrics,
        "model_dir": str(model_dir),
    }

    output_text = json.dumps(result, ensure_ascii=False, indent=2)
    if args.output == "-":
        print(output_text)
    else:
        out_path = Path(args.output)
        out_path.parent.mkdir(parents=True, exist_ok=True)
        out_path.write_text(output_text, encoding="utf-8")
        logger.info("指标已保存: %s", out_path)

    return 0


if __name__ == "__main__":
    sys.exit(main())
