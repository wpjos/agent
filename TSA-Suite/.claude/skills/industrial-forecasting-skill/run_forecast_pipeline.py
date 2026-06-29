#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
时序预测端到端流程脚本。

流程：
1. 生成/读取示例数据（单一 CSV，内含 train/val/test split 列）
2. 使用 train split 训练 ITransformerForecaster
3. 保存并加载模型
4. 在 test split 上执行预测
5. 用 ForecastingMetrics 评估
6. 输出报告与指标 JSON
"""

import json
import logging
import sys
import warnings
from datetime import datetime
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd

# ---------------------------------------------------------------------------
# 0. 环境准备
# ---------------------------------------------------------------------------
SCRIPT_DIR = Path(__file__).resolve().parent
from config import setup_paths

PATHS = setup_paths()
REPO_ROOT = PATHS["REPO_ROOT"]
SRC_DIR = PATHS["SRC_DIR"]
DATA_DIR = PATHS["DATA_DIR"]
MODEL_DIR = PATHS["MODEL_DIR_FORECAST"]
FORECAST_DIR = PATHS["FORECAST_DIR"]
METRICS_DIR = PATHS["METRICS_DIR"]

warnings.filterwarnings("ignore")

# 延迟导入 torch，便于在环境检查阶段给出友好提示
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
# 1. 配置
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
D_MODEL = 64
NHEAD = 2
NUM_LAYERS = 1
EPOCHS = 10
BATCH_SIZE = 32
LR = 0.001
DEVICE = "cpu"


def setup_logging(level: int = logging.INFO) -> logging.Logger:
    logging.basicConfig(
        level=level,
        format="%(asctime)s [%(levelname)s] %(message)s",
        datefmt="%Y-%m-%d %H:%M:%S",
    )
    return logging.getLogger("forecast_pipeline")


# ---------------------------------------------------------------------------
# 2. 数据
# ---------------------------------------------------------------------------
def ensure_data(logger: logging.Logger) -> pd.DataFrame:
    """确保示例数据存在，不存在则调用生成脚本。"""
    data_path = DATA_DIR / "forecast_demo.csv"
    if not data_path.exists():
        logger.info("示例数据不存在，正在生成...")
        gen_script = SCRIPT_DIR / "generate_synthetic_data.py"
        import subprocess

        result = subprocess.run(
            [sys.executable, str(gen_script)],
            cwd=REPO_ROOT,
            capture_output=True,
            text=True,
        )
        if result.returncode != 0:
            logger.error("生成数据失败:\n%s", result.stderr)
            raise RuntimeError("生成示例数据失败")
        logger.info("示例数据生成完成。")

    df = pd.read_csv(data_path)
    logger.info("加载预测数据: %s, shape=%s", data_path, df.shape)
    return df


# ---------------------------------------------------------------------------
# 3. Pipeline
# ---------------------------------------------------------------------------
class ForecastPipeline:
    def __init__(self, logger: logging.Logger) -> None:
        self.logger = logger
        self.df: pd.DataFrame | None = None
        self.train_df: pd.DataFrame | None = None
        self.test_df: pd.DataFrame | None = None
        self.forecaster: ITransformerForecaster | None = None
        self.metrics: Any = None

    def load_data(self) -> pd.DataFrame:
        self.df = ensure_data(self.logger)
        assert set(FEATURE_COLS + ["split"]).issubset(self.df.columns)

        # 按 split 列分离训练集与测试集
        self.train_df = self.df[self.df["split"] == "train"].reset_index(drop=True)
        self.test_df = self.df[self.df["split"] == "test"].reset_index(drop=True)
        self.logger.info(
            "数据集划分: train=%s, test=%s",
            self.train_df.shape,
            self.test_df.shape,
        )
        return self.df

    def build_forecaster(self) -> ITransformerForecaster:
        self.logger.info("构造 ITransformerForecaster...")
        config = ITransformerForecasterConfig(
            seq_len=SEQ_LEN,
            pred_len=PRED_LEN,
            d_model=D_MODEL,
            nhead=NHEAD,
            num_layers=NUM_LAYERS,
            dim_feedforward=D_MODEL * 2,
            dropout=0.1,
            lag_aware=True,
            lag_max=8,
            kan_grid_size=3,
            target_idx=TARGET_IDX,
            epochs=EPOCHS,
            batch_size=BATCH_SIZE,
            lr=LR,
            early_stop_patience=5,
            # 已经在外部按 70:15:15 划分好 train/val/test，
            # 因此传入 forecaster 的仅为 train split；
            # 这里把 train split 中少量样本作为内部早停验证集。
            train_ratio=0.9,
            val_ratio=0.05,
            device=DEVICE,
        )
        return ITransformerForecaster(config=config)

    def train(self) -> ITransformerForecaster:
        df = self.train_df[FEATURE_COLS]
        x = df.values.astype(np.float32)
        y = df[[TARGET_COL]].values.astype(np.float32)

        self.forecaster = self.build_forecaster()
        self.logger.info(
            "开始训练: x=%s, y=%s, device=%s",
            x.shape,
            y.shape,
            self.forecaster._device,
        )
        self.forecaster.fit(x, y)
        self.logger.info("训练完成。")

        MODEL_DIR.mkdir(parents=True, exist_ok=True)
        self.forecaster.save(MODEL_DIR)
        self.logger.info("模型已保存到: %s", MODEL_DIR)
        return self.forecaster

    def predict(self) -> tuple[np.ndarray, np.ndarray]:
        """在 test split 上构造 seq_len+pred_len 窗口并执行预测。"""
        loaded = ITransformerForecaster.load(MODEL_DIR)
        self.logger.info("模型已加载。")

        df = self.test_df[FEATURE_COLS]
        total_len = len(df)
        if total_len < SEQ_LEN + PRED_LEN:
            raise ValueError(f"测试集长度 {total_len} 不足以构造 seq_len+pred_len 窗口")

        start_idx = total_len - SEQ_LEN - PRED_LEN
        x_window = df.iloc[start_idx : start_idx + SEQ_LEN].values.astype(np.float32)
        y_true = (
            df.iloc[start_idx + SEQ_LEN : start_idx + SEQ_LEN + PRED_LEN][[TARGET_COL]]
            .values.astype(np.float32)
        )

        self.logger.info("推理窗口: x=%s, y_true=%s", x_window.shape, y_true.shape)
        y_pred = loaded.run(x_window)
        self.logger.info("预测输出 shape: %s", y_pred.shape)

        # 保存预测结果
        FORECAST_DIR.mkdir(parents=True, exist_ok=True)
        pred_df = pd.DataFrame(
            {
                "y_true": y_true.ravel(),
                "y_pred": y_pred.ravel(),
            }
        )
        pred_path = FORECAST_DIR / "pred.csv"
        pred_df.to_csv(pred_path, index=False)
        self.logger.info("预测结果已保存: %s", pred_path)
        return y_true, y_pred

    def evaluate(self, y_true: np.ndarray, y_pred: np.ndarray) -> Any:
        # 使用训练集目标列计算 naive_error，用于 MASE
        train_y = self.train_df[TARGET_COL].values.astype(np.float32)
        naive_error = float(np.mean(np.abs(train_y[1:] - train_y[:-1])))

        metric_config = ForecastingMetricConfig(naive_error=naive_error)
        metrics_op = ForecastingMetrics(config=metric_config)
        self.metrics = metrics_op.run((y_true, y_pred))

        METRICS_DIR.mkdir(parents=True, exist_ok=True)
        metrics_path = METRICS_DIR / "forecast_metrics.json"
        with open(metrics_path, "w", encoding="utf-8") as f:
            json.dump(self.metrics.model_dump(), f, ensure_ascii=False, indent=2)
        self.logger.info("指标已保存: %s", metrics_path)
        return self.metrics

    def report(self) -> str:
        lines = [
            "\n========== 时序预测端到端流程报告 ==========",
            f"数据文件    : {DATA_DIR / 'forecast_demo.csv'}",
            f"训练样本数 : {len(self.train_df)}",
            f"测试样本数 : {len(self.test_df)}",
            f"模型目录    : {MODEL_DIR}",
            f"预测结果    : {FORECAST_DIR / 'pred.csv'}",
            f"指标文件    : {METRICS_DIR / 'forecast_metrics.json'}",
            "",
            "关键指标:",
            f"  MSE   : {self.metrics.mse:.6f}",
            f"  RMSE  : {self.metrics.rmse:.6f}",
            f"  MAE   : {self.metrics.mae:.6f}",
            f"  MAPE  : {self.metrics.mape:.6f} %",
            f"  SMAPE : {self.metrics.smape:.6f} %",
            f"  MASE  : {self.metrics.mase:.6f}",
            f"  DTW   : {self.metrics.dtw:.6f}",
            f"  R²    : {self.metrics.r2:.6f}",
            "============================================\n",
        ]
        summary = "\n".join(lines)
        self.logger.info(summary)
        return summary

    def run(self) -> dict[str, Any]:
        start = datetime.now()
        self.logger.info("启动时序预测端到端流程: %s", start.strftime("%Y-%m-%d %H:%M:%S"))

        self.load_data()
        self.train()
        y_true, y_pred = self.predict()
        self.evaluate(y_true, y_pred)
        self.report()

        end = datetime.now()
        self.logger.info("流程结束，总耗时: %.2f 秒", (end - start).total_seconds())

        return {
            "metrics": self.metrics.model_dump() if self.metrics else None,
            "model_dir": str(MODEL_DIR),
            "forecast_csv": str(FORECAST_DIR / "pred.csv"),
            "metrics_json": str(METRICS_DIR / "forecast_metrics.json"),
        }


# ---------------------------------------------------------------------------
# 4. 入口
# ---------------------------------------------------------------------------
def main() -> int:
    logger = setup_logging()
    pipeline = ForecastPipeline(logger)
    result = pipeline.run()
    logger.info("输出汇总: %s", result)
    return 0


if __name__ == "__main__":
    sys.exit(main())
