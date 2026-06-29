#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
时序预测端到端流程脚本。

流程：
1. 读取用户提供的时序数据 CSV
2. 按默认 70:15:15 顺序划分为 train/val/test
3. 使用 train split 训练 ITransformerForecaster
4. 保存并加载模型
5. 在 test split 上执行预测
6. 用 ForecastingMetrics 评估
7. 输出报告与指标 JSON

注意：本脚本不再自动生成示例数据，调用方必须通过 --input 提供时序数据集；
      如未提供或文件不存在，脚本会提示用户并退出。
"""

import argparse
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

# 默认按时间顺序划分 train/val/test 的比例
TRAIN_RATIO = 0.70
VAL_RATIO = 0.15
TEST_RATIO = 0.15


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
def prompt_for_dataset(missing_path: Path) -> str:
    """返回缺失时序数据集时的友好提示信息。"""
    return (
        "\n"
        "========================================\n"
        "未找到时序数据集。\n"
        "----------------------------------------\n"
        f"指定路径不存在: {missing_path}\n"
        "\n"
        "调用本 skill 时，请提供原始时序数据 CSV 文件，例如：\n"
        "  python skills/industrial-forecasting-skill/run_forecast_pipeline.py \\\n"
        "      --input data/your_dataset.csv\n"
        "\n"
        "输入 CSV 要求：\n"
        "  - 按时间顺序排列的时序数据；\n"
        "  - 包含与本 skill 配置一致的特征列与目标列；\n"
        f"  - 脚本默认按 {TRAIN_RATIO*100:.0f}:{VAL_RATIO*100:.0f}:{TEST_RATIO*100:.0f} "
        "顺序划分为 train/val/test。\n"
        "========================================\n"
    )


def split_data(df: pd.DataFrame, train_ratio: float, val_ratio: float) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    """按时间顺序将数据划分为 train/val/test。"""
    n = len(df)
    n_train = int(n * train_ratio)
    n_val = int(n * val_ratio)
    train_df = df.iloc[:n_train].reset_index(drop=True)
    val_df = df.iloc[n_train : n_train + n_val].reset_index(drop=True)
    test_df = df.iloc[n_train + n_val :].reset_index(drop=True)
    return train_df, val_df, test_df


def load_data(input_path: Path, logger: logging.Logger) -> pd.DataFrame:
    """加载用户提供的时序数据；如不存在则提示用户并退出。"""
    if not input_path.exists():
        message = prompt_for_dataset(input_path)
        logger.error(message)
        raise FileNotFoundError(message)

    df = pd.read_csv(input_path)
    logger.info("加载预测数据: %s, shape=%s", input_path, df.shape)
    return df


# ---------------------------------------------------------------------------
# 3. Pipeline
# ---------------------------------------------------------------------------
class ForecastPipeline:
    def __init__(
        self,
        input_path: Path,
        logger: logging.Logger,
        train_ratio: float = TRAIN_RATIO,
        val_ratio: float = VAL_RATIO,
        test_ratio: float = TEST_RATIO,
    ) -> None:
        self.input_path = input_path
        self.logger = logger
        self.train_ratio = train_ratio
        self.val_ratio = val_ratio
        self.test_ratio = test_ratio
        self.df: pd.DataFrame | None = None
        self.train_df: pd.DataFrame | None = None
        self.val_df: pd.DataFrame | None = None
        self.test_df: pd.DataFrame | None = None
        self.forecaster: ITransformerForecaster | None = None
        self.metrics: Any = None

    def load_data(self) -> pd.DataFrame:
        self.df = load_data(self.input_path, self.logger)
        assert set(FEATURE_COLS).issubset(self.df.columns)

        # 按时间顺序 70:15:15 划分 train/val/test
        self.train_df, self.val_df, self.test_df = split_data(
            self.df, self.train_ratio, self.val_ratio
        )
        self.logger.info(
            "数据集划分 (%.0f:%.0f:%.0f): train=%s, val=%s, test=%s",
            self.train_ratio * 100,
            self.val_ratio * 100,
            self.test_ratio * 100,
            self.train_df.shape,
            self.val_df.shape,
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
            f"数据文件    : {self.input_path}",
            f"训练样本数 : {len(self.train_df)}",
            f"验证样本数 : {len(self.val_df)}",
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
    parser = argparse.ArgumentParser(
        description="工业时序预测端到端流程（需用户提供时序数据集）",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog=(
            "使用示例:\n"
            "  python run_forecast_pipeline.py --input data/your_dataset.csv\n"
            "\n"
            "输入 CSV 应为按时间顺序排列的原始时序数据；"
            "脚本默认按 70:15:15 顺序划分为 train/val/test。"
        ),
    )
    parser.add_argument(
        "--input",
        "-i",
        type=Path,
        required=True,
        help="时序数据集路径（CSV，按时间顺序排列）",
    )
    parser.add_argument(
        "--train-ratio",
        type=float,
        default=TRAIN_RATIO,
        help=f"训练集占比，默认 {TRAIN_RATIO}",
    )
    parser.add_argument(
        "--val-ratio",
        type=float,
        default=VAL_RATIO,
        help=f"验证集占比，默认 {VAL_RATIO}",
    )
    parser.add_argument(
        "--test-ratio",
        type=float,
        default=TEST_RATIO,
        help=f"测试集占比，默认 {TEST_RATIO}",
    )
    args = parser.parse_args()

    total_ratio = args.train_ratio + args.val_ratio + args.test_ratio
    if not (0.99 <= total_ratio <= 1.01):
        parser.error(f"train/val/test 比例之和必须等于 1，当前为 {total_ratio}")

    logger = setup_logging()
    pipeline = ForecastPipeline(
        input_path=args.input,
        logger=logger,
        train_ratio=args.train_ratio,
        val_ratio=args.val_ratio,
        test_ratio=args.test_ratio,
    )
    result = pipeline.run()
    logger.info("输出汇总: %s", result)
    return 0


if __name__ == "__main__":
    sys.exit(main())
