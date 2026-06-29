#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
TSA-Suite 时序预测端到端流程统一入口脚本。

执行时序预测流程（训练、推理、评估）。调用方必须通过 --input 提供
按时间顺序排列的时序数据 CSV；脚本默认按 70:15:15 顺序划分为
train/val/test。若未提供，子脚本会提示用户并退出。

使用方式：
    python skills/industrial-forecasting-skill/run_all_pipelines.py \
        --input data/your_dataset.csv
"""

import argparse
import logging
import sys
from datetime import datetime
from pathlib import Path

SCRIPT_DIR = Path(__file__).resolve().parent


def setup_logging(level: int = logging.INFO) -> logging.Logger:
    logging.basicConfig(
        level=level,
        format="%(asctime)s [%(levelname)s] %(message)s",
        datefmt="%Y-%m-%d %H:%M:%S",
    )
    return logging.getLogger("tsas_pipeline")


def run_script(script_name: str, args: list[str], logger: logging.Logger) -> int:
    """运行子脚本并返回退出码。"""
    script_path = SCRIPT_DIR / script_name
    if not script_path.exists():
        logger.error("脚本不存在: %s", script_path)
        return 1

    import subprocess

    logger.info("=" * 60)
    logger.info("开始执行: %s %s", script_name, " ".join(args))
    logger.info("=" * 60)

    result = subprocess.run(
        [sys.executable, str(script_path), *args],
        cwd=str(SCRIPT_DIR),
    )

    if result.returncode != 0:
        logger.error("%s 执行失败，退出码: %d", script_name, result.returncode)
    else:
        logger.info("%s 执行完成", script_name)

    return result.returncode


def main() -> int:
    parser = argparse.ArgumentParser(
        description="TSA-Suite 时序预测端到端流程统一入口（需用户提供时序数据集）",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog=(
            "使用示例:\n"
            "  python run_all_pipelines.py --input data/your_dataset.csv\n"
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
        default=0.70,
        help="训练集占比，默认 0.70",
    )
    parser.add_argument(
        "--val-ratio",
        type=float,
        default=0.15,
        help="验证集占比，默认 0.15",
    )
    parser.add_argument(
        "--test-ratio",
        type=float,
        default=0.15,
        help="测试集占比，默认 0.15",
    )
    args = parser.parse_args()

    total_ratio = args.train_ratio + args.val_ratio + args.test_ratio
    if not (0.99 <= total_ratio <= 1.01):
        parser.error(f"train/val/test 比例之和必须等于 1，当前为 {total_ratio}")

    logger = setup_logging()
    start = datetime.now()
    logger.info("TSA-Suite 时序预测端到端流程启动: %s", start.strftime("%Y-%m-%d %H:%M:%S"))

    exit_codes = []

    # 时序预测（必须提供输入数据）
    logger.info("步骤 1/1: 时序预测流程")
    forecast_args = [
        "--input", str(args.input),
        "--train-ratio", str(args.train_ratio),
        "--val-ratio", str(args.val_ratio),
        "--test-ratio", str(args.test_ratio),
    ]
    exit_codes.append(("forecast", run_script("run_forecast_pipeline.py", forecast_args, logger)))

    end = datetime.now()
    logger.info("=" * 60)
    logger.info("全部流程结束，总耗时: %.2f 秒", (end - start).total_seconds())
    logger.info("执行结果汇总:")
    for name, code in exit_codes:
        status = "✓ 成功" if code == 0 else f"✗ 失败 (退出码 {code})"
        logger.info("  %s: %s", name, status)
    logger.info("=" * 60)

    # 如果有任何一个失败，返回非零退出码
    return 1 if any(code != 0 for _, code in exit_codes) else 0


if __name__ == "__main__":
    sys.exit(main())
