#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
TSA-Suite 时序预测端到端流程统一入口脚本。

依次执行：
1. 数据生成（如不存在）
2. 时序预测流程

使用方式：
    python .claude/skills/industrial-forecasting-skill/run_all_pipelines.py
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


def run_script(script_name: str, logger: logging.Logger) -> int:
    """运行子脚本并返回退出码。"""
    script_path = SCRIPT_DIR / script_name
    if not script_path.exists():
        logger.error("脚本不存在: %s", script_path)
        return 1

    import subprocess

    logger.info("=" * 60)
    logger.info("开始执行: %s", script_name)
    logger.info("=" * 60)

    result = subprocess.run(
        [sys.executable, str(script_path)],
        cwd=str(SCRIPT_DIR),
    )

    if result.returncode != 0:
        logger.error("%s 执行失败，退出码: %d", script_name, result.returncode)
    else:
        logger.info("%s 执行完成", script_name)

    return result.returncode


def main() -> int:
    parser = argparse.ArgumentParser(description="TSA-Suite 时序预测端到端流程统一入口")
    args = parser.parse_args()

    logger = setup_logging()
    start = datetime.now()
    logger.info("TSA-Suite 时序预测端到端流程启动: %s", start.strftime("%Y-%m-%d %H:%M:%S"))

    exit_codes = []

    # 数据生成（始终先执行）
    logger.info("步骤 1/2: 检查并生成示例数据")
    exit_codes.append(("generate_data", run_script("generate_synthetic_data.py", logger)))

    # 时序预测
    logger.info("步骤 2/2: 时序预测流程")
    exit_codes.append(("forecast", run_script("run_forecast_pipeline.py", logger)))

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
