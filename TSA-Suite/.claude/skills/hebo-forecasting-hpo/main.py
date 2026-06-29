#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
main.py — hebo-forecasting-hpo：工业时序预测超参数 HEBO 多目标寻优 Orchestrator

本文件不再是完整 HPO 执行入口，而是轻量说明/帮助入口。

本 skill 为编排器（orchestrator），自身不实现 HEBO 引擎，也不直接训练模型。
实际执行由 Claude Code 框架根据 SKILL.md 中的编排说明，通过 Skill 工具调用以下两个子 skill 完成：

  - /bo-for-experiment          : HEBO 参数空间管理与推荐
  - /industrial-forecasting-skill : 工业时序预测训练与评估

使用方式：
  在 Claude Code 对话中输入 "/hebo-forecasting-hpo" 或相关自然语言请求，
  Claude 将自动执行：
    1. /bo-for-experiment init（非交互）
    2. 循环调用 /bo-for-experiment iterate 获取推荐参数
    3. 调用 /industrial-forecasting-skill/run_forecast_with_params.py 训练评估
    4. 将指标反馈给 /bo-for-experiment
    5. 达到 max_trials 后输出 rank-sum 综合最优参数

如需脚本化执行，请直接使用：
  python .claude/skills/bo-for-experiment/main.py ...
  python .claude/skills/industrial-forecasting-skill/run_forecast_with_params.py ...
"""

import argparse
import sys


WORKFLOW_TEXT = """
[hebo-forecasting-hpo] 编排工作流
================================

本 skill 通过调用以下两个子 skill 完成闭环超参数寻优：

1. /bo-for-experiment
   - init:  创建 HPO 任务，定义 10 维超参数空间和 8 项目标
   - iterate: 基于历史记录推荐下一组候选参数

2. /industrial-forecasting-skill
   - run_forecast_with_params.py: 使用给定超参数训练 ITransformerForecaster，
                                  返回 MSE/RMSE/MAE/MAPE/SMAPE/MASE/DTW/R²

Conversation 执行流程：
----------------------
Step 1: 调用 /bo-for-experiment init（非交互模式）
        --params_config <10-dim space JSON>
        --objectives    <8-objective JSON>
        --task_id       HPO_FORECAST_YYYYMMDD_HHMMSS
        --data_dir      ./hpo_results

Step 2: 循环直到 max_trials
        a. /bo-for-experiment iterate --format json
           → 获取 suggestions（首次可传空列表以触发随机推荐）
        b. 对每组 suggestion 调用 run_forecast_with_params.py --params <json>
           → 获取 metrics JSON
        c. 将 (params, metrics) 追加为下一轮 iterate 的 x_new / y_new

Step 3: 读取 {Task_ID}_history.json，使用 rank-sum 方法计算综合最优参数，
        保存为 {Task_ID}_best_params.json 并返回给用户。

注意：本 main.py 仅作说明，不执行实际 HPO。请在 Claude Code 对话中触发本 skill。
"""


def main() -> int:
    parser = argparse.ArgumentParser(
        description="hebo-forecasting-hpo: orchestrator entry point (informational only)",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog=__doc__,
    )
    parser.add_argument(
        "--workflow",
        action="store_true",
        help="打印完整编排工作流说明",
    )

    args = parser.parse_args()

    if args.workflow:
        print(WORKFLOW_TEXT)
    else:
        print(WORKFLOW_TEXT)
        print('\n提示：本 skill 为 orchestrator，请在 Claude Code 对话中使用 "/hebo-forecasting-hpo" 触发。')
        print("      如需脚本化执行，请直接调用 bo-for-experiment 与 run_forecast_with_params.py。")

    return 0


if __name__ == "__main__":
    sys.exit(main())
