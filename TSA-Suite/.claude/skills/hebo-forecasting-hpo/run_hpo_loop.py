#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
run_hpo_loop.py — HEBO 工业时序预测超参数寻优闭环脚本。

协调 bo-for-experiment 与 industrial-forecasting-skill 完成自动寻优。
"""

import argparse
import json
import os
import subprocess
import sys
import time
from pathlib import Path
from typing import Any

ROOT = Path(__file__).resolve().parents[3]
BO_MAIN = ROOT / ".claude" / "skills" / "bo-for-experiment" / "main.py"
FORECAST_MAIN = (
    ROOT / ".claude" / "skills" / "industrial-forecasting-skill" / "run_forecast_with_params.py"
)
PYTHON = sys.executable
OBJECTIVE_NAMES = ["mse", "rmse", "mae", "mape", "smape", "mase", "dtw", "r2"]


def run_cmd(cmd: list[str], env: dict[str, str] | None = None) -> tuple[int, str, str]:
    merged_env = os.environ.copy()
    if env:
        merged_env.update(env)
    if "PYTHONPATH" not in merged_env:
        merged_env["PYTHONPATH"] = str(ROOT / "src")
    result = subprocess.run(cmd, cwd=ROOT, env=merged_env, capture_output=True, text=True)
    return result.returncode, result.stdout, result.stderr


def bo_iterate(
    task_id: str,
    data_dir: str,
    x_new: list[dict],
    y_new: list[dict],
    n_suggest: int,
) -> list[dict[str, Any]]:
    cmd = [
        PYTHON,
        str(BO_MAIN),
        "--mode", "iterate",
        "--task_id", task_id,
        "--x_new", json.dumps(x_new, ensure_ascii=False),
        "--y_new", json.dumps(y_new, ensure_ascii=False),
        "--n_suggest", str(n_suggest),
        "--data_dir", data_dir,
        "--format", "json",
    ]
    rc, stdout, stderr = run_cmd(cmd)
    if rc != 0:
        print(f"[ERROR] BO iterate 失败:\n{stderr}\n{stdout}", file=sys.stderr)
        return []
    lines = [line.strip() for line in stdout.splitlines() if line.strip()]
    if not lines:
        print("[ERROR] BO iterate 无输出", file=sys.stderr)
        return []
    try:
        result = json.loads(lines[-1])
    except json.JSONDecodeError as e:
        print(f"[ERROR] 无法解析 BO 输出: {e}\n{lines[-1]}", file=sys.stderr)
        return []
    return result.get("suggestions", [])


def evaluate_params(params: dict[str, Any], model_tag: str) -> dict[str, float] | None:
    cmd = [
        PYTHON,
        str(FORECAST_MAIN),
        "--params", json.dumps(params, ensure_ascii=False),
        "--model_tag", model_tag,
        "--output", "-",
    ]
    rc, stdout, stderr = run_cmd(cmd)
    if rc != 0:
        print(f"[ERROR] 训练评估失败 ({model_tag}):\n{stderr}\n{stdout}", file=sys.stderr)
        return None

    last_brace = stdout.rfind("{")
    if last_brace == -1:
        print(f"[ERROR] 训练输出中无 JSON ({model_tag}):\n{stdout}", file=sys.stderr)
        return None

    for start in range(last_brace, -1, -1):
        if stdout[start] != "{":
            continue
        try:
            result = json.loads(stdout[start:])
            return result.get("metrics")
        except json.JSONDecodeError:
            continue

    print(f"[ERROR] 无法解析训练输出 ({model_tag}):\n{stdout}", file=sys.stderr)
    return None


def rank_sum_selection(records: list[dict]) -> dict[str, Any]:
    import numpy as np

    if not records:
        return {}
    X = np.array([[r["y"][name] for name in OBJECTIVE_NAMES] for r in records])
    X[:, OBJECTIVE_NAMES.index("r2")] *= -1
    ranks = np.zeros_like(X)
    for j in range(X.shape[1]):
        ranks[:, j] = np.argsort(np.argsort(X[:, j])) + 1
    rank_sums = ranks.sum(axis=1)
    best_idx = int(np.argmin(rank_sums))
    best = records[best_idx].copy()
    best["rank_sum"] = float(rank_sums[best_idx])
    return best


def main() -> int:
    parser = argparse.ArgumentParser(description="HEBO 工业时序预测超参数闭环寻优")
    parser.add_argument("--task_id", required=True)
    parser.add_argument("--data_dir", default="./hpo_results")
    parser.add_argument("--max_trials", type=int, default=10)
    parser.add_argument("--n_suggest", type=int, default=1)
    parser.add_argument(
        "--pilot_params",
        default='{"d_model": 64, "nhead": 2, "num_layers": 1, "dropout": 0.1, "lag_max": 8, "kan_grid_size": 3, "epochs": 10, "batch_size": 32, "lr": 0.001, "trend_weight": 1.0}',
    )
    parser.add_argument(
        "--pilot_metrics",
        default="",
        help='Pilot 指标 JSON；留空则真实运行 pilot 评估',
    )
    parser.add_argument("--skip_pilot", action="store_true")
    parser.add_argument("--resume", action="store_true")
    args = parser.parse_args()

    x_new: list[dict] = []
    y_new: list[dict] = []
    records: list[dict] = []
    trial_counter = 0

    if args.resume:
        history_path = Path(args.data_dir) / f"{args.task_id}_history.json"
        if history_path.exists():
            history = json.loads(history_path.read_text(encoding="utf-8"))
            records = history.get("records", [])
            trial_counter = len(records)
            print(f"[Resume] 从历史文件恢复 {trial_counter} 条记录")

    if not args.skip_pilot and not args.resume:
        pilot_params = json.loads(args.pilot_params)
        if args.pilot_metrics:
            pilot_metrics = json.loads(args.pilot_metrics)
            print(f"[Pilot] 使用给定指标: {pilot_metrics}")
        else:
            print("[Pilot] 真实运行 pilot 评估...")
            pilot_metrics = evaluate_params(pilot_params, "trial_0001")
            if pilot_metrics is None:
                print("[ERROR] pilot 评估失败，退出。", file=sys.stderr)
                return 1
        x_new.append(pilot_params)
        y_new.append(pilot_metrics)
        records.append({"x": pilot_params, "y": pilot_metrics})
        trial_counter += 1
        print(f"[Pilot] trial_{trial_counter:04d} metrics={pilot_metrics}")

    while trial_counter < args.max_trials:
        suggestions = bo_iterate(args.task_id, args.data_dir, x_new, y_new, n_suggest=args.n_suggest)
        if not suggestions:
            print("[WARN] 未获取到推荐参数，终止循环。", file=sys.stderr)
            break
        x_new = []
        y_new = []

        for suggestion in suggestions:
            trial_counter += 1
            model_tag = f"trial_{trial_counter:04d}"
            print(f"\n[Trial {trial_counter}/{args.max_trials}] tag={model_tag} params={suggestion}")
            start = time.time()
            metrics = evaluate_params(suggestion, model_tag)
            elapsed = time.time() - start

            if metrics is None:
                print(f"[WARN] {model_tag} 评估失败，跳过。")
                continue

            print(f"[Trial {trial_counter}] elapsed={elapsed:.1f}s metrics={metrics}")
            records.append({"x": suggestion, "y": metrics})
            x_new.append(suggestion)
            y_new.append(metrics)

            if trial_counter >= args.max_trials:
                break

    best = rank_sum_selection(records)
    print("\n" + "=" * 70)
    print("[HPO 完成] 最优参数（rank-sum 最小）:")
    print(json.dumps(best, ensure_ascii=False, indent=2))
    print("=" * 70)

    best_path = Path(args.data_dir) / f"{args.task_id}_best_params.json"
    best_path.write_text(json.dumps(best, ensure_ascii=False, indent=2), encoding="utf-8")
    print(f"[文件] 最优参数已保存: {best_path}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
