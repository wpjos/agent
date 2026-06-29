"""
main.py — bo4experiment 技能执行入口

两种运行模式：
  --mode init      Scenario 1：解析自然语言 → 展示建议书 → 等待确认 → 初始化任务
  --mode iterate   Scenario 2：追加观测数据 → 拟合模型 → 推荐下一组实验参数

使用示例：

  # Scenario 1：初始化任务（交互式）
  python .claude/skills/bo-for-experiment/main.py --mode init \\
    --description "催化剂合成：温度200-400°C，压力1-10bar，目标最大化转化率"

  # Scenario 1：非交互式初始化（适用于 orchestrator 调用）
  python .claude/skills/bo-for-experiment/main.py --mode init \\
    --non_interactive \\
    --params_config '[{"name":"temperature","type":"num","lb":200,"ub":400},...]' \\
    --objectives '[{"name":"conversion_rate","direction":"max"}]' \\
    --task_id BO20260626_143022

  # Scenario 2：追加数据并推荐
  python .claude/skills/bo-for-experiment/main.py --mode iterate \\
    --task_id BO20260415_143022 \\
    --x_new '{"temperature": 300, "pressure": 5}' \\
    --y_new '{"conversion_rate": 0.85}' \\
    --n_suggest 3

  # Scenario 2：空历史时随机推荐（orchestrator 首次迭代）
  python .claude/skills/bo-for-experiment/main.py --mode iterate \\
    --task_id BO20260415_143022 \\
    --x_new '[]' --y_new '[]' \\
    --n_suggest 3 --format json

  # 指定历史文件目录（默认当前目录）
  python .claude/skills/bo-for-experiment/main.py --mode iterate \\
    --task_id BO20260415_143022 \\
    --x_new '{"temperature": 300, "pressure": 5}' \\
    --y_new '{"conversion_rate": 0.85}' \\
    --data_dir ./experiments
"""

import argparse
import json
import os
import sys
from typing import List, Dict, Optional, Any

# 将 agents/ 目录加入路径，以便直接导入子模块
_SKILL_DIR  = os.path.dirname(os.path.abspath(__file__))
_AGENTS_DIR = os.path.join(_SKILL_DIR, "agents")
if _AGENTS_DIR not in sys.path:
    sys.path.insert(0, _AGENTS_DIR)

from space_parser   import _get_client, parse_space, format_proposal_table
from history_manager import (
    init_history, generate_task_id,
    load_history, append_records, save_history,
    get_record_count,
)

# bo_engine 在运行时动态导入，避免 init 模式（可能不调用 HEBO）强制依赖 hebo 包
_suggest_next = None
_compute_pareto_summary = None

def _load_bo_engine():
    global _suggest_next, _compute_pareto_summary
    if _suggest_next is None:
        from bo_engine import suggest_next, compute_pareto_summary
        _suggest_next = suggest_next
        _compute_pareto_summary = compute_pareto_summary
    return _suggest_next, _compute_pareto_summary


# ─────────────────────────────────────────────────────────────────────────────
# Scenario 1：任务初始化
# ─────────────────────────────────────────────────────────────────────────────

def run_init(
    description: str = "",
    data_dir: str = ".",
    non_interactive: bool = False,
    params_config: Optional[List[Dict[str, Any]]] = None,
    objectives: Optional[List[Dict[str, Any]]] = None,
    constraints: Optional[List[Dict[str, Any]]] = None,
    task_id: Optional[str] = None,
) -> str:
    """
    Scenario 1：创建任务。

    交互模式（默认）：
      解析自然语言描述 → 展示《寻优任务建议书》 → 等待用户确认 → 初始化任务。

    非交互模式（--non_interactive）：
      当 --params_config 与 --objectives 同时提供时，直接创建任务，不调用 LLM，不阻塞。
    """
    # 如果调用方直接提供了结构化的参数空间与目标，则跳过 LLM 解析
    if params_config is not None and objectives is not None:
        parsed = {
            "params_config": params_config,
            "objectives": objectives,
            "constraints": constraints or [],
        }
        print("[非交互模式] 使用调用方提供的参数空间与目标，跳过 LLM 解析。")
    else:
        if not description:
            raise ValueError("非交互模式下必须同时提供 --params_config 和 --objectives；交互模式下必须提供 --description。")

        print("\n[1/2] 正在解析实验空间...\n")
        client = _get_client()
        try:
            parsed = parse_space(client, description)
        except Exception as e:
            print(f"❌ [错误] 解析失败：{e}")
            sys.exit(1)

    # 展示《寻优任务建议书》（非交互模式下仅打印，不阻塞）
    print(format_proposal_table(parsed))
    print()

    if not non_interactive:
        # CLI 阻塞模式
        print("  [确认] 直接按回车 或 输入 '确认' — 创建任务")
        print("  [修改] 输入修改意见（如'把温度上限改为500'）— 重新解析")
        print("  [退出] 输入 '取消' — 放弃，不创建任务")
        print()
        current_desc = description  # 累积所有修改意见
        while True:
            try:
                user_input = input("请输入您的选择: ").strip()
            except (EOFError, KeyboardInterrupt):
                print("\n已取消。")
                sys.exit(0)

            if not user_input or user_input in ("确认", "confirm", "yes", "y", "ok", "开始"):
                break
            elif user_input in ("取消", "cancel", "no", "n", "退出", "quit"):
                print("已取消，未创建任务。")
                sys.exit(0)
            else:
                # 用户想要修改
                print(f"\n[调整] 您的修改意见：{user_input}")
                print("正在重新解析...\n")
                try:
                    current_desc = current_desc + f"\n\n[用户修改意见] {user_input}"
                    parsed       = parse_space(client, current_desc)
                except Exception as e:
                    print(f"[错误] 重新解析失败：{e}")
                    continue
                print(format_proposal_table(parsed))
                print()
    else:
        print("[非交互模式] 跳过确认，直接创建任务。\n")

    # 用户确认，创建任务
    print("\n[确认] 正在创建任务...")

    final_task_id = task_id or generate_task_id()
    history = init_history(
        task_id       = final_task_id,
        description   = description or "非交互模式创建的任务",
        params_config = parsed["params_config"],
        objectives    = parsed["objectives"],
        constraints   = parsed.get("constraints", []),
    )
    history_path = save_history(history, data_dir)

    print(f"[OK] 任务已创建！Task_ID: {final_task_id}")
    print(f"[文件] 历史文件：{os.path.abspath(history_path)}")
    print()
    print("下一步：完成一批实验后，使用以下命令提交结果并获取推荐：")

    # 构造示例命令
    obj_names   = [o["name"] for o in parsed["objectives"]]
    param_names = [p["name"] for p in parsed["params_config"]]
    x_example   = {n: "..." for n in param_names}
    y_example   = {n: "..." for n in obj_names}
    print(f"""
  python main.py --mode iterate \\
    --task_id {final_task_id} \\
    --x_new '{json.dumps(x_example, ensure_ascii=False)}' \\
    --y_new '{json.dumps(y_example, ensure_ascii=False)}'
""")

    return final_task_id


# ─────────────────────────────────────────────────────────────────────────────
# Scenario 2：数据迭代与参数推荐
# ─────────────────────────────────────────────────────────────────────────────

def run_iterate(
    task_id: str,
    x_new,          # dict 或 list[dict]
    y_new,          # dict 或 list[dict]
    n_suggest: int = 3,
    data_dir: str = ".",
    output_format: str = "text",
) -> List[Dict[str, Any]]:
    """
    Scenario 2：追加新的实验观测数据，基于全量历史推荐下一组实验参数。
    支持单条（dict）或批量（list[dict]）输入；支持空历史时直接随机推荐。

    当 output_format == "json" 时，stdout 最后一行为 JSON：
      {"suggestions": [...], "task_id": "...", "n_records": N}
    """
    # ── 载入历史 ────────────────────────────────────────────────────────────
    try:
        history = load_history(task_id, data_dir)
    except FileNotFoundError as e:
        print(f"[错误] {e}")
        sys.exit(1)

    n_before = get_record_count(history)

    # ── 统一转为列表，支持批量输入 ──────────────────────────────────────────
    x_list = x_new if isinstance(x_new, list) else [x_new]
    y_list = y_new if isinstance(y_new, list) else [y_new]

    if len(x_list) != len(y_list):
        print(f"[错误] x_new 和 y_new 长度不一致：{len(x_list)} vs {len(y_list)}")
        sys.exit(1)

    # ── 批量追加新记录 ──────────────────────────────────────────────────────
    for x, y in zip(x_list, y_list):
        try:
            history = append_records(history, x, y)
        except ValueError as e:
            print(f"[错误] {e}")
            sys.exit(1)

    n_records = get_record_count(history)
    n_added   = n_records - n_before
    if n_added > 0:
        timestamp = history["records"][-1]["timestamp"]
        print(f"[数据更新] 已追加 {n_added} 条新记录（历史累计 {n_records} 条）({timestamp})")
    else:
        print(f"[数据更新] 无新观测，当前历史累计 {n_records} 条")

    # ── 样本数检查 ──────────────────────────────────────────────────────────
    if n_records < 5:
        print(
            f"[警告] 当前历史数据仅 {n_records} 条 (<5)，"
            "代理模型可能不稳定，推荐结果仅供参考。"
        )

    # ── 保存更新后的历史 ─────────────────────────────────────────────────────
    try:
        history_path = save_history(history, data_dir)
    except Exception as e:
        print(f"❌ [错误] 保存历史文件失败：{e}")
        sys.exit(1)

    # ── 调用 HEBO 推荐 ───────────────────────────────────────────────────────
    print(f"\n[推荐] 正在基于 {n_records} 条历史数据生成 {n_suggest} 个推荐点...\n")

    try:
        if n_records == 0:
            # 空历史：直接随机采样（首次迭代场景）
            from hebo.design_space.design_space import DesignSpace
            from hebo.optimizers.general import GeneralBO
            from contextlib import redirect_stdout
            space = DesignSpace().parse(history["params_config"])
            opt = GeneralBO(
                space=space,
                num_obj=len(history["objectives"]),
                num_constr=0,
                rand_sample=n_suggest,
            )
            with open(os.devnull, "w") as devnull, redirect_stdout(devnull):
                rec_df = opt.suggest(n_suggestions=n_suggest)
        else:
            suggest_next_fn, _ = _load_bo_engine()
            rec_df = suggest_next_fn(
                params_config = history["params_config"],
                records       = history["records"],
                objectives    = history["objectives"],
                n_suggestions = n_suggest,
            )
    except Exception as e:
        print(f"❌ [错误] HEBO 推荐失败：{e}")
        sys.exit(1)

    # ── 输出推荐结果 ─────────────────────────────────────────────────────────
    suggestions = [row.to_dict() for _, row in rec_df.iterrows()]

    if output_format == "json":
        result = {
            "suggestions": suggestions,
            "task_id": task_id,
            "n_records": n_records,
        }
        print(json.dumps(result, ensure_ascii=False))
    else:
        print("[推荐的下一组实验参数]:\n")
        _print_recommendation_table(rec_df)

    # ── 多目标：Pareto 简报 ──────────────────────────────────────────────────
    objectives = history["objectives"]
    if output_format == "text" and len(objectives) >= 2:
        _, compute_pareto_summary_fn = _load_bo_engine()
        pareto_summary = compute_pareto_summary_fn(history["records"], objectives)
        if pareto_summary:
            print(pareto_summary)

    if output_format == "text":
        print(f"\n[文件] 历史记录已更新：{os.path.abspath(history_path)} ({n_records} 条记录)")

    return suggestions


def _print_recommendation_table(rec_df) -> None:
    """将推荐的 DataFrame 输出为 Markdown 表格。"""
    cols  = list(rec_df.columns)
    header = "| # | " + " | ".join(cols) + " |"
    sep    = "|---|" + "|".join(["---"] * len(cols)) + "|"
    print(header)
    print(sep)
    for i, row in rec_df.iterrows():
        vals = []
        for col in cols:
            v = row[col]
            # 数值型：保留 4 位小数；其他类型直接转字符串
            if isinstance(v, float):
                vals.append(f"{v:.4f}")
            else:
                vals.append(str(v))
        print(f"| {i+1} | " + " | ".join(vals) + " |")
    print()


# ─────────────────────────────────────────────────────────────────────────────
# CLI 入口
# ─────────────────────────────────────────────────────────────────────────────

def main():
    # Windows 控制台 UTF-8 兼容
    if sys.platform == "win32":
        import io
        sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding="utf-8", errors="replace")

    parser = argparse.ArgumentParser(
        description="bo4experiment：贝叶斯优化闭环实验助手",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog=__doc__,
    )
    parser.add_argument(
        "--mode", required=True, choices=["init", "iterate"],
        help="运行模式：init（初始化任务）或 iterate（追加数据并推荐）",
    )

    # Scenario 1 参数
    parser.add_argument(
        "--description", default="",
        help="[init 模式] 自然语言实验描述",
    )
    parser.add_argument(
        "--non_interactive", "--yes", action="store_true",
        help="[init 模式] 非交互模式，跳过确认循环（需同时提供 --params_config 和 --objectives）",
    )
    parser.add_argument(
        "--params_config", default="",
        help="[init 模式] 参数空间 JSON 字符串（非交互模式使用）",
    )
    parser.add_argument(
        "--objectives", default="",
        help="[init 模式] 优化目标 JSON 字符串（非交互模式使用）",
    )
    parser.add_argument(
        "--constraints", default="",
        help="[init 模式] 约束条件 JSON 字符串（非交互模式可选）",
    )

    # Scenario 2 参数
    parser.add_argument(
        "--task_id", default="",
        help="[iterate 模式] 任务 ID（由 init 模式生成）",
    )
    parser.add_argument(
        "--x_new", default="",
        help=(
            "[iterate 模式] 新实验的参数值（JSON 字符串或数组）。"
            "单条：'{\"temperature\": 300}'；"
            "批量：'[{\"temperature\": 300}, {\"temperature\": 350}]'；"
            "空历史随机推荐：'[]'"
        ),
    )
    parser.add_argument(
        "--y_new", default="",
        help=(
            "[iterate 模式] 新实验的目标值（JSON 字符串或数组）。"
            "单条：'{\"conversion_rate\": 0.85}'；"
            "批量：'[{\"conversion_rate\": 0.85}, {\"conversion_rate\": 0.71}]'；"
            "空历史随机推荐：'[]'"
        ),
    )
    parser.add_argument(
        "--n_suggest", type=int, default=3,
        help="[iterate 模式] 推荐参数组数，默认 3",
    )

    # 通用参数
    parser.add_argument(
        "--data_dir", default=".",
        help="历史文件存储目录，默认为当前目录",
    )
    parser.add_argument(
        "--format", dest="output_format", default="text", choices=["text", "json"],
        help="输出格式：text（默认，Markdown 表格）或 json（stdout 最后一行输出 JSON）",
    )

    args = parser.parse_args()

    if args.mode == "init":
        params_config = None
        objectives = None
        constraints = None

        if args.params_config:
            try:
                params_config = json.loads(args.params_config)
            except json.JSONDecodeError as e:
                parser.error(f"--params_config 不是合法的 JSON：{e}")

        if args.objectives:
            try:
                objectives = json.loads(args.objectives)
            except json.JSONDecodeError as e:
                parser.error(f"--objectives 不是合法的 JSON：{e}")

        if args.constraints:
            try:
                constraints = json.loads(args.constraints)
            except json.JSONDecodeError as e:
                parser.error(f"--constraints 不是合法的 JSON：{e}")

        if args.non_interactive and (params_config is None or objectives is None):
            parser.error("--non_interactive 必须同时提供 --params_config 和 --objectives")

        if not args.description and (params_config is None or objectives is None):
            parser.error("--mode init 需要提供 --description，或在非交互模式下提供 --params_config 和 --objectives")

        run_init(
            description     = args.description,
            data_dir        = args.data_dir,
            non_interactive = args.non_interactive,
            params_config   = params_config,
            objectives      = objectives,
            constraints     = constraints,
            task_id         = args.task_id or None,
        )

    elif args.mode == "iterate":
        if not args.task_id:
            parser.error("--mode iterate 需要提供 --task_id 参数")

        x_new = []
        if args.x_new:
            try:
                x_new = json.loads(args.x_new)
            except json.JSONDecodeError as e:
                parser.error(f"--x_new 不是合法的 JSON：{e}\n示例：--x_new '{{\"temperature\": 300}}'")

        y_new = []
        if args.y_new:
            try:
                y_new = json.loads(args.y_new)
            except json.JSONDecodeError as e:
                parser.error(f"--y_new 不是合法的 JSON：{e}\n示例：--y_new '{{\"conversion_rate\": 0.85}}'")

        # 统一转换为列表
        if not isinstance(x_new, list):
            x_new = [x_new]
        if not isinstance(y_new, list):
            y_new = [y_new]

        if len(x_new) != len(y_new):
            parser.error("--x_new 和 --y_new 长度不一致")

        run_iterate(
            task_id       = args.task_id,
            x_new         = x_new,
            y_new         = y_new,
            n_suggest     = args.n_suggest,
            data_dir      = args.data_dir,
            output_format = args.output_format,
        )


if __name__ == "__main__":
    main()
