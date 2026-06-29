"""
bo_engine.py — HEBO BO 引擎（纯计算模块，零文件 IO）

核心函数：
  suggest_next(params_config, records, objectives, n_suggestions=3) -> pd.DataFrame
    基于历史实验记录，调用 HEBO 推荐下一组实验参数。

设计原则：
  - 完全无文件 IO，所有数据通过参数传入
  - 每次调用重新实例化 GeneralBO + observe 全量历史（HEBO 不支持增量更新）
  - max 目标自动取负传入 HEBO，输出时不做还原（由 main.py 负责展示）
  - 屏蔽 GPy 内部日志，保持输出整洁
"""

import os
import warnings
import logging
from contextlib import redirect_stdout

import numpy as np
import pandas as pd

warnings.filterwarnings("ignore")
logging.getLogger("GP").setLevel(logging.ERROR)
logging.getLogger("GPy").setLevel(logging.ERROR)

# ─────────────────────────────────────────────────────────────────────────────
# HEBO 导入（带友好错误提示）
# ─────────────────────────────────────────────────────────────────────────────

try:
    from hebo.design_space.design_space import DesignSpace
    from hebo.optimizers.general import GeneralBO
except ImportError as _e:
    raise ImportError(
        "HEBO 未安装或路径错误。\n"
        "请执行：pip install -e path/to/HEBO\n"
        f"原始错误：{_e}"
    )


# ─────────────────────────────────────────────────────────────────────────────
# 核心推荐函数
# ─────────────────────────────────────────────────────────────────────────────

def suggest_next(
    params_config: list,
    records: list,
    objectives: list,
    n_suggestions: int = 3,
) -> pd.DataFrame:
    """
    基于历史实验记录，推荐下一组实验参数。

    Parameters
    ----------
    params_config : list
        HEBO 参数配置列表，格式同 DesignSpace.parse() 的输入。
        例：[{"name": "temperature", "type": "num", "lb": 200, "ub": 400}, ...]

    records : list
        历史实验记录列表，每条记录为 dict，包含：
        - "x": dict，参数名 → 参数值
        - "y": dict，目标名 → 目标值
        例：[{"x": {"temperature": 250}, "y": {"conversion_rate": 0.62}}, ...]

    objectives : list
        优化目标列表，每项为 dict：
        - "name": str，目标名（与 records[i]["y"] 的键对应）
        - "direction": "min" 或 "max"
        例：[{"name": "conversion_rate", "direction": "max"}]

    n_suggestions : int
        推荐点数量，默认 3。

    Returns
    -------
    pd.DataFrame
        推荐的实验参数，列名与 params_config 中的 name 一致。

    Raises
    ------
    ValueError
        当 records 为空，或数据中存在 NaN，或参数名不匹配时。
    """
    if not records:
        raise ValueError("历史记录为空，无法拟合代理模型。请先提供至少 1 条实验数据。")

    # ── 构建 DesignSpace ────────────────────────────────────────────────────
    space = DesignSpace().parse(params_config)
    param_names = [p["name"] for p in params_config]
    obj_names   = [o["name"] for o in objectives]
    n_obj       = len(objectives)

    # ── 构建 X DataFrame ────────────────────────────────────────────────────
    x_rows = []
    for i, rec in enumerate(records):
        x_dict = rec.get("x", {})
        row = {}
        for pname in param_names:
            if pname not in x_dict:
                raise ValueError(
                    f"第 {i+1} 条记录缺少参数 '{pname}'。\n"
                    f"记录内容：{x_dict}\n"
                    f"期望参数：{param_names}"
                )
            row[pname] = x_dict[pname]
        x_rows.append(row)

    X_df = pd.DataFrame(x_rows)

    # ── 校验 X 中无 NaN ─────────────────────────────────────────────────────
    nan_cols = X_df.columns[X_df.isnull().any()].tolist()
    if nan_cols:
        raise ValueError(
            f"参数数据中存在 NaN，列：{nan_cols}。\n"
            "请检查历史记录中对应字段是否有缺失值。"
        )

    # ── 构建 Y 矩阵（HEBO 统一最小化，max 目标取负）────────────────────────
    y_rows = []
    for i, rec in enumerate(records):
        y_dict = rec.get("y", {})
        row = []
        for obj in objectives:
            oname = obj["name"]
            if oname not in y_dict:
                raise ValueError(
                    f"第 {i+1} 条记录缺少目标值 '{oname}'。\n"
                    f"记录内容：{y_dict}\n"
                    f"期望目标：{obj_names}"
                )
            val = float(y_dict[oname])
            # max 目标取负，使 HEBO 最小化等价于最大化原始值
            if obj["direction"] == "max":
                val = -val
            row.append(val)
        y_rows.append(row)

    Y_arr = np.array(y_rows, dtype=float)  # shape: (n_records, n_obj)

    # ── 校验 Y 中无 NaN ─────────────────────────────────────────────────────
    if np.any(np.isnan(Y_arr)):
        nan_indices = np.argwhere(np.isnan(Y_arr))
        raise ValueError(
            f"目标值数据中存在 NaN，位置（行,列）：{nan_indices.tolist()}。\n"
            "请检查历史记录中对应目标字段是否有缺失值。"
        )

    # ── 实例化 GeneralBO 并注入历史数据 ─────────────────────────────────────
    # rand_sample 设为较小值（5），因为已有历史数据，不需要大量随机初始化
    opt = GeneralBO(
        space=space,
        num_obj=n_obj,
        num_constr=0,
        rand_sample=5,
    )

    # 屏蔽 GPy 拟合时的内部日志
    with open(os.devnull, "w") as devnull, redirect_stdout(devnull):
        opt.observe(X_df, Y_arr)

    # ── 生成推荐点 ───────────────────────────────────────────────────────────
    with open(os.devnull, "w") as devnull, redirect_stdout(devnull):
        rec_df = opt.suggest(n_suggestions=n_suggestions)

    return rec_df


# ─────────────────────────────────────────────────────────────────────────────
# Pareto 前沿简报（多目标时使用）
# ─────────────────────────────────────────────────────────────────────────────

def compute_pareto_summary(records: list, objectives: list) -> str:
    """
    基于历史记录计算 Pareto 前沿简报（rank-sum 方法）。

    Parameters
    ----------
    records    : 历史实验记录列表（同 suggest_next 的 records 参数）
    objectives : 优化目标列表（同 suggest_next 的 objectives 参数）

    Returns
    -------
    str : Markdown 格式的 Pareto 简报文本
    """
    if len(objectives) < 2:
        return ""  # 单目标不需要 Pareto 简报

    obj_names = [o["name"] for o in objectives]
    n_obj     = len(objectives)

    # 提取目标值矩阵（原始值，不取负）
    y_rows = []
    for rec in records:
        y_dict = rec.get("y", {})
        row = [float(y_dict.get(oname, float("nan"))) for oname in obj_names]
        y_rows.append(row)

    Y_raw = np.array(y_rows, dtype=float)

    # 过滤含 NaN 的行
    valid_mask = ~np.any(np.isnan(Y_raw), axis=1)
    if valid_mask.sum() < 2:
        return ""

    Y_valid = Y_raw[valid_mask]
    n_valid = len(Y_valid)

    # 转换为最小化方向（max 目标取负）用于 rank 计算
    Y_min = Y_valid.copy()
    for j, obj in enumerate(objectives):
        if obj["direction"] == "max":
            Y_min[:, j] = -Y_min[:, j]

    # rank-sum 方法选 Top-5 均衡解
    ranks    = np.argsort(np.argsort(Y_min, axis=0), axis=0)
    rank_sum = ranks.sum(axis=1)
    top_k    = min(5, n_valid)
    top_idx  = np.argsort(rank_sum)[:top_k]

    lines = []
    lines.append("")
    lines.append(f"[Pareto 前沿简报] ({n_valid} 条有效记录)")
    lines.append("")
    lines.append(f"Top-{top_k} 均衡解 (rank-sum 方法):")
    lines.append("")

    # 表头
    header_cols = ["#", "RankSum"] + [
        f"{o['name']} ({'max' if o['direction']=='max' else 'min'})"
        for o in objectives
    ]
    lines.append("| " + " | ".join(header_cols) + " |")
    lines.append("|" + "|".join(["---"] * len(header_cols)) + "|")

    for rank, sol_i in enumerate(top_idx, 1):
        vals = [f"{Y_valid[sol_i, j]:.4f}" for j in range(n_obj)]
        row  = [str(rank), str(rank_sum[sol_i])] + vals
        lines.append("| " + " | ".join(row) + " |")

    lines.append("")

    # 各目标最优解
    lines.append("各目标最优解:")
    for j, obj in enumerate(objectives):
        if obj["direction"] == "max":
            best_i = int(np.argmax(Y_valid[:, j]))
            best_v = Y_valid[best_i, j]
            lines.append(f"  >> 最大化 {obj['name']}: {best_v:.4f}")
        else:
            best_i = int(np.argmin(Y_valid[:, j]))
            best_v = Y_valid[best_i, j]
            lines.append(f"  >> 最小化 {obj['name']}: {best_v:.4f}")

    return "\n".join(lines)
