#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
tsas-num-prep-resampler: 时间序列数据时间轴治理工具

读取一个 CSV（首列为时间列）+ 一份元信息 MD，执行：
1. 名义采样间隔推断（众数）
2. 目标网格选择（口径 B：对称绝对数，平局选粗）
3. 重采样（下采样取历史最近值 / 上采样前向填充 / 标注列最近邻）
4. 偶发/区间缺失判定（K=10）
5. 偶发缺失填充（不跨区间缺失）
6. 区间缺失标记

输出 resampled_data.csv + resampled_data_report.md。

用法:
    python resampler.py <input_csv> \
        --metadata <metadata.md> \
        [--output-dir <dir>] \
        [--output <csv文件名>] \
        [--report-output <报告文件名>] \
        [--grid-interval <秒>] \
        [--k <连续缺失阈值>]
"""

import argparse
import os
import re
import sys
from collections import Counter, OrderedDict
from datetime import datetime, timedelta
from pathlib import Path

# 输出编码统一为 utf-8（兼容 Windows GBK 终端）
try:
    sys.stdout.reconfigure(encoding="utf-8")
    sys.stderr.reconfigure(encoding="utf-8")
except (AttributeError, OSError):
    pass

# 依赖自检：启动时立刻检查，缺失给出友好提示
def _check_deps():
    missing = []
    try:
        import pandas  # noqa: F401
    except ImportError:
        missing.append("pandas")
    try:
        import numpy  # noqa: F401
    except ImportError:
        missing.append("numpy")
    if missing:
        print(f"[FATAL] 缺少依赖: {', '.join(missing)}", file=sys.stderr)
        print(f"        请确保当前 Python 环境已安装 pandas 和 numpy", file=sys.stderr)
        print(f"        或换用一个已配置的 Python 解释器启动本脚本", file=sys.stderr)
        print(f"        当前 Python: {sys.executable}", file=sys.stderr)
        sys.exit(1)


_check_deps()

import pandas as pd
import numpy as np


# ============================================================
# 常量
# ============================================================

NA_TOKENS = {"", "na", "n/a", "#n/a", "nan", "null", "none", "-", "--", "nil"}

DEFAULT_K = 10


# ============================================================
# 工具函数
# ============================================================

def is_empty(val):
    """判断值是否为空（空字符串或 NA 标记）"""
    if val is None:
        return True
    s = str(val).strip()
    return s == "" or s.lower() in NA_TOKENS


def try_parse_float(val):
    """尝试解析为浮点数"""
    if val is None:
        return None
    s = str(val).strip()
    if not s or s.lower() in NA_TOKENS:
        return None
    try:
        return float(s)
    except ValueError:
        return None


def try_parse_int(val):
    """尝试解析为整数"""
    if val is None:
        return None
    s = str(val).strip()
    if not s:
        return None
    if re.match(r'^[+-]?\d+$', s):
        return int(s)
    return None


# ============================================================
# 元信息解析
# ============================================================

def parse_metadata(metadata_path):
    """
    解析元信息 MD，提取以下字段：

    必须：
      - columns: {col_name: {"role": str, "type": str}}  — 来自 `## 列清单` 表格

    可选：
      - label_cols: List[str]            — 来自 `## 标注列` 列表 / 列清单角色为"标注"
      - grid_interval: float (seconds)   — 来自 `## 目标网格间隔` 单值（CLI 可覆盖）
      - k: int                            — 来自 `## 区间缺失阈值` 单值（CLI 可覆盖）

    返回 dict。
    """
    if not os.path.exists(metadata_path):
        raise FileNotFoundError(f"元信息文件不存在: {metadata_path}")

    with open(metadata_path, "r", encoding="utf-8") as f:
        content = f.read()

    info = {
        "columns": OrderedDict(),  # 列名 -> {role, type}
        "label_cols": [],
        "grid_interval": None,
        "k": None,
    }

    # --- 列清单（必须）---
    col_section = re.search(
        r'##\s*列清单\s*\n(.*?)(?=\n##\s|\Z)', content, re.DOTALL
    )
    if col_section:
        for line in col_section.group(1).split("\n"):
            line = line.strip()
            if not line or not line.startswith("|"):
                continue
            parts = [p.strip() for p in line.strip("|").split("|")]
            if len(parts) < 2:
                continue
            col_name = parts[0].strip().strip("`")
            if col_name in ("列名", "") or col_name.startswith("---") or col_name.startswith(":-"):
                continue
            role = parts[1] if len(parts) >= 2 else ""
            col_type = parts[2] if len(parts) >= 3 else ""
            info["columns"][col_name] = {"role": role, "type": col_type}
            if role and ("标注" in role or "label" in role.lower()):
                if col_name not in info["label_cols"]:
                    info["label_cols"].append(col_name)

    # --- 标注列（可选）---
    label_section = re.search(
        r'##\s*标注列\s*\n(.*?)(?=\n##\s|\Z)', content, re.DOTALL
    )
    if label_section:
        for line in label_section.group(1).split("\n"):
            line = line.strip()
            if not line:
                continue
            m = re.match(r'^[-*]\s*`?([^`\s]+)`?\s*$', line)
            if m:
                col = m.group(1).strip()
                if col and col not in info["label_cols"]:
                    info["label_cols"].append(col)

    # --- 目标网格间隔（可选）---
    grid_section = re.search(
        r'##\s*目标网格间隔\s*\n(.*?)(?=\n##\s|\Z)', content, re.DOTALL
    )
    if grid_section:
        text = grid_section.group(1).strip()
        m = re.search(r'(\d+(?:\.\d+)?)', text)
        if m:
            info["grid_interval"] = float(m.group(1))

    # --- 区间缺失阈值（可选）---
    k_section = re.search(
        r'##\s*区间缺失阈值\s*\n(.*?)(?=\n##\s|\Z)', content, re.DOTALL
    )
    if k_section:
        text = k_section.group(1).strip()
        m = re.search(r'(\d+)', text)
        if m:
            info["k"] = int(m.group(1))

    return info


# ============================================================
# 1. 名义采样间隔推断
# ============================================================

def infer_nominal_interval(times):
    """对相邻时间戳差值取众数。"""
    if len(times) < 2:
        return timedelta(seconds=1)

    diffs = times.diff().dropna()
    if len(diffs) == 0:
        return timedelta(seconds=1)

    mode_diff = diffs.mode()
    if len(mode_diff) > 0:
        return mode_diff.iloc[0]
    return diffs.iloc[0]


# ============================================================
# 2. 目标网格选择
# ============================================================

def select_target_grid(times, user_interval, columns_data):
    """
    目标网格选择（口径B：逐列单元格修改数据点数最少，平局选粗）。
    候选间隔来源：各列（非空值）的众数间隔集合。
    """
    if user_interval:
        target_td = timedelta(seconds=user_interval)
        return target_td, {
            "method": "用户指定",
            "interval": target_td,
            "candidates": [],
        }

    candidate_intervals = set()
    col_time_sets = {}

    for col, values in columns_data.items():
        col_times = []
        for i, v in enumerate(values):
            if not is_empty(v):
                col_times.append(times.iloc[i])

        if len(col_times) < 2:
            if col_times:
                col_time_sets[col] = set(col_times)
            continue

        col_time_sets[col] = set(col_times)

        col_series = pd.Series(col_times)
        col_diffs = col_series.diff().dropna()
        if len(col_diffs) == 0:
            continue

        mode_diff = col_diffs.mode()
        if len(mode_diff) > 0:
            candidate_intervals.add(mode_diff.iloc[0])

    if not candidate_intervals:
        diffs = times.diff().dropna()
        if len(diffs) > 0:
            mode_diff = diffs.mode()
            if len(mode_diff) > 0:
                candidate_intervals.add(mode_diff.iloc[0])

    if not candidate_intervals:
        return timedelta(seconds=1), {
            "method": "默认（数据不足）",
            "interval": timedelta(seconds=1),
            "candidates": [],
        }

    unique_diffs = sorted(candidate_intervals)

    time_min = times.iloc[0]
    time_max = times.iloc[-1]

    candidates = []
    for cand_td in unique_diffs:
        grid_points = _generate_grid(time_min, time_max, cand_td)
        grid_set = set(grid_points)

        total_to_add = 0
        total_to_delete = 0
        col_details = []

        for col, col_ts_set in col_time_sets.items():
            aligned = len(col_ts_set & grid_set)
            to_add = len(grid_set) - aligned
            to_delete = len(col_ts_set) - aligned

            total_to_add += to_add
            total_to_delete += to_delete

            col_details.append({
                "col": col,
                "non_null": len(col_ts_set),
                "to_add": to_add,
                "to_delete": to_delete,
            })

        total_modifications = total_to_add + total_to_delete

        candidates.append({
            "interval": cand_td,
            "total_seconds": cand_td.total_seconds(),
            "grid_points": len(grid_points),
            "to_delete": total_to_delete,
            "to_add": total_to_add,
            "total_modifications": total_modifications,
            "col_details": col_details,
        })

    candidates.sort(key=lambda c: (c["total_modifications"], -c["total_seconds"]))

    best = candidates[0]
    return best["interval"], {
        "method": "口径B自动选择（修改数据点数最少，平局选粗）",
        "interval": best["interval"],
        "candidates": candidates,
    }


def _generate_grid(time_min, time_max, interval):
    """生成从 time_min 到 time_max 的等间隔网格点列表"""
    grid = []
    current = time_min
    while current <= time_max:
        grid.append(current)
        current += interval
    return grid


# ============================================================
# 3. 重采样
# ============================================================

def resample_data(df, target_interval, label_cols):
    """按目标网格重采样所有列。返回: (resampled_df, resample_info)"""
    times = pd.to_datetime(df["time"])
    time_min = times.iloc[0]
    time_max = times.iloc[-1]

    grid = _generate_grid(time_min, time_max, target_interval)

    resample_info = {}
    result_data = {"time": grid}

    for col in df.columns:
        if col == "time":
            continue

        col_values = df[col].tolist()
        pairs = []
        for i, v in enumerate(col_values):
            if not is_empty(v):
                pairs.append((times.iloc[i], v))

        if not pairs:
            result_data[col] = [""] * len(grid)
            resample_info[col] = {"method": "全空", "details": "原始数据无有效值"}
            continue

        col_times = [p[0] for p in pairs]
        if len(col_times) >= 2:
            col_diffs = pd.Series(col_times).diff().dropna()
            col_mode_interval = col_diffs.mode()
            if len(col_mode_interval) > 0:
                col_interval = col_mode_interval.iloc[0]
            else:
                col_interval = col_diffs.iloc[0]
        else:
            col_interval = target_interval

        is_downsample = col_interval < target_interval
        is_upsample = col_interval > target_interval

        is_label = col in label_cols

        if is_downsample:
            method = "下采样（取历史最近值）"
            result_data[col] = _downsample_history_nearest(pairs, grid)
        elif is_upsample:
            if is_label:
                method = "上采样（标注列最近邻）"
                result_data[col] = _upsample_label_nearest(pairs, grid)
            else:
                method = "上采样（前向填充）"
                result_data[col] = _upsample_forward_fill(pairs, grid)
        else:
            method = "网格对齐（间隔一致）"
            result_data[col] = _align_to_grid(pairs, grid)

        resample_info[col] = {"method": method, "original_interval": str(col_interval)}

    resampled_df = pd.DataFrame(result_data)
    return resampled_df, resample_info


def _downsample_history_nearest(pairs, grid):
    """下采样：取历史上 <=t 的最大时间点对应的值。"""
    sorted_pairs = sorted(pairs, key=lambda x: x[0])
    orig_times = [p[0] for p in sorted_pairs]
    orig_values = [p[1] for p in sorted_pairs]

    result = []
    for t in grid:
        import bisect
        pos = bisect.bisect_right(orig_times, t) - 1
        if pos >= 0:
            result.append(orig_values[pos])
        else:
            result.append(orig_values[0])
    return result


def _upsample_forward_fill(pairs, grid):
    """上采样：前向填充。"""
    sorted_pairs = sorted(pairs, key=lambda x: x[0])
    orig_times = [p[0] for p in sorted_pairs]
    orig_values = [p[1] for p in sorted_pairs]

    result = []
    for t in grid:
        import bisect
        pos = bisect.bisect_right(orig_times, t) - 1
        if pos >= 0:
            result.append(orig_values[pos])
        else:
            result.append(orig_values[0])
    return result


def _upsample_label_nearest(pairs, grid):
    """标注列上采样：最近邻（双向），前后等距取前值。"""
    sorted_pairs = sorted(pairs, key=lambda x: x[0])
    orig_times = [p[0] for p in sorted_pairs]
    orig_values = [p[1] for p in sorted_pairs]

    result = []
    for t in grid:
        import bisect
        pos = bisect.bisect_left(orig_times, t)

        if pos < len(orig_times) and orig_times[pos] == t:
            result.append(orig_values[pos])
            continue

        prev_pos = pos - 1
        next_pos = pos

        prev_exists = prev_pos >= 0
        next_exists = next_pos < len(orig_times)

        if prev_exists and next_exists:
            dist_prev = (t - orig_times[prev_pos]).total_seconds()
            dist_next = (orig_times[next_pos] - t).total_seconds()
            if dist_prev <= dist_next:
                result.append(orig_values[prev_pos])
            else:
                result.append(orig_values[next_pos])
        elif prev_exists:
            result.append(orig_values[prev_pos])
        elif next_exists:
            result.append(orig_values[next_pos])
        else:
            result.append("")

    return result


def _align_to_grid(pairs, grid):
    """间隔一致时对齐到网格：找最近的原始值"""
    sorted_pairs = sorted(pairs, key=lambda x: x[0])
    orig_times = [p[0] for p in sorted_pairs]
    orig_values = [p[1] for p in sorted_pairs]

    result = []
    for t in grid:
        import bisect
        pos = bisect.bisect_left(orig_times, t)
        if pos < len(orig_times) and orig_times[pos] == t:
            result.append(orig_values[pos])
        elif pos > 0 and pos < len(orig_times):
            dist_prev = (t - orig_times[pos - 1]).total_seconds()
            dist_next = (orig_times[pos] - t).total_seconds()
            if dist_prev <= dist_next:
                result.append(orig_values[pos - 1])
            else:
                result.append(orig_values[pos])
        elif pos == 0:
            result.append(orig_values[0])
        else:
            result.append(orig_values[-1])
    return result


# ============================================================
# 4. 偶发/区间缺失判定
# ============================================================

def classify_missing(resampled_df, label_cols, k):
    grid_times = pd.to_datetime(resampled_df["time"])

    interval_gaps = []
    sporadic_info = {}

    for col in resampled_df.columns:
        if col == "time":
            continue

        values = resampled_df[col].tolist()
        is_missing = [is_empty(v) for v in values]

        i = 0
        col_sporadic = 0
        while i < len(is_missing):
            if not is_missing[i]:
                i += 1
                continue

            j = i
            while j < len(is_missing) and is_missing[j]:
                j += 1
            gap_length = j - i

            if gap_length >= k:
                interval_gaps.append({
                    "col": col,
                    "start_idx": i,
                    "end_idx": j - 1,
                    "length": gap_length,
                    "start_time": str(grid_times.iloc[i]),
                    "end_time": str(grid_times.iloc[j - 1]),
                })
            else:
                col_sporadic += gap_length

            i = j

        sporadic_info[col] = col_sporadic

    return interval_gaps, sporadic_info


# ============================================================
# 5. 偶发缺失填充（不跨区间缺失）
# ============================================================

def fill_sporadic(resampled_df, interval_gaps, label_cols):
    grid_times = pd.to_datetime(resampled_df["time"])

    gap_indices = {}
    for gap in interval_gaps:
        col = gap["col"]
        if col not in gap_indices:
            gap_indices[col] = set()
        for idx in range(gap["start_idx"], gap["end_idx"] + 1):
            gap_indices[col].add(idx)

    fill_info = {}
    filled_df = resampled_df.copy()

    for col in filled_df.columns:
        if col == "time":
            continue

        is_label = col in label_cols
        values = filled_df[col].tolist()
        col_gaps = gap_indices.get(col, set())

        fill_count = 0
        for i in range(len(values)):
            if not is_empty(values[i]):
                continue
            if i in col_gaps:
                continue

            if is_label:
                new_val = _find_nearest_value(values, i, col_gaps)
            else:
                new_val = _find_forward_value(values, i, col_gaps)

            if new_val is not None:
                values[i] = new_val
                fill_count += 1

        filled_df[col] = values
        fill_info[col] = fill_count

    return filled_df, fill_info


def _find_forward_value(values, target_idx, gap_indices):
    for i in range(target_idx - 1, -1, -1):
        if i in gap_indices:
            return None
        if not is_empty(values[i]):
            return values[i]
    for i in range(target_idx + 1, len(values)):
        if i in gap_indices:
            return None
        if not is_empty(values[i]):
            return values[i]
    return None


def _find_nearest_value(values, target_idx, gap_indices):
    prev_val = None
    prev_dist = float('inf')
    for i in range(target_idx - 1, -1, -1):
        if i in gap_indices:
            break
        if not is_empty(values[i]):
            prev_val = values[i]
            prev_dist = target_idx - i
            break

    next_val = None
    next_dist = float('inf')
    for i in range(target_idx + 1, len(values)):
        if i in gap_indices:
            break
        if not is_empty(values[i]):
            next_val = values[i]
            next_dist = i - target_idx
            break

    if prev_val is None and next_val is None:
        return None
    if prev_val is None:
        return next_val
    if next_val is None:
        return prev_val
    return prev_val if prev_dist <= next_dist else next_val


# ============================================================
# 报告生成
# ============================================================

def generate_report(resampled_df, nominal_interval, grid_selection, resample_info,
                    interval_gaps, sporadic_info, fill_info, label_cols,
                    output_path, input_csv, k):
    lines = []
    lines.append("# 重采样报告（tsas-num-prep-resampler）\n")
    lines.append(f"> 生成时间：{datetime.now().strftime('%Y-%m-%d %H:%M:%S')}\n")

    lines.append("## 1. 概览\n")
    lines.append(f"- **输入文件**：{os.path.basename(input_csv)}")
    lines.append(f"- **名义采样间隔（众数）**：{nominal_interval}")
    lines.append(f"- **目标网格间隔**：{grid_selection['interval']}")
    lines.append(f"- **网格选择方法**：{grid_selection['method']}")
    lines.append(f"- **网格点数（行数）**：{len(resampled_df)}")
    lines.append(f"- **列数**：{len(resampled_df.columns)}")
    lines.append(f"- **区间缺失判定阈值 K**：{k}")
    lines.append(f"- **标注列**：{', '.join(label_cols) if label_cols else '无'}")
    lines.append("")

    lines.append("## 2. 网格选择过程\n")
    candidates = grid_selection.get("candidates", [])
    if candidates:
        lines.append("| 候选间隔 | 网格点数 | 需删除数据点 | 需新增数据点 | 总修改数据点数 |")
        lines.append("|---|---|---|---|---|")
        for c in candidates:
            is_best = c["interval"] == grid_selection["interval"]
            marker = " ← **选中**" if is_best else ""
            lines.append(
                f"| {c['interval']}{marker} | {c['grid_points']} | "
                f"{c['to_delete']} | {c['to_add']} | {c['total_modifications']} |"
            )
        lines.append("")
        lines.append("**逐列明细：**\n")
        for c in candidates:
            is_best = c["interval"] == grid_selection["interval"]
            marker = " ← **选中**" if is_best else ""
            lines.append(f"- 候选间隔 `{c['interval']}`（网格 {c['grid_points']} 点）{marker}：")
            lines.append("")
            lines.append("  | 列名 | 非空值数 | 需新增 | 需删除 | 列修改数据点数 |")
            lines.append("  |---|---|---|---|---|")
            for cd in c.get("col_details", []):
                col_mod = cd["to_add"] + cd["to_delete"]
                lines.append(
                    f"  | {cd['col']} | {cd['non_null']} | "
                    f"{cd['to_add']} | {cd['to_delete']} | {col_mod} |"
                )
            lines.append("")
    else:
        lines.append("用户指定间隔或数据不足，未执行自动选择。")
    lines.append("")

    lines.append("## 3. 各列重采样方法\n")
    lines.append("| 列名 | 角色 | 重采样方法 | 原始间隔 |")
    lines.append("|---|---|---|---|")
    for col in resampled_df.columns:
        if col == "time":
            continue
        ri = resample_info.get(col, {})
        method = ri.get("method", "-")
        orig_int = ri.get("original_interval", "-")
        role = "标注列" if col in label_cols else "业务数据"
        lines.append(f"| {col} | {role} | {method} | {orig_int} |")
    lines.append("")

    lines.append("## 4. 缺失判定与填充\n")
    lines.append("| 列名 | 偶发缺失数 | 偶发已填充数 | 区间缺失段数 | 区间缺失总长度 |")
    lines.append("|---|---|---|---|---|")
    for col in resampled_df.columns:
        if col == "time":
            continue
        sporadic = sporadic_info.get(col, 0)
        filled = fill_info.get(col, 0)
        col_gaps = [g for g in interval_gaps if g["col"] == col]
        gap_count = len(col_gaps)
        gap_total = sum(g["length"] for g in col_gaps)
        lines.append(f"| {col} | {sporadic} | {filled} | {gap_count} | {gap_total} |")
    lines.append("")

    lines.append("## 5. 区间缺失位置\n")
    if interval_gaps:
        lines.append("| 列名 | 起始时间 | 结束时间 | 连续缺失长度 |")
        lines.append("|---|---|---|---|")
        for gap in interval_gaps:
            lines.append(
                f"| {gap['col']} | {gap['start_time']} | {gap['end_time']} | {gap['length']} |"
            )
    else:
        lines.append("无区间缺失。")
    lines.append("")

    lines.append("## 6. 错误/警告\n")
    lines.append("无错误和警告。" if not interval_gaps
                 else f"共有 {len(interval_gaps)} 处区间缺失（已标记，留待下游切分）。")
    lines.append("")

    report_text = "\n".join(lines)
    with open(output_path, "w", encoding="utf-8") as f:
        f.write(report_text)

    return report_text


# ============================================================
# 主流程
# ============================================================

def main():
    parser = argparse.ArgumentParser(description="时间序列数据时间轴治理工具")
    parser.add_argument("input_csv", help="治理后的 CSV 文件路径")
    parser.add_argument("--metadata", "-m", required=True,
                        help="元信息文件路径（MD 格式）")
    parser.add_argument("--output-dir", "-o", default=None,
                        help="输出目录（缺省为输入文件同目录）")
    parser.add_argument("--output", default=None,
                        help="重采样数据输出文件名（缺省 resampled_data.csv）")
    parser.add_argument("--report-output", default=None,
                        help="报告输出文件名（缺省 resampled_data_report.md）")
    parser.add_argument("--grid-interval", type=float, default=None,
                        help="目标网格间隔（秒），覆盖自动选择")
    parser.add_argument("--k", type=int, default=None,
                        help=f"区间缺失判定阈值（缺省 {DEFAULT_K}）")

    args = parser.parse_args()

    # 解析元信息
    print(f"读取元信息：{args.metadata}")
    meta = parse_metadata(args.metadata)
    label_cols = meta["label_cols"]
    columns_info = meta["columns"]
    print(f"  列清单：{', '.join(columns_info.keys())}")
    print(f"  标注列：{label_cols}")

    if not columns_info:
        print("[FATAL] 元信息中缺少 `## 列清单`，无法确定列角色", file=sys.stderr)
        sys.exit(2)

    # 元信息中的 grid_interval 和 k 可以被 CLI 覆盖
    user_grid_interval = args.grid_interval or meta.get("grid_interval")
    user_k = args.k if args.k is not None else (meta.get("k") or DEFAULT_K)

    # 确定输出目录
    input_path = Path(args.input_csv)
    if args.output_dir:
        output_dir = Path(args.output_dir)
    else:
        output_dir = input_path.parent
    output_dir.mkdir(parents=True, exist_ok=True)

    csv_name = args.output or "resampled_data.csv"
    report_name = args.report_output or "resampled_data_report.md"

    # 读取数据
    print(f"\n读取治理后数据：{args.input_csv}")
    df = pd.read_csv(args.input_csv, dtype=str, keep_default_na=False, na_values=[])
    print(f"  行数：{len(df)}，列数：{len(df.columns)}")

    times = pd.to_datetime(df["time"])
    df["time"] = times

    # 1. 名义采样间隔推断
    print("\n[1/5] 名义采样间隔推断...")
    nominal_interval = infer_nominal_interval(times)
    print(f"  名义间隔（众数）：{nominal_interval}")

    # 2. 目标网格选择
    print("[2/5] 目标网格选择...")
    columns_data = {}
    for col in df.columns:
        if col != "time":
            columns_data[col] = df[col].tolist()

    target_interval, grid_selection = select_target_grid(
        times, user_grid_interval, columns_data
    )
    print(f"  目标网格间隔：{target_interval}")
    print(f"  选择方法：{grid_selection['method']}")
    if grid_selection.get("candidates"):
        for c in grid_selection["candidates"][:3]:
            print(f"    候选 {c['interval']}: 修改{c['total_modifications']}个数据点"
                  f"（删{c['to_delete']}+增{c['to_add']}）")

    # 3. 重采样
    print("\n[3/5] 重采样...")
    df["time"] = df["time"].astype(str)
    resampled_df, resample_info = resample_data(df, target_interval, set(label_cols))
    print(f"  重采样后行数：{len(resampled_df)}")

    for col, info in resample_info.items():
        if col != "time":
            print(f"    {col}: {info['method']}")

    # 4. 偶发/区间缺失判定
    print(f"\n[4/5] 缺失判定（K={user_k}）...")
    interval_gaps, sporadic_info = classify_missing(resampled_df, set(label_cols), user_k)
    total_gaps = len(interval_gaps)
    total_sporadic = sum(sporadic_info.values())
    print(f"  区间缺失：{total_gaps} 段")
    print(f"  偶发缺失：{total_sporadic} 个")

    # 5. 偶发缺失填充
    print("\n[5/5] 偶发缺失填充...")
    filled_df, fill_info = fill_sporadic(resampled_df, interval_gaps, set(label_cols))
    total_filled = sum(fill_info.values())
    print(f"  已填充：{total_filled} 个")

    # 写入输出
    csv_path = str(output_dir / csv_name)
    filled_df.to_csv(csv_path, index=False, encoding="utf-8-sig")
    print(f"\n重采样数据已写入：{csv_path}")
    print(f"  行数：{len(filled_df)}，列数：{len(filled_df.columns)}")

    # 生成报告
    report_path = str(output_dir / report_name)
    generate_report(filled_df, nominal_interval, grid_selection, resample_info,
                    interval_gaps, sporadic_info, fill_info, label_cols,
                    report_path, args.input_csv, user_k)
    print(f"重采样报告已生成：{report_path}")

    return 0


if __name__ == "__main__":
    sys.exit(main())