#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
tsas-num-prep-exporter: 时间序列数据输出归一化工具

读取一个 CSV（首列为时间列）+ 一份元信息 MD，执行输出归一化：
1. 区间缺失切分（在全空行处截断，产出多个连续段）
2. 列排序（time → 标注列 → 业务数据，按元信息中的列清单顺序）
3. 最终质量校验（空值检查、行数检查、时间等间隔检查）
4. 生成最终预处理报告

输出 final_data.csv（或多个 final_data_XXX.csv）+ final_report.md。

用法:
    python exporter.py <input_csv> \
        --metadata <metadata.md> \
        [--output-dir <dir>] \
        [--output <csv文件名>] \
        [--report-output <报告文件名>] \
        [--no-split]
"""

import argparse
import os
import re
import sys
from collections import OrderedDict
from datetime import datetime
from pathlib import Path

# 输出编码统一为 utf-8（兼容 Windows GBK 终端）
try:
    sys.stdout.reconfigure(encoding="utf-8")
    sys.stderr.reconfigure(encoding="utf-8")
except (AttributeError, OSError):
    pass

# 依赖自检：启动时立刻检查，缺失给出友好提示
def _check_deps():
    try:
        import pandas  # noqa: F401
    except ImportError:
        print("[FATAL] 缺少依赖: pandas", file=sys.stderr)
        print("        请确保当前 Python 环境已安装 pandas", file=sys.stderr)
        print("        或换用一个已配置的 Python 解释器启动本脚本", file=sys.stderr)
        print(f"        当前 Python: {sys.executable}", file=sys.stderr)
        sys.exit(1)


_check_deps()

import pandas as pd


# ============================================================
# 工具函数
# ============================================================

NA_TOKENS = {"", "na", "n/a", "#n/a", "nan", "null", "none", "-", "--", "nil"}

def is_empty(val):
    """判断值是否为空"""
    if val is None:
        return True
    s = str(val).strip()
    return s == "" or s.lower() in NA_TOKENS


# ============================================================
# 元信息解析
# ============================================================

def parse_metadata(metadata_path):
    """
    解析元信息 MD，提取以下字段：

    必须：
      - columns: OrderedDict{col_name: {"role": str, "source": str}}
                  — 来自 `## 列清单` 表格，**顺序敏感**（决定最终列序）

    可选（仅用于报告展示，不影响业务逻辑）：
      - label_cols: List[str]       — `## 列清单` 中角色为"标注"的列 + `## 标注列` 显式列表
      - flow_summary: List[Dict]    — 来自 `## 全流程摘要` 表格
      - notes: str                  — 来自 `## 备注` 自由文本

    返回 dict。
    """
    if not os.path.exists(metadata_path):
        raise FileNotFoundError(f"元信息文件不存在: {metadata_path}")

    with open(metadata_path, "r", encoding="utf-8") as f:
        content = f.read()

    info = {
        "columns": OrderedDict(),
        "label_cols": [],
        "flow_summary": [],
        "notes": "",
    }

    # --- 列清单（必须，顺序敏感）---
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
            source = parts[2] if len(parts) >= 3 else ""
            info["columns"][col_name] = {"role": role, "source": source}
            if role and ("标注" in role or "label" in role.lower()):
                if col_name not in info["label_cols"]:
                    info["label_cols"].append(col_name)

    # --- 标注列（可选，独立小节）---
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

    # --- 全流程摘要（可选，仅用于报告展示）---
    flow_section = re.search(
        r'##\s*全流程摘要\s*\n(.*?)(?=\n##\s|\Z)', content, re.DOTALL
    )
    if flow_section:
        for line in flow_section.group(1).split("\n"):
            line = line.strip()
            if not line or not line.startswith("|"):
                continue
            parts = [p.strip() for p in line.strip("|").split("|")]
            if len(parts) < 2:
                continue
            step = parts[0].strip()
            if step in ("步骤", "") or step.startswith("---") or step.startswith(":-"):
                continue
            info["flow_summary"].append({
                "step": step,
                "skill": parts[1] if len(parts) >= 2 else "",
                "result": parts[2] if len(parts) >= 3 else "",
            })

    # --- 备注（可选）---
    notes_section = re.search(
        r'##\s*备注\s*\n(.*?)(?=\n##\s|\Z)', content, re.DOTALL
    )
    if notes_section:
        info["notes"] = notes_section.group(1).strip()

    return info


# ============================================================
# 1. 区间缺失切分
# ============================================================

def split_by_full_empty_rows(df):
    """
    根据全空行切分数据。直接扫描数据找全空行（所有非时间列都为空），
    不依赖任何外部报告。单列区间缺失不触发切分。

    返回: (segments, split_info)
    """
    business_cols = [c for c in df.columns if c != "time"]

    # 直接在数据中找全空行
    cut_indices = set()
    for idx in range(len(df)):
        row_all_empty = all(is_empty(df[col].iloc[idx]) for col in business_cols)
        if row_all_empty:
            cut_indices.add(idx)

    if not cut_indices:
        # 无全空行 → 整体一个段
        return [df], {
            "total_segments": 1,
            "details": [{"segment": 1, "start": str(df["time"].iloc[0]),
                         "end": str(df["time"].iloc[-1]), "rows": len(df)}],
        }

    # 按切分点分段
    sorted_cuts = sorted(cut_indices)
    segments = []
    segment_start = 0

    for cut_idx in sorted_cuts:
        # cut_idx 是区间缺失的第一行
        # 向后扫描找到连续缺失段的结尾
        end_of_gap = cut_idx
        while end_of_gap + 1 < len(df) and end_of_gap + 1 in cut_indices:
            end_of_gap += 1

        # 段 = [segment_start, cut_idx - 1]
        if cut_idx > segment_start:
            seg = df.iloc[segment_start:cut_idx].reset_index(drop=True)
            if len(seg) > 0:
                segments.append(seg)

        # 跳过缺失段
        segment_start = end_of_gap + 1

    # 最后一段
    if segment_start < len(df):
        seg = df.iloc[segment_start:].reset_index(drop=True)
        segments.append(seg)

    details = []
    for i, seg in enumerate(segments):
        details.append({
            "segment": i + 1,
            "start": str(seg["time"].iloc[0]),
            "end": str(seg["time"].iloc[-1]),
            "rows": len(seg),
        })

    return segments, {
        "total_segments": len(segments),
        "details": details,
        "cut_points": len(sorted_cuts),
    }


# ============================================================
# 2. 列排序
# ============================================================

def reorder_columns(df, columns_meta, label_cols):
    """
    列排序：time → 标注列（按元信息中顺序）→ 业务数据（按元信息中顺序）

    columns_meta: OrderedDict{col_name: {"role", "source"}}
                  — 顺序敏感，决定业务数据列的排列顺序
    label_cols: 标注列名集合（用于把标注列提到 time 之后）

    返回: 排序后的 DataFrame
    """
    cols = list(df.columns)
    ordered = []

    # 1. time 列在最前
    if "time" in cols:
        ordered.append("time")
        cols.remove("time")

    # 2. 标注列（按元信息中出现顺序）
    for col in columns_meta:
        if col in label_cols and col in cols:
            ordered.append(col)
            cols.remove(col)

    # 3. 元信息中显式声明的剩余列（按元信息顺序）
    for col in columns_meta:
        if col in cols and col != "time":
            ordered.append(col)
            cols.remove(col)

    # 4. 未在元信息中出现的列，保持原顺序追加
    ordered.extend(cols)

    return df[ordered]


# ============================================================
# 3. 最终质量校验
# ============================================================

def quality_check(df):
    """最终质量校验"""
    results = []

    # 空值检查
    for col in df.columns:
        if col == "time":
            continue
        empty_count = sum(1 for v in df[col] if is_empty(v))
        if empty_count > 0:
            results.append(("warning", f"列 `{col}` 仍有 {empty_count} 个空值"))

    # 时间列检查
    times = pd.to_datetime(df["time"])
    if len(times) < 2:
        results.append(("error", "数据行数不足 2 行，无法校验时间间隔"))
    else:
        diffs = times.diff().dropna()
        unique_diffs = diffs.unique()
        if len(unique_diffs) > 1:
            results.append(("warning", f"时间轴不等间隔：发现 {len(unique_diffs)} 种不同间隔"))
        elif len(unique_diffs) == 1:
            results.append(("info", f"时间轴等间隔：{unique_diffs[0]}"))

    # 行数检查
    results.append(("info", f"最终行数：{len(df)}，列数：{len(df.columns)}"))

    return results


# ============================================================
# 4. 最终报告生成
# ============================================================

def generate_final_report(segments, split_info, qc_results, columns_meta,
                          label_cols, final_column_order, flow_summary,
                          notes, output_path, input_csv, output_files):
    """生成最终预处理报告"""
    lines = []
    lines.append("# 最终预处理报告（tsas-num-prep-exporter）\n")
    lines.append(f"> 生成时间：{datetime.now().strftime('%Y-%m-%d %H:%M:%S')}\n")

    # 1. 全流程摘要（如果元信息提供了 `## 全流程摘要`）
    if flow_summary:
        lines.append("## 1. 全流程摘要\n")
        lines.append("| 步骤 | SKILL | 关键结果 |")
        lines.append("|---|---|---|")
        for fs in flow_summary:
            lines.append(f"| {fs['step']} | {fs['skill']} | {fs['result']} |")
        lines.append("")

    # 2. 输出段信息
    section_idx = 2 if flow_summary else 1
    lines.append(f"## {section_idx}. 输出段信息\n")
    if split_info["total_segments"] == 1:
        d = split_info["details"][0]
        lines.append(f"- **段数**：1（连续无切分）")
        lines.append(f"- **时间范围**：{d['start']} ~ {d['end']}")
        lines.append(f"- **行数**：{d['rows']}")
        if "note" in split_info:
            lines.append(f"- **备注**：{split_info['note']}")
    else:
        lines.append(f"- **段数**：{split_info['total_segments']}")
        lines.append(f"- **切分原因**：全空行断档（{split_info.get('cut_points', 0)} 个切分点）")
        lines.append("")
        lines.append("| 段号 | 起始时间 | 结束时间 | 行数 |")
        lines.append("|---|---|---|---|")
        for d in split_info["details"]:
            lines.append(f"| {d['segment']} | {d['start']} | {d['end']} | {d['rows']} |")
    lines.append("")

    # 3. 最终列清单
    section_idx += 1
    lines.append(f"## {section_idx}. 最终列清单\n")
    lines.append("| 顺序 | 列名 | 角色 | 来源 |")
    lines.append("|---|---|---|---|")
    for i, col in enumerate(final_column_order):
        if col == "time":
            role = "时间列"
            source = "-"
        elif col in label_cols:
            role = "标注列"
            source = columns_meta.get(col, {}).get("source", "-") or "-"
        else:
            meta = columns_meta.get(col, {})
            role = meta.get("role") or "业务数据"
            source = meta.get("source", "-") or "-"
        lines.append(f"| {i + 1} | `{col}` | {role} | {source} |")
    lines.append("")

    # 4. 质量校验
    section_idx += 1
    lines.append(f"## {section_idx}. 质量校验\n")
    if not qc_results:
        lines.append("无校验项。")
    else:
        lines.append("| 级别 | 信息 |")
        lines.append("|---|---|")
        for level, msg in qc_results:
            marker = {"error": "❌", "warning": "⚠️", "info": "✅"}.get(level, "")
            lines.append(f"| {marker} {level} | {msg} |")
    lines.append("")

    # 5. 输出文件
    section_idx += 1
    lines.append(f"## {section_idx}. 输出文件\n")
    for f in output_files:
        lines.append(f"- `{os.path.basename(f)}`")
    lines.append("")

    # 6. 错误/警告
    section_idx += 1
    lines.append(f"## {section_idx}. 错误/警告\n")
    errors = [m for l, m in qc_results if l == "error"]
    warnings = [m for l, m in qc_results if l == "warning"]
    if errors:
        for e in errors:
            lines.append(f"- ❌ {e}")
    if warnings:
        for w in warnings:
            lines.append(f"- ⚠️ {w}")
    if not errors and not warnings:
        lines.append("无错误和警告。")
    lines.append("")

    # 7. 备注（如有）
    if notes:
        section_idx += 1
        lines.append(f"## {section_idx}. 备注\n")
        lines.append(notes)
        lines.append("")

    report_text = "\n".join(lines)
    with open(output_path, "w", encoding="utf-8") as f:
        f.write(report_text)

    return report_text


# ============================================================
# 主流程
# ============================================================

def main():
    parser = argparse.ArgumentParser(description="时间序列数据输出归一化工具")
    parser.add_argument("input_csv", help="输入 CSV 文件路径（首列为 time）")
    parser.add_argument("--metadata", "-m", required=True,
                        help="元信息文件路径（MD 格式）")
    parser.add_argument("--output-dir", "-o", default=None,
                        help="输出目录（缺省为输入文件同目录）")
    parser.add_argument("--output", default=None,
                        help="最终数据输出文件名（缺省 final_data.csv）")
    parser.add_argument("--report-output", default=None,
                        help="最终报告输出文件名（缺省 final_report.md）")
    parser.add_argument("--no-split", action="store_true",
                        help="禁用区间缺失切分（保留为整段）")

    args = parser.parse_args()

    # 解析元信息
    print(f"读取元信息：{args.metadata}")
    try:
        meta = parse_metadata(args.metadata)
    except FileNotFoundError as e:
        print(f"[FATAL] {e}", file=sys.stderr)
        sys.exit(2)
    columns_meta = meta["columns"]
    label_cols = meta["label_cols"]
    flow_summary = meta["flow_summary"]
    notes = meta["notes"]
    print(f"  列清单：{', '.join(columns_meta.keys())}")
    print(f"  标注列：{label_cols}")

    if not columns_meta:
        print("[FATAL] 元信息中缺少 `## 列清单`，无法确定列序", file=sys.stderr)
        sys.exit(2)

    # 确定输出目录
    input_path = Path(args.input_csv)
    if not input_path.exists():
        print(f"[FATAL] 输入 CSV 不存在: {args.input_csv}", file=sys.stderr)
        sys.exit(2)
    if args.output_dir:
        output_dir = Path(args.output_dir)
    else:
        output_dir = input_path.parent
    output_dir.mkdir(parents=True, exist_ok=True)

    csv_name = args.output or "final_data.csv"
    report_name = args.report_output or "final_report.md"

    # 读取数据
    print(f"\n读取输入数据：{args.input_csv}")
    df = pd.read_csv(args.input_csv, dtype=str, keep_default_na=False, na_values=[])
    print(f"  行数：{len(df)}，列数：{len(df.columns)}")

    if "time" not in df.columns:
        print("[FATAL] 输入 CSV 中缺少 `time` 列", file=sys.stderr)
        sys.exit(2)

    # 1. 全空行切分
    print("\n[1/3] 全空行切分...")
    if args.no_split:
        print("  已禁用切分（--no-split）")
        segments = [df]
        split_info = {
            "total_segments": 1,
            "details": [{"segment": 1, "start": str(df["time"].iloc[0]),
                         "end": str(df["time"].iloc[-1]), "rows": len(df)}],
            "note": "已禁用切分（--no-split）",
        }
    else:
        segments, split_info = split_by_full_empty_rows(df)
    print(f"  切分为 {split_info['total_segments']} 段")
    for d in split_info["details"]:
        print(f"    段{d['segment']}: {d['start']} ~ {d['end']} ({d['rows']} 行)")

    # 2. 列排序
    print("\n[2/3] 列排序...")
    label_set = set(label_cols)
    reordered_segments = []
    for seg in segments:
        reordered = reorder_columns(seg, columns_meta, label_set)
        reordered_segments.append(reordered)
    print(f"  列顺序：{list(reordered_segments[0].columns)}")

    # 3. 质量校验
    print("\n[3/3] 质量校验...")
    if len(reordered_segments) == 1:
        qc_df = reordered_segments[0]
    else:
        qc_df = pd.concat(reordered_segments, ignore_index=True)
    qc_results = quality_check(qc_df)
    for level, msg in qc_results:
        prefix = {"error": "[E]", "warning": "[W]", "info": "[I]"}.get(level, "")
        print(f"  {prefix} {msg}")

    # 写入输出
    output_files = []
    if len(reordered_segments) == 1:
        csv_path = str(output_dir / csv_name)
        reordered_segments[0].to_csv(csv_path, index=False, encoding="utf-8-sig")
        output_files.append(csv_path)
        print(f"\n最终数据已写入：{csv_path}")
        print(f"  行数：{len(reordered_segments[0])}，列数：{len(reordered_segments[0].columns)}")
    else:
        base, ext = os.path.splitext(csv_name)
        for i, seg in enumerate(reordered_segments):
            seg_name = f"{base}_{i + 1:03d}{ext}"
            csv_path = str(output_dir / seg_name)
            seg.to_csv(csv_path, index=False, encoding="utf-8-sig")
            output_files.append(csv_path)
            print(f"\n段{i + 1}已写入：{csv_path} ({len(seg)} 行)")

    # 生成最终报告
    report_path = str(output_dir / report_name)
    generate_final_report(
        reordered_segments, split_info, qc_results, columns_meta,
        label_set, list(reordered_segments[0].columns),
        flow_summary, notes,
        report_path, args.input_csv, output_files
    )
    print(f"最终报告已生成：{report_path}")

    return 0


if __name__ == "__main__":
    sys.exit(main())
