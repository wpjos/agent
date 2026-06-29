#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
tsas-num-prep-merger: 时间序列数据合并对齐工具

读取原始数据文件 + 元信息，执行：
  3a. 单文件时间预处理（格式解析 + 精度恢复）
  2.  拼接合并（A/B/C/D/E/F 形态）
  3b. 全局时间治理（时区转换 + 排序）
输出合并后的 CSV 和 Markdown 报告。

用法:
    python merger.py <data_path> --metadata <metadata_path>
        [--output-dir <dir>] [--output <csv名>] [--report-output <报告名>]
        [--timezone <tz>] [--use-row-index]
"""

import argparse
import os
import re
import sys
from collections import Counter
from datetime import datetime, timedelta, timezone
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

DEFAULT_PREFIX = "merged_data"


# ============================================================
# 工具函数
# ============================================================

def is_na_token(val):
    if val is None:
        return True
    s = str(val).strip().lower()
    return s in NA_TOKENS


def format_size(size_bytes):
    if size_bytes < 1024:
        return f"{size_bytes} B"
    elif size_bytes < 1024 * 1024:
        return f"{size_bytes / 1024:.1f} KB"
    elif size_bytes < 1024 * 1024 * 1024:
        return f"{size_bytes / (1024 * 1024):.1f} MB"
    else:
        return f"{size_bytes / (1024 * 1024 * 1024):.2f} GB"


def remove_unit_parentheses(col_name):
    """去除列名中的单位括号（半角/全角），返回 (基础名, 单位或None)"""
    # 半角括号
    m = re.search(r'\(([^)]+)\)\s*$', col_name)
    if m:
        base = col_name[:m.start()].strip()
        return base if base else col_name, m.group(1).strip()
    # 全角括号
    m = re.search(r'[（(]([^）)]+)[）)]\s*$', col_name)
    if m:
        base = col_name[:m.start()].strip()
        return base if base else col_name, m.group(1).strip()
    return col_name, None


def normalize_col_name(col_name, is_time_col=False):
    """列名归一化"""
    if is_time_col:
        return "time"
    base, _ = remove_unit_parentheses(col_name)
    return base


# ============================================================
# 元信息解析
# ============================================================

class ProfilerInfo:
    """从元信息文件中提取的关键信息"""

    def __init__(self):
        self.shard_pattern = None
        self.shard_reason = None
        self.files = {}  # filename -> {time_col, time_type, columns: {name: {type, unit}}}

    def get_time_col(self, filename):
        info = self.files.get(filename)
        if info:
            return info.get("time_col")
        return None

    def get_column_types(self, filename):
        info = self.files.get(filename)
        if info:
            return info.get("columns", {})
        return {}


def parse_metadata(metadata_path):
    """
    解析元信息 MD 文件，提取分片形态、各文件时间列和类型、列清单。

    元信息文件格式见 references/metadata_spec.md。
    必须包含：分片形态、各文件的时间列和时间类型。
    可选包含：各文件列清单、目标时区。

    返回:
        info: ProfilerInfo 对象
    """
    if not os.path.exists(metadata_path):
        raise FileNotFoundError(f"元信息文件不存在: {metadata_path}")

    with open(metadata_path, "r", encoding="utf-8") as f:
        content = f.read()

    info = ProfilerInfo()

    # 解析分片形态
    m = re.search(r'## 分片形态\s*\n\s*([A-F])\s*(?:\n|$)', content)
    if m:
        info.shard_pattern = m.group(1)

    # 解析各文件小节
    # 格式：## 文件：xxx.csv\n- 时间列：col_name\n- 时间类型：timestamp/sequence\n[列清单表...]
    file_blocks = re.split(r'## 文件[：:](.+?)\n', content)
    # file_blocks[0] 是头部，之后交替是 filename 和 content

    for i in range(1, len(file_blocks), 2):
        filename = file_blocks[i].strip()
        block = file_blocks[i + 1] if i + 1 < len(file_blocks) else ""

        file_info = {"time_col": None, "time_type": None, "columns": {}}

        # 时间列
        m = re.search(r'-\s*\*?\*?时间列\*?\*?\s*[：:]\s*[`"]?(.+?)[`"]?\s*(?:\n|$)', block)
        if m:
            file_info["time_col"] = m.group(1).strip()

        # 时间类型
        m = re.search(r'-\s*\*?\*?时间类型\*?\*?\s*[：:]\s*[`"]?(.+?)[`"]?\s*(?:\n|$)', block)
        if m:
            file_info["time_type"] = m.group(1).strip()

        # 列清单表（可选，格式：| 列名 | 数据类型 | 单位 |）
        in_table = False
        for line in block.split("\n"):
            stripped = line.strip()
            if stripped.startswith("| 列名") or stripped.startswith("|列名"):
                in_table = True
                continue
            if in_table:
                if stripped.startswith("|---") or stripped.startswith("| ---"):
                    continue
                if stripped.startswith("|"):
                    parts = [p.strip() for p in stripped.split("|")]
                    if len(parts) >= 4:
                        col_name = parts[1].strip().strip("`")
                        col_type = parts[2].strip()
                        col_unit = parts[3].strip()
                        if col_name and col_name != "列名" and not col_name.startswith("---"):
                            if col_unit == "-":
                                col_unit = None
                            file_info["columns"][col_name] = {
                                "type": col_type,
                                "unit": col_unit
                            }
                else:
                    in_table = False

        info.files[filename] = file_info

    return info


# ============================================================
# 精度恢复
# ============================================================

class PrecisionRecoveryResult:
    def __init__(self):
        self.recovered = False
        self.repeat_count = None       # 众数重复次数
        self.recovered_interval = None # 恢复后的间隔（秒）
        self.recovered_times = None    # 恢复后的时间列表（仅 recovered=True 时有值）
        self.error = None


def _do_precision_recovery(time_list, filename):
    """对时间列做精度恢复（简洁实现）"""
    result = PrecisionRecoveryResult()

    if not time_list:
        return result

    # 过滤 NaT
    valid_times = [t for t in time_list if t is not None and not (hasattr(t, 'year') and pd.isna(t))]
    if not valid_times:
        return result

    # 统计每个时间戳的重复次数
    from collections import OrderedDict
    groups = OrderedDict()  # time -> count
    for t in time_list:
        if t is None or (hasattr(t, 'year') and pd.isna(t)):
            # NaT 作为单元素组
            groups[t] = groups.get(t, 0) + 1
        else:
            groups[t] = groups.get(t, 0) + 1

    repeat_counts = [c for c in groups.values() if c > 0]

    # 如果所有值只出现一次，无需恢复
    if max(repeat_counts) == 1:
        return result

    # 计算重复次数的众数
    repeat_counter = Counter(repeat_counts)
    mode_repeat = repeat_counter.most_common(1)[0][0]
    mode_ratio = repeat_counter[mode_repeat] / len(groups)

    # 众数占比 > 50% 才判定为精度缺失
    if mode_ratio <= 0.5:
        return result

    result.recovered = True
    result.repeat_count = mode_repeat

    # 尝试计算众数的间隔
    mode_td = _calc_interval(mode_repeat)
    if mode_td is None:
        result.recovered = False
        result.error = f"文件 {filename}：重复次数 {mode_repeat} 无法整除 60 秒（即使展开到微秒）"
        return result

    result.recovered_interval = mode_td.total_seconds()

    # 执行恢复：遍历原始列表，按连续相同值分组
    recovered_times = []
    used_times = set()
    i = 0
    while i < len(time_list):
        current_val = time_list[i]
        # 跳过 NaT
        if current_val is None or (hasattr(current_val, 'year') and pd.isna(current_val)):
            recovered_times.append(current_val)
            i += 1
            continue

        # 找连续相同组
        j = i
        while j < len(time_list) and time_list[j] == current_val:
            j += 1
        group_size = j - i

        # 计算该组间隔
        group_td = _calc_interval(group_size)
        if group_td is None:
            # 无法整除（即使微秒）：报错，不跳过
            result.error = f"文件 {filename}：时间戳 '{current_val}' 重复 {group_size} 次无法整除 60 秒（即使展开到微秒）"
            return result

        # 均匀分配
        for k in range(group_size):
            new_time = current_val + group_td * k
            if new_time in used_times:
                result.error = f"文件 {filename}：精度恢复后时间点重叠（{new_time}），数据可能存在矛盾"
                return result
            used_times.add(new_time)
            recovered_times.append(new_time)

        i = j

    result.recovered_times = recovered_times
    return result


def _calc_interval(n):
    """计算 60/N 的间隔，返回 timedelta 或 None（除不尽）"""
    for unit, multiplier in [("seconds", 1), ("milliseconds", 1000), ("microseconds", 1000000)]:
        total = 60 * multiplier
        if total % n == 0:
            interval = total // n
            return timedelta(**{unit: interval})
    return None


# ============================================================
# 文件处理
# ============================================================

class FileMergeRecord:
    """单文件的合并处理记录"""
    def __init__(self, filename):
        self.filename = filename
        self.original_row_count = 0
        self.time_col_original = None
        self.time_col_type = None
        self.precision_recovery = None
        self.timezone_detected = None
        self.unit_map = {}          # 原始列名 -> (归一名, 单位)
        self.errors = []
        self.warnings = []


def read_and_preprocess_file(filepath, profiler_info, user_timezone=None):
    """读取并预处理单个文件"""
    filename = os.path.basename(filepath)
    record = FileMergeRecord(filename)

    # 从 profiler 信息获取时间列
    time_col = None
    time_type = None
    if profiler_info:
        file_info = profiler_info.files.get(filename)
        if file_info:
            time_col = file_info.get("time_col")
            time_type = file_info.get("time_type")

    if not time_col:
        record.errors.append("无法从元信息获取时间列信息")
        return None, record

    record.time_col_original = time_col
    record.time_col_type = time_type

    # 读取 CSV
    try:
        df = pd.read_csv(filepath, dtype=str, keep_default_na=False, na_values=[])
    except Exception as e:
        record.errors.append(f"读取失败: {e}")
        return None, record

    record.original_row_count = len(df)

    if time_col not in df.columns:
        record.errors.append(f"时间列 '{time_col}' 不在文件中")
        return None, record

    # 检查单位行（表头下1~3行）
    unit_row_index = None
    if len(df) >= 3:
        for i in range(min(3, len(df))):
            row = df.iloc[i]
            non_num_count = sum(
                1 for v in row
                if str(v).strip() and not is_na_token(v)
                and _try_parse_number(v) is None
            )
            if non_num_count >= len(df.columns) * 0.7:
                unit_row_index = i
                break

    # 如果有单位行，移除并记录
    if unit_row_index is not None:
        unit_row = df.iloc[unit_row_index]
        df = df.drop(unit_row_index).reset_index(drop=True)

    # 解析时间列
    raw_times = df[time_col].tolist()

    # 先尝试 pd.to_datetime
    try:
        parsed_times = pd.to_datetime(raw_times, errors="coerce")
    except Exception:
        parsed_times = pd.Series([pd.NaT] * len(raw_times))

    # 确保 parsed_times 是 Series（pd.to_datetime 返回 Series 或 DatetimeIndex）
    if isinstance(parsed_times, pd.DatetimeIndex):
        parsed_times = pd.Series(parsed_times)

    # 检查解析成功率
    valid_count = parsed_times.notna().sum()
    if valid_count < len(raw_times) * 0.5:
        record.errors.append(f"时间列解析成功率过低 ({valid_count}/{len(raw_times)})")
        return None, record

    # 检测时区（在精度恢复之前完成，确保即使后续报错中止也能记录时区信息）
    has_timezone = parsed_times.dt.tz is not None
    tz_detected = str(parsed_times.dt.tz) if has_timezone else None

    # 记录时区信息（无论后续是否报错，都先写入 record）
    if has_timezone:
        record.timezone_detected = tz_detected
    else:
        record.timezone_detected = "未注明（默认 +08:00）"

    # 转为 naive datetime
    if has_timezone:
        naive_times = parsed_times.dt.tz_localize(None)
    else:
        naive_times = parsed_times

    # 精度恢复（仅对 timestamp 类型）
    if time_type in ("timestamp", None):
        pr_result = _do_precision_recovery(naive_times.tolist(), filename)
        record.precision_recovery = pr_result

        if pr_result.error:
            record.errors.append(pr_result.error)
            return None, record

        if pr_result.recovered and pr_result.recovered_times:
            naive_times = pd.Series(pr_result.recovered_times)

    # 赋值 time 列（先移除原始时间列避免冲突）
    if time_col != "time":
        df = df.drop(columns=[time_col])
    else:
        # 原列名就是 time，直接覆盖
        pass
    df["time"] = naive_times.values

    # 列名归一
    new_cols = {}
    for col in df.columns:
        if col == "time":
            new_cols[col] = ("time", None)
            continue
        base, unit = remove_unit_parentheses(col)
        # 检查 profiler 报告中是否有单位信息
        if not unit and profiler_info:
            file_info = profiler_info.files.get(filename, {})
            col_info = file_info.get("columns", {}).get(col, {})
            unit = col_info.get("unit")
        new_cols[col] = (base, unit)
        record.unit_map[col] = (base, unit)

    # 重命名列
    rename_map = {}
    for old_name, (new_name, _) in new_cols.items():
        if old_name != new_name:
            rename_map[old_name] = new_name
    if rename_map:
        df = df.rename(columns=rename_map)

    return df, record


def _try_parse_number(val):
    if val is None:
        return None
    s = str(val).strip()
    if not s:
        return None
    try:
        return int(s)
    except ValueError:
        pass
    try:
        return float(s)
    except ValueError:
        return None


# ============================================================
# 合并逻辑
# ============================================================

def merge_files(file_dfs, shard_pattern, records):
    """根据分片形态合并文件"""
    errors = []
    warnings = []

    if shard_pattern == "A" or len(file_dfs) == 1:
        # 单文件，无需合并
        merged = file_dfs[0]
        return merged, errors, warnings

    if shard_pattern == "B":
        # 按行划分：纵向拼接
        merged = pd.concat(file_dfs, ignore_index=True)
        return merged, errors, warnings

    if shard_pattern == "C":
        # 按列划分：time 并集 outer join
        return _merge_by_column(file_dfs, errors, warnings)

    if shard_pattern == "D":
        # 对齐行列划分
        # 先按列名集合分组
        col_groups = {}
        for i, df in enumerate(file_dfs):
            cols_key = frozenset(df.columns)
            if cols_key not in col_groups:
                col_groups[cols_key] = []
            col_groups[cols_key].append(df)

        # 组内按 C 列合并
        group_merged = []
        for cols_key, dfs in col_groups.items():
            if len(dfs) == 1:
                group_merged.append(dfs[0])
            else:
                merged_group, grp_errors, grp_warnings = _merge_by_column(dfs, errors, warnings)
                group_merged.append(merged_group)

        # 组间按 B 行合并
        merged = pd.concat(group_merged, ignore_index=True)
        return merged, errors, warnings

    if shard_pattern in ("E", "F"):
        # 不对齐：outer join + 重叠验证
        return _merge_unaligned(file_dfs, errors, warnings, records)

    # 默认：outer join
    return _merge_by_column(file_dfs, errors, warnings)


def _merge_by_column(file_dfs, errors, warnings):
    """C 类合并：time 并集 outer join"""
    # 收集所有时间点
    all_times = set()
    for df in file_dfs:
        if "time" in df.columns:
            all_times.update(df["time"].dropna().tolist())

    all_times_sorted = sorted(all_times)

    # 每个文件 reindex 到统一时间索引
    merged = pd.DataFrame({"time": all_times_sorted})

    for i, df in enumerate(file_dfs):
        df_indexed = df.set_index("time")
        # 只取非 time 列
        non_time_cols = [c for c in df_indexed.columns if c != "time"]
        df_indexed = df_indexed[non_time_cols]
        merged = merged.merge(df_indexed, left_on="time", right_index=True, how="left")

    return merged, errors, warnings


def _merge_unaligned(file_dfs, errors, warnings, records):
    """E/F 类合并：outer join + 重叠区验证"""
    # 先做重叠区验证
    for i in range(len(file_dfs)):
        for j in range(i + 1, len(file_dfs)):
            df_a = file_dfs[i]
            df_b = file_dfs[j]
            rec_a = records[i] if i < len(records) else None
            rec_b = records[j] if j < len(records) else None

            # 找重叠时间点
            times_a = set(df_a["time"].dropna().tolist()) if "time" in df_a.columns else set()
            times_b = set(df_b["time"].dropna().tolist()) if "time" in df_b.columns else set()
            overlap = times_a & times_b

            if not overlap:
                continue

            # 找共有列
            common_cols = set(df_a.columns) & set(df_b.columns) - {"time"}
            if not common_cols:
                continue

            # 验证重叠区
            df_a_idx = df_a.set_index("time")
            df_b_idx = df_b.set_index("time")

            for t in overlap:
                for col in common_cols:
                    val_a = df_a_idx.loc[t, col]
                    val_b = df_b_idx.loc[t, col]

                    # 跳过 NaN
                    if pd.isna(val_a) or pd.isna(val_b):
                        continue

                    # 字面相等比较
                    if str(val_a).strip() != str(val_b).strip():
                        name_a = rec_a.filename if rec_a else f"文件{i}"
                        name_b = rec_b.filename if rec_b else f"文件{j}"
                        errors.append(
                            f"重叠区验证失败：{name_a} 和 {name_b} 在时间 {t} 的列 '{col}' "
                            f"值不一致（'{val_a}' != '{val_b}'）"
                        )

    if errors:
        return None, errors, warnings

    # 验证通过，做 outer join
    return _merge_by_column(file_dfs, errors, warnings)


# ============================================================
# 时区处理
# ============================================================

def determine_target_timezone(records, user_timezone):
    """确定目标时区"""
    if user_timezone:
        return user_timezone, f"用户指定: {user_timezone}"

    # 检查数据中的时区
    has_utc8 = False
    all_tz = []
    for rec in records:
        if rec.timezone_detected:
            tz = rec.timezone_detected.lower()
            all_tz.append(tz)
            if "+08:00" in tz or "东八区" in tz or "未注明" in tz:
                has_utc8 = True

    if has_utc8:
        return "+08:00", f"数据中包含 +08:00（含未注明默认），统一到 +08:00"

    if all_tz:
        tz_counter = Counter(all_tz)
        mode_tz = tz_counter.most_common(1)[0][0]
        return mode_tz, f"取众数时区: {mode_tz}"

    return "+08:00", "未检测到时区信息，默认 +08:00"


# ============================================================
# 报告生成
# ============================================================

def generate_report(records, shard_pattern, merge_strategy, target_tz, tz_reason,
                    merged_df, col_sources, output_path, data_path, profiler_info=None):
    """生成合并报告"""
    lines = []
    lines.append("# 合并报告（tsas-num-prep-merger）\n")
    lines.append(f"> 生成时间：{datetime.now().strftime('%Y-%m-%d %H:%M:%S')}\n")

    # 概览
    lines.append("## 1. 概览\n")
    lines.append(f"- **分片形态**：{shard_pattern}")
    lines.append(f"- **实际合并策略**：{merge_strategy}")
    lines.append(f"- **合并后行数 / 列数**：{len(merged_df)} / {len(merged_df.columns)}")
    if "time" in merged_df.columns:
        time_col = merged_df["time"].dropna()
        if len(time_col) > 0:
            lines.append(f"- **时间范围**：{time_col.iloc[0]} ~ {time_col.iloc[-1]}")
    lines.append("")

    # 单文件时间预处理记录
    lines.append("## 2. 单文件时间预处理记录\n")
    for rec in records:
        lines.append(f"### 文件：{rec.filename}\n")
        lines.append(f"- **原始行数**：{rec.original_row_count}")
        lines.append(f"- **时间列**：`{rec.time_col_original}`（类型：{rec.time_col_type}）")

        if rec.precision_recovery and rec.precision_recovery.recovered:
            pr = rec.precision_recovery
            lines.append(f"- **精度恢复**：是（重复次数众数 N={pr.repeat_count}，恢复间隔 {pr.recovered_interval}s）")
        else:
            lines.append(f"- **精度恢复**：否")

        lines.append(f"- **检测到的时区**：{rec.timezone_detected}")

        if rec.errors:
            lines.append(f"- **错误**：")
            for err in rec.errors:
                lines.append(f"  - {err}")
        lines.append("")

    # 列名归一记录
    lines.append("## 3. 列名归一记录\n")
    has_renames = False
    lines.append("| 原始列名 | 归一列名 | 去除的单位 |")
    lines.append("|---|---|---|")
    for rec in records:
        for old_name, (new_name, unit) in rec.unit_map.items():
            if old_name != new_name or unit:
                has_renames = True
                lines.append(f"| {old_name} | {new_name} | {unit or '-'} |")
    if not has_renames:
        lines.append("| - | - | 无需归一 |")
    lines.append("")

    # 合并过程记录
    lines.append("## 4. 合并过程记录\n")
    lines.append(f"- **合并策略路径**：{merge_strategy}")
    lines.append(f"- **各文件来源**：")
    for rec in records:
        lines.append(f"  - {rec.filename}（{rec.original_row_count} 行）")
    lines.append("")

    # 时区处理记录
    lines.append("## 5. 时区处理记录\n")
    lines.append(f"- **检测到的时区**：{', '.join(r.timezone_detected or '未检测到' for r in records)}")
    lines.append(f"- **目标时区**：{target_tz}")
    lines.append(f"- **转换说明**：{tz_reason}")
    lines.append(f"- **输出时间列不带时区信息**（naive datetime）")
    lines.append("")

    # 合并后列清单
    lines.append("## 6. 合并后列清单\n")
    lines.append("| 列名 | 来源文件 | 说明 |")
    lines.append("|---|---|---|")
    # 从 profiler 报告收集标注列名
    label_cols = set()
    if profiler_info:
        for finfo in profiler_info.files.values():
            for cname, cinfo in finfo.get("columns", {}).items():
                if cinfo.get("type") == "label":
                    label_cols.add(cname)
    for col in merged_df.columns:
        sources = col_sources.get(col, [])
        sources_str = ", ".join(sources) if sources else "-"
        if col == "time":
            role = "时间列"
        elif col in label_cols:
            role = "标注列"
        else:
            role = "业务数据"
        lines.append(f"| {col} | {sources_str} | {role} |")
    lines.append("")

    # 时间范围
    lines.append("## 7. 时间范围\n")
    if "time" in merged_df.columns:
        time_col = merged_df["time"].dropna()
        if len(time_col) > 0:
            t_min = time_col.iloc[0]
            t_max = time_col.iloc[-1]
            lines.append(f"- **起始时间**：{t_min}")
            lines.append(f"- **结束时间**：{t_max}")
            if hasattr(t_max, "year") and hasattr(t_min, "year"):
                span = t_max - t_min
                lines.append(f"- **总时间跨度**：{span}")
    lines.append("")

    # 错误/警告
    lines.append("## 8. 错误/警告\n")
    all_errors = []
    all_warnings = []
    for rec in records:
        for err in rec.errors:
            all_errors.append(f"[{rec.filename}] {err}")
        for w in rec.warnings:
            all_warnings.append(f"[{rec.filename}] {w}")
    if all_errors:
        lines.append("### 错误（致命）\n")
        for err in all_errors:
            lines.append(f"- {err}")
    if all_warnings:
        lines.append("\n### 警告\n")
        for w in all_warnings:
            lines.append(f"- {w}")
    if not all_errors and not all_warnings:
        lines.append("无错误和警告。")
    lines.append("")

    report_text = "\n".join(lines)
    with open(output_path, "w", encoding="utf-8") as f:
        f.write(report_text)

    return report_text


# ============================================================
# 主流程
# ============================================================

def main():
    parser = argparse.ArgumentParser(description="时间序列数据合并对齐工具")
    parser.add_argument("data_path", help="原始数据文件或目录路径")
    parser.add_argument("--metadata", "-m", required=True,
                        help="元信息文件路径（格式见 references/metadata_spec.md）")
    parser.add_argument("--output-dir", "-o", default=None,
                        help="输出目录（缺省为数据同目录）")
    parser.add_argument("--output", default=None,
                        help="合并数据输出文件名（缺省 merged_data.csv）")
    parser.add_argument("--report-output", default=None,
                        help="合并报告输出文件名（缺省 merged_data_report.md）")
    parser.add_argument("--timezone", default=None,
                        help="目标时区（如 +08:00），覆盖元信息中的时区")
    parser.add_argument("--use-row-index", action="store_true",
                        help="声明用行号作为序号")

    args = parser.parse_args()

    # 解析元信息
    print(f"读取元信息：{args.metadata}")
    try:
        profiler_info = parse_metadata(args.metadata)
    except FileNotFoundError as e:
        print(f"错误：{e}", file=sys.stderr)
        sys.exit(1)

    if not profiler_info.shard_pattern:
        print("错误：元信息中未找到分片形态", file=sys.stderr)
        sys.exit(1)

    if not profiler_info.files:
        print("错误：元信息中未找到文件信息", file=sys.stderr)
        sys.exit(1)

    print(f"分片形态：{profiler_info.shard_pattern}")
    print(f"文件数量：{len(profiler_info.files)}")

    # 确定输出目录
    data_path = Path(args.data_path)
    if args.output_dir:
        output_dir = Path(args.output_dir)
    else:
        output_dir = data_path.parent if data_path.is_file() else data_path
    output_dir.mkdir(parents=True, exist_ok=True)

    # 发现文件（按文件名字典序）
    if data_path.is_dir():
        files = sorted(data_path.rglob("*.csv"), key=lambda p: str(p).lower())
    else:
        files = [data_path]

    # 过滤跳过的文件（profiler 报告中标记为跳过的）
    valid_filenames = set(profiler_info.files.keys())
    files = [f for f in files if os.path.basename(f) in valid_filenames]

    if not files:
        print("错误：未找到有效文件", file=sys.stderr)
        sys.exit(1)

    # 逐文件读取 + 预处理
    print("\n逐文件预处理...")
    file_dfs = []
    records = []
    for filepath in files:
        filename = os.path.basename(filepath)
        print(f"  处理：{filename}")
        df, record = read_and_preprocess_file(filepath, profiler_info, args.timezone)
        if df is None:
            print(f"  ❌ {filename}: {'; '.join(record.errors)}", file=sys.stderr)
            records.append(record)
            continue
        file_dfs.append(df)
        records.append(record)

    # 确定输出文件名
    csv_name = args.output or "merged_data.csv"
    report_name = args.report_output or "merged_data_report.md"

    # 检查是否有错误 —— 致命错误必须中止，不产出 CSV
    # 注意：即使报错中止，也要正确执行时区判定，使报告信息完整
    has_errors = any(rec.errors for rec in records)
    if has_errors:
        print("\n❌ 预处理阶段存在致命错误，流程中止（不产出合并数据）。", file=sys.stderr)
        target_tz, tz_reason = determine_target_timezone(records, args.timezone)
        report_path = str(output_dir / report_name)
        generate_report(records, profiler_info.shard_pattern or "?", "未执行（预处理失败）",
                        target_tz, tz_reason, pd.DataFrame(), {}, report_path, data_path, profiler_info)
        print(f"报告已生成：{report_path}")
        sys.exit(2)

    if not file_dfs:
        print("\n错误：无有效文件可合并", file=sys.stderr)
        target_tz, tz_reason = determine_target_timezone(records, args.timezone)
        # 仍然生成报告
        report_path = str(output_dir / report_name)
        generate_report(records, profiler_info.shard_pattern or "?", "未执行（预处理失败）",
                        target_tz, tz_reason, pd.DataFrame(), {}, report_path, data_path, profiler_info)
        print(f"报告已生成：{report_path}")
        sys.exit(2)

    # 确定目标时区
    target_tz, tz_reason = determine_target_timezone(records, args.timezone)
    print(f"\n目标时区：{target_tz}（{tz_reason}）")

    # 合并
    print(f"\n执行合并（形态 {profiler_info.shard_pattern}）...")
    merged_df, merge_errors, merge_warnings = merge_files(
        file_dfs, profiler_info.shard_pattern, records
    )

    if merge_errors:
        print("\n❌ 合并阶段存在错误：", file=sys.stderr)
        for err in merge_errors:
            print(f"  - {err}", file=sys.stderr)

        # 仍然生成报告
        report_path = str(output_dir / report_name)
        generate_report(records, profiler_info.shard_pattern or "?", "合并失败",
                        target_tz, tz_reason, pd.DataFrame(), {}, report_path, data_path, profiler_info)
        print(f"报告已生成：{report_path}")
        sys.exit(2)

    # 排序
    if "time" in merged_df.columns:
        merged_df = merged_df.sort_values("time").reset_index(drop=True)

    # 构建列来源映射
    col_sources = {}
    for rec, df in zip(records, file_dfs):
        for col in df.columns:
            if col not in col_sources:
                col_sources[col] = []
            if rec.filename not in col_sources[col]:
                col_sources[col].append(rec.filename)

    # 确定合并策略描述
    strategy_map = {
        "A": "单文件，无需合并",
        "B": "按行划分 → 纵向拼接",
        "C": "按列划分 → time 并集 outer join",
        "D": "对齐行列 → 组内列合并 + 组间行合并",
        "E": "不对齐 → outer join + 重叠区验证",
        "F": "混合 → 分离对齐/不对齐分别处理"
    }
    merge_strategy = strategy_map.get(profiler_info.shard_pattern, "未知策略")

    # 写入合并后的 CSV
    csv_path = str(output_dir / csv_name)
    merged_df.to_csv(csv_path, index=False, encoding="utf-8-sig")
    print(f"\n合并数据已写入：{csv_path}")
    print(f"  行数：{len(merged_df)}，列数：{len(merged_df.columns)}")

    # 生成报告
    report_path = str(output_dir / report_name)
    generate_report(records, profiler_info.shard_pattern, merge_strategy,
                    target_tz, tz_reason, merged_df, col_sources, report_path, data_path, profiler_info)
    print(f"合并报告已生成：{report_path}")

    # 追加合并阶段的警告到 records
    if merge_warnings:
        for w in merge_warnings:
            records[0].warnings.append(w) if records else None

    # 最终错误检查
    all_errors = [err for rec in records for err in rec.errors] + merge_errors
    if all_errors:
        print("\n⚠️ 存在错误，请查看报告。", file=sys.stderr)
        return 2

    return 0


if __name__ == "__main__":
    sys.exit(main())
