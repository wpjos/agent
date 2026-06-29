"""
history_manager.py — 历史数据 JSON 的读/写/追加（零 BO 逻辑）

核心函数：
  init_history(task_id, description, params_config, objectives, constraints) -> dict
  load_history(task_id, data_dir=".") -> dict
  append_records(history, x_new, y_new) -> dict
  save_history(history, data_dir=".")

JSON 结构：
  {
    "task_id": "BO20260415_143022",
    "created_at": "2026-04-15T14:30:22",
    "description": "用户原始描述",
    "params_config": [...],
    "objectives": [...],
    "constraints": [...],
    "records": [
      {
        "index": 1,
        "timestamp": "2026-04-15T15:00:00",
        "x": {"temperature": 250, "pressure": 3},
        "y": {"conversion_rate": 0.62}
      },
      ...
    ]
  }
"""

import json
import os
import shutil
from datetime import datetime


# ─────────────────────────────────────────────────────────────────────────────
# 初始化
# ─────────────────────────────────────────────────────────────────────────────

def init_history(
    task_id: str,
    description: str,
    params_config: list,
    objectives: list,
    constraints: list,
) -> dict:
    """
    创建初始历史数据结构（内存中，不写文件）。

    Parameters
    ----------
    task_id       : 唯一任务 ID，如 "BO20260415_143022"
    description   : 用户原始自然语言描述
    params_config : HEBO 参数配置列表
    objectives    : 优化目标列表
    constraints   : 约束条件列表

    Returns
    -------
    dict : 初始化的历史数据结构
    """
    return {
        "task_id":      task_id,
        "created_at":   datetime.now().isoformat(timespec="seconds"),
        "description":  description,
        "params_config": params_config,
        "objectives":   objectives,
        "constraints":  constraints,
        "records":      [],
    }


def generate_task_id() -> str:
    """生成基于时间戳的唯一 Task_ID。"""
    return "BO" + datetime.now().strftime("%Y%m%d_%H%M%S")


# ─────────────────────────────────────────────────────────────────────────────
# 读取
# ─────────────────────────────────────────────────────────────────────────────

def load_history(task_id: str, data_dir: str = ".") -> dict:
    """
    从文件读取历史数据。

    Parameters
    ----------
    task_id  : 任务 ID
    data_dir : 历史文件所在目录，默认当前目录

    Returns
    -------
    dict : 历史数据结构

    Raises
    ------
    FileNotFoundError : 文件不存在时
    """
    path = _history_path(task_id, data_dir)
    if not os.path.exists(path):
        raise FileNotFoundError(
            f"历史文件不存在：{path}\n"
            f"请确认 Task_ID '{task_id}' 是否正确，或使用 --mode init 先创建任务。"
        )
    with open(path, encoding="utf-8") as f:
        return json.load(f)


# ─────────────────────────────────────────────────────────────────────────────
# 追加记录
# ─────────────────────────────────────────────────────────────────────────────

def append_records(history: dict, x_new: dict, y_new: dict) -> dict:
    """
    向历史数据追加一条新的实验记录（内存操作，不写文件）。

    Parameters
    ----------
    history : 当前历史数据结构（由 load_history 或 init_history 返回）
    x_new   : 新实验的参数值，dict，键名必须与 params_config 中的 name 一致
    y_new   : 新实验的目标值，dict，键名必须与 objectives 中的 name 一致

    Returns
    -------
    dict : 追加后的历史数据结构（原地修改并返回）

    Raises
    ------
    ValueError : 参数名或目标名不匹配时
    """
    # 校验参数名
    param_names = [p["name"] for p in history["params_config"]]
    for key in x_new:
        if key not in param_names:
            raise ValueError(
                f"输入参数 '{key}' 不在参数空间中。\n"
                f"已知参数名：{param_names}\n"
                "请检查 --x_new 中的键名是否与初始化时一致。"
            )
    for pname in param_names:
        if pname not in x_new:
            raise ValueError(
                f"缺少参数 '{pname}' 的值。\n"
                f"期望参数：{param_names}\n"
                f"实际提供：{list(x_new.keys())}"
            )

    # 校验目标名
    obj_names = [o["name"] for o in history["objectives"]]
    for key in y_new:
        if key not in obj_names:
            raise ValueError(
                f"目标值 '{key}' 不在目标列表中。\n"
                f"已知目标名：{obj_names}\n"
                "请检查 --y_new 中的键名是否与初始化时一致。"
            )
    for oname in obj_names:
        if oname not in y_new:
            raise ValueError(
                f"缺少目标 '{oname}' 的值。\n"
                f"期望目标：{obj_names}\n"
                f"实际提供：{list(y_new.keys())}"
            )

    # 追加记录
    index = len(history["records"]) + 1
    record = {
        "index":     index,
        "timestamp": datetime.now().isoformat(timespec="seconds"),
        "x":         dict(x_new),
        "y":         dict(y_new),
    }
    history["records"].append(record)
    return history


# ─────────────────────────────────────────────────────────────────────────────
# 保存
# ─────────────────────────────────────────────────────────────────────────────

def save_history(history: dict, data_dir: str = ".") -> str:
    """
    将历史数据写入文件（原子写入：先写 .tmp，再重命名）。

    Parameters
    ----------
    history  : 历史数据结构
    data_dir : 保存目录，默认当前目录

    Returns
    -------
    str : 保存的文件路径
    """
    os.makedirs(data_dir, exist_ok=True)
    path     = _history_path(history["task_id"], data_dir)
    tmp_path = path + ".tmp"

    with open(tmp_path, "w", encoding="utf-8") as f:
        json.dump(history, f, indent=2, ensure_ascii=False)

    shutil.move(tmp_path, path)
    return path


# ─────────────────────────────────────────────────────────────────────────────
# 工具函数
# ─────────────────────────────────────────────────────────────────────────────

def _history_path(task_id: str, data_dir: str) -> str:
    """返回历史文件的完整路径。"""
    return os.path.join(data_dir, f"{task_id}_history.json")


def get_record_count(history: dict) -> int:
    """返回当前历史记录条数。"""
    return len(history.get("records", []))
