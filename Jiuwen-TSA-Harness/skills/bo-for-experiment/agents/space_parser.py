"""
space_parser.py — 自然语言 → HEBO DesignSpace 解析器

核心功能：
  - parse_space(client, description) -> dict
    调用 LLM 将用户自然语言描述解析为结构化的 HEBO 参数配置
  - format_proposal_table(parsed) -> str
    将解析结果格式化为 Markdown《寻优任务建议书》

依赖：openai（LLM 客户端，OpenAI 兼容协议）
"""

import json
import os
import re


# ─────────────────────────────────────────────────────────────────────────────
# LLM 工具函数（轻量版，不依赖 _llm_utils 模块）
# ─────────────────────────────────────────────────────────────────────────────

def _get_client():
    """创建 OpenAI 兼容客户端，从环境变量读取配置。"""
    try:
        from openai import OpenAI
    except ImportError:
        raise ImportError("请安装 openai：pip install openai")
    api_key  = os.environ.get("OPENAI_API_KEY", "")
    base_url = os.environ.get("OPENAI_BASE_URL") or None
    if not api_key:
        raise EnvironmentError(
            "未设置 OPENAI_API_KEY 环境变量。\n"
            "请执行：export OPENAI_API_KEY=your_key"
        )
    return OpenAI(api_key=api_key, base_url=base_url)


def _call_llm(client, system: str, user: str, max_tokens: int = 2000) -> str:
    """调用 LLM，返回原始文本响应。"""
    model       = os.environ.get("OPENAI_MODEL", "gpt-4o")
    temperature = float(os.environ.get("OPENAI_TEMPERATURE", "0.2"))
    resp = client.chat.completions.create(
        model=model,
        max_tokens=max_tokens,
        temperature=temperature,
        messages=[
            {"role": "system", "content": system},
            {"role": "user",   "content": user},
        ],
    )
    return resp.choices[0].message.content.strip()


def _strip_code_fence(text: str) -> str:
    """去除 LLM 输出中的 ```json / ``` 代码块包裹。"""
    for marker in ["```json", "```python", "```"]:
        if marker in text:
            text = text.split(marker, 1)[1]
            if "```" in text:
                text = text.split("```")[0]
            break
    return text.strip()


def _parse_json_with_retry(client, system: str, user: str,
                            max_retries: int = 3) -> dict:
    """
    调用 LLM 并解析 JSON，最多重试 max_retries 次。
    返回解析后的 dict，失败则抛出异常。
    """
    last_err = None
    for attempt in range(max_retries):
        try:
            raw  = _call_llm(client, system, user)
            text = _strip_code_fence(raw)
            return json.loads(text)
        except json.JSONDecodeError as e:
            last_err = e
            if attempt < max_retries - 1:
                # 在 user prompt 后追加错误提示，引导 LLM 修正
                user = user + f"\n\n[上一次响应 JSON 解析失败：{e}，请重新生成合法 JSON，不要有多余文字]"
    raise ValueError(f"LLM JSON 解析失败（{max_retries} 次重试后仍然失败）：{last_err}")


# ─────────────────────────────────────────────────────────────────────────────
# System Prompt
# ─────────────────────────────────────────────────────────────────────────────

_SYSTEM_PROMPT = """你是一个实验设计专家，擅长将科研人员的自然语言实验描述转化为结构化的贝叶斯优化参数配置。

你的任务：
1. 解析用户描述中的实验参数空间（名称、类型、范围/类别）
2. 识别优化目标（名称、方向：min 或 max）
3. 提取约束条件（如有）

**HEBO 参数类型规则**：
- `num`：连续数值型（最常用），字段：name, type, lb, ub
- `int`：整数型（步骤数、批次数等离散整数），字段：name, type, lb, ub
- `pow`：对数尺度连续型（浓度、pH、学习率等跨越多个数量级的参数），字段：name, type, lb, ub, base（默认10）
- `cat`：类别型（材料种类、溶剂类型等），字段：name, type, categories（字符串列表，必须完整枚举）
- `bool`：布尔型（是/否），字段：name, type

**类型判断优先级**：
- 浓度 / pH / 学习率 / 跨越 2+ 个数量级的正值 → `pow`（base=10）
- 步骤数 / 批次数 / 整数计数 → `int`
- 材料种类 / 溶剂 / 气氛 / 方法名 → `cat`
- 是否 / 开关 → `bool`
- 其余连续量（温度、压力、时间、质量等）→ `num`

**输出格式**（严格 JSON，不要有任何额外文字）：
```json
{
  "params_config": [
    {"name": "参数名", "type": "num", "lb": 下界, "ub": 上界},
    {"name": "参数名", "type": "cat", "categories": ["A", "B", "C"]},
    {"name": "参数名", "type": "pow", "lb": 0.001, "ub": 1.0, "base": 10},
    {"name": "参数名", "type": "bool"}
  ],
  "objectives": [
    {"name": "目标名", "direction": "max", "description": "简短说明"},
    {"name": "目标名", "direction": "min", "description": "简短说明"}
  ],
  "constraints": [
    {"description": "约束的自然语言说明", "expression": "可选的数学表达式，如 x1 + x2 <= 10"}
  ]
}
```

**注意**：
- 参数名使用英文小写+下划线（如 temperature, reaction_time）
- 目标名使用英文小写+下划线
- `cat` 类型必须列出用户描述中提到的所有类别
- `pow` 类型的 lb 和 ub 为真实物理值（不是对数值）
- 如果没有约束，"constraints" 为空列表 []
"""


# ─────────────────────────────────────────────────────────────────────────────
# 核心解析函数
# ─────────────────────────────────────────────────────────────────────────────

def parse_space(client, description: str) -> dict:
    """
    将用户自然语言描述解析为结构化的 HEBO 参数配置。

    Parameters
    ----------
    client      : OpenAI 兼容客户端
    description : 用户自然语言描述

    Returns
    -------
    dict with keys:
      - params_config : list of HEBO param dicts
      - objectives    : list of {"name", "direction", "description"}
      - constraints   : list of {"description", "expression"}
    """
    user_prompt = f"请解析以下实验描述，输出符合规范的 JSON：\n\n{description}"
    parsed = _parse_json_with_retry(client, _SYSTEM_PROMPT, user_prompt)

    # 基本结构校验
    if "params_config" not in parsed or not parsed["params_config"]:
        raise ValueError("解析结果缺少 'params_config' 字段或为空，请检查描述是否包含参数信息。")
    if "objectives" not in parsed or not parsed["objectives"]:
        raise ValueError("解析结果缺少 'objectives' 字段或为空，请检查描述是否包含优化目标。")

    # 确保 constraints 字段存在
    parsed.setdefault("constraints", [])

    return parsed


# ─────────────────────────────────────────────────────────────────────────────
# 格式化输出：《寻优任务建议书》
# ─────────────────────────────────────────────────────────────────────────────

def format_proposal_table(parsed: dict) -> str:
    """
    将解析结果格式化为 Markdown《寻优任务建议书》。

    Parameters
    ----------
    parsed : parse_space() 的返回值

    Returns
    -------
    str : Markdown 格式的建议书文本
    """
    lines = []
    lines.append("=" * 60)
    lines.append("  [寻优任务建议书]")
    lines.append("=" * 60)
    lines.append("")

    # ── 参数空间表 ──────────────────────────────────────────────────────────
    lines.append("### 参数空间")
    lines.append("")
    lines.append("| 参数名 | 类型 | 下界 | 上界 / 类别 | 备注 |")
    lines.append("|--------|------|------|-------------|------|")

    for p in parsed["params_config"]:
        name = p["name"]
        ptype = p["type"]

        if ptype == "num":
            lb, ub = p["lb"], p["ub"]
            upper  = str(ub)
            note   = "连续数值"
        elif ptype == "int":
            lb, ub = p["lb"], p["ub"]
            upper  = str(ub)
            note   = "整数"
        elif ptype == "pow":
            lb, ub = p["lb"], p["ub"]
            base   = p.get("base", 10)
            upper  = str(ub)
            note   = f"对数尺度，base={base}"
        elif ptype == "cat":
            lb     = "—"
            upper  = " / ".join(f"'{c}'" for c in p.get("categories", []))
            note   = f"类别型（{len(p.get('categories', []))} 个）"
        elif ptype == "bool":
            lb     = "—"
            upper  = "True / False"
            note   = "布尔型"
        else:
            lb, ub = p.get("lb", "?"), p.get("ub", "?")
            upper  = str(ub)
            note   = ptype

        lines.append(f"| {name} | {ptype} | {lb} | {upper} | {note} |")

    lines.append("")

    # ── 优化目标表 ──────────────────────────────────────────────────────────
    lines.append("### 优化目标")
    lines.append("")
    lines.append("| 目标名 | 优化方向 | 说明 |")
    lines.append("|--------|---------|------|")

    for obj in parsed["objectives"]:
        direction_cn = "最大化 (max)" if obj["direction"] == "max" else "最小化 (min)"
        desc = obj.get("description", "")
        lines.append(f"| {obj['name']} | {direction_cn} | {desc} |")

    lines.append("")

    # ── 约束条件 ────────────────────────────────────────────────────────────
    lines.append("### 约束条件")
    lines.append("")
    constraints = parsed.get("constraints", [])
    if not constraints:
        lines.append("（无约束）")
    else:
        lines.append("| # | 约束说明 | 表达式 |")
        lines.append("|---|---------|--------|")
        for i, c in enumerate(constraints, 1):
            expr = c.get("expression", "—")
            lines.append(f"| {i} | {c['description']} | `{expr}` |")

    lines.append("")
    lines.append("=" * 60)
    lines.append("  [提示] 请检查以上解析是否符合您的实验设计。")
    lines.append("  输入 \"确认\" 创建任务，或告诉我需要修改的地方。")
    lines.append("=" * 60)

    return "\n".join(lines)


# ─────────────────────────────────────────────────────────────────────────────
# 独立测试入口
# ─────────────────────────────────────────────────────────────────────────────

if __name__ == "__main__":
    description = (
        "催化剂合成实验：温度范围200-400°C，压力1-10bar，催化剂用量0.1-2g，"
        "溶剂选择（乙醇、丙酮、甲醇），是否搅拌（是/否）。"
        "目标最大化反应转化率，同时最小化副产物浓度。"
        "约束：温度高于300°C时必须使用乙醇。"
    )
    cli = _get_client()
    result = parse_space(cli, description)
    print(format_proposal_table(result))
    print("\n[原始 JSON]")
    print(json.dumps(result, indent=2, ensure_ascii=False))
