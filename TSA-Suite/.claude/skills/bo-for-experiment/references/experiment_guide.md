# 实验容错与健壮性指南

本文档描述 `bo4experiment` 技能在实际使用中的常见问题、预防策略和处理方式。

---

## 常见问题与处理

### 1. 样本数不足（< 5 条历史数据）

**现象**：代理模型（高斯过程）在极少数据点时拟合不稳定，推荐结果随机性强。

**策略**：
- 在 `bo_engine.py` 的 `suggest_next()` 中，`len(records) < 5` 时必须打印警告
- 不阻断执行，但警告需醒目（前缀 `⚠️  [警告]`）
- 建议用户先用随机/均匀设计积累至少 5-10 条数据后再启用 BO

**代码示例**：
```python
if len(records) < 5:
    print(f"⚠️  [警告] 当前历史数据仅 {len(records)} 条 (<5)，"
          f"代理模型可能不稳定，推荐结果仅供参考。")
```

---

### 2. 参数类型解析错误

**现象**：LLM 将连续参数识别为整数型，或未能识别对数尺度参数。

**策略**：
- `space_parser.py` 的 system prompt 中包含明确的类型判断规则：
  - 浓度、pH、学习率等 → 优先考虑 `pow` 类型
  - 步骤数、批次数等 → `int` 类型
  - 催化剂种类、溶剂类型等 → `cat` 类型
- 《寻优任务建议书》中明确展示类型，让用户有机会纠正
- 用户确认后才写文件，杜绝错误参数空间被持久化

---

### 3. 目标值方向混淆

**现象**：用户说"最大化转化率"，但 HEBO 内部只支持最小化。

**策略**：
- 在 `history.json` 的 `objectives` 中记录 `direction: "max"` 或 `"min"`
- `bo_engine.py` 构建 Y 矩阵时，对 `max` 目标乘以 `-1` 后传入 HEBO
- 输出推荐结果时，不展示取负的内部值，仅展示参数

---

### 4. 历史记录的数据类型一致性

**现象**：用户在不同批次输入数据时，参数名或单位不一致（如 `temp` vs `temperature`）。

**策略**：
- `history_manager.py` 的 `append_records()` 中，检查 `X_new` 的键名是否与 `params_config` 中的 `name` 完全匹配
- 发现不匹配时，打印清晰的错误信息并退出，不写脏数据：
  ```
  ❌ [错误] 输入参数 'temp' 不在参数空间中。
     已知参数名：temperature, pressure, catalyst_amount
     请检查 --x_new 中的键名是否与初始化时一致。
  ```

---

### 5. `opt.observe()` 报错（数据类型问题）

**现象**：DataFrame 中包含 `None`、`NaN` 或类型不符的值导致 HEBO 崩溃。

**策略**：
- `bo_engine.py` 在调用 `opt.observe()` 前，校验 X 和 Y 中无 NaN
- 发现 NaN 时，打印哪条记录有问题，并建议用户修复后重试：
  ```
  ❌ [错误] 第 3 条历史记录中 'pressure' 值为 NaN，无法拟合模型。
     请检查 {task_id}_history.json 的 records[2] 字段。
  ```

---

### 6. 多目标 Pareto 前沿计算

**注意事项**：
- 当 `num_obj >= 2` 时，`opt.y` 中存储的是内部最小化后的值（max 目标已取负）
- Pareto 分析时需注意还原符号
- 使用 rank-sum 方法选取均衡解，避免单一目标的极端解

---

### 7. HEBO 导入失败

**现象**：`ImportError: No module named 'hebo'`

**策略**：在 `bo_engine.py` 顶部加保护性导入：
```python
try:
    from hebo.design_space.design_space import DesignSpace
    from hebo.optimizers.general import GeneralBO
except ImportError as e:
    raise ImportError(
        "HEBO 未安装或路径错误。\n"
        "请执行：pip install -e path/to/HEBO\n"
        f"原始错误：{e}"
    )
```

---

### 8. JSON 文件读写失败

**现象**：历史文件被损坏或编码错误。

**策略**：
- `history_manager.py` 所有文件读写均使用 `encoding='utf-8'`
- 写文件前先写临时文件 `.tmp`，写完成后再重命名，防止中断导致文件损坏：
  ```python
  import tempfile, os, json, shutil
  tmp_path = history_path + '.tmp'
  with open(tmp_path, 'w', encoding='utf-8') as f:
      json.dump(history, f, indent=2, ensure_ascii=False)
  shutil.move(tmp_path, history_path)
  ```
