# tsas-num-prep-cleaner 元信息格式规范

> 本文档定义 cleaner 接受的元信息 MD 文件的字段和格式。
> AI 负责将用户提供的任意来源信息整理成此格式后传给脚本。

---

## 1. 元信息文件是必须的

cleaner 需要知道每列的**数据类型**、**单位**和**角色**才能正确执行类型归一、单位归一、标注编码和类别编码。这些信息必须通过元信息提供。

---

## 2. 元信息字段

| 字段 | 必须/可选 | 说明 |
|---|---|---|
| 列清单 | 必须 | 所有非时间列的列名、数据类型、单位、角色 |
| 标注列 | 可选 | 标注列名列表，覆盖列清单中角色字段 |
| 类别列 | 可选 | 类别列名列表，覆盖列清单中角色字段 |
| 目标单位 | 可选 | 列名 → 目标单位映射，指定后执行单位换算 |
| 类别映射 | 可选 | 列名 → 编码映射，指定后按映射编码 |
| 有效值域 | 可选 | 列名 → [min, max]，指定后开启异常值检测 |
| 哨兵值 | 可选 | 列名 → 值列表，指定后开启哨兵值检测 |

---

## 3. 标准 MD 格式模板

元信息文件使用以下 Markdown 格式。必须提供列清单表，其余可省略。

```markdown
# 元信息

## 列清单

| 列名 | 数据类型 | 单位 | 角色 |
|---|---|---|---|
| current_amps | numeric_float | amps | 业务数据 |
| label | category_int | - | 标注列 |
| temperature_celsius | numeric_float | celsius | 业务数据 |
| vibration_rms | numeric_float | rms | 业务数据 |

## 标注列
label

## 类别列
label

## 目标单位
current_amps: ma

## 类别映射
```json
{"label": {"正常": 0, "异常": 1}}
```

## 有效值域
```json
{"current_amps": [0, 100], "temperature_celsius": [-50, 200]}
```

## 哨兵值
```json
{"vibration_rms": [-99, -100]}
```
```

### 字段说明

- **列清单**：表格，每行一个非时间列，四列：
  - `列名`：列名
  - `数据类型`：`numeric_float` / `numeric_int` / `category_int` / `category_str` / `label` 等
  - `单位`：`-` 表示无单位
  - `角色`：`时间列` / `标注列` / `类别列` / `业务数据`
- **标注列**：逗号分隔的列名列表。指定的列角色会被设为"标注列"（覆盖列清单中的角色）。
- **类别列**：逗号分隔的列名列表。指定的列角色会被设为"类别列"。
- **目标单位**：`列名: 目标单位`，每行一个。指定后执行单位换算，未指定则不转换。
- **类别映射**：JSON 对象，`列名 → {原始值 → 编码}`。未指定则按首次出现顺序编码。
- **有效值域**：JSON 对象，`列名 → [min, max]`。指定后对超出范围的值置空。
- **哨兵值**：JSON 对象，`列名 → [值列表]`。指定后匹配哨兵值的单元格置空。

### 格式细节

- `## 列清单` 是 cleaner 的解析锚点，必须精确匹配
- 列清单表的列标题必须精确匹配：`列名 | 数据类型 | 单位 | 角色`
- 标注列/类别列/目标单位可同时通过元信息文件和 CLI 参数提供，CLI 参数优先级更高
- 类别映射/有效值域/哨兵值小节中的 JSON 可放在代码块（```json```）中或直接写
- 空行忽略

---

## 4. 示例

### 示例 1：完整元信息（带目标单位和值域）

```markdown
# 元信息

## 列清单

| 列名 | 数据类型 | 单位 | 角色 |
|---|---|---|---|
| current_amps | numeric_float | amps | 业务数据 |
| label | category_int | - | 标注列 |
| temperature_celsius | numeric_float | celsius | 业务数据 |
| vibration_rms | numeric_float | rms | 业务数据 |

## 目标单位
current_amps: ma
temperature_celsius: fahrenheit

## 有效值域
```json
{"current_amps": [0, 100], "temperature_celsius": [-50, 200]}
```
```

### 示例 2：最小元信息（仅列清单，全自动处理）

```markdown
# 元信息

## 列清单

| 列名 | 数据类型 | 单位 | 角色 |
|---|---|---|---|
| current_amps | numeric_float | amps | 业务数据 |
| label | category_int | - | 标注列 |
| temperature_celsius | numeric_float | celsius | 业务数据 |
| vibration_rms | numeric_float | rms | 业务数据 |
```

### 示例 3：带类别映射

```markdown
# 元信息

## 列清单

| 列名 | 数据类型 | 单位 | 角色 |
|---|---|---|---|
| status | category_str | - | 类别列 |
| value | numeric_float | v | 业务数据 |

## 类别映射
```json
{"status": {"Y": 0, "N": 1, "UNKNOWN": 2}}
```
```