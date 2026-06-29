# tsas-num-prep-merger 元信息格式规范

> 本文档定义 merger 接受的元信息 MD 文件的字段和格式。
> AI 负责将用户提供的任意来源信息整理成此格式后传给脚本。

---

## 1. 元信息文件是必须的

merger 必须知道每个文件的时间列名、时间类型和分片形态才能工作。这些信息无法从 CSV 数据本身可靠推断，必须通过元信息提供。

---

## 2. 元信息字段

| 字段 | 必须/可选 | 说明 |
|---|---|---|
| 分片形态 | 必须 | A/B/C/D/E/F 六种之一，决定合并策略 |
| 各文件时间列名 | 必须 | 每个文件用哪个列作为时间列 |
| 各文件时间类型 | 必须 | timestamp（时间戳）或 sequence（整数序号） |
| 各文件列清单 | 可选 | 列名+数据类型+单位，用于报告记录列来源 |
| 目标时区 | 可选 | 统一时间到指定时区（如 +08:00） |

---

## 3. 标准 MD 格式模板

元信息文件使用以下 Markdown 格式。必须提供分片形态和各文件小节，其余可省略。

```markdown
# 元信息

## 分片形态
C

## 文件：current_amps.csv
- 时间列：time
- 时间类型：timestamp
- 列清单：
  | 列名 | 数据类型 | 单位 |
  |---|---|---|
  | time | time | - |
  | current_amps | numeric_float | amps |

## 文件：temperature_celsius.csv
- 时间列：timestamp
- 时间类型：timestamp
- 列清单：
  | 列名 | 数据类型 | 单位 |
  |---|---|---|
  | timestamp | time | - |
  | temperature_celsius | numeric_float | celsius |

## 目标时区
+08:00
```

### 字段说明

- **分片形态**：单个字母（A/B/C/D/E/F）。含义见 `references/algorithm_spec.md` §4。
- **文件：xxx**：每个有效文件一个小节，标题包含文件名。小节内容：
  - `时间列`：该文件中作为时间列的列名
  - `时间类型`：`timestamp`（可解析为 datetime）或 `sequence`（单调递增整数）
  - `列清单`（可选）：该文件所有列的类型和单位表
- **目标时区**（可选）：如 `+08:00`、`UTC`。也可通过 `--timezone` CLI 参数提供。

### 格式细节

- `## 分片形态` 和 `## 文件：xxx` 是 merger 的解析锚点，必须精确匹配
- 文件小节标题格式：`## 文件：<文件名>`（注意冒号是全角或半角均可）
- 列清单表格是可选的，只用于报告中的来源记录；缺失不影响合并
- 空行忽略

---

## 4. 示例

### 示例 1：C 类分片（按列划分，各文件不同物理量）

```markdown
# 元信息

## 分片形态
C

## 文件：current_amps.csv
- 时间列：time
- 时间类型：timestamp

## 文件：label.csv
- 时间列：datetime
- 时间类型：timestamp

## 文件：temperature_celsius.csv
- 时间列：timestamp
- 时间类型：timestamp

## 文件：vibration_rms.csv
- 时间列：datetime
- 时间类型：timestamp
```

### 示例 2：B 类分片（按行划分，同结构文件），带列清单

```markdown
# 元信息

## 分片形态
B

## 文件：batch1.csv
- 时间列：time
- 时间类型：timestamp
- 列清单：
  | 列名 | 数据类型 | 单位 |
  |---|---|---|
  | time | time | - |
  | current_amps | numeric_float | amps |
  | label | label | - |

## 文件：batch2.csv
- 时间列：time
- 时间类型：timestamp
- 列清单：
  | 列名 | 数据类型 | 单位 |
  |---|---|---|
  | time | time | - |
  | current_amps | numeric_float | amps |
  | label | label | - |
```

### 示例 3：A 类（单文件），带目标时区

```markdown
# 元信息

## 分片形态
A

## 文件：sensor_data.csv
- 时间列：timestamp
- 时间类型：timestamp

## 目标时区
+08:00
```
