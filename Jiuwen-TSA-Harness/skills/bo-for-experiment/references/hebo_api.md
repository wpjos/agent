# HEBO API 接口说明

本文档记录 `bo4experiment` 技能所使用的 HEBO 核心接口，供 `bo_engine.py` 实现参考。

---

## DesignSpace

**导入路径**：`from hebo.design_space.design_space import DesignSpace`

### `DesignSpace.parse(params_config: list) -> DesignSpace`

将参数配置列表解析为 HEBO 设计空间对象。

**参数类型规范**：

| 类型 | 必填字段 | 可选字段 | 说明 |
|------|---------|---------|------|
| `num` | `name`, `type`, `lb`, `ub` | — | 连续数值型 |
| `int` | `name`, `type`, `lb`, `ub` | — | 整数型 |
| `pow` | `name`, `type`, `lb`, `ub` | `base`（默认10） | 对数尺度连续型 |
| `pow_int` | `name`, `type`, `lb`, `ub` | `base` | 对数尺度整数型 |
| `int_exponent` | `name`, `type`, `lb`, `ub` | `base` | 指数整数型 |
| `step_int` | `name`, `type`, `lb`, `ub`, `step` | — | 步长整数型 |
| `cat` | `name`, `type`, `categories` | — | 类别型（字符串列表） |
| `bool` | `name`, `type` | — | 布尔型（True/False） |

**示例**：
```python
from hebo.design_space.design_space import DesignSpace

params_config = [
    {'name': 'temperature', 'type': 'num',  'lb': 200,  'ub': 400},
    {'name': 'pressure',    'type': 'num',  'lb': 1,    'ub': 10},
    {'name': 'steps',       'type': 'int',  'lb': 1,    'ub': 5},
    {'name': 'lr',          'type': 'pow',  'lb': 1e-4, 'ub': 1e-1, 'base': 10},
    {'name': 'solvent',     'type': 'cat',  'categories': ['ethanol', 'acetone', 'methanol']},
    {'name': 'stir',        'type': 'bool'},
]
space = DesignSpace().parse(params_config)
```

**常用属性**：
- `space.num_paras`：参数总数
- `space.num_numeric`：数值型参数数量
- `space.num_categorical`：类别型参数数量
- `space.para_names`：参数名列表

---

## GeneralBO

**导入路径**：`from hebo.optimizers.general import GeneralBO`

### `GeneralBO.__init__(space, num_obj, num_constr, rand_sample)`

| 参数 | 类型 | 说明 |
|------|------|------|
| `space` | `DesignSpace` | 设计空间对象 |
| `num_obj` | `int` | 目标函数数量（≥1） |
| `num_constr` | `int` | 约束函数数量（bo4experiment 中通常为 0） |
| `rand_sample` | `int` | 随机初始化样本数（已有历史数据时设为较小值，如 5） |

### `opt.observe(X: pd.DataFrame, y: np.ndarray)`

注入观测数据，拟合代理模型。

- `X`：参数 DataFrame，列名必须与 `space.para_names` 一致
- `y`：目标值数组，shape `(n_samples, num_obj + num_constr)`
  - 前 `num_obj` 列为目标值（HEBO 统一**最小化**，`max` 目标需取负）
  - 后 `num_constr` 列为约束值（≤0 表示满足约束）

### `opt.suggest(n_suggestions: int) -> pd.DataFrame`

基于当前代理模型生成推荐点。

- 返回 DataFrame，列名与 `space.para_names` 一致
- 建议用 `redirect_stdout` 屏蔽 GPy 内部日志

**完整使用示例（"只观测不随机"模式）**：
```python
import os
import numpy as np
import pandas as pd
from contextlib import redirect_stdout
from hebo.design_space.design_space import DesignSpace
from hebo.optimizers.general import GeneralBO

params_config = [
    {'name': 'temperature', 'type': 'num', 'lb': 200, 'ub': 400},
    {'name': 'pressure',    'type': 'num', 'lb': 1,   'ub': 10},
]
space = DesignSpace().parse(params_config)

# 历史数据（X 为 DataFrame，y 为 ndarray）
X_history = pd.DataFrame({
    'temperature': [250, 300, 350],
    'pressure':    [3,   6,   8],
})
y_history = np.array([[0.7], [0.5], [0.6]])  # 单目标，已取负（原始为 max）

opt = GeneralBO(space=space, num_obj=1, num_constr=0, rand_sample=5)
opt.observe(X_history, y_history)

with open(os.devnull, 'w') as f, redirect_stdout(f):
    rec_df = opt.suggest(n_suggestions=3)

print(rec_df)
```

---

## 注意事项

1. **目标方向**：HEBO 只支持最小化。对 `max` 目标，传入 `y` 时取负值，展示时还原。
2. **`observe` 方法名**：确认为 `opt.observe(X, y)`，不是 `observe_new_data`。
3. **重新实例化**：每次迭代都需重新 `GeneralBO(...)` + `opt.observe(全量历史)`，不支持增量更新。
4. **类别变量**：`cat` 类型的 X 列传入字符串值即可，HEBO 内部自动编码。
5. **bool 变量**：传入 `True`/`False` 或 `1`/`0` 均可。
