# 自定义传统增强算法

传统低层视觉方法继承 `openLLV/tradition/algorithms/BaseModel.py` 中的 `LLVEnhancer`。请将实现放在与其领域匹配的包中，例如 `BaseMethods`、`Dehazing` 或 `LLIE`。

## 1. 基类约定

`LLVEnhancer` 提供公共 RGB 图像读取/输出、内部 BGR 转换、值域归一化/恢复、验证、数据类型保留、裁剪、自动注册和工厂构建。子类只需实现 `_enhance()` 以及自己的参数。

## 2. 最小算法示例

```python
from typing import Any

import numpy as np

from openLLV.tradition.algorithms import LLVEnhancer


class MyAlgorithm(LLVEnhancer):
    name = "my_algorithm"
    aliases = ["myalgo"]
    working_range = "byte"

    def __init__(self, strength: float = 1.0, **kwargs: Any):
        super().__init__(**kwargs)
        if strength < 0:
            raise ValueError("strength must be non-negative.")
        self.strength = float(strength)

    def _enhance(self, image: np.ndarray, **kwargs: Any) -> np.ndarray:
        strength = float(kwargs.get("strength", self.strength))
        result = image.astype(np.float32) * strength
        return result

    def get_params(self):
        params = super().get_params()
        params["strength"] = self.strength
        return params
```

`_enhance()` 接收和返回的三通道数组采用 OpenCV 风格的 BGR 顺序。基类会在调用前把公共 RGB 输入转为 BGR，并在裁剪和可选的数据类型恢复后把结果转回 RGB；子类不要重复执行边界转换。

当 `_enhance()` 接收并返回 `[0,255]` 的 `uint8` 值时，设置 `working_range = "byte"`（默认值，适合 OpenCV 直方图方法）；当它接收并返回 `[0,1]` 的 `float32` 值时，设置 `working_range = "unit"`。基类会把调用方解析出的输入值域映射到该工作值域，并在算法执行后恢复调用方值域。

## 3. 基础选项

| 选项 | 默认值 | 含义 |
| --- | --- | --- |
| `output_type` | `"numpy"` | `numpy`、`pil`、`bytes`、`base64` 或 `file` |
| `keep_dtype` | `True` | 将结果转换回输入数据类型 |
| `clip_output` | `True` | 将结果裁剪到解析后的输入值域 |
| `value_range` | `"auto"` | `"auto"`、`"unit"`、`"byte"` 或自定义 `(min, max)`/`[min, max]`；控制输入值域解释 |

`"auto"` 将最大值 `<= 1` 的浮点数组解释为 `[0,1]`，将最大值 `<= 255` 的浮点数组解释为 `[0,255]`。极暗的字节值域浮点图也可能满足最大值 `<= 1`，此时应显式传 `value_range="byte"`。负值或大于 `255` 的浮点值需要显式提供有效的自定义值域；非有限输入值始终会被拒绝。

## 4. 注册与使用

从所在领域的 `__init__.py` 以及 `openLLV/tradition/algorithms/__init__.py` 中导出该类，使导入 openLLV 时触发注册。

```python
import openLLV as llv

enhanced, saved_path = llv.predict(
    "myalgo",
    "input.jpg",
    output="results/myalgo.png",
    strength=1.2,
)
```

也可以直接使用工厂：

```python
from openLLV.tradition.algorithms import LLVEnhancer

enhancer = LLVEnhancer.create_enhancer(
    "my_algorithm",
    output_type="pil",
    strength=1.2,
)
result = enhancer("input.jpg")
```

使用 `llv.list_algorithms()` 确认类名、声明名称和别名均已注册。
