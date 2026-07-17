# 自定义传统增强算法

传统低层视觉方法继承 `openLLV/tradition/algorithms/BaseModel.py` 中的 `LLVEnhancer`。请将实现放在与其领域匹配的包中，例如 `BaseMethods`、`Dehazing` 或 `LLIE`。

## 1. 基类约定

`LLVEnhancer` 提供图像加载、BGR NumPy 转换、验证、数据类型保留、裁剪、输出转换、自动注册和工厂构建。子类只需实现 `_enhance()` 以及自己的参数。

## 2. 最小算法示例

```python
from typing import Any

import numpy as np

from openLLV.tradition.algorithms import LLVEnhancer


class MyAlgorithm(LLVEnhancer):
    name = "my_algorithm"
    aliases = ["myalgo"]

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

`_enhance()` 接收的三通道数组采用 OpenCV 风格的 BGR 顺序。请返回 NumPy 数组；基类负责裁剪和可选的数据类型恢复。

## 3. 基础选项

| 选项 | 默认值 | 含义 |
| --- | --- | --- |
| `output_type` | `"numpy"` | `numpy`、`pil`、`bytes`、`base64` 或 `file` |
| `keep_dtype` | `True` | 将结果转换回输入数据类型 |
| `clip_output` | `True` | 将结果裁剪到数据类型的有效范围 |

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

