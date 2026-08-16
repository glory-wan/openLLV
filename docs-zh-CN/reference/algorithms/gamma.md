# Gamma

> 文档分组：传统低光图像增强

Gamma 将图像归一化至 `[0, 1]` 后执行逐通道幂律校正。

## Links

| 类型 | URL |
| --- | --- |
| 论文 | 无 |
| 官方源码 | 无 |
| 官方项目页 | 无 |

## Location in openLLV

| 项目 | 位置 |
| --- | --- |
| 实现 | `openLLV/tradition/algorithms/LLIE/Gamma.py` |
| 类名 | `Gamma` |
| 注册名 | `Gamma`（无别名；查找忽略大小写并去除首尾空白） |
| 基类 | `openLLV/tradition/algorithms/BaseModel.py` 中的 `LLVEnhancer` |

## Implementation Notes

公共三通道输入和 NumPy 输出使用 RGB；基类在 `_enhance()` 边界完成内部 BGR 往返转换。基类会解析输入的语义值域，将其映射到 Gamma 的 `[0,1]` 浮点工作值域，并在处理后恢复相同的源值域与 dtype。因此浮点 `[0,1]` 和浮点 `[0,255]` 输入会分别保持自己的输出约定。传给 `enhance()` 的 `gamma` 仅覆盖本次调用，不修改已存值。小于 `1` 的值提亮图像。工厂创建会忽略不支持的构造键并发出 `UserWarning`。

## Parameters

| 参数 | 类型 | 默认值 | 含义 | 约束 |
| --- | --- | --- | --- | --- |
| `gamma` | `float` | `0.6` | 幂律指数。 | 必须为 `int` 或 `float` 且 `> 0`，否则抛 `TypeError` 或 `ValueError`；可运行时覆盖。 |
| `output_type` | `Literal["numpy", "pil", "bytes", "base64", "file"]` | `"numpy"` | 基类输出格式。 | 其他值抛 `ValueError`。 |
| `keep_dtype` | `bool` | `True` | 将输出转回输入 dtype。 | 非 `bool` 抛 `TypeError`。 |
| `clip_output` | `bool` | `True` | 裁剪到解析后的输入值域。 | 非 `bool` 抛 `TypeError`。 |
| `value_range` | `"auto" \| "unit" \| "byte" \| Tuple[float, float] \| List[float]` | `"auto"` | 输入值域；自动区分通常的浮点 `[0,1]` 与 `[0,255]` 输入。 | 最大值 `<= 1` 的暗部字节值域浮点图须用 `"byte"`。自定义边界须有限且递增；图像值越界抛 `ValueError`。 |

## Usage Example

```python
import openLLV as llv
enhanced, saved_path = llv.predict("Gamma", "input.jpg", output="results/gamma/output.png", gamma=0.45)
```

```python
from openLLV.tradition.algorithms.LLIE.Gamma import Gamma
enhancer = Gamma(gamma=0.7)
result = enhancer("input.jpg", gamma=0.5)
```
