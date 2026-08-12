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

整数输入按 dtype 最大值归一化，计算 `image ** gamma`，取整并恢复原 dtype 值域；浮点输入裁剪到 `[0, 1]`。传给 `enhance()` 的 `gamma` 仅覆盖本次调用，不修改已存值。小于 `1` 的值提亮图像。工厂创建会忽略不支持的构造键并发出 `UserWarning`。

## Parameters

| 参数 | 类型 | 默认值 | 含义 | 约束 |
| --- | --- | --- | --- | --- |
| `gamma` | `float` | `0.6` | 幂律指数。 | 必须为 `int` 或 `float` 且 `> 0`，否则抛 `TypeError` 或 `ValueError`；可运行时覆盖。 |
| `output_type` | `Literal["numpy", "pil", "bytes", "base64", "file"]` | `"numpy"` | 基类输出格式。 | 其他值抛 `ValueError`。 |
| `keep_dtype` | `bool` | `True` | 将输出转回输入 dtype。 | 非 `bool` 抛 `TypeError`。 |
| `clip_output` | `bool` | `True` | 裁剪到目标 dtype 有效范围。 | 非 `bool` 抛 `TypeError`。 |

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
